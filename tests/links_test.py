"""Live task links are separate from successful result publication."""

import asyncio
import json
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from lxm3 import execution
from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.clusters import ssh
from lxm3.xm_cluster import experiment as experiment_lib
from lxm3.xm_cluster import inspection
from lxm3.xm_cluster.execution.slurm import SlurmJobScriptBuilder


@pytest.fixture(autouse=True)
def isolated_loop_policy(monkeypatch):
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    monkeypatch.delenv("LXM_PROJECT", raising=False)
    monkeypatch.delenv("LXM_CLUSTER", raising=False)
    monkeypatch.delenv("LXM_LINKS_FILE", raising=False)
    yield
    asyncio.set_event_loop_policy(previous)


def test_atomic_named_links_and_noop_outside_execution(tmp_path, monkeypatch):
    execution.link("unused", "https://example.org/outside")
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / "links.json"
    monkeypatch.setenv("LXM_LINKS_FILE", str(path))
    execution.link("wandb", "https://example.org/first")
    execution.link("report $ ' /", "https://example.org/report")
    execution.link("wandb", "https://example.org/replacement")
    assert json.loads(path.read_text()) == {
        "wandb": "https://example.org/replacement",
        "report $ ' /": "https://example.org/report",
    }
    with mock.patch("os.replace", side_effect=OSError("not published")):
        with pytest.raises(OSError):
            execution.link("wandb", "https://example.org/lost")
    assert json.loads(path.read_text())["wandb"].endswith("replacement")
    assert list(tmp_path.iterdir()) == [path]
    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text("{}")
    assert path.stat().st_mode == ordinary.stat().st_mode


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("array", [False, True])
def test_reopen_links_after_success_or_failure_with_task_isolation(
    tmp_path, exit_code, array
):
    store = tmp_path / "store $ ' \n literal"
    config = xc.Config({"local": {"storage": {"staging": str(store)}}})
    root = tmp_path / "work"
    experiment = xc.create_experiment("links", config=config)
    code = (
        "from lxm3 import execution; import os,sys; "
        "execution.link('report', 'https://example.org/' + os.environ['VALUE']); "
        f"sys.exit({exit_code})"
    )

    def launch():
        with experiment:
            source = xc.SourceTree(
                xc.CommandList(
                    [f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"]
                ),
                tmp_path,
                files=[],
            )
            [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
            executor = xc.Local(workdir_root=str(root))
            job = (
                xc.ArrayJob(
                    executable, executor, env_vars=[{"VALUE": "one"}, {"VALUE": "two"}]
                )
                if array
                else xm.Job(
                    executable,
                    executor,
                    env_vars={"VALUE": "one", "LXM_LINKS_FILE": "/wrong"},
                )
            )
            experiment.add(job)

    if exit_code:
        with pytest.raises(subprocess.CalledProcessError):
            launch()
    else:
        launch()
    unit = xc.get_experiment(experiment.experiment_id, config=config).work_units()[1]
    assert unit.get_links(task=0) == {"report": "https://example.org/one"}
    if array:
        assert unit.get_links(task=1) == {"report": "https://example.org/two"}
        with pytest.raises(ValueError, match="zero-based"):
            unit.get_links()
    else:
        assert unit.get_links() == unit.get_links(task=0)
    for task in (-1, 2):
        with pytest.raises(ValueError, match="range"):
            unit.get_links(task=task)
    assert list(root.iterdir()) == []
    assert unit.artifacts(task=0) == {}


def test_reader_uses_recorded_site_and_missing_receipt_is_empty():
    record = dict(
        task_count=2,
        links_directory="/shared/links",
        hostname="nersc-recorded",
        username="recorded-user",
    )
    with mock.patch.object(
        ssh,
        "run",
        return_value=SimpleNamespace(stdout='{"report":"https://example.org"}'),
    ) as call:
        assert inspection.get_links(record, task=1) == {"report": "https://example.org"}
        assert call.call_args.kwargs == {
            "hostname": "nersc-recorded",
            "username": "recorded-user",
        }
        assert call.call_args.args[0][-1] == "/shared/links/1/links.json"
    with mock.patch.object(ssh, "run", return_value=SimpleNamespace(stdout="")):
        assert inspection.get_links(record, task=0) == {}
    assert inspection.get_links({"task_count": 1}) == {}


def test_link_is_readable_in_another_process_before_payload_finishes(tmp_path):
    config = xc.Config({"local": {"storage": {"staging": str(tmp_path / "store")}}})
    release = tmp_path / "release"
    code = f"""
from lxm3 import execution
from pathlib import Path
import time
execution.link("report", "https://example.org/live")
deadline = time.monotonic() + 15
while not Path({str(release)!r}).exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("reader never released payload")
    time.sleep(0.02)
"""
    with xc.create_experiment("live-link", config=config) as experiment:
        source = xc.SourceTree(
            xc.CommandList([f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"]),
            tmp_path,
            files=[],
        )
        [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
        experiment.add(xm.Job(executable, xc.Local()))
        experiment._wait_for_tasks()
        unit = experiment.work_units()[1]
        try:
            deadline = time.monotonic() + 8
            while not unit.get_links():
                assert time.monotonic() < deadline, "payload did not report its link"
                time.sleep(0.02)
            reader = (
                "from lxm3 import xm_cluster as xc; "
                f"config = xc.Config({config._data!r}); "
                f"unit = xc.get_experiment({experiment.experiment_id}, config=config).work_units()[1]; "
                "assert unit.get_links() == {'report': 'https://example.org/live'}"
            )
            subprocess.run([sys.executable, "-c", reader], check=True, timeout=5)
        finally:
            release.touch()


@pytest.mark.parametrize("kind", ["singularity", "docker", "shifter"])
def test_container_script_exposes_retained_link_directory(kind, tmp_path):
    from lxm3.xm_cluster.executables import ContainerImage
    from lxm3.xm_cluster.executables import ContainerImageType

    bundle = xc.AppBundle(
        "test",
        "true",
        "/source.zip",
        container_image=ContainerImage("image", ContainerImageType(kind)),
    )
    script = SlurmJobScriptBuilder().build(
        xm.Job(bundle, xc.Slurm()), "test", str(tmp_path / "logs")
    )
    assert 'mkdir -p -- "$LXM_LINK_DIR"' in script
    if kind == "shifter":
        assert "--volume=" not in script
        assert str(tmp_path / "logs" / "links") in script
    else:
        assert "/run/lxm3/links" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
