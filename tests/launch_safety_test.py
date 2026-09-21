"""Execute generated scripts; inspect behavior rather than shell spelling."""

import asyncio
import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import fsspec
import pytest

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import executables
from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import local
from lxm3.xm_cluster.execution import slurm
from lxm3.xm_cluster.packaging import router


@pytest.fixture(autouse=True)
def isolated_event_loop_policy():
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    yield
    asyncio.set_event_loop_policy(previous)


def bundle(tmp_path, script):
    archive = tmp_path / "archive $literal ' with spaces.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        entry = zipfile.ZipInfo("entrypoint.sh")
        entry.external_attr = 0o755 << 16
        zipped.writestr(entry, script)
    return xc.AppBundle("probe", "./entrypoint.sh", str(archive))


@pytest.mark.parametrize(
    "executor,builder",
    [
        (xc.Local(), local.LocalJobScriptBuilder()),
        (xc.Slurm(), slurm.SlurmJobScriptBuilder()),
        (xc.GridEngine(), gridengine.GridEngineJobScriptBuilder()),
    ],
)
@pytest.mark.parametrize("array", [False, True])
def test_literal_arguments_and_environment(tmp_path, executor, builder, array):
    marker = tmp_path / "must-not-exist"
    values = [
        "",
        "two words",
        "quote'\"",
        "$HOME",
        f"$(touch {marker})",
        "a\nb",
        "*?[abc]",
        "\\",
    ]
    env_value = f"first\nEOF\n$(touch {marker})\nlast ' \" \\"
    probe = 'import json,os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps([sys.argv[2:],os.getenv("FOO")]))'
    executable = bundle(
        tmp_path,
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(probe)} "$@"\n',
    )
    outputs = [tmp_path / f"result-{index}.json" for index in range(2 if array else 1)]
    args = [[str(output), *values] for output in outputs]
    job = (
        xc.ArrayJob(executable, executor, args=args, env_vars=[{"FOO": env_value}] * 2)
        if array
        else xm.Job(executable, executor, args=args[0], env_vars={"FOO": env_value})
    )
    script = tmp_path / "job.sh"
    script.write_text(builder.build(job, "literal", str(tmp_path / "logs")))
    for index, output in enumerate(outputs, builder.ARRAY_TASK_OFFSET):
        subprocess.run(
            ["bash", str(script)],
            env={**os.environ, builder.ARRAY_TASK_ID: str(index)},
            check=True,
            capture_output=True,
        )
        assert json.loads(output.read_text()) == [values, env_value]
    assert not marker.exists()


@pytest.mark.parametrize(
    "executor_type,builder",
    [
        (xc.Local, local.LocalJobScriptBuilder()),
        (xc.Slurm, slurm.SlurmJobScriptBuilder()),
    ],
)
@pytest.mark.parametrize("custom_root", [False, True])
@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_workdir_placement_and_cleanup(
    tmp_path, executor_type, builder, custom_root, array, exit_code
):
    system_tmp = tmp_path / "system tmp"
    system_tmp.mkdir()
    root = tmp_path / "experiment ' $(touch unexpected)" if custom_root else system_tmp
    root.mkdir(exist_ok=True)
    sentinel = root / "keep.txt"
    sentinel.write_text("existing experiment data")
    executor = executor_type(workdir_root=str(root) if custom_root else None)
    executable = bundle(
        tmp_path,
        '#!/bin/sh\nprintf \'%s\\n\' "$PWD" "$TMPDIR" > "$1"\nexit "$2"\n',
    )
    outputs = [tmp_path / f"result-{i}" for i in range(2 if array else 1)]
    args = [[str(output), str(exit_code)] for output in outputs]
    job = (
        xc.ArrayJob(executable, executor, args=args)
        if array
        else xm.Job(executable, executor, args=args[0])
    )
    script = tmp_path / "job.sh"
    script.write_text(builder.build(job, "workdir", str(tmp_path / "logs")))
    workdirs = []
    for index, output in enumerate(outputs, builder.ARRAY_TASK_OFFSET):
        result = subprocess.run(
            ["bash", str(script)],
            cwd=tmp_path,
            env={
                **os.environ,
                "TMPDIR": str(system_tmp),
                builder.ARRAY_TASK_ID: str(index),
            },
            capture_output=True,
        )
        assert result.returncode == exit_code, result.stderr.decode()
        workdir, application_tmp = output.read_text().splitlines()
        workdir = Path(workdir)
        assert workdir.parent == root
        assert application_tmp == str(system_tmp)
        assert not workdir.exists()
        workdirs.append(workdir)
    assert len(set(workdirs)) == len(outputs)
    assert list(root.iterdir()) == [sentinel]
    assert sentinel.read_text() == "existing experiment data"
    assert not (tmp_path / "unexpected").exists()


@pytest.mark.parametrize(
    "executor_type,builder",
    [
        (xc.Local, local.LocalJobScriptBuilder()),
        (xc.Slurm, slurm.SlurmJobScriptBuilder()),
    ],
)
@pytest.mark.parametrize("blocked", [False, True])
def test_workdir_parent_setup(tmp_path, executor_type, builder, blocked):
    parent = tmp_path / "new parent"
    if blocked:
        parent.write_text("not a directory")
    executable = bundle(tmp_path, "#!/bin/sh\necho executed\n")
    executor = executor_type(workdir_root="new parent/nested")
    script = builder.build(xm.Job(executable, executor), "workdir", str(tmp_path))
    result = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True
    )
    if blocked:
        assert result.returncode != 0
        assert "executed" not in result.stdout
        assert parent.read_text() == "not a directory"
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "executed"
        assert list((parent / "nested").iterdir()) == []


@pytest.mark.parametrize("runtime", ["singularity", "docker"])
@pytest.mark.parametrize("custom_root", [False, True])
def test_container_mounts_and_image_are_literal(
    tmp_path, monkeypatch, runtime, custom_root
):
    capture = tmp_path / "container-argv.json"
    fake = tmp_path / runtime
    fake.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    source, destination = (
        "/data/space $literal ' quote",
        "/mount/space $literal ' quote",
    )
    image = "/images/space $literal ' quote.sif"
    executable = bundle(tmp_path, "#!/bin/sh\ntrue\n")
    executable.container_image = executables.ContainerImage(
        image, executables.ContainerImageType(runtime)
    )
    executor = xc.Local(
        singularity_options=xc.SingularityOptions(bind={source: destination}),
        docker_options=xc.DockerOptions(volumes={source: destination}),
        workdir_root=str(tmp_path / "work ' $literal") if custom_root else None,
    )
    script = tmp_path / "job.sh"
    script.write_text(
        local.LocalJobScriptBuilder().build(
            xm.Job(executable, executor), "mounts", str(tmp_path / "logs")
        )
    )
    subprocess.run(["bash", str(script)], check=True, capture_output=True)
    argv = json.loads(capture.read_text())
    expected = (
        f"--bind={source}:{destination}"
        if runtime == "singularity"
        else f"--mount=type=bind,source={source},target={destination}"
    )
    assert expected in argv
    assert image in argv
    assert not any("$LXM_WORKDIR" in arg for arg in argv)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_failure_propagates_and_closes_context(tmp_path, asynchronous):
    config = xc.Config({"local": {"storage": {"staging": str(tmp_path / "staging")}}})
    experiment = xc.create_experiment("failure", project="probe", config=config)
    executable = bundle(tmp_path, "#!/bin/sh\necho intentional-failure\nexit 7\n")
    job = xm.Job(executable, xc.Local())

    async def launch():
        async with experiment:
            await experiment.add(job)

    with pytest.raises(subprocess.CalledProcessError) as error:
        if asynchronous:
            asyncio.run(launch())
        else:
            with experiment:
                experiment.add(job)
    assert error.value.returncode == 7
    if not asynchronous:
        assert not experiment._event_loop_thread.is_alive()
        assert experiment._event_loop.is_closed()
    with pytest.raises(RuntimeError, match="requires an experiment context"):
        xc.get_current_experiment()
    logs = list((tmp_path / "staging").rglob("task-0.log"))
    assert len(logs) == 1
    assert "intentional-failure" in logs[0].read_text()


def test_staged_content_cannot_be_overwritten_by_a_later_package(tmp_path):
    store = artifacts.ArtifactStore(
        fsspec.filesystem("file"), str(tmp_path / "staging")
    )
    source = tmp_path / "same-name.zip"
    source.write_bytes(b"first")
    timestamp = source.stat().st_mtime
    first = router._transfer_file(store, str(source), "archives/same-name.zip")
    source.write_bytes(b"other")
    os.utime(source, (timestamp, timestamp))
    second = router._transfer_file(store, str(source), "archives/same-name.zip")
    assert first != second
    assert Path(first).read_bytes() == b"first"
    assert Path(second).read_bytes() == b"other"
    assert router._transfer_file(store, str(source), "archives/same-name.zip") == second


def test_failed_transfer_does_not_expose_partial_package(tmp_path, monkeypatch):
    store = artifacts.ArtifactStore(
        fsspec.filesystem("file"), str(tmp_path / "staging")
    )
    source = tmp_path / "source"
    source.write_bytes(b"complete")

    def interrupted(lpath, rpath):
        Path(rpath).write_bytes(b"partial")
        raise OSError("interrupted transfer")

    monkeypatch.setattr(store.filesystem, "put_file", interrupted)
    with pytest.raises(OSError, match="interrupted transfer"):
        store.put_file(str(source), "archives/final.zip")
    assert not store.exists("archives/final.zip")


@pytest.mark.parametrize(
    "resources,expected",
    [
        ({}, False),
        ({"gpus": 0}, False),
        ({"gpus": 1}, True),
        ({"gpus-per-node": "a100:4"}, True),
        ({"gres": "gpu"}, True),
        ({"gres": "gpu:a100:1"}, True),
        ({"gres": "gpu:0"}, False),
        ({"gres": "shard:4"}, False),
    ],
)
def test_slurm_gpu_flags_follow_requested_resources(resources, expected):
    assert (
        slurm.SlurmJobScriptBuilder._is_gpu_requested(xc.Slurm(resources=resources))
        is expected
    )
