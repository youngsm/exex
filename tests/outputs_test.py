"""Retained results survive working-directory cleanup and launcher exit."""

import asyncio
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest import mock

import pytest

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.clusters import ssh
from lxm3.xm_cluster import experiment as experiment_lib
from lxm3.xm_cluster import outputs
from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import slurm
from lxm3.xm_cluster.execution.output_capture import capture
from lxm3.xm_cluster.execution.output_capture import digest_file


@pytest.fixture(autouse=True)
def isolated_loop_policy(monkeypatch):
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    monkeypatch.delenv("LXM_PROJECT", raising=False)
    monkeypatch.delenv("LXM_CLUSTER", raising=False)
    yield
    asyncio.set_event_loop_policy(previous)


@pytest.fixture
def config(tmp_path):
    return xc.Config({"local": {"storage": {"staging": str(tmp_path / "store")}}})


def package(experiment, tmp_path, commands):
    source = xc.SourceTree(xc.CommandList(commands), tmp_path, files=[])
    return experiment.package([xm.Packageable(source, xc.Local.Spec())])[0]


def test_reopened_outputs_survive_cleanup_and_cannot_overwrite(config, tmp_path):
    with xc.create_experiment("results", config=config) as experiment:
        executable = package(
            experiment,
            tmp_path,
            [
                'printf "%s" "$PWD" > "$LXM_OUTPUT_DIR/result.txt"',
                'mkdir -p "$LXM_OUTPUT_DIR/checkpoint/empty"',
                'printf weights > "$LXM_OUTPUT_DIR/checkpoint/weights.bin"',
            ],
        )
        declared = {"result": "result.txt", "checkpoint": "checkpoint"}
        experiment.add(xm.Job(executable, xc.Local()), outputs=declared)
        declared["result"] = "caller-changed-this"
    unit = xc.get_experiment(experiment.experiment_id, config=config).work_units()[1]
    artifacts = unit.artifacts()
    assert set(artifacts) == {"result", "checkpoint"}
    result = artifacts["result"].fetch(tmp_path / "download/result")
    assert not Path(result.read_text()).exists()
    checkpoint = artifacts["checkpoint"].fetch(tmp_path / "download/checkpoint")
    assert (checkpoint / "weights.bin").read_text() == "weights"
    assert (checkpoint / "empty").is_dir()
    for name, destination in [("result", result), ("checkpoint", checkpoint)]:
        with pytest.raises(FileExistsError):
            artifacts[name].fetch(destination)
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        artifacts["result"].fetch(link)
    assert link.is_symlink()
    assert json.loads(unit._record["outputs"]) == {
        "result": "result.txt",
        "checkpoint": "checkpoint",
    }


@pytest.mark.parametrize("payload", ["job", "group", "generator"])
def test_add_preserves_defaults_and_overrides(config, tmp_path, payload):
    with xc.create_experiment("arguments", config=config) as experiment:
        executable = package(
            experiment,
            tmp_path,
            ['printf "%s\\n" "$@" "$VALUE" > "$LXM_OUTPUT_DIR/result"; :'],
        )
        executable.args = xm.SequentialArgs.from_collection(["default"])
        executable.env_vars = {"VALUE": "default"}
        job = xm.Job(executable, xc.Local(), args=["job"])

        async def generator(unit, value):
            await unit.add(job, {"args": [value], "env_vars": {"VALUE": value}})

        override = {"args": ["override"], "env_vars": {"VALUE": "override"}}
        choices = {
            "job": (job, override),
            "group": (xm.JobGroup(worker=job), {"worker": override}),
            "generator": (generator, {"value": "override"}),
        }
        selected, args = choices[payload]
        experiment.add(selected, args, outputs={"result": "result"})
    unit = experiment.work_units()[1]
    result = unit.artifacts()["result"].fetch(tmp_path / "result")
    assert result.read_text().splitlines() == ["default", "job", "override", "override"]
    assert unit.job.env_vars == {"VALUE": "override"}


@pytest.mark.parametrize("failure", ["missing", "exit"])
def test_failure_does_not_publish_partial_results(config, tmp_path, failure):
    experiment = xc.create_experiment("failure", config=config)
    with pytest.raises(subprocess.CalledProcessError) as error:
        with experiment:
            commands = ['printf present > "$LXM_OUTPUT_DIR/first"']
            if failure == "exit":
                commands.append("exit 7")
            executable = package(experiment, tmp_path, commands)
            experiment.add(
                xm.Job(executable, xc.Local()),
                outputs={"first": "first", "missing": "missing"},
            )
    unit = experiment.work_units()[1]
    assert unit.get_status().is_failed
    assert unit.artifacts() == {}
    root = Path(unit._record["artifact_directory"])
    assert not root.exists() or list(root.iterdir()) == []
    assert error.value.returncode == (7 if failure == "exit" else 1)


def test_array_results_are_independent_even_with_a_failed_sibling(config, tmp_path):
    experiment = xc.create_experiment("array", config=config)
    with pytest.raises(subprocess.CalledProcessError):
        with experiment:
            executable = package(
                experiment,
                tmp_path,
                [
                    'test "$VALUE" != fail',
                    'printf "%s" "$VALUE" > "$LXM_OUTPUT_DIR/result"',
                ],
            )
            experiment.add(
                xc.ArrayJob(
                    executable,
                    xc.Local(),
                    env_vars=[
                        {"VALUE": "first"},
                        {"VALUE": "second"},
                        {"VALUE": "fail"},
                    ],
                ),
                outputs={"result": "result"},
            )
    unit = experiment.work_units()[1]
    with pytest.raises(ValueError, match="zero-based"):
        unit.artifacts()
    for task in [-1, 3]:
        with pytest.raises(ValueError, match="range"):
            unit.artifacts(task=task)
    for task, expected in enumerate(["first", "second"]):
        result = unit.artifacts(task=task)["result"].fetch(tmp_path / f"result-{task}")
        assert result.read_text() == expected
    assert unit.artifacts(task=2) == {}


@pytest.mark.parametrize("path", ["", ".", "../escape", "a/../b", "/absolute"])
def test_invalid_paths_rejected_before_allocation(config, path):
    experiment = xc.create_experiment("invalid", config=config)
    with pytest.raises(ValueError, match="beneath"):
        experiment.add(None, outputs={"result": path})
    assert experiment.work_units() == {}


def test_outputs_preserve_auxiliary_role_rejection(config):
    experiment = xc.create_experiment("auxiliary", config=config)
    job = xm.AuxiliaryUnitJob(None, termination_delay_secs=0)
    with pytest.raises(NotImplementedError, match="Auxiliary"):
        experiment.add(job, outputs={"result": "result"})
    assert experiment.work_units() == {}


@pytest.mark.parametrize("kind", ["symlink", "nested-link", "fifo"])
def test_capture_rejects_links_and_special_files(tmp_path, kind):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    path = source / "result"
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "nested-link":
        path.mkdir()
        (path / "link").symlink_to(outside)
    else:
        os.mkfifo(path)
    destination = tmp_path / "artifacts/0"
    with pytest.raises(ValueError):
        capture(source, destination, {"result": "result"})
    assert not destination.exists()
    assert outside.read_text() == "untouched"


def test_content_identity_and_snapshot_are_independent_of_name_and_mtime(tmp_path):
    for name in ["first", "second"]:
        (tmp_path / name).write_text("same content")
    os.utime(tmp_path / "second", (100, 100))
    destination = tmp_path / "artifacts/0"
    capture(tmp_path, destination, {"a": "first", "b": "second"})
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["a"] == manifest["b"]
    (tmp_path / "first").write_text("changed after capture")
    artifact = xc.Artifact(manifest["a"], str(destination / (manifest["a"] + ".tar")))
    assert artifact.fetch(tmp_path / "result").read_text() == "same content"
    with pytest.raises(OSError):
        capture(tmp_path, destination, {"a": "first"})
    assert json.loads((destination / "manifest.json").read_text()) == manifest


def test_corrupt_or_unsafe_archives_are_not_exposed(tmp_path):
    archive_path = tmp_path / "archive.tar"
    archive_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="identity"):
        xc.Artifact("0" * 64, str(archive_path)).fetch(tmp_path / "download")
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("../../escaped")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        xc.Artifact(digest_file(archive_path), str(archive_path)).fetch(
            tmp_path / "download"
        )
    assert not (tmp_path / "download").exists()
    assert not (tmp_path / "escaped").exists()
    assert not list(tmp_path.glob(".lxm-fetch-*"))


def test_remote_receipt_and_streaming_fetch_use_recorded_endpoint(tmp_path):
    source = tmp_path / "source"
    source.write_text("remote bytes")
    capture(tmp_path, tmp_path / "results/0", {"result": "source"})
    manifest = (tmp_path / "results/0/manifest.json").read_text()
    record = dict(
        task_count=1,
        artifact_directory="/remote/results",
        backend="slurm",
        hostname="nersc",
        username="saved-user",
    )
    with mock.patch.object(
        ssh, "run", return_value=subprocess.CompletedProcess([], 0, manifest)
    ) as command:
        artifact = outputs.artifacts(record)["result"]
    assert command.call_args.kwargs == {"hostname": "nersc", "username": "saved-user"}
    assert command.call_args.args[0][-1] == "/remote/results/0/manifest.json"
    data = (tmp_path / "results/0" / (artifact.id + ".tar")).read_bytes()

    def transfer(argv, **options):
        assert argv == ["cat", "--", artifact._archive_path]
        assert options["hostname"] == "nersc"
        assert options["username"] == "saved-user"
        assert options["text"] is False
        options["stdout"].write(data)

    with mock.patch.object(ssh, "run", side_effect=transfer) as command:
        assert artifact.fetch(tmp_path / "download").read_text() == "remote bytes"
    assert command.call_count == 1
    with mock.patch.object(
        ssh, "run", side_effect=subprocess.CalledProcessError(255, "ssh")
    ) as command:
        with pytest.raises(subprocess.CalledProcessError):
            artifact.fetch(tmp_path / "failed")
    assert command.call_count == 1
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".lxm-fetch-*"))


@pytest.mark.parametrize(
    "executor,builder",
    [
        (xc.Slurm(log_directory="/custom/logs"), slurm.SlurmJobScriptBuilder()),
        (xc.GridEngine(), gridengine.GridEngineJobScriptBuilder()),
    ],
)
def test_native_array_offset_and_retention_location(
    config, tmp_path, executor, builder
):
    experiment = xc.create_experiment("native", config=config)
    executable = package(
        experiment, tmp_path, ['printf result > "$LXM_OUTPUT_DIR/result"']
    )
    script = builder.build(
        xc.ArrayJob(executable, executor, args=[[], []]),
        "native",
        str(tmp_path / "logs"),
        outputs={"result": "result"},
    )
    subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        env={**os.environ, builder.ARRAY_TASK_ID: str(builder.ARRAY_TASK_OFFSET + 1)},
    )
    assert (tmp_path / "logs/artifacts/1/manifest.json").exists()
    assert not (tmp_path / "logs/artifacts/0").exists()


def test_fresh_process_retrieval_requires_no_launcher_or_worker_tree(tmp_path):
    setup = f"""
from lxm3 import xm, xm_cluster as xc
config = xc.Config({{"local": {{"storage": {{"staging": {str(tmp_path / "store")!r}}}}}}})
"""
    launcher = (
        setup
        + f"""
with xc.create_experiment("fresh", config=config) as experiment:
    source = xc.SourceTree(xc.CommandList(['printf value > "$LXM_OUTPUT_DIR/result"']), {str(tmp_path)!r}, files=[])
    [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
    experiment.add(xm.Job(executable, xc.Local()), outputs={{"result": "result"}})
print(experiment.experiment_id)
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", launcher], check=True, capture_output=True, text=True
    )
    experiment_id = int(result.stdout.splitlines()[-1])
    reader = (
        setup
        + f"""
from unittest import mock
from lxm3.xm_cluster import packaging
with mock.patch.object(packaging, "package", side_effect=AssertionError("must not package")):
    unit = xc.get_experiment({experiment_id}, config=config).work_units()[1]
    assert unit.artifacts()["result"].fetch({str(tmp_path / "download")!r}).read_text() == "value"
"""
    )
    subprocess.run(
        [sys.executable, "-c", reader], check=True, capture_output=True, text=True
    )
