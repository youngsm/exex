"""Consumers use verified private copies of retained artifacts across sites."""

import asyncio
import getpass
import io
import json
import os
import shlex
import socket
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest import mock

import pytest
from fsspec.implementations.local import LocalFileSystem

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.clusters import slurm as native_slurm
from lxm3.clusters import ssh
from lxm3.xm_cluster import experiment as experiment_lib
from lxm3.xm_cluster import inputs
from lxm3.xm_cluster.execution import artifact_io
from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import slurm
from lxm3.xm_cluster.execution.artifact_io import capture
from lxm3.xm_cluster.execution.artifact_io import digest_file
from lxm3.xm_cluster.execution.artifact_io import prepare


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
    return xc.Config(
        {
            "local": {"storage": {"staging": str(tmp_path / "store")}},
            "clusters": [
                {"name": "site", "storage": {"staging": str(tmp_path / "site")}}
            ],
        }
    )


def package(experiment, tmp_path, commands, executor=None):
    source = xc.SourceTree(xc.CommandList(commands), tmp_path, files=[])
    executor = executor or xc.Local()
    return experiment.package([xm.Packageable(source, executor.Spec())])[0]


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "checkpoint/weights").write_text("original")
    capture(tmp_path, tmp_path / "retained", {"checkpoint": "checkpoint"})
    identity = json.loads((tmp_path / "retained/manifest.json").read_text())[
        "checkpoint"
    ]
    return xc.Artifact(identity, str(tmp_path / "retained" / (identity + ".tar")))


@pytest.mark.parametrize("payload", ["job", "group", "generator", "array"])
def test_inputs_are_frozen_private_copies_with_original_invocation(
    config, tmp_path, artifact, payload
):
    original_bytes = Path(artifact._archive_path).read_bytes()
    with mock.patch.object(
        xc.Artifact, "fetch", side_effect=AssertionError("No author fetch")
    ):
        with mock.patch.object(
            ssh.OpenSSHFileSystem, "get_file", side_effect=AssertionError("No transfer")
        ):
            with xc.create_experiment("consumer", config=config) as experiment:
                executable = package(
                    experiment,
                    tmp_path,
                    [
                        'cat "$LXM_INPUT_DIR/checkpoint/weights" > "$LXM_OUTPUT_DIR/result"',
                        'printf changed > "$LXM_INPUT_DIR/checkpoint/weights"',
                        'printf "%s\\n" "$VALUE" "$@" > "$LXM_OUTPUT_DIR/invocation"; :',
                    ],
                )
                job = xm.Job(executable, xc.Local())
                override = {
                    "args": ["override"],
                    "env_vars": {"VALUE": "value", "LXM_INPUT_DIR": "/wrong"},
                }

                async def generator(unit, value):
                    await unit.add(job, value)

                choices = {
                    "job": (job, override),
                    "group": (xm.JobGroup(worker=job), {"worker": override}),
                    "generator": (generator, {"value": override}),
                    "array": (
                        xc.ArrayJob(
                            executable,
                            xc.Local(),
                            args=[["override"]],
                            env_vars=[override["env_vars"]] * 2,
                        ),
                        None,
                    ),
                }
                selected, args = choices[payload]
                bindings = {"checkpoint": artifact}
                experiment.add(
                    selected,
                    args,
                    inputs=bindings,
                    outputs={"result": "result", "invocation": "invocation"},
                )
                bindings.clear()
    unit = xc.get_experiment(experiment.experiment_id, config=config).work_units()[1]
    saved = json.loads(unit._record["inputs"])
    assert saved == {
        "checkpoint": dict(
            id=artifact.id,
            archive_path=artifact._archive_path,
            hostname=socket.gethostname(),
            username=getpass.getuser(),
        )
    }
    for task in range(2 if payload == "array" else 1):
        result = unit.artifacts(task=task)
        assert (
            result["result"].fetch(tmp_path / f"result-{task}").read_text()
            == "original"
        )
        assert result["invocation"].fetch(
            tmp_path / f"args-{task}"
        ).read_text().splitlines() == ["value", "override"]
    assert Path(artifact._archive_path).read_bytes() == original_bytes


def test_input_only_and_literal_names(config, tmp_path, artifact):
    name = "literal $value ' with spaces"
    escaped = shlex.quote(name + "/weights")
    result = tmp_path / "result"
    with xc.create_experiment("input-only", config=config) as experiment:
        executable = package(
            experiment,
            tmp_path,
            [f'cat "$LXM_INPUT_DIR"/{escaped} > {shlex.quote(str(result))}'],
        )
        experiment.add(xm.Job(executable, xc.Local()), inputs={name: artifact})
    assert result.read_text() == "original"
    assert experiment.work_units()[1].artifacts() == {}


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "/absolute", "a/", "a/../b"])
def test_invalid_names_fail_before_allocating_work_unit(config, artifact, name):
    experiment = xc.create_experiment("invalid", config=config)
    with pytest.raises(ValueError, match="single path component"):
        experiment.add(None, inputs={name: artifact})
    assert experiment.work_units() == {}


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_bad_input_prevents_payload_start_and_cleans_workdir(
    config, tmp_path, artifact, damage
):
    archive = Path(artifact._archive_path)
    if damage == "missing":
        archive.unlink()
    else:
        archive.write_bytes(b"corrupted")
    marker = tmp_path / "payload-ran"
    workdir = tmp_path / "work"
    experiment = xc.create_experiment("bad-input", config=config)
    with pytest.raises(subprocess.CalledProcessError):
        with experiment:
            executable = package(
                experiment, tmp_path, [f"touch {shlex.quote(str(marker))}"]
            )
            experiment.add(
                xm.Job(executable, xc.Local(workdir_root=str(workdir))),
                inputs={"checkpoint": artifact},
            )
    assert not marker.exists()
    assert list(workdir.iterdir()) == []
    assert experiment.work_units()[1].get_status().is_failed


@pytest.mark.parametrize(
    "member_name,kind",
    [
        ("../escaped", tarfile.REGTYPE),
        ("/tmp/escaped", tarfile.REGTYPE),
        ("data/../../escaped", tarfile.REGTYPE),
        ("other", tarfile.REGTYPE),
        ("data", tarfile.SYMTYPE),
        ("data", tarfile.LNKTYPE),
        ("data", tarfile.FIFOTYPE),
    ],
)
def test_unsafe_archive_is_rejected_before_extraction(tmp_path, member_name, kind):
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo(member_name)
        member.type = kind
        member.linkname = "/tmp/escaped"
        member.size = 1 if kind == tarfile.REGTYPE else 0
        archive.addfile(member, io.BytesIO(b"x"))
    root = tmp_path / "inputs"
    root.mkdir()
    with mock.patch.object(tarfile.TarFile, "extractall") as extraction:
        with pytest.raises(ValueError, match="Invalid input archive member"):
            prepare(
                root,
                {
                    "input": {
                        "id": digest_file(archive_path),
                        "archive_path": str(archive_path),
                    }
                },
            )
    extraction.assert_not_called()
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "producer_host,producer_user,consumer_host,consumer_user,allowed",
    [
        (None, None, None, None, True),
        (None, None, None, "ignored-for-on-site", True),
        (socket.getfqdn(), getpass.getuser(), None, None, True),
        ("nersc", "sam", "nersc", "sam", True),
        ("nersc", None, "nersc", None, True),
        ("nersc", "sam", None, None, False),
        ("nersc", "sam", "different-alias", "sam", False),
        ("nersc", "sam", "nersc", "different-user", False),
    ],
)
def test_same_endpoint_is_explicit(
    config,
    artifact,
    producer_host,
    producer_user,
    consumer_host,
    consumer_user,
    allowed,
):
    value = xc.Artifact(
        artifact.id, artifact._archive_path, producer_host, producer_user
    )
    binding = inputs.declarations({"checkpoint": value})["checkpoint"]
    endpoint = inputs._endpoint(consumer_host, consumer_user if consumer_host else None)
    assert ({key: binding[key] for key in endpoint} == endpoint) == allowed


@pytest.mark.parametrize("failure", ["download", "digest", "upload"])
def test_cross_site_errors_prevent_submission(config, tmp_path, artifact, failure):
    foreign = xc.Artifact(artifact.id, artifact._archive_path, "nersc", "sam")
    experiment = xc.create_experiment("cross-site", config=config)
    executor = xc.Slurm(cluster="site")
    executable = package(experiment, tmp_path, ["true"], executor)
    if failure == "digest":
        Path(artifact._archive_path).write_bytes(b"bad archive")
    with mock.patch.object(ssh, "OpenSSHFileSystem", return_value=LocalFileSystem()):
        with mock.patch.object(native_slurm.SlurmCluster, "launch") as submission:
            download = LocalFileSystem.get_file
            upload = inputs.job_script_builder.artifacts.ArtifactStore.put_file
            with (
                mock.patch.object(
                    LocalFileSystem,
                    "get_file",
                    autospec=True,
                    side_effect=OSError("disconnected")
                    if failure == "download"
                    else download,
                ),
                mock.patch.object(
                    inputs.job_script_builder.artifacts.ArtifactStore,
                    "put_file",
                    autospec=True,
                    side_effect=OSError("disconnected")
                    if failure == "upload"
                    else upload,
                ),
                pytest.raises((OSError, ValueError)),
            ):
                with experiment:
                    experiment.add(
                        xm.Job(executable, executor), inputs={"checkpoint": foreign}
                    )
    submission.assert_not_called()


@pytest.mark.parametrize(
    "source_host,target_host",
    [(None, "target"), ("source", None), ("source", "target")],
)
def test_cross_site_staging_preserves_identity_and_reuses_destination(
    config, tmp_path, artifact, source_host, target_host
):
    config._data["clusters"][0].update(server=target_host)
    executor = xc.Slurm(cluster="site") if target_host else xc.Local()
    original = inputs.declarations(
        {"checkpoint": xc.Artifact(artifact.id, artifact._archive_path, source_host)}
    )
    filesystem = LocalFileSystem()
    filesystem.abspath = os.path.abspath
    with (
        mock.patch.object(ssh, "OpenSSHFileSystem", return_value=filesystem),
        mock.patch.object(
            inputs.job_script_builder, "OpenSSHFileSystem", return_value=filesystem
        ),
        mock.patch.object(filesystem, "put_file", wraps=filesystem.put_file) as upload,
    ):
        staged = inputs.stage(original, executor, config, "consumer")
        assert staged["checkpoint"]["id"] == artifact.id
        assert staged["checkpoint"]["archive_path"] != artifact._archive_path
        assert original["checkpoint"]["archive_path"] == artifact._archive_path
        assert digest_file(staged["checkpoint"]["archive_path"]) == artifact.id
        # A retained destination copy no longer needs the original endpoint.
        Path(artifact._archive_path).unlink()
        assert inputs.stage(original, executor, config, "consumer") == staged
        assert upload.call_count == 1
    root = tmp_path / "unpacked"
    root.mkdir()
    prepare(root, staged)
    assert (root / "checkpoint/weights").read_text() == "original"


def test_cross_site_local_execution_keeps_provenance(config, tmp_path, artifact):
    foreign = xc.Artifact(artifact.id, artifact._archive_path, "source")
    with mock.patch.object(ssh, "OpenSSHFileSystem", return_value=LocalFileSystem()):
        with xc.create_experiment("transferred", config=config) as experiment:
            executable = package(
                experiment,
                tmp_path,
                ['cat "$LXM_INPUT_DIR/checkpoint/weights" > "$LXM_OUTPUT_DIR/result"'],
            )
            experiment.add(
                xm.Job(executable, xc.Local()),
                inputs={"checkpoint": foreign},
                outputs={"result": "result"},
            )
    unit = xc.get_experiment(experiment.experiment_id, config=config).work_units()[1]
    assert json.loads(unit._record["inputs"])["checkpoint"]["hostname"] == "source"
    assert str(tmp_path / "store/inputs" / (artifact.id + ".tar")) in unit.get_script()
    assert (
        unit.artifacts()["result"].fetch(tmp_path / "result").read_text() == "original"
    )


@pytest.mark.parametrize(
    "executor,builder",
    [
        (xc.Slurm(), slurm.SlurmJobScriptBuilder()),
        (xc.GridEngine(), gridengine.GridEngineJobScriptBuilder()),
    ],
)
def test_native_array_tasks_get_isolated_inputs(
    config, tmp_path, artifact, executor, builder
):
    experiment = xc.create_experiment("native-array", config=config)
    executable = package(
        experiment,
        tmp_path,
        [
            'test "$(cat "$LXM_INPUT_DIR/checkpoint/weights")" = original',
            'printf changed > "$LXM_INPUT_DIR/checkpoint/weights"',
        ],
    )
    script = builder.build(
        xc.ArrayJob(executable, executor, args=[[], []]),
        "input-array",
        str(tmp_path / "logs"),
        inputs=inputs.declarations({"checkpoint": artifact}),
    )
    for task in range(2):
        subprocess.run(
            ["bash", "-c", script],
            check=True,
            capture_output=True,
            env={
                **os.environ,
                builder.ARRAY_TASK_ID: str(task + builder.ARRAY_TASK_OFFSET),
            },
        )


def test_producer_and_consumer_are_separate_processes(config, tmp_path):
    setup = f"""
from lxm3 import xm, xm_cluster as xc
config = xc.Config({{"local": {{"storage": {{"staging": {str(tmp_path / "store")!r}}}}}}})
"""
    producer = (
        setup
        + f"""
with xc.create_experiment("producer", config=config) as experiment:
    source = xc.SourceTree(xc.CommandList(['printf original > "$LXM_OUTPUT_DIR/checkpoint"']), {str(tmp_path)!r}, files=[])
    [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
    experiment.add(xm.Job(executable, xc.Local()), outputs={{"checkpoint": "checkpoint"}})
print(experiment.experiment_id)
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", producer], check=True, capture_output=True, text=True
    )
    producer_id = int(result.stdout.splitlines()[-1])
    consumer = (
        setup
        + f"""
artifact = xc.get_experiment({producer_id}, config=config).work_units()[1].artifacts()["checkpoint"]
with xc.create_experiment("consumer", config=config) as experiment:
    source = xc.SourceTree(xc.CommandList(['cat "$LXM_INPUT_DIR/checkpoint" > "$LXM_OUTPUT_DIR/result"']), {str(tmp_path)!r}, files=[])
    [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
    experiment.add(xm.Job(executable, xc.Local()), inputs={{"checkpoint": artifact}}, outputs={{"result": "result"}})
assert experiment.work_units()[1].artifacts()["result"].fetch({str(tmp_path / "result")!r}).read_text() == "original"
"""
    )
    subprocess.run(
        [sys.executable, "-c", consumer], check=True, capture_output=True, text=True
    )


def test_preparation_needs_only_stdlib_and_preserves_executable_files(tmp_path):
    source = tmp_path / "program"
    source.write_text("#!/bin/sh\nprintf executable")
    source.chmod(0o755)
    capture(tmp_path, tmp_path / "retained", {"program": "program"})
    identity = json.loads((tmp_path / "retained/manifest.json").read_text())["program"]
    bindings = {
        "program": {
            "id": identity,
            "archive_path": str(tmp_path / "retained" / (identity + ".tar")),
        }
    }
    root = tmp_path / "inputs"
    root.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            artifact_io.__file__,
            "prepare",
            str(root),
            json.dumps(bindings),
        ],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [str(root / "program")], check=True, capture_output=True, text=True
    )
    assert result.stdout == "executable"


def test_corrupt_staged_cache_is_checked_before_payload(config, tmp_path, artifact):
    foreign = xc.Artifact(artifact.id, artifact._archive_path, "source")
    with mock.patch.object(ssh, "OpenSSHFileSystem", return_value=LocalFileSystem()):
        staged = inputs.stage(
            inputs.declarations({"checkpoint": foreign}), xc.Local(), config, None
        )
        Path(staged["checkpoint"]["archive_path"]).write_bytes(
            b"damaged after transfer"
        )
        marker = tmp_path / "payload-ran"
        with pytest.raises(subprocess.CalledProcessError):
            with xc.create_experiment("bad-cache", config=config) as experiment:
                executable = package(
                    experiment, tmp_path, [f"touch {shlex.quote(str(marker))}"]
                )
                experiment.add(
                    xm.Job(executable, xc.Local()), inputs={"checkpoint": foreign}
                )
    assert not marker.exists()


def test_interrupted_upload_never_exposes_final_archive(config, tmp_path, artifact):
    config._data["clusters"][0].update(server="target")
    filesystem = LocalFileSystem()
    filesystem.abspath = os.path.abspath

    def interrupted(local, remote):
        Path(remote).write_bytes(b"partial upload")
        raise OSError("disconnected")

    with (
        mock.patch.object(
            inputs.job_script_builder, "OpenSSHFileSystem", return_value=filesystem
        ),
        mock.patch.object(filesystem, "put_file", side_effect=interrupted),
        pytest.raises(OSError, match="disconnected"),
    ):
        inputs.stage(
            inputs.declarations({"checkpoint": artifact}),
            xc.Slurm(cluster="site"),
            config,
            None,
        )
    assert not (tmp_path / "site/inputs" / (artifact.id + ".tar")).exists()
