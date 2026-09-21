import os
import shlex
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from lxm3.clusters import slurm
from lxm3.clusters import ssh
from lxm3.xm_cluster import config
from lxm3.xm_cluster.execution import job_script_builder


def test_command_uses_system_ssh_without_retries():
    command = ["sbatch", "--parsable", "--", "/path with 'quotes/$literal.sbatch"]
    with mock.patch.object(ssh.subprocess, "run") as run:
        ssh.run(command, hostname="nersc", username="selected-user")
    run.assert_called_once()
    argv = run.call_args.args[0]
    assert argv[:3] == ["ssh", "-T", "-x"]
    assert argv[-5:-3] == ["-l", "selected-user"]
    assert argv[-3:-1] == ["--", "nersc"]
    assert shlex.split(argv[-1]) == command
    assert "BatchMode=yes" in argv


def test_onsite_command_is_not_sent_through_a_shell():
    with mock.patch.object(ssh.subprocess, "run") as run:
        ssh.run(["sbatch", "--parsable", "--", "/path with spaces/job.sh"])
    assert run.call_args.args[0] == [
        "sbatch",
        "--parsable",
        "--",
        "/path with spaces/job.sh",
    ]
    assert "shell" not in run.call_args.kwargs


def test_staging_commands_preserve_paths_and_contents(tmp_path, monkeypatch):
    # Execute the transmitted command locally, exercising its actual shell quoting.
    real_run = subprocess.run
    calls = []

    def fake_ssh(argv, **options):
        calls.append(argv)
        assert argv[-2] == "selected-host"
        return real_run(["sh", "-c", argv[-1]], **options)

    monkeypatch.setattr(ssh.subprocess, "run", fake_ssh)
    settings = config.ClusterSettings(
        {
            "server": "selected-host",
            "user": "selected-user",
            "storage": {"staging": str(tmp_path / "stage $literal ' with space")},
        }
    )
    store = job_script_builder.create_artifact_store(
        settings=settings, project="project", use_openssh=True
    )
    assert isinstance(store.filesystem, ssh.OpenSSHFileSystem)
    source = tmp_path / "source"
    source.write_bytes(bytes(range(256)) * 100)
    result = store.put_file(str(source), "archives/data.zip")
    assert Path(result).read_bytes() == source.read_bytes()
    store.put_text("literal $value ' \n", "jobs/job.sh")
    assert (
        Path(store.normalize_path("jobs/job.sh")).read_text() == "literal $value ' \n"
    )
    assert store.get_file_info("archives/data.zip").size == source.stat().st_size
    assert not store.exists("missing")
    assert store.filesystem.abspath("~") == os.path.expanduser("~")
    assert store.filesystem.abspath("~/a b") == os.path.expanduser("~/a b")
    assert all("selected-user" in argv for argv in calls)
    assert not list(Path(store.storage_root).rglob("*.tmp"))


@pytest.mark.parametrize("operation", ["submit", "exists", "upload"])
def test_connection_failure_is_an_error_not_missing_or_retried(tmp_path, operation):
    source = tmp_path / "source"
    source.touch()
    filesystem = ssh.OpenSSHFileSystem("nersc")
    with mock.patch.object(
        ssh.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 255, stdout=""),
    ) as run:
        # Model subprocess.run(check=True), including the check=False existence probe.
        def disconnected(argv, **options):
            result = subprocess.CompletedProcess(argv, 255, stdout="")
            if options["check"]:
                result.check_returncode()
            return result

        run.side_effect = disconnected
        with pytest.raises(subprocess.CalledProcessError) as error:
            if operation == "submit":
                slurm.SlurmCluster("nersc").launch("job.sh")
            elif operation == "exists":
                filesystem.exists("archive.zip")
            else:
                filesystem.put_file(str(source), "archive.zip")
        assert error.value.returncode == 255
        run.assert_called_once()


def test_federated_job_reference_is_not_truncated():
    with mock.patch.object(
        ssh,
        "run",
        return_value=subprocess.CompletedProcess([], 0, stdout="42;cluster\n"),
    ):
        assert slurm.SlurmCluster("nersc").launch("job.sh") == "42;cluster"
