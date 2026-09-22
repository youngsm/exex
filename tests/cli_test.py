"""Discovery and CLI control reuse the existing catalog and WorkUnit methods."""

import json
import os
import socket
import subprocess
import sys
from unittest import mock

import pytest
from absl.testing import flagsaver

from exex import xm
from exex import xm_cluster as xc
from exex.cli import cli
from exex.clusters import slurm
from exex.xm_cluster import catalog
from exex.xm_cluster import config as config_lib
from exex.xm_cluster import experiment as experiment_lib


@pytest.fixture(autouse=True)
def isolated_defaults(monkeypatch):
    monkeypatch.delenv("EXEX_PROJECT", raising=False)
    monkeypatch.delenv("EXEX_CLUSTER", raising=False)
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    config_lib.default.cache_clear()
    with flagsaver.flagsaver(exex_config=None):
        yield
    config_lib.default.cache_clear()


@pytest.fixture
def config(tmp_path):
    return xc.Config({"local": {"storage": {"staging": str(tmp_path / "author")}}})


@pytest.mark.parametrize("existing_directory", [False, True])
def test_discovery_without_a_catalog_creates_nothing(
    config, tmp_path, existing_directory
):
    root = tmp_path / "author"
    if existing_directory:
        root.mkdir()
    assert xc.list_experiments(config=config) == []
    assert root.exists() == existing_directory
    assert not list(tmp_path.rglob("experiments.sqlite3"))


def test_discovery_filters_orders_and_returns_read_only_handles(config, monkeypatch):
    with mock.patch.object(catalog.time, "time_ns", side_effect=[101, 202, 303]):
        for project in ("quoted ' project", "other", "quoted ' project"):
            xc.create_experiment("same title", project=project, config=config)
    database = catalog.Catalog(config.local_settings().storage_root).path
    before = database.read_bytes()
    monkeypatch.setenv("EXEX_PROJECT", "not-an-implicit-filter")
    with mock.patch.object(
        subprocess, "Popen", side_effect=AssertionError("External operation")
    ):
        experiments = xc.list_experiments(config=config)
        assert [e.experiment_id for e in experiments] == [303, 202, 101]
        assert all(e.work_units() == {} for e in experiments)
        assert [
            e.experiment_id
            for e in xc.list_experiments(project="quoted ' project", config=config)
        ] == [303, 101]
        assert xc.list_experiments(project="missing", config=config) == []
        assert xc.list_experiments(project="' OR 1=1 --", config=config) == []
    assert database.read_bytes() == before
    config._data["local"]["storage"]["staging"] = "/unused-after-discovery"
    assert experiments[0].work_units() == {}


def test_discovery_resolves_default_config_once(config):
    xc.create_experiment("one", config=config)
    xc.create_experiment("two", config=config)
    with mock.patch.object(config_lib, "default", return_value=config) as default:
        assert len(xc.list_experiments()) == 2
    default.assert_called_once_with()


@pytest.fixture
def saved(config, tmp_path, monkeypatch):
    with mock.patch.object(catalog.time, "time_ns", return_value=101):
        experiment = xc.create_experiment(
            "literal [red] title", project="demo", config=config
        )
    logs = tmp_path / "logs with ' $ spaces"
    logs.mkdir()
    for state in ("completed", "failed"):
        unit_id = experiment._catalog.create_work_unit(101)
        experiment._catalog.update_work_unit(
            101,
            unit_id,
            backend="local",
            hostname=socket.gethostname(),
            log_directory=str(logs),
            state=state,
            message="literal [blue] message",
        )
    (logs / "task-0.log").write_text(
        "first\nliteral [red] $ ' second\nlast-without-newline"
    )
    (logs / "task-1.log").write_text("array-task-one\n")
    config_file = tmp_path / "exex.toml"
    config_file.write_text(
        "[local.storage]\nstaging = "
        + json.dumps(config.local_settings().storage_root)
        + "\n"
    )
    monkeypatch.setenv("EXEX_CONFIG", str(config_file))
    return experiment, config_file


def invoke(*arguments):
    cli.main(cli._parse_flags(["exex", *map(str, arguments)]))


def run_cli(*arguments, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "exex.cli.cli", *map(str, arguments)],
        capture_output=True,
        text=True,
        timeout=30,
        **kwargs,
    )


def test_experiments_cli_is_metadata_only_and_preserves_literal_titles(saved, capsys):
    experiment, _ = saved
    experiment.work_units()[2]._save(
        backend="slurm", hostname="unreachable", native_id="789"
    )
    with mock.patch.object(
        subprocess, "Popen", side_effect=AssertionError("External operation")
    ):
        invoke("experiments", "--project", "demo")
    output = capsys.readouterr().out
    assert "101" in output and "demo" in output and "literal [red] title" in output
    invoke("experiments", "--project", "missing")
    assert "101" not in capsys.readouterr().out


@pytest.mark.parametrize("command", [("status", "101"), ("status", "101", "2")])
def test_status_cli_selects_actual_ids(saved, command, capsys):
    with mock.patch.object(
        xc.ClusterWorkUnit,
        "get_status",
        autospec=True,
        return_value=xc.WorkUnitStatus("unknown", "no evidence"),
    ) as status:
        invoke(*command)
    assert [call.args[0].work_unit_id for call in status.call_args_list] == (
        [1, 2] if len(command) == 2 else [2]
    )
    assert "unknown" in capsys.readouterr().out


def test_logs_cli_forwards_array_selection_and_tail(saved, capsys):
    with mock.patch.object(
        xc.ClusterWorkUnit,
        "get_logs",
        autospec=True,
        return_value="[red] literal\nno newline",
    ) as logs:
        invoke("logs", "101", "2", "--task", "0", "--tail", "17")
    assert logs.call_args.args[0].work_unit_id == 2
    assert logs.call_args.kwargs == {"task": 0, "tail": 17}
    assert capsys.readouterr().out == "[red] literal\nno newline"


def test_stop_cli_uses_recorded_target_once_without_polling(saved, capsys):
    experiment, _ = saved
    unit = experiment.work_units()[2]
    unit._save(
        backend="slurm",
        hostname="recorded-site",
        username="recorded-user",
        native_id="789",
        job_name="recorded-name",
    )
    with mock.patch.object(slurm.ssh, "run") as run:
        invoke("stop", "101", "2")
    run.assert_called_once_with(
        ["scancel", "--ctld", "--name=recorded-name", "789"],
        hostname="recorded-site",
        username="recorded-user",
    )
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "command",
    [
        ("status", "101", "2"),
        ("logs", "101", "2"),
        ("stop", "101", "2"),
        ("script", "101", "2"),
    ],
)
def test_ssh_error_propagates_without_retry(saved, command):
    experiment, _ = saved
    experiment.work_units()[2]._save(
        backend="slurm",
        hostname="unreachable",
        native_id="789",
        job_name="recorded-name",
        script_path="/saved/job.sh",
    )
    with mock.patch.object(
        slurm.ssh, "run", side_effect=subprocess.CalledProcessError(255, "ssh")
    ) as run:
        with pytest.raises(subprocess.CalledProcessError):
            invoke(*command)
    assert run.call_count == 1


@pytest.mark.parametrize("command", ["status", "logs", "stop", "script"])
def test_missing_unit_is_an_error_not_an_index_or_a_new_unit(saved, command):
    experiment, _ = saved
    before = experiment._catalog.path.read_bytes()
    with pytest.raises(xm.NotFoundError, match="WorkUnit 3 in experiment 101"):
        invoke(command, "101", "3")
    assert experiment._catalog.path.read_bytes() == before


def test_cli_in_fresh_processes_reads_catalog_status_and_literal_logs(saved):
    experiment, _ = saved
    before = experiment._catalog.path.read_bytes()
    result = run_cli("experiments", check=True)
    assert "literal [red] title" in result.stdout
    result = run_cli("status", "101", check=True)
    assert "completed" in result.stdout and "failed" in result.stdout
    assert "literal [blue] message" in result.stdout
    result = run_cli("logs", "101", "1", "--tail", "2", check=True)
    assert result.stdout == "literal [red] $ ' second\nlast-without-newline"
    assert run_cli("logs", "101", "1", "--tail", "0", check=True).stdout == ""
    assert experiment._catalog.path.read_bytes() == before


def test_cli_reads_zero_based_array_logs_and_rejects_ambiguous_requests(saved):
    experiment, _ = saved
    experiment.work_units()[2]._save(is_array=True, task_count=2)
    result = run_cli("logs", "101", "2", "--task", "1", check=True)
    assert result.stdout == "array-task-one\n"
    result = run_cli("logs", "101", "2")
    assert result.returncode != 0 and "zero-based task" in result.stderr


@pytest.mark.parametrize(
    "command,diagnostic",
    [
        (("status", "404"), "404"),
        (("status", "101", "404"), "WorkUnit 404"),
        (("logs", "101", "1", "--tail", "-1"), "nonnegative"),
        (("logs", "101", "1", "--task", "2"), "in range"),
        (("stop", "101", "1"), "only for Slurm"),
        (("stop", "101"), "work_unit_id"),
        (("script", "101", "1"), "No submission script"),
        (("script", "101"), "work_unit_id"),
    ],
)
def test_cli_errors_exit_nonzero_without_changing_records(saved, command, diagnostic):
    experiment, _ = saved
    before = experiment._catalog.path.read_bytes()
    result = run_cli(*command)
    assert result.returncode != 0 and diagnostic in result.stderr
    assert experiment._catalog.path.read_bytes() == before


@pytest.mark.parametrize("before_command", [False, True])
def test_explicit_config_flag_overrides_environment(saved, before_command):
    _, config_file = saved
    flag = f"--exex_config={config_file}"
    arguments = [flag, "experiments"] if before_command else ["experiments", flag]
    result = run_cli(
        *arguments, env={**os.environ, "EXEX_CONFIG": "/missing/config"}, check=True
    )
    assert "literal [red] title" in result.stdout


def test_script_cli_uses_saved_endpoint_and_exact_path(saved, capsys):
    experiment, _ = saved
    experiment.work_units()[1]._save(
        hostname="saved-host", username="saved-user", script_path="/saved ' $/job.sh"
    )
    with mock.patch.object(
        slurm.ssh,
        "run",
        return_value=subprocess.CompletedProcess([], 0, "literal [red] $ ' text"),
    ) as run:
        invoke("script", "101", "1")
    run.assert_called_once_with(
        ["cat", "--", "/saved ' $/job.sh"], hostname="saved-host", username="saved-user"
    )
    assert capsys.readouterr().out == "literal [red] $ ' text"


@pytest.mark.parametrize("before_command", [False, True])
def test_script_cli_fresh_process_is_read_only_and_file_errors_propagate(
    saved, tmp_path, before_command
):
    experiment, config_file = saved
    path = tmp_path / "script ' $ [red]"
    path.write_text("#!/bin/bash\necho 'literal $ [red]'\n# no final newline")
    experiment.work_units()[1]._save(script_path=str(path))
    before = experiment._catalog.path.read_bytes()
    flag = f"--exex_config={config_file}"
    arguments = (
        [flag, "script", "101", "1"] if before_command else ["script", "101", "1", flag]
    )
    result = run_cli(
        *arguments, env={**os.environ, "EXEX_CONFIG": "/missing/config"}, check=True
    )
    assert result.stdout == path.read_text()
    path.unlink()
    result = run_cli("script", "101", "1")
    assert result.returncode != 0
    assert experiment._catalog.path.read_bytes() == before


def test_version_help_and_launch_argument_forwarding_are_preserved(tmp_path):
    environment = {**os.environ, "EXEX_CONFIG": "/missing/config"}
    assert run_cli("version", env=environment, check=True).stdout.startswith("exex ")
    assert "experiments" in run_cli("--help", env=environment, check=True).stdout
    launcher = tmp_path / "launcher.py"
    launcher.write_text("""from absl import flags
from exex.xm_cluster import config
message = flags.DEFINE_string("message", "", "Test forwarded argument.")
def main(argv):
    print(repr(message.value))
    print(config.EXEX_CONFIG.value)
""")
    result = run_cli(
        "launch",
        launcher,
        "--",
        "--exex_config=/forwarded/config",
        "--message=literal $ ' --flag",
        env=environment,
        check=True,
    )
    assert repr("literal $ ' --flag") in result.stdout
    assert "/forwarded/config" in result.stdout
