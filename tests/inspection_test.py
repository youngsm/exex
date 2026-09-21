"""Durable inspection through public APIs, including two independent processes."""

import asyncio
import json
import socket
import sqlite3
import subprocess
import sys
from concurrent import futures
from pathlib import Path
from unittest import mock

import pytest

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.clusters import slurm
from lxm3.xm_cluster import catalog
from lxm3.xm_cluster import experiment as experiment_lib
from lxm3.xm_cluster import inspection


@pytest.fixture(autouse=True)
def isolated_loop_policy(monkeypatch):
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    monkeypatch.delenv("LXM_PROJECT", raising=False)
    monkeypatch.delenv("LXM_CLUSTER", raising=False)
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    yield
    asyncio.set_event_loop_policy(previous)


@pytest.fixture
def config(tmp_path):
    return xc.Config(
        {
            "local": {"storage": {"staging": str(tmp_path / "author")}},
            "clusters": [
                {"name": "site", "storage": {"staging": str(tmp_path / "site")}}
            ],
        }
    )


def submit(config, tmp_path, *, count=None, log_directory=None):
    with mock.patch.object(slurm.SlurmCluster, "launch", return_value="789"):
        with xc.create_experiment(
            "inspect", project="test", config=config
        ) as experiment:
            executor = xc.Slurm(cluster="site", log_directory=log_directory)
            [executable] = experiment.package(
                [
                    xm.Packageable(
                        xc.SourceTree(xc.CommandList(["true"]), tmp_path, files=[]),
                        executor.Spec(),
                    )
                ]
            )
            payload = (
                xc.ArrayJob(executable, executor, args=[[str(i)] for i in range(count)])
                if count is not None
                else xm.Job(executable, executor)
            )
            experiment.add(payload)
    return experiment


def reopen(experiment, config):
    return xc.get_experiment(experiment.experiment_id, config=config).work_units()[1]


def accounting_row(unit, state, code="0:0", native_id="789", name=None):
    return f"{native_id}|{name or unit._record['job_name']}|{state}|{code}\n"


def observe(unit, output):
    with mock.patch.object(
        slurm.ssh, "run", return_value=subprocess.CompletedProcess([], 0, output)
    ):
        return unit.get_status()


@pytest.mark.parametrize("exit_code,expected", [(0, "completed"), (7, "failed")])
def test_local_launch_exit_and_reopen_in_a_fresh_process(tmp_path, exit_code, expected):
    launch = """
import json, subprocess, sys
from lxm3 import xm, xm_cluster as xc
config = xc.Config({'local': {'storage': {'staging': sys.argv[1]}}})
experiment = xc.create_experiment('fresh process', project='test', config=config)
try:
    with experiment:
        source = xc.SourceTree(xc.CommandList(["printf 'marker-a\\nmarker-b\\n'", 'exit ' + sys.argv[2]]), sys.argv[1], files=[])
        [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
        experiment.add(xm.Job(executable, xc.Local()))
except subprocess.CalledProcessError:
    pass
print(json.dumps(experiment.experiment_id))
"""
    root = tmp_path / "store with ' $ spaces"
    result = subprocess.run(
        [sys.executable, "-c", launch, str(root), str(exit_code)],
        capture_output=True,
        text=True,
        check=True,
    )
    experiment_id = json.loads(result.stdout.splitlines()[-1])
    read = """
import asyncio, json, sys
from lxm3 import xm, xm_cluster as xc
config = xc.Config({'local': {'storage': {'staging': sys.argv[1]}}})
experiment = xc.get_experiment(int(sys.argv[2]), config=config)
unit = experiment.work_units()[1]
with experiment:
    pass  # Merely entering a reopened context must not overwrite saved outcomes.
async def wait():
    try:
        assert await unit.wait_until_complete() is unit
    except xm.ExperimentUnitFailedError as error:
        assert error.work_unit is unit and unit.get_status().is_failed
asyncio.run(wait())
print(json.dumps({'id': experiment.experiment_id, 'units': list(experiment.work_units()), 'state': unit.get_status().state, 'logs': unit.get_logs(tail=100)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", read, str(root), str(experiment_id)],
        capture_output=True,
        text=True,
        check=True,
    )
    record = json.loads(result.stdout)
    assert record["id"] == experiment_id
    assert record["units"] == [1]
    assert record["state"] == expected
    assert "marker-a\nmarker-b" in record["logs"]
    config_file = tmp_path / "config.toml"
    config_file.write_text(f"[local.storage]\nstaging = {json.dumps(str(root))}\n")
    example = Path(__file__).parents[1] / "examples/inspection/reopen.py"
    output = subprocess.check_output(
        [
            sys.executable,
            str(example),
            str(experiment_id),
            "--config",
            str(config_file),
            "--logs",
        ],
        text=True,
    )
    assert f"WorkUnit 1: {expected}" in output
    assert "marker-a\nmarker-b" in output


def test_missing_experiment_does_not_create_storage(config, tmp_path):
    with pytest.raises(xm.NotFoundError):
        xc.get_experiment(1, config=config)
    assert not (tmp_path / "author").exists()
    xc.create_experiment("existing", config=config)
    with pytest.raises(xm.NotFoundError):
        xc.get_experiment(1, config=config)


def test_reopen_is_read_only_and_does_not_replay_or_poll(config, tmp_path):
    experiment = submit(config, tmp_path)
    database = tmp_path / "author/experiments.sqlite3"
    before = database.read_bytes()
    with mock.patch.object(
        subprocess, "Popen", side_effect=AssertionError("Unexpected external operation")
    ):
        reopened = xc.get_experiment(experiment.experiment_id, config=config)
        assert list(reopened.work_units()) == [1]
        assert reopened.work_units()[1].work_unit_id == 1
        assert reopened._project == "test"
        assert len(reopened.sources()) == 1
    assert database.read_bytes() == before


def test_database_records_package_version_without_schema_counter(config, tmp_path):
    import lxm3

    experiment = xc.create_experiment("version", config=config)
    with sqlite3.connect(tmp_path / "author/experiments.sqlite3") as db:
        assert (
            db.execute(
                "SELECT package_version FROM experiments WHERE id = ?",
                (experiment.experiment_id,),
            ).fetchone()[0]
            == lxm3.__version__
        )
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0


def test_ids_are_durable_and_allocated_transactionally(config):
    experiment = xc.create_experiment("ids", config=config)
    store = catalog.Catalog(config.local_settings().storage_root)
    with futures.ThreadPoolExecutor(4) as pool:
        ids = list(
            pool.map(
                lambda _: store.create_work_unit(experiment.experiment_id), range(12)
            )
        )
    assert sorted(ids) == list(range(1, 13))
    assert list(
        xc.get_experiment(experiment.experiment_id, config=config).work_units()
    ) == list(range(1, 13))


@pytest.mark.parametrize(
    "native,code,state",
    [
        ("PENDING", "0:0", "queued"),
        ("RUNNING", "0:0", "running"),
        ("COMPLETING", "0:0", "running"),
        ("SUSPENDED", "0:0", "paused"),
        ("COMPLETED", "0:0", "completed"),
        ("COMPLETED", "0:9", "failed"),
        ("COMPLETED", "7:0", "failed"),
        ("COMPLETED", "", "unknown"),
        ("FAILED", "1:0", "failed"),
        ("TIMEOUT", "0:15", "failed"),
        ("CANCELLED by 123", "0:0", "stopped"),
        ("FUTURE_STATE", "0:0", "unknown"),
    ],
)
def test_slurm_status_mapping(config, tmp_path, native, code, state):
    unit = reopen(submit(config, tmp_path), config)
    status = observe(unit, accounting_row(unit, native, code))
    assert status.state == state
    assert status.is_active == (state in {"queued", "running"})
    assert status.is_completed == (state == "completed")
    assert status.is_failed == (state == "failed")


def test_missing_or_recycled_slurm_id_is_unknown_not_cached_success(config, tmp_path):
    unit = reopen(submit(config, tmp_path), config)
    assert observe(unit, accounting_row(unit, "COMPLETED")).is_completed
    assert observe(unit, "").state == "unknown"
    assert (
        observe(
            unit, accounting_row(unit, "COMPLETED", name="different-experiment")
        ).state
        == "unknown"
    )


@pytest.mark.parametrize(
    "rows,state",
    [
        ([("789_1", "COMPLETED"), ("789_2", "COMPLETED")], "completed"),
        ([("789_1", "COMPLETED")], "unknown"),
        ([("789", "COMPLETED")], "unknown"),
        ([("789_1.batch", "COMPLETED"), ("789_2.batch", "COMPLETED")], "unknown"),
        ([("789_1", "RUNNING"), ("789_2", "PENDING")], "running"),
        ([("789_1", "FAILED"), ("789_2", "RUNNING")], "failed"),
    ],
)
def test_array_status_requires_all_expected_tasks(config, tmp_path, rows, state):
    unit = reopen(submit(config, tmp_path, count=2), config)
    output = "".join(
        accounting_row(unit, native, native_id=job_id) for job_id, native in rows
    )
    assert observe(unit, output).state == state


def test_inspection_uses_recorded_site_not_changed_profile(config, tmp_path):
    unit = reopen(submit(config, tmp_path), config)
    # Simulate the saved endpoint of a remote submission, then change the TOML.
    unit._save(hostname="original-alias", username="original-user")
    config._data["clusters"][0]["server"] = "different-alias"
    unit = xc.get_experiment(unit.experiment_id, config=config).work_units()[1]
    with mock.patch.object(
        slurm.ssh,
        "run",
        return_value=subprocess.CompletedProcess(
            [], 0, accounting_row(unit, "RUNNING")
        ),
    ) as run:
        assert unit.get_status().state == "running"
    assert run.call_args.kwargs == {
        "hostname": "original-alias",
        "username": "original-user",
    }
    argv = run.call_args.args[0]
    assert "--array" in argv and "--allocations" in argv and "--jobs=789" in argv


def test_ssh_failure_is_raised_without_retry(config, tmp_path):
    unit = reopen(submit(config, tmp_path), config)
    with mock.patch.object(
        slurm.ssh, "run", side_effect=subprocess.CalledProcessError(255, "ssh")
    ) as run:
        with pytest.raises(subprocess.CalledProcessError):
            unit.get_status()
    assert run.call_count == 1


@pytest.mark.parametrize(
    "count,task,suffix", [(None, None, "789"), (1, None, "789_1"), (2, 1, "789_2")]
)
def test_logs_use_recorded_custom_directory_and_zero_based_task(
    config, tmp_path, count, task, suffix
):
    directory = tmp_path / "logs ' $literal"
    directory.mkdir()
    unit = reopen(
        submit(config, tmp_path, count=count, log_directory=str(directory)), config
    )
    logfile = directory / f"{unit._record['job_name']}-{suffix}.out"
    logfile.write_text("first\nsecond\nlast\n")
    assert unit.get_logs(task=task, tail=2) == "second\nlast\n"
    assert unit.get_logs(task=task, tail=0) == ""


def test_array_logs_require_valid_task_and_missing_logs_raise(config, tmp_path):
    unit = reopen(submit(config, tmp_path, count=2), config)
    with pytest.raises(ValueError, match="task index"):
        unit.get_logs()
    for task, tail in [(-1, 2), (2, 2), (0, -1)]:
        with pytest.raises(ValueError):
            unit.get_logs(task=task, tail=tail)
    with pytest.raises(subprocess.CalledProcessError):
        unit.get_logs(task=0)


def test_submission_error_is_recorded_and_reopening_never_resubmits(config, tmp_path):
    experiment = xc.create_experiment("broken submission", config=config)
    job = xm.Job(xc.AppBundle("source", "true", "unused.zip"), xc.Slurm(cluster="site"))
    with mock.patch.object(
        slurm.SlurmCluster,
        "launch",
        side_effect=subprocess.CalledProcessError(255, "ssh"),
    ) as launch:
        with pytest.raises(subprocess.CalledProcessError):
            with experiment:
                experiment.add(job)
        unit = reopen(experiment, config)
        assert unit.get_status().state == "unknown"
        assert "Submission error" in unit.get_status().message
        assert launch.call_count == 1


def test_unfinished_reopened_local_is_unknown(config):
    experiment = xc.create_experiment("unfinished", config=config)
    unit_id = experiment._catalog.create_work_unit(experiment.experiment_id)
    experiment._catalog.update_work_unit(
        experiment.experiment_id, unit_id, backend="local", hostname=socket.getfqdn()
    )
    assert reopen(experiment, config).get_status().state == "unknown"


def test_status_predicates_do_not_mistake_unknown_for_completion():
    status = xc.WorkUnitStatus("unknown", "No matching scheduler evidence")
    assert not status.is_active and not status.is_completed and not status.is_failed
    assert status.message == "No matching scheduler evidence"


def test_local_live_status_uses_futures():
    future = futures.Future()
    handle = mock.Mock(future=future)
    assert inspection.local_status([handle]).state == "queued"
    future.set_running_or_notify_cancel()
    assert inspection.local_status([handle]).state == "running"
    future.set_result(None)
    assert inspection.local_status([handle]).state == "completed"


def test_local_array_preserves_failure_and_individual_logs(config, tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import sys\nprint('marker:' + sys.argv[1])\nsys.exit(int(sys.argv[1]))\n"
    )
    experiment = xc.create_experiment("local array", config=config)
    with pytest.raises(subprocess.CalledProcessError):
        with experiment:
            source = xc.SourceTree(
                xc.ModuleName("worker"), tmp_path, files=["worker.py"]
            )
            [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
            experiment.add(xc.ArrayJob(executable, xc.Local(), args=[["0"], ["7"]]))
    unit = reopen(experiment, config)
    assert unit.get_status().is_failed
    assert "marker:0" in unit.get_logs(task=0)
    assert "marker:7" in unit.get_logs(task=1)


def test_async_context_remains_awaitable_and_inspectable(config, tmp_path):
    async def launch():
        async with xc.create_experiment("async", config=config) as experiment:
            source = xc.SourceTree(xc.CommandList(["true"]), tmp_path, files=[])
            [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
            unit = await experiment.add(xm.Job(executable, xc.Local()))
            assert experiment.work_units()[unit.work_unit_id] is unit
            assert await unit.wait_until_complete() is unit
        return experiment

    experiment = asyncio.run(launch())
    assert reopen(experiment, config).get_status().is_completed


def test_federated_id_selects_the_recorded_native_cluster(config, tmp_path):
    unit = reopen(submit(config, tmp_path), config)
    unit._save(native_id="789;federated-site")
    with mock.patch.object(
        slurm.ssh,
        "run",
        return_value=subprocess.CompletedProcess(
            [], 0, accounting_row(unit, "COMPLETED")
        ),
    ) as run:
        assert unit.get_status().is_completed
    assert "--clusters=federated-site" in run.call_args.args[0]
    assert (
        f"--format=JobID%64,JobName%{len(unit._record['job_name']) + 1},State%32,ExitCode"
        in run.call_args.args[0]
    )


def test_log_connection_error_is_not_an_empty_success(config, tmp_path):
    unit = reopen(submit(config, tmp_path), config)
    with mock.patch.object(
        slurm.ssh, "run", side_effect=subprocess.CalledProcessError(255, "ssh")
    ) as run:
        with pytest.raises(subprocess.CalledProcessError):
            unit.get_logs()
    assert run.call_count == 1


def test_gridengine_inspection_is_explicitly_outside_this_slice(config, tmp_path):
    unit = reopen(submit(config, tmp_path), config)
    unit._save(backend="gridengine")
    with pytest.raises(NotImplementedError, match="GridEngine"):
        unit.get_status()


def test_on_site_inspection_does_not_depend_on_dns(config, tmp_path):
    unit = reopen(submit(config, tmp_path), config)
    assert unit._record["hostname"] == socket.gethostname()
    with mock.patch.object(
        socket, "getfqdn", side_effect=AssertionError("DNS must not be needed")
    ):
        with mock.patch.object(
            slurm.ssh,
            "run",
            return_value=subprocess.CompletedProcess(
                [], 0, accounting_row(unit, "COMPLETED")
            ),
        ) as run:
            assert unit.get_status().is_completed
    assert run.call_args.kwargs["hostname"] is None


def test_same_machine_different_user_keeps_ssh_identity(config, tmp_path):
    unit = reopen(submit(config, tmp_path), config)
    unit._save(username="other-submitter")
    with mock.patch.object(
        slurm.ssh,
        "run",
        return_value=subprocess.CompletedProcess(
            [], 0, accounting_row(unit, "COMPLETED")
        ),
    ) as run:
        assert unit.get_status().is_completed
    assert run.call_args.kwargs == {
        "hostname": socket.gethostname(),
        "username": "other-submitter",
    }
