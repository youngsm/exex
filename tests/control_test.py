"""Native cancellation and read-only completion waiting through XM's API."""

import asyncio
import socket
import subprocess
import threading
from concurrent import futures
from unittest import mock

import pytest

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.clusters import slurm
from lxm3.xm_cluster import experiment as experiment_lib
from lxm3.xm_cluster.execution.local import LocalExecutionHandle


@pytest.fixture
def unit(tmp_path, monkeypatch):
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    monkeypatch.setattr(experiment_lib, "_STATUS_POLL_INTERVAL", 0)
    config = xc.Config({"local": {"storage": {"staging": str(tmp_path)}}})
    experiment = xc.create_experiment("control", project="test", config=config)
    unit_id = experiment._catalog.create_work_unit(experiment.experiment_id)
    experiment._catalog.update_work_unit(
        experiment.experiment_id,
        unit_id,
        backend="slurm",
        hostname="recorded-site",
        username="recorded-user",
        native_id="789",
        job_name=f"control_{experiment.experiment_id}_{unit_id}",
    )
    yield xc.get_experiment(experiment.experiment_id, config=config).work_units()[
        unit_id
    ]
    asyncio.set_event_loop_policy(previous)


def accounting(unit, *rows):
    return subprocess.CompletedProcess(
        [],
        0,
        "".join(
            f"{job}|{unit._record['job_name']}|{state}|{code}\n"
            for job, state, code in rows
        ),
    )


@pytest.mark.parametrize("native_id", ["789", "789;native-cluster"])
@pytest.mark.parametrize("array", [False, True])
def test_stop_targets_recorded_endpoint_id_and_name(unit, native_id, array):
    unit._save(native_id=native_id, is_array=array, task_count=2 if array else 1)
    with mock.patch.object(slurm.ssh, "run") as run:
        unit.stop(message="Requested by researcher")
    run.assert_called_once_with(
        [
            "scancel",
            "--ctld",
            f"--name={unit._record['job_name']}",
            *(["--clusters=native-cluster"] if ";" in native_id else []),
            "789",
        ],
        hostname="recorded-site",
        username="recorded-user",
    )
    assert unit._record["message"] == "Requested by researcher"


def test_stop_on_author_host_does_not_use_ssh(unit):
    unit._save(hostname=socket.gethostname(), username=None)
    with mock.patch.object(slurm.ssh, "run") as run:
        unit.stop()
    assert run.call_args.kwargs["hostname"] is None


def test_stop_error_propagates_and_only_explicit_calls_redeliver(unit):
    with mock.patch.object(
        slurm.ssh, "run", side_effect=subprocess.CalledProcessError(255, "ssh")
    ) as run:
        with pytest.raises(subprocess.CalledProcessError):
            unit.stop()
        assert run.call_count == 1
        with pytest.raises(subprocess.CalledProcessError):
            unit.stop()
        assert run.call_count == 2
    with mock.patch.object(slurm.ssh, "run", return_value=accounting(unit)):
        assert unit.get_status().state == "unknown"


@pytest.mark.parametrize("mark_failed", [False, True])
def test_stop_intent_never_replaces_native_evidence_and_survives_reopen(
    unit, mark_failed
):
    with mock.patch.object(slurm.ssh, "run"):
        unit.stop(mark_as_failed=mark_failed, message="stop reason")
    unit = xc.get_experiment(
        unit.experiment_id, config=unit.experiment._config
    ).work_units()[1]
    for native, expected in [
        ("RUNNING", "running"),
        ("COMPLETED", "completed"),
        ("UNKNOWN", "unknown"),
        ("CANCELLED", "failed" if mark_failed else "stopped"),
    ]:
        with mock.patch.object(
            slurm.ssh, "run", return_value=accounting(unit, ("789", native, "0:0"))
        ):
            status = unit.get_status()
        assert status.state == expected
        if native == "CANCELLED":
            assert "stop reason" in status.message


def test_stop_without_accepted_handle_does_not_discover_or_claim_cancellation(unit):
    unit._save(backend=None, native_id=None, message="Submission error")
    before = unit._record
    with mock.patch.object(slurm.ssh, "run") as run:
        unit.stop(mark_as_failed=True, message="XM launch-error cleanup")
    run.assert_not_called()
    assert unit._record == before


@pytest.mark.parametrize(
    "backend,completed", [("slurm", True), ("local", False), ("gridengine", False)]
)
def test_unsupported_stop_is_non_mutating(unit, backend, completed):
    unit._save(backend=backend)
    before = unit._record
    with mock.patch.object(slurm.ssh, "run") as run:
        with pytest.raises(NotImplementedError):
            unit.stop(mark_as_completed=completed)
    run.assert_not_called()
    assert unit._record == before


def test_wait_preserves_xm_wrapper_and_returns_same_unit_without_writes(unit):
    before = unit._record
    responses = [accounting(unit)] + [
        accounting(unit, ("789", state, "0:0"))
        for state in ("PENDING", "RUNNING", "COMPLETED")
    ]
    waiter = unit.wait_until_complete()
    assert isinstance(waiter, xm.WorkUnitCompletedAwaitable)
    assert waiter.work_unit is unit
    with mock.patch.object(slurm.ssh, "run", side_effect=responses) as run:
        assert asyncio.run(waiter) is unit
    assert run.call_count == 4
    assert all(call.args[0][0] == "sacct" for call in run.call_args_list)
    assert unit._record == before


@pytest.mark.parametrize(
    "native,code,error_type",
    [
        ("FAILED", "1:0", xm.ExperimentUnitFailedError),
        ("COMPLETED", "7:0", xm.ExperimentUnitFailedError),
        ("CANCELLED by 123", "0:15", xm.ExperimentUnitNotCompletedError),
        ("SUSPENDED", "0:0", xm.ExperimentUnitNotCompletedError),
    ],
)
def test_wait_raises_existing_xm_error_with_unit(unit, native, code, error_type):
    with mock.patch.object(
        slurm.ssh, "run", return_value=accounting(unit, ("789", native, code))
    ):
        with pytest.raises(error_type) as caught:
            asyncio.run(unit.wait_until_complete())
    assert caught.value.work_unit is unit
    assert native in str(caught.value)


def test_wait_ssh_error_is_not_retried_or_converted_to_unknown(unit):
    with mock.patch.object(
        slurm.ssh, "run", side_effect=subprocess.CalledProcessError(255, "ssh")
    ) as run:
        with pytest.raises(subprocess.CalledProcessError):
            asyncio.run(unit.wait_until_complete())
    assert run.call_count == 1


@pytest.mark.parametrize("second", [None, "RUNNING", "COMPLETED", "FAILED"])
def test_array_wait_requires_every_success_and_never_cancels_siblings(unit, second):
    unit._save(is_array=True, task_count=2)
    rows = [("789_1", "COMPLETED", "0:0")]
    if second:
        rows.append(("789_2", second, "0:0"))

    async def wait():
        return await asyncio.wait_for(unit.wait_until_complete(), timeout=0.05)

    with mock.patch.object(
        slurm.ssh, "run", return_value=accounting(unit, *rows)
    ) as run:
        if second == "COMPLETED":
            assert asyncio.run(wait()) is unit
        else:
            with pytest.raises(
                xm.ExperimentUnitFailedError if second == "FAILED" else TimeoutError
            ):
                asyncio.run(wait())
    assert all(call.args[0][0] == "sacct" for call in run.call_args_list)


def test_blocked_status_query_does_not_block_event_loop_or_cancel_job(unit):
    release = threading.Event()
    response = accounting(unit, ("789", "RUNNING", "0:0"))

    def slow_query(*args, **kwargs):
        assert release.wait(timeout=2), "event loop was blocked"
        return response

    async def wait():
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(unit.wait_until_complete(), timeout=0.05)
        finally:
            release.set()

    with mock.patch.object(slurm.ssh, "run", side_effect=slow_query) as run:
        asyncio.run(wait())
    assert run.call_count == 1
    assert run.call_args.args[0][0] == "sacct"


@pytest.mark.parametrize("running", [False, True])
def test_local_wait_timeout_leaves_execution_future_alive(unit, running):
    future = futures.Future()
    if running:
        future.set_running_or_notify_cancel()
    unit._local_handles = [LocalExecutionHandle(future, "unused")]
    unit._save(backend="local", native_id=None)

    async def wait():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(unit.wait_until_complete(), timeout=0.02)
        assert not future.done()
        future.set_result(None)
        assert await unit.wait_until_complete() is unit

    with mock.patch.object(slurm.ssh, "run") as run:
        asyncio.run(wait())
    run.assert_not_called()


@pytest.mark.parametrize("outcome", ["completed", "failed", "stopped"])
def test_local_live_wait_uses_real_future_outcomes(unit, outcome):
    future = futures.Future()
    if outcome == "completed":
        future.set_result(None)
    elif outcome == "failed":
        future.set_exception(subprocess.CalledProcessError(7, "worker"))
    else:
        future.cancel()
    unit._local_handles = [LocalExecutionHandle(future, "unused")]
    if outcome == "completed":
        assert asyncio.run(unit.wait_until_complete()) is unit
    else:
        error_type = (
            xm.ExperimentUnitFailedError
            if outcome == "failed"
            else xm.ExperimentUnitNotCompletedError
        )
        with pytest.raises(error_type) as caught:
            asyncio.run(unit.wait_until_complete())
        assert caught.value.work_unit is unit


def test_local_wait_timeout_and_loop_shutdown_do_not_cancel_queued_execution(unit):
    future = futures.Future()
    unit._local_handles = [LocalExecutionHandle(future, "unused")]

    async def wait():
        await asyncio.wait_for(unit.wait_until_complete(), timeout=0.02)

    with pytest.raises(TimeoutError):
        asyncio.run(wait())
    assert not future.done()
    future.set_result(None)
    assert asyncio.run(unit.wait_until_complete()) is unit


def test_reopened_local_wait_observes_author_record_without_process_discovery(unit):
    unit._save(backend="local", native_id=None)

    async def wait():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(unit.wait_until_complete(), timeout=0.02)
        unit._save(state="completed")
        assert await unit.wait_until_complete() is unit

    with mock.patch.object(slurm.ssh, "run") as run:
        asyncio.run(wait())
    run.assert_not_called()
