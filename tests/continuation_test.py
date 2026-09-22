"""Run real generated scripts; replace only native Slurm commands."""

import ast
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

import lxm3
from lxm3 import execution
from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.clusters import slurm
from lxm3.xm_cluster import experiment as experiment_lib
from lxm3.xm_cluster.executables import ContainerImage
from lxm3.xm_cluster.executables import ContainerImageType
from lxm3.xm_cluster.execution import continuation as driver
from lxm3.xm_cluster.execution.slurm import AttachedHandle
from lxm3.xm_cluster.execution.slurm import SlurmJobScriptBuilder

WORKER = """
import os, time
from pathlib import Path
from lxm3 import execution
root = Path(os.environ["LXM_OUTPUT_DIR"])
resume = Path(os.environ.get("LXM_INPUT_DIR", "/absent")) / "checkpoint"
step = int(resume.read_text()) + 1 if resume.exists() else 1
print("step", step, flush=True)
execution.link("report", "https://example.org/step/" + str(step))
if os.environ.get("WAIT"):
    time.sleep(30)
if not os.environ.get("MISSING"):
    (root / "checkpoint").write_text(str(step))
if step < int(os.environ.get("STEPS", "3")):
    deadline = time.monotonic() + 5
    while not execution.pause_requested():
        assert time.monotonic() < deadline, "No pause request received"
        time.sleep(0.01)
    execution.mark_paused()
    if os.environ.get("CRASH") or (os.environ.get("CRASH_SECOND") and step == 2):
        raise RuntimeError("crash after checkpoint and marker")
else:
    (root / "metrics.json").write_text('{"finished": true}')
"""

NATIVE = """
import json, os, subprocess, sys, time
from pathlib import Path
command = Path(sys.argv[0]).name
root = Path(os.environ["FAKE_SLURM"])
args = sys.argv[1:]
if command == "scontrol":
    if args[0] == "requeue":
        if os.environ.get("REQUEUE_FAIL"):
            sys.exit(1)
        with (root / "requeues").open("a") as stream:
            stream.write(args[1] + "\\n")
    else:
        print("EndTime=" + time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 1)))
elif command in ("salloc", "srun"):
    with (root / "commands").open("a") as stream:
        stream.write(json.dumps([command, *args]) + "\\n")
    while args and args[0].startswith("--"):
        args.pop(0)
    environment = dict(os.environ)
    if command == "salloc":
        assert "SLURM_JOB_ID" not in environment, "borrowed caller allocation"
        counter = root / "allocations"
        number = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(number))
        if os.environ.get("PENDING"):
            time.sleep(30)
        environment.update(SLURM_JOB_ID=str(100 + number), SLURM_JOB_END_TIME=str(time.time() + 1))
    sys.exit(subprocess.call(args, env=environment))
elif command == "sacct":
    job_id = next(item.split("=", 1)[1] for item in args if item.startswith("--jobs="))
    print(job_id + "|" + os.environ["FAKE_JOB_NAME"] + "|COMPLETED|0:0")
"""


@pytest.fixture
def site(tmp_path, monkeypatch):
    old = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    for key in ("LXM_PROJECT", "LXM_CLUSTER", "LXM_PAUSE_REQUEST", "LXM_PAUSE_READY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHONPATH", str(Path(lxm3.__file__).parent.parent))
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("salloc", "srun", "scontrol", "sacct"):
        path = binary / name
        path.write_text("#!" + sys.executable + "\n" + NATIVE)
        path.chmod(0o755)
    (binary / "python3").write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
    )
    (binary / "python3").chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + ":" + os.environ["PATH"])
    monkeypatch.setenv("FAKE_SLURM", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 1))
    (tmp_path / "worker.py").write_text(WORKER)
    config = xc.Config(
        {
            "local": {"storage": {"staging": str(tmp_path / "author")}},
            "clusters": [
                {
                    "name": "test",
                    "storage": {"staging": str(tmp_path / "site $ ' space")},
                }
            ],
        }
    )
    yield config
    asyncio.set_event_loop_policy(old)


def launch(site, tmp_path, *, mode="sbatch", budget=3, env=None, continuation=True):
    executor = xc.Slurm(
        cluster="test", mode=mode, walltime=60, workdir_root=str(tmp_path / "unpack")
    )
    with mock.patch.object(slurm.SlurmCluster, "launch", return_value="42"):
        with xc.create_experiment("continuation", config=site) as experiment:
            source = xc.SourceTree(
                xc.ModuleName("worker"), tmp_path, files=["worker.py"]
            )
            [bundle] = experiment.package([xm.Packageable(source, executor.Spec())])
            experiment.add(
                xm.Job(bundle, executor, env_vars=env or {}),
                outputs={"checkpoint": "checkpoint", "metrics": "metrics.json"},
                continuation=xc.Continuation(
                    checkpoint="checkpoint", max_attempts=budget, pause_before=10
                )
                if continuation
                else None,
            )
    return xc.get_experiment(experiment.experiment_id, config=site).work_units()[1]


def state(unit):
    return json.loads(
        (Path(unit._record["execution_directory"]) / "state.json").read_text()
    )


def run_batch(unit):
    return subprocess.run(
        ["bash", unit._record["script_path"]],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_worker_helpers_are_optional_and_do_not_exit(tmp_path, monkeypatch):
    monkeypatch.delenv("LXM_PAUSE_REQUEST", raising=False)
    assert not execution.pause_requested()
    request, ready = tmp_path / "request", tmp_path / "ready"
    monkeypatch.setenv("LXM_PAUSE_REQUEST", str(request))
    monkeypatch.setenv("LXM_PAUSE_READY", str(ready))
    request.touch()
    assert execution.pause_requested()
    assert execution.mark_paused() is None
    assert ready.exists()


def test_batch_requeues_only_verified_pause_and_reopens_results(
    site, tmp_path, monkeypatch
):
    unit = launch(site, tmp_path)
    monkeypatch.setenv("FAKE_JOB_NAME", unit._record["job_name"])
    for number in range(1, 4):
        result = run_batch(unit)
        assert result.returncode == 0, result.stderr + unit.get_logs()
        assert state(unit)["attempt"] == number
        assert state(unit)["state"] == ("completed" if number == 3 else "paused")
        assert unit.get_status().state == ("completed" if number == 3 else "queued")
        artifacts = unit.artifacts()
        assert set(artifacts) == (
            {"checkpoint", "metrics"} if number == 3 else {"checkpoint"}
        )
        assert artifacts["checkpoint"].fetch(
            tmp_path / f"download-{number}"
        ).read_text() == str(number)
        assert unit.get_links()["report"].endswith(str(number))
        assert f"step {number}" in unit.get_logs()
    assert (tmp_path / "requeues").read_text().splitlines() == ["42", "42"]
    assert not list((tmp_path / "unpack").iterdir())
    root = Path(unit._record["execution_directory"])
    assert len(list(root.glob("attempts/*/record.json"))) == 3
    assert run_batch(unit).returncode != 0  # never replay completed work
    assert state(unit)["attempt"] == 3


@pytest.mark.parametrize("budget", [1, 2])
def test_budget_limit_is_paused_not_completed(site, tmp_path, budget):
    unit = launch(site, tmp_path, budget=budget)
    for _ in range(budget):
        assert run_batch(unit).returncode == 0
    assert state(unit)["state"] == "paused"
    assert state(unit)["attempt"] == budget
    requeues = tmp_path / "requeues"
    assert (
        len(requeues.read_text().splitlines()) if requeues.exists() else 0
    ) == budget - 1
    assert run_batch(unit).returncode != 0


@pytest.mark.parametrize("env", [{"CRASH": "1"}, {"MISSING": "1"}])
def test_marker_does_not_override_payload_or_capture_failure(site, tmp_path, env):
    unit = launch(site, tmp_path, env=env)
    result = run_batch(unit)
    assert result.returncode != 0
    assert state(unit)["state"] == "failed"
    assert unit.artifacts() == {}
    assert not (tmp_path / "requeues").exists()
    assert run_batch(unit).returncode != 0
    assert state(unit)["attempt"] == 1


def test_requeue_failure_is_not_retried_and_keeps_checkpoint(
    site, tmp_path, monkeypatch
):
    unit = launch(site, tmp_path)
    monkeypatch.setenv("REQUEUE_FAIL", "1")
    assert run_batch(unit).returncode != 0
    assert state(unit)["state"] == "failed"
    assert set(unit.artifacts()) == {"checkpoint"}
    assert run_batch(unit).returncode != 0


def test_native_end_time_fallback_and_incomplete_restart_guard(
    site, tmp_path, monkeypatch
):
    unit = launch(site, tmp_path)
    monkeypatch.delenv("SLURM_JOB_END_TIME")
    assert run_batch(unit).returncode == 0
    path = Path(unit._record["execution_directory"]) / "state.json"
    interrupted = state(unit)
    interrupted["state"] = "running"
    path.write_text(json.dumps(interrupted))
    assert run_batch(unit).returncode != 0
    assert state(unit)["attempt"] == 1


@pytest.mark.parametrize("continuation", [False, True])
def test_attached_chain_waits_for_completion_and_owns_new_allocations(
    site, tmp_path, continuation
):
    unit = launch(
        site,
        tmp_path,
        mode="salloc",
        continuation=continuation,
        env={} if continuation else {"STEPS": "1"},
    )
    expected = 3 if continuation else 1
    assert state(unit)["state"] == "completed"
    assert state(unit)["attempt"] == expected
    assert (tmp_path / "allocations").read_text() == str(expected)
    assert unit.artifacts()["checkpoint"].fetch(tmp_path / "final").read_text() == str(
        expected
    )
    assert not (tmp_path / "requeues").exists()
    receipts = sorted(
        (Path(unit._record["execution_directory"]) / "attempts").glob("*/record.json")
    )
    assert [json.loads(path.read_text())["native_id"] for path in receipts] == [
        str(101 + i) for i in range(expected)
    ]


def test_batch_signal_requests_pause_at_safe_point(site, tmp_path, monkeypatch):
    unit = launch(site, tmp_path, budget=1)
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 3600))
    process = subprocess.Popen(
        ["bash", unit._record["script_path"]], stdout=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + 5
        while not (Path(unit._record["execution_directory"]) / "state.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.send_signal(signal.SIGUSR1)
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert state(unit)["state"] == "paused"


def test_stop_prevents_requeue_and_retains_completed_previous_checkpoint(
    site, tmp_path
):
    unit = launch(site, tmp_path)
    assert run_batch(unit).returncode == 0
    with mock.patch.object(slurm.SlurmCluster, "cancel") as cancel:
        unit.stop()
    cancel.assert_called_once_with("42", unit._record["job_name"])
    assert run_batch(unit).returncode != 0
    assert state(unit)["attempt"] == 1
    assert set(unit.artifacts()) == {"checkpoint"}


def test_invalid_policy_is_rejected_before_native_submission(site, tmp_path):
    for fields in [
        dict(checkpoint="../bad", max_attempts=1),
        dict(checkpoint="checkpoint", max_attempts=0),
    ]:
        with pytest.raises(ValueError):
            xc.Continuation(**fields)
    with pytest.raises(ValueError):
        xc.Slurm(mode="guess")
    experiment = xc.create_experiment("bad", config=site)
    with pytest.raises(ValueError, match="declared output"):
        experiment.add(
            None, continuation=xc.Continuation(checkpoint="checkpoint", max_attempts=2)
        )
    assert experiment.work_units() == {}


def test_later_failure_preserves_previous_checkpoint(site, tmp_path):
    unit = launch(site, tmp_path, env={"CRASH_SECOND": "1"})
    assert run_batch(unit).returncode == 0
    artifact = unit.artifacts()["checkpoint"]
    assert run_batch(unit).returncode != 0
    assert state(unit)["state"] == "failed"
    assert unit.artifacts()["checkpoint"].id == artifact.id
    assert artifact.fetch(tmp_path / "previous").read_text() == "1"


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("mark_failed", [False, True])
def test_stopping_attached_job_cancels_driver_and_prevents_future_slots(
    site, tmp_path, monkeypatch, pending, mark_failed
):
    if pending:
        monkeypatch.setenv("PENDING", "1")
    executor = xc.Slurm(cluster="test", mode="salloc", walltime=60)
    experiment = xc.create_experiment("stop-attached", config=site)
    with pytest.raises(subprocess.CalledProcessError):
        with experiment:
            source = xc.SourceTree(
                xc.ModuleName("worker"), tmp_path, files=["worker.py"]
            )
            [bundle] = experiment.package([xm.Packageable(source, executor.Spec())])
            experiment.add(
                xm.Job(bundle, executor, env_vars={"WAIT": "1"}),
                outputs={"checkpoint": "checkpoint"},
                continuation=xc.Continuation(
                    checkpoint="checkpoint", max_attempts=3, pause_before=10
                ),
            )
            experiment._wait_for_tasks()
            unit = experiment.work_units()[1]
            monkeypatch.setenv("FAKE_JOB_NAME", unit._record["job_name"])
            deadline = time.monotonic() + 10
            while not (tmp_path / "allocations").exists() or (
                not pending and state(unit)["state"] != "running"
            ):
                assert time.monotonic() < deadline
                time.sleep(0.02)
            with mock.patch.object(slurm.SlurmCluster, "cancel"):
                unit.stop(mark_as_failed=mark_failed, message="cancel test")
    assert state(unit)["state"] == "stopped"
    assert (tmp_path / "allocations").read_text() == "1"
    assert state(unit)["attempt"] == 1
    assert unit.get_status().state == ("failed" if mark_failed else "stopped")


@pytest.mark.parametrize("kind", ["singularity", "shifter"])
def test_container_controls_use_visible_attempt_paths(site, tmp_path, kind):
    bundle = xc.AppBundle(
        "probe",
        "true",
        "/source.tar",
        container_image=ContainerImage("image", ContainerImageType(kind)),
    )
    script = SlurmJobScriptBuilder().build(
        xm.Job(bundle, xc.Slurm()),
        "probe",
        str(tmp_path),
        outputs={"checkpoint": "state"},
        continuation={
            "checkpoint": "checkpoint",
            "max_attempts": 2,
            "pause_before": 10,
        },
    )
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert '"$LXM_ATTEMPT_DIR/links/0"' in script
    if kind == "singularity":
        assert "LXM_PAUSE_REQUEST=/run/lxm3/links/pause-request" in script
    else:
        assert 'LXM_PAUSE_REQUEST="$LXM_ATTEMPT_DIR/links/0"/pause-request' in script
    assert "LXM_PAUSE_READY=" in script


@pytest.mark.parametrize("kind", ["local", "array", "walltime"])
def test_unsupported_execution_is_rejected_before_submission(site, tmp_path, kind):
    executor = (
        xc.Local()
        if kind == "local"
        else xc.Slurm(cluster="test", walltime=10 if kind == "walltime" else 60)
    )
    with mock.patch.object(slurm.SlurmCluster, "launch") as submit:
        with pytest.raises(ValueError):
            with xc.create_experiment("unsupported", config=site) as experiment:
                source = xc.SourceTree(
                    xc.ModuleName("worker"), tmp_path, files=["worker.py"]
                )
                [bundle] = experiment.package([xm.Packageable(source, executor.Spec())])
                job = (
                    xc.ArrayJob(bundle, executor, args=[[], []])
                    if kind == "array"
                    else xm.Job(bundle, executor)
                )
                experiment.add(
                    job,
                    outputs={"checkpoint": "state"},
                    continuation=xc.Continuation(
                        checkpoint="checkpoint", max_attempts=2, pause_before=10
                    ),
                )
        submit.assert_not_called()


def test_attached_transport_failure_propagates_once_without_retry():
    error = subprocess.CalledProcessError(255, "ssh")
    with mock.patch("lxm3.clusters.ssh.run", side_effect=error) as run:
        handle = AttachedHandle(
            ["bash", "/saved/job.sh"], hostname="nersc", username="user"
        )
        with pytest.raises(subprocess.CalledProcessError) as caught:
            handle.future.result(timeout=2)
    assert caught.value is error
    run.assert_called_once()


def test_site_driver_has_no_modern_python_syntax_requirement():
    ast.parse(Path(driver.__file__).read_text(), feature_version=(3, 6))
