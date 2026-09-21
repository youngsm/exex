"""Read-only execution inspection using recorded sites, not current profiles."""

import getpass
import os
import socket
from dataclasses import dataclass

from lxm3 import xm
from lxm3.clusters import slurm
from lxm3.clusters import ssh


@dataclass(frozen=True)
class WorkUnitStatus(xm.ExperimentUnitStatus):
    state: str
    _message: str = ""

    @property
    def message(self):
        return self._message

    @property
    def is_active(self):
        return self.state in {"created", "queued", "running"}

    @property
    def is_completed(self):
        return self.state == "completed"

    @property
    def is_failed(self):
        return self.state == "failed"


_SLURM_STATES = {
    **dict.fromkeys(("PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD"), "queued"),
    **dict.fromkeys(("RUNNING", "COMPLETING", "RESIZING", "STAGE_OUT"), "running"),
    **dict.fromkeys(
        (
            "FAILED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "BOOT_FAIL",
            "DEADLINE",
            "PREEMPTED",
        ),
        "failed",
    ),
    "COMPLETED": "completed",
    "CANCELLED": "stopped",
    "SUSPENDED": "paused",
}


def aggregate(statuses):
    for state in (
        "failed",
        "stopped",
        "unknown",
        "running",
        "paused",
        "queued",
        "completed",
    ):
        matches = [status for status in statuses if status.state == state]
        if matches:
            return matches[0]
    return WorkUnitStatus("unknown", "No execution has been recorded")


def local_status(handles):
    statuses = []
    for handle in handles:
        future = handle.future
        if future.cancelled():
            status = WorkUnitStatus("stopped")
        elif not future.done():
            status = WorkUnitStatus("running" if future.running() else "queued")
        else:
            error = future.exception()
            status = (
                WorkUnitStatus("failed", str(error))
                if error
                else WorkUnitStatus("completed")
            )
        statuses.append(status)
    return aggregate(statuses)


def _hostname(record):
    host = record["hostname"]
    same_host = host == socket.gethostname() or host == socket.getfqdn()
    same_user = record["username"] in (None, getpass.getuser())
    return None if same_host and same_user else host


def get_status(record):
    if record["backend"] == "gridengine":
        raise NotImplementedError("GridEngine inspection is not implemented")
    if record["backend"] != "slurm" or not record["native_id"]:
        return WorkUnitStatus(record["state"], record["message"])
    cluster = slurm.SlurmCluster(_hostname(record), record["username"])
    rows = cluster.accounting(record["native_id"], record["job_name"])
    job_id = record["native_id"].split(";", 1)[0]
    statuses = []
    for task in range(record["task_count"]):
        native_id = f"{job_id}_{task + 1}" if record["is_array"] else job_id
        native_state, exit_code = rows.get(native_id, ("UNKNOWN", ""))
        state = _SLURM_STATES.get(native_state.split()[0].rstrip("+"), "unknown")
        if state == "completed" and exit_code != "0:0":
            state = "failed" if exit_code else "unknown"
        statuses.append(
            WorkUnitStatus(state, f"{native_id}: {native_state} {exit_code}".strip())
        )
    return aggregate(statuses)


def get_logs(record, *, task=None, tail=200):
    if record["backend"] not in {"local", "slurm"}:
        raise NotImplementedError("Logs require a recorded Local or Slurm execution")
    if task is None and record["task_count"] > 1:
        raise ValueError("Choose a zero-based task index for array logs")
    task = 0 if task is None else task
    if not 0 <= task < record["task_count"] or tail < 0:
        raise ValueError("task must be in range and tail must be nonnegative")
    if record["backend"] == "local":
        filename = f"task-{task}.log"
    else:
        job_id = record["native_id"].split(";", 1)[0]
        suffix = f"{job_id}_{task + 1}" if record["is_array"] else job_id
        filename = f"{record['job_name']}-{suffix}.out"
    return ssh.run(
        [
            "tail",
            "-n",
            str(tail),
            "--",
            os.path.join(record["log_directory"], filename),
        ],
        hostname=_hostname(record),
        username=record["username"],
    ).stdout
