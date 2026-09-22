"""Execution-site Slurm driver. Staged verbatim; Python 3.6+ standard library only.

Batch runs one attempt then asks Slurm to requeue. Attached execution owns its
salloc children. Neither path imports the author library or opens its catalog.
"""

import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def write_json(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class Driver:
    def __init__(self, config_path):
        self.config_path = str(config_path)
        self.config = json.loads(Path(config_path).read_text())
        self.root = Path(self.config["directory"])
        self.stop_file = self.root / "stop"
        self.cancelled = False
        self.pause = False
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self.cancel)
        signal.signal(signal.SIGUSR1, self.request_pause)

    def cancel(self, *_):
        self.cancelled = True

    def request_pause(self, *_):
        self.pause = True

    def read(self):
        path = self.root / "state.json"
        return json.loads(path.read_text()) if path.exists() else None

    def record(self, state):
        directory = self.root / "attempts" / str(state["attempt"])
        write_json(directory / "record.json", state)
        write_json(self.root / "state.json", state)

    def stopped(self):
        return self.cancelled or self.stop_file.exists()

    def begin(self, previous):
        number = previous["attempt"] + 1 if previous else 1
        directory = self.root / "attempts" / str(number)
        (directory / "links" / "0").mkdir(parents=True)
        state = dict(
            attempt=number,
            native_id=None,
            state="queued",
            message="Waiting for allocation",
            artifact_directory=previous["artifact_directory"]
            if previous
            else str(directory / "artifacts"),
            links_directory=str(directory / "links"),
            log_path=str(directory / "allocation.log"),
        )
        self.record(state)
        return state

    def inputs(self, number):
        bindings = dict(self.config["inputs"])
        if number > 1:
            directory = self.root / "attempts" / str(number - 1) / "artifacts" / "0"
            name = self.config["continuation"]["checkpoint"]
            identity = json.loads((directory / "manifest.json").read_text())[name]
            bindings[name] = dict(
                id=identity, archive_path=str(directory / (identity + ".tar"))
            )
        return bindings

    def deadline(self):
        value = os.environ.get("SLURM_JOB_END_TIME")
        if value:
            return float(value)
        # salloc does not export JOB_END_TIME on every cluster. Query once after
        # the allocation is granted, rather than starting the clock after setup.
        output = subprocess.check_output(
            ["scontrol", "show", "job", "--oneliner", os.environ["SLURM_JOB_ID"]],
            universal_newlines=True,
        )
        value = next(
            item.split("=", 1)[1]
            for item in output.split()
            if item.startswith("EndTime=")
        )
        return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").timestamp()

    def wait(self, process, *, request=None, deadline=float("inf"), allocation=False):
        while process.poll() is None:
            if self.stopped():
                self.stop_file.touch()
                # HUP makes salloc release both pending and granted allocations.
                sig = signal.SIGHUP if allocation else signal.SIGTERM
                try:
                    os.killpg(process.pid, sig)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            if request is not None and (self.pause or time.time() >= deadline):
                request.touch()
                request = None
            time.sleep(0.2)
        return process.wait()

    def attempt(self, number):
        state = self.read()
        if state["attempt"] != number or state["state"] != "queued":
            raise RuntimeError("Attempt has already started; refusing to replay it")
        if self.stopped():
            state.update(state="stopped", message="Cancelled before payload startup")
            self.record(state)
            return state
        directory = self.root / "attempts" / str(number)
        state.update(
            native_id=os.environ["SLURM_JOB_ID"],
            state="running",
            message="",
            started_at=time.time(),
            log_path=str(
                Path(self.config["log_directory"])
                / (self.config["job_name"] + "-attempt-" + str(number) + ".out")
            ),
        )
        self.record(state)
        request = directory / "links" / "0" / "pause-request"
        marker = directory / "links" / "0" / "paused"
        policy = self.config["continuation"]
        try:
            deadline = (
                self.deadline() - policy["pause_before"] if policy else float("inf")
            )
            Path(state["log_path"]).parent.mkdir(parents=True, exist_ok=True)
            with open(state["log_path"], "x") as log:
                with subprocess.Popen(
                    ["bash", self.config["payload"]],
                    env=dict(
                        os.environ,
                        EXEX_ATTEMPT_DIR=str(directory),
                        EXEX_ATTEMPT_INPUTS=json.dumps(self.inputs(number)),
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process:
                    code = self.wait(
                        process, request=request if policy else None, deadline=deadline
                    )
            if self.stopped():
                state.update(state="stopped", message="Execution cancelled")
            elif code:
                state.update(state="failed", message="Payload exited " + str(code))
            else:
                state.update(
                    state="paused" if marker.exists() else "completed",
                    message="",
                )
            state["exit_code"] = code
        except Exception as error:
            state.update(state="failed", message=str(error), exit_code=1)
            raise
        finally:
            if (directory / "artifacts" / "0" / "manifest.json").exists():
                state["artifact_directory"] = str(directory / "artifacts")
            state["finished_at"] = time.time()
            state["continuing"] = self.can_continue(state)
            self.record(state)
        return state

    def can_continue(self, state):
        policy = self.config["continuation"]
        return (
            policy is not None
            and state["state"] == "paused"
            and state["attempt"] < policy["max_attempts"]
            and not self.stopped()
        )

    def batch(self):
        previous = self.read()
        if self.stopped():
            return 1
        if previous and not self.can_continue(previous):
            raise RuntimeError("Scheduler restart has no verified, budgeted pause")
        state = self.attempt(self.begin(previous)["attempt"])
        if self.can_continue(state):
            try:
                subprocess.run(["scontrol", "requeue", state["native_id"]], check=True)
            except subprocess.CalledProcessError as error:
                state.update(state="failed", message=str(error), continuing=False)
                self.record(state)
                raise
        return 0 if state["state"] in ("completed", "paused") else 1

    def attached(self):
        previous = self.read()
        if previous is not None:
            raise RuntimeError("Attached driver has already run; refusing to replay it")
        while not self.stopped():
            state = self.begin(previous)
            command = [
                "salloc",
                "--kill-command=TERM",
                *self.config["salloc_options"],
                "srun",
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                *self.config["step_options"],
                "python3",
                str(Path(__file__).resolve()),
                "attempt",
                self.config_path,
                str(state["attempt"]),
            ]
            # An existing caller allocation must not turn this into a borrowed
            # step. Resource requests come exclusively from the frozen config.
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("SLURM_", "SALLOC_", "SBATCH_"))
                or key == "SLURM_CONF"
            }
            with open(state["log_path"], "x") as log:
                with subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process:
                    code = self.wait(process, allocation=True)
            state = self.read()
            if self.stopped():
                state.update(state="stopped", message="Attached execution cancelled")
                self.record(state)
                return 1
            # salloc can return zero on signals; require the worker's own receipt.
            if code or state["state"] not in ("completed", "paused"):
                state.update(
                    state="failed",
                    message=state["message"]
                    or "Allocation ended without a successful attempt",
                )
                self.record(state)
                return 1
            if not self.can_continue(state):
                return 0
            previous = state
        return 1


if __name__ == "__main__":
    driver = Driver(sys.argv[2])
    if sys.argv[1] == "attempt":
        outcome = driver.attempt(int(sys.argv[3]))
        sys.exit(0 if outcome["state"] in ("paused", "completed") else 1)
    with open(driver.root / "driver.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(driver.batch() if sys.argv[1] == "batch" else driver.attached())
