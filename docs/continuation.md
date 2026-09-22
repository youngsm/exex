# Cooperative continuation

Attach a bounded policy to a singleton Slurm job:

```python
executor = xc.Slurm(
    cluster="mycluster", mode="sbatch", walltime=3600,
    resources={"account": "my-account", "gpus": 1},
)
experiment.add(
    xm.Job(executable, executor),
    outputs={"checkpoint": "checkpoint", "metrics": "metrics.json"},
    continuation=xc.Continuation(
        checkpoint="checkpoint", max_attempts=3, pause_before=120,
    ),
)
```

The worker owns safe points and checkpoint contents:

```python
from exex import execution

if execution.pause_requested():
    save_complete_checkpoint()  # Application-owned; finish asynchronous writes.
    execution.mark_paused()
    return                     # Exit successfully on every rank.
```

On the next attempt, restore `EXEX_INPUT_DIR/checkpoint`. Distributed applications
must agree on pausing across all ranks and checkpoint collectively where required.
Call `mark_paused()` only from the reporting process. The application runtime
needs exex installed to import these helpers; non-Python workers can poll
`EXEX_PAUSE_REQUEST` and create the empty file `EXEX_PAUSE_READY` instead.

`mode="sbatch"` uses native requeue and survives launcher exit. `mode="salloc"`
owns successive attached allocations, keeps the experiment context open, and
does not promise survival across terminal/SSH loss. Select the appropriate QoS
explicitly. Attached execution runs a one-task driver on a compute node.

The next attempt requires a pause marker, successful worker exit, successful
checkpoint capture, and remaining budget. `max_attempts` includes the initial
attempt. Budget exhaustion leaves the WorkUnit **paused**, not completed.
The pause margin must cover safe-point latency, checkpoint saving, and capture;
explicit walltime longer than the margin is required. During a pause only the
checkpoint output is captured; on completion every declared output is required.

Attempts have separate logs, links, checkpoints and receipts, even when Slurm
reuses its job ID. Existing inspection and cancellation APIs operate on the
logical WorkUnit. Failed payloads, missing checkpoints, unexpected scheduler
restarts, and SSH failures are not retried. Abrupt preemption may give no warning.

No array continuation, borrowed allocations, manual budget extension, automatic
cross-site placement, or hidden watchdog is included. Application-owned workers
are supported, but every distributed topology needs qualification.

`examples/continuation/launch.py` is a complete counter example with selectable
mode and runtime. It demonstrates the contract without an ML framework.
