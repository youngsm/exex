# Cooperative Slurm continuation

## API

```python
executor = xc.Slurm(..., mode="sbatch")  # or "salloc"
experiment.add(
    xm.Job(executable, executor),
    outputs={"checkpoint": "checkpoint", "metrics": "metrics.json"},
    continuation=xc.Continuation(
        checkpoint="checkpoint", max_attempts=8, pause_before=120,
    ),
)
```

The worker checks `execution.pause_requested()` at a safe point, finishes writing
the checkpoint beneath `LXM_OUTPUT_DIR`, calls `execution.mark_paused()` from its
reporting process, and exits successfully. Distributed applications coordinate
their own ranks and finish asynchronous saves before marking the pause. The next
attempt restores the retained artifact from `LXM_INPUT_DIR/checkpoint`.

These optional helpers live in `lxm3.execution` and use only the standard library.
Install this fork in the application environment to import them. They do not
import an ML framework or install signal handlers. A non-Python program can use
the same environment/file contract: poll `LXM_PAUSE_REQUEST`, then create the
empty file at `LXM_PAUSE_READY` after saving and before successful exit. LXM3
supplies fresh paths for each attempt, including under container cleanenv modes.

The [launcher](../examples/continuation/launch.py) runs a
[standard-library counter](../examples/continuation/worker.py) using a configured
cluster and existing runtime. For example, with LXM3 available in the image:

```bash
lxm3 launch examples/continuation/launch.py \
  --target=nersc --mode=salloc --runtime=shifter --image="$WORKER_IMAGE" \
  --resource=account=m5238_g --resource=qos=interactive \
  --resource=constraint=gpu --resource=nodes=1 --resource=gpus-per-node=4 \
  --workdir_root=/pscratch/sd/y/youngsam/lxm3/work
```

Choose `--mode=sbatch` and a batch QoS explicitly for native requeue. Both modes
run the entrypoint once; a multi-node application still launches its own workers.
The attached mode uses a one-task overlapping `srun` driver on a compute node,
not the login node. Its entrypoint can launch application-owned steps across the
allocation; LXM3 does not pick a framework's distributed launch protocol.

## Lifecycle and inspection

`max_attempts` includes the first attempt. `pause_before` is seconds before the
allocation deadline; the margin must include both application checkpointing and
artifact capture. Explicit walltime is required. Only the checkpoint output is
required during a pause; all declared outputs are required on completion.

Batch continuation requeues the native job after a verified checkpoint capture;
it does not depend on the author process. Attached `salloc` execution holds the
experiment context open and acquires successive allocations until completion or
the attempt limit. QoS is explicit; choosing a mode does not choose a QoS. An
attached chain is not a detached service and does not promise survival across
driver/SSH loss. Submission/SSH errors propagate without retry or reconciliation.

Successful pauses at the budget limit remain paused, not completed. Application
errors, missing outputs, capture errors, and uncheckpointed scheduler restarts
do not silently continue. `stop()` cancels owned execution and prevents future
attempts. Checkpoints, logs and receipts have attempt-specific locations; a later
failure cannot overwrite an earlier checkpoint. Ordinary jobs are unchanged.

The catalog stores the policy and the execution-site directory. The site stores
an atomic current-state receipt plus separate `attempts/<number>/record.json`,
links and immutable artifacts. Workers never open the author's SQLite database.
The existing `get_status()`, `get_logs()`, `get_links()` and `artifacts()` work
after reopening. Logs/links refer to the latest attempt; artifacts expose the
latest successfully retained set, including an earlier checkpoint after a later
failure. Older attempt receipts and logs remain in their recorded directories.

Batch mode requests a Slurm batch-shell warning; the site driver also checks the
allocation's native end time, which provides the warning for attached execution.
Warnings can arrive early, and abrupt preemption may provide no warning at all.
On cancellation LXM3 records a stop flag before cancelling the owned allocation;
the attached driver also observes it while waiting for pending allocations.

Scope: singleton Slurm jobs, including application-owned distributed workers.
No array continuation, borrowed allocations, manual resume/budget extension,
automatic cross-site placement, abrupt-failure retry policy, or hidden watchdog.

## Qualification

The full LXM3/vendored-XManager suite passes **625 tests**, with the two existing
upstream integration tests deselected. The continuation tests run real generated
scripts and artifact capture, replacing only native Slurm commands. They cover
same-ID batch restarts, new-ID attached allocations, signal and deadline pauses,
budget exhaustion, payload/capture/requeue failures, cancellation while pending
and running, clean container control paths, preserved earlier checkpoints,
fresh-process inspection, and ordinary jobs without a continuation policy.
SSH failures propagate without replay. The staged site driver is standard-library
only and also runs on NERSC's Python 3.6 host environment.

### Live GPU-allocation checks, 2026-09-22 UTC

| Site/runtime | Mode | Experiment | Native IDs | Outcome |
| --- | --- | --- | --- | --- |
| S3DF/Singularity | Batch requeue | `1790055848912238616` | `38773544`, reused | Two attempts, completed |
| S3DF/Singularity | Attached chain | `1790055969340319760` | `38773587`, `38773609` | Two allocations, completed |
| NERSC/Shifter | Batch requeue | `1790055856943108785` | `58736056`, reused | Two attempts, completed |
| NERSC/Shifter | Attached chain | `1790055975758934541` | `58736134`, `58736168` | Two allocations, completed |
| S3DF/Singularity | Cancel attached | `1790056404036850740` | `38773752` | Stopped in attempt one; no continuation |
| NERSC/Shifter | Cancel attached | `1790056411126025612` | `58736587` | Stopped in attempt one; no continuation |

All launchers originated on S3DF. Each case was capped at two two-minute
allocations. S3DF used one A100 with `neutrino:default@ampere`, partition `ampere`,
and QoS `preemptable`; NERSC used one four-GPU node with `m5238_g`, debug QoS for
batch and interactive QoS for attached execution. Existing images were reused;
no image builds/imports or worker environment installations were needed.

The non-ML probe stages the unmodified `lxm3/execution.py` helper as a standalone
module, avoiding a new SDK install in the existing images. Each attempt records
its native ID, host, visible GPU UUIDs and restored counter. It pauses with a
retained checkpoint in attempt one, restores it in attempt two, and completes.
Both attached chains moved to different compute nodes. Cancellation was requested
from a fresh author process after the worker reported its link. Slurm reported
the released `salloc` allocations as `COMPLETED 0:0`; LXM3 correctly reports the
logical WorkUnit as stopped, using the explicit cancellation/driver outcome.

This qualifies allocation lifecycle, checkpoint handoff and container paths, not
ML numerical trajectories or arbitrary multi-node/nested-step topologies. Pimm
and its existing launcher were not modified by this slice.

Evidence/catalog/fetched outputs:
`/sdf/group/neutrino/youngsam/representations/lxm3-cooperative-qualification.gL162h`.
NERSC staging: `/pscratch/sd/y/youngsam/lxm3-cooperative-qualification-gL162h`.
S3DF reused the SIF cache in `lxm3-qualification.jmeYsG/s3df`; all new job and
log directories have unique experiment names. Requeue keeps attempt artifacts
and payload logs separate, even when the scheduler ID stays the same.
