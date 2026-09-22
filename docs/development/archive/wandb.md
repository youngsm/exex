# W&B and task links

The launcher sets tracking defaults; the application owns its W&B run, metrics
and checkpoints. There is no LXM3 metric writer or framework-specific dependency.
Install this fork with its optional SDK dependency (`pip install '.[wandb]'` from
the checkout). The worker's Python environment also needs this fork and W&B;
SourceTree does not install dependencies into an existing runtime.

## Runnable local example

From this checkout, with a configured local staging directory:

```sh
lxm3 launch examples/wandb/launch.py
```

This runs offline by default, logs three scalar rows and retains the tracking
state and scalar-history file as ordinary artifacts. It creates no remote W&B run.
To use your existing W&B login and publish a real run URL:

```sh
lxm3 launch examples/wandb/launch.py --wandb_mode=online \
  --wandb_entity=YOUR_TEAM --wandb_project=YOUR_PROJECT
```

The [launcher](../examples/wandb/launch.py) uses the existing
`configure_wandb(project, entity, group="{title}_{xid}_{wid}", mode="online")`
wrapper. It preserves the Job/ArrayJob and merges tracking defaults into the job
environment. Explicit application init arguments still take precedence. It sets
neither credentials nor run IDs, and no longer injects Git metadata from the
launcher's current directory. LXM3's retained source is the code provenance;
W&B's own SDK metadata collection remains its responsibility.

## Application API

```python
from lxm3.contrib import wandb as tracking

# checkpoint is application-owned and already contains restored training state.
saved = checkpoint.get("wandb")
with tracking.init(state=saved, history="new", config={"lr": 0.001}) as run:
    run.define_metric("train/global_step")
    run.define_metric("train/*", step_metric="train/global_step")
    run.log({"train/global_step": 100, "train/loss": 0.25})
    checkpoint["wandb"] = tracking.checkpoint_state(run)
    # Save the complete training checkpoint using your existing checkpoint code.
```

`init(*, state=None, history="new", **wandb_kwargs)` returns the native `wandb.Run`.
Use its logging, tables, images, artifacts and lifecycle directly. Invoke it only
in the reporting process, normally rank zero. It also reports the actual run URL
to the current LXM3 task, if there is one. Outside LXM3, tracking still works and
link reporting is a no-op.

| History | Behavior |
| --- | --- |
| `new` (default) | Fresh ID and empty history; records `lxm3/parent` when state is supplied. |
| `append` | Uses the saved entity/project/ID with `resume="must"`. Appends to existing history; does not remove later rows. |
| `fork` | Fresh ID, same saved entity/project, native `fork_from` through the saved last row. Requires server-side permission. |

Append/fork require saved state and online mode. Fork errors propagate unchanged;
there is no permission probe, retry, fallback or rewind implementation. A
checkpoint behind the server's history is **not** made consistent by append:
choose a new run, or explicitly request a permitted fork.

History selection belongs to `history=`, not native resume/fork/reinit arguments
or their `WANDB_*` environment equivalents. Native run settings and ordinary SDK
arguments otherwise pass through. Mode is selected from the init argument,
its settings, `WANDB_MODE`, then the SDK's global settings (normally `online`);
an existing `wandb.setup(wandb.Settings(mode="offline"))` setting is respected.
The selected mode is passed explicitly to the SDK.
Existing global runs are not reused. A default new ID does not inherit
`WANDB_RUN_ID`. Append uses the saved ID regardless of an explicit ID; append/fork
use the saved project/entity. New runs may choose a different project/entity.

`checkpoint_state(run)` returns entity, project, group, actual run ID and
`next_step` (the SDK's next history-row index), or `None` for disabled tracking.
It uses `max(run.step, run.starting_step)`: W&B 0.28.0 can report `step=0`
before the first new log after append/fork, despite existing history.
Commit pending metric rows before calling it. This does not log another row,
flush uploads, save a checkpoint or guarantee that the server has received the
history. The history index is **not** the training iteration; log your training
step as a separate metric as above. Replaying older checkpoints cannot make
already-uploaded history disappear.

Already using native W&B initialization? Keep it and call `tracking.attach(run)`
to publish its URL without changing its lifecycle or resume semantics. The helper
module imports no scheduler libraries; the SDK is imported only by `init()`.

Offline runs have no remote URL. Choose persistent storage with native `dir=` or
`WANDB_DIR`; otherwise temporary-workdir cleanup deletes the offline data. The
example retains the completed scalar-history file through successful output capture.
For offline data to survive a failed job, use a persistent site directory instead
of relying on success-only outputs.
For media, retain the associated W&B files too. Copying only the history file is
sufficient for this scalar-only example, not for arbitrary tracking data.

## Named links, without W&B

```python
from lxm3 import execution

execution.link("report", "https://example.org/results/123")
```

This writes a small atomic JSON receipt beside the retained job logs, not to
the author's database and not through success-only output capture. From another
process, using the same author catalog:

```python
from lxm3 import xm_cluster as xc

experiment = xc.get_experiment(1790000000000000000)  # replace with the printed ID
unit = experiment.work_units()[1]
print(unit.get_links())
# Arrays: unit.get_links(task=0), using zero-based task indices.
```

Links can be read while a task runs and survive payload failure/cancellation once
written. Missing receipts return `{}`; connection and storage errors propagate.
Repeated names replace their URL. Each task has its own file; use one reporting
process per task. The file is mutable: a requeued native task may replace a link;
this is not an attempt-history API. Receipts remain available only as long as the
site's retained log storage does.

The language-neutral contract is `LXM_LINKS_FILE`, a writable path to a JSON object
mapping names to URL strings. LXM3 exposes its parent in supported containers.
For Shifter, the retained-log directory must already be visible at its host path,
just like `workdir_root`; no additional bind mount is requested. Site mounts or
explicit `ShifterOptions.bind` control that visibility.
Non-Python programs can atomically replace that file themselves. Reporting never
contacts the URL, copies the external artifact or publishes credentials.

No pimm migration, scheduler cancellation policy, automatic continuation or W&B
permission management is included in this slice.

## Qualification (2026-09-21)

All 592 regressions pass; 2 upstream integration tests are deselected. Changed
Python files also pass formatting and lint checks.

The regression suite covers argument/environment precedence, preserved Job/ArrayJob
fields, all three history translations, unchanged SDK errors without fallback,
disabled tracking and worker imports without scheduler/ML libraries. Real W&B
0.28.0 offline subprocess tests check actual IDs/history counters and native Settings
precedence. The runnable example is launched and its offline history/state artifacts
are fetched after reopening. Live links are read from a separate process, with
success/failure and array isolation covered locally.

Container probes imported the built fork wheel without installing into or modifying
the existing runtimes. These use test URLs, not remote W&B runs:

| Site/runtime | Experiment / native job | Evidence |
| --- | --- | --- |
| S3DF/Singularity (`--cleanenv`, one A100) | `1790033978893653949` / `38754303` | Named links/replacement read while RUNNING; cancelled through reopened WorkUnit; receipts still readable afterward. |
| S3DF → NERSC/Shifter (`--clearenv`, GPU module) | `1790034478366444029` / `58718229` | Same checks, with receipt reads and cancellation issued from S3DF. |

The first NERSC probe (`58717666`) failed before the payload because Shifter rejected
an unnecessary same-path bind mount. Removing that mount and using the existing
shared-filesystem visibility passed the second probe. No image rebuild or runtime
installation was required. These jobs are terminal; scripts, logs and the author
catalog are retained under
`/sdf/group/neutrino/youngsam/representations/lxm3-links-qualification.5i1fk8`.

Separate CPU-only Local jobs on S3DF live-qualified W&B 0.28.0 in the isolated
`dune-ml/lxm3-qualification` project: new history, append without truncation,
fork at the saved row, native errors for duplicate-new/missing-append IDs, and
retained WorkUnit links. Fork permission was verified only for this org;
the capability remains explicit and optional. No rewind was requested.

Checkpointing before any new log was also verified against server history:

| Case | W&B run | Saved `next_step` | Unchanged server history |
| --- | --- | --- | --- |
| Append | [`b4ais04d`](https://wandb.ai/dune-ml/lxm3-qualification/runs/b4ais04d) | 7 | Rows 0–6 |
| Fork | [`ghw4wp5n`](https://wandb.ai/dune-ml/lxm3-qualification/runs/ghw4wp5n) | 2 | Inherited rows 0–1 |

Both runs finished; outputs were fetched through reopened WorkUnits. Diagnostic
scripts, retained outputs and server observations are under
`/lscratch/youngsam/tmp/lxm3-wandb-online.XmG7uF` (temporary storage).
These online SDK tests did not run inside the GPU containers. Docker generation
is regression-tested, not live-qualified; no GPU computation is claimed by the
link probes.
