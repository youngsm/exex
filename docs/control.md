# Cancel and wait for recorded WorkUnits

## Exact API

Implement the inherited XManager interfaces; no new public methods:

```python
unit.stop(
    *, mark_as_failed: bool = False,
    mark_as_completed: bool = False, message: str | None = None,
) -> None
unit.wait_until_complete() -> xm.WorkUnitCompletedAwaitable
```

`stop()` supports Slurm, including a reopened WorkUnit and a whole array. Waiting
supports Slurm and attached/reopened Local. Local cancellation and GridEngine
control are not implemented. Reopening still cannot submit more work.

This keeps the existing Experiment/WorkUnit model and async wrapper. No new
dependencies, schema changes, process manager, scheduler-discovery path or retries
are needed. Native cancellation lives in `clusters/slurm.py`; the WorkUnit owns
the small stop/wait policy. Source packaging and execution are unchanged.

## Actual usage

Submit the bounded shell probe from S3DF, using an existing NERSC SSH alias and
the site's account/resources in your TOML/configuration:

```bash
lxm3 launch examples/control/launch.py -- \
  --lxm_config=/path/to/lxm.toml --target=nersc --duration=90 \
  --resource=account=m5238_g --resource=qos=debug \
  --resource=constraint=gpu --resource=nodes=1
```

The launcher exits after submission and prints the experiment ID. In a different
Python process, using the same author catalog:

```python
import asyncio
from lxm3 import xm, xm_cluster as xc

config = xc.Config.from_file("/path/to/lxm.toml")
experiment = xc.get_experiment(experiment_id, config=config)
unit = experiment.work_units()[1]

async def wait():
    return await asyncio.wait_for(unit.wait_until_complete(), timeout=120)

assert asyncio.run(wait()) is unit  # Returns the same WorkUnit on success.
```

To cancel instead, explicitly call `unit.stop(message="Finished testing")` before
waiting. A confirmed cancellation raises `xm.ExperimentUnitNotCompletedError`;
the error's `work_unit` is the unit being monitored. A failed job raises
`xm.ExperimentUnitFailedError`. Catch these when failure/stop is an expected result.

The standalone example implements these operations without importing a launcher:

```bash
python examples/control/control.py EXPERIMENT_ID --config=/path/to/lxm.toml --timeout=120
python examples/control/control.py EXPERIMENT_ID --config=/path/to/lxm.toml --stop --timeout=120
```

The second invocation deliberately raises XM's not-completed error on confirmed
cancellation. A timeout raises `TimeoutError`; neither timeout nor Ctrl-C on the
reader requests job cancellation. There are no new `lxm3` CLI subcommands.

For Local, launch with `--target=local --duration=0`, optionally `--tasks=2`.
Local still waits on context exit; the subsequent reader observes its recorded
outcome. `--exit_code=7` tests failure; `--tasks=2` makes an array on either backend.

## Cancellation contract

- Await `experiment.add(...)`, or reopen after the launcher exits, before stopping.
  With no accepted execution handle, `stop()` is a no-op. This accommodates XM's
  launch-error cleanup without searching for possibly accepted jobs. It does not
  cancel an in-flight submission or replace its recorded uncertainty with success.
- The recorded host/user, native ID and generated job name determine the target.
  The native operation is `scancel --ctld --name=NAME ID`, with `--clusters=...`
  for a recorded federated ID. The name restricts cancellation so a recycled ID
  with a different name is not targeted. There is no preflight query/cancel race
  added by the adapter. Preserve generated job names; manual rename/reuse and raw
  name overrides are outside this ownership contract.
- Native cancellation covers the allocation and its steps, or the entire array.
  It uses Slurm's normal termination behavior, not a custom signal/checkpoint
  protocol. See the [native command contract](https://slurm.schedmd.com/scancel.html).
- `stop()` returns after the command succeeds, not after termination. No matching
  job is not evidence of successful termination. SSH/native errors propagate;
  there is no automatic redelivery. Another explicit `stop()` calls Slurm again.
- Existing catalog `state`/`message` fields hold Local outcomes; for Slurm, they
  hold the requested stop classification/reason. Intent is saved before delivery
  but never treated as evidence of cancellation. No cached Slurm status is added.
  `mark_as_failed=True` classifies a natively cancelled job as failed; completion,
  running, failure and missing evidence retain their native interpretation.
  The reason is attached when cancellation is observed, including after reopening.
- `mark_as_completed=True` raises without mutation. Cancellation cannot invent a
  successful result. Local/GridEngine cancellation also raises without mutation.

## Waiting contract

- The inherited XM wrapper still waits for submission, returns the same WorkUnit,
  and exposes `.work_unit`. Submission failures keep XM's existing error behavior.
- Slurm polls the existing status adapter every ten seconds. Native reads run off
  the async event loop. Missing/purged/delayed accounting stays unknown and waiting
  continues until evidence changes or the caller times out. SSH errors raise on
  the first failed command; polling is not a transport-error retry policy.
- Native completed/exit-zero succeeds; failure raises the existing failed error;
  stopped or suspended raises the existing not-completed error. Suspending does
  not imply checkpoint readiness, and waiting never resumes/requeues anything.
- Array success requires evidence for every expected element. A failed/stopped
  Slurm element may raise while siblings remain active, matching the existing
  aggregate status. An exceptional wait is **not** a resource-release barrier;
  siblings are not automatically cancelled. Explicit `stop()` targets the whole
  array. Local waiting drains its execution futures before classifying the result.
- Local futures are shielded individually from waiter cancellation, including
  event-loop shutdown after a timeout. No PID/process-group control is added.
  Reopened Local waits for its author-recorded outcome; if the author died without
  recording one, it stays unknown. There is no process discovery.
- A timeout cancels observation, not execution. An already-dispatched blocking
  SSH call may finish after the timeout; `asyncio.run()` can wait for its thread
  during shutdown. This is not a hard deadline on the SSH subprocess itself.

## Verification

`tests/control_test.py` covers recorded cancellation targets, arrays, federated IDs,
explicit redelivery, no fake completion, native evidence after stop, XM errors,
read-only waiting, nonblocking polling, Local timeout safety and reopened outcomes.
Existing separate-process Local inspection tests now also exercise completion
waiting; the existing async launch-context test awaits the inherited wrapper.

Regression result: **438 passed, 2 integration tests deselected**, including 30
new control cases. Existing separate-process Local success/failure and async
launch tests also exercise waiting. Ruff, formatting and `git diff --check` pass.
The test environment includes the existing optional PEX extra; no dependency was
added to the project. Remaining warnings come from upstream async deprecations.

### Live qualification, 2026-09-21

The launcher and control readers ran as separate processes on S3DF. The author
catalog/configuration remains under
`/sdf/group/neutrino/youngsam/representations/lxm3-qualification.jmeYsG/`.

| Execution | Experiment ID | Native job | Verified result |
| --- | --- | --- | --- |
| Local two-task array | 1790012816850415562 | — | Standalone launcher exits; standalone reader awaits completion |
| S3DF → NERSC control probe | 1790012942481605864 | 58701332 | Timeout leaves job queued; mismatched name refuses cancellation; explicit WorkUnit stop cancels the recorded job |
| Previously completed NERSC probe | 1789976056632035902 | 58682890 | New standalone reader returns successfully from completion waiting |

The new NERSC probe requested one GPU-partition node, debug QoS, a 90-second shell
sleep and a 150-second walltime. It was queued during the timeout/name-filter
checks, then began running before the explicit stop arrived. Independent native
accounting confirms allocation cancellation after 20 seconds on `nid001593`, batch
`CANCELLED` with exit `0:15`, and extern completion after 22 seconds. The native
queue has no remaining entry for this probe. This qualifies running batch-job
cancellation, not GPU computation, checkpoint handling or distributed workers.

A further fresh reader reported `stopped`, retained the supplied stop reason and
retrieved real remote stdout: the start marker, Slurm termination message and
temporary-source cleanup message. As designed, the control example raised
`xm.ExperimentUnitNotCompletedError` after cancellation rather than reporting
success. Remote logs/scripts are retained under
`/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/staging/projects/qualification/`,
with slug `control_probe_1790012942481605864_1`.

Slurm arrays, failure interpretation and transport-error behavior are regression
tested, not newly live-qualified here. No detached Local execution, Local
cancellation, GridEngine control, submission retries or pimm changes were added.
