# Inspect and control jobs

Use the same catalog configuration as the launcher. These commands work after
the launcher exits; none reruns the original launch script.

```bash
exex experiments
exex status EXPERIMENT_ID
exex logs EXPERIMENT_ID WORK_UNIT_ID --tail 100
exex script EXPERIMENT_ID WORK_UNIT_ID
exex stop EXPERIMENT_ID WORK_UNIT_ID
```

`experiments` is catalog-only. Slurm status queries native accounting; an
unavailable record is not proof of success. Logs are a bounded tail, not a
streaming monitor. Array logs take a zero-based `--task` index.

The equivalent Python interface:

```python
from exex import xm_cluster as xc

experiment = xc.get_experiment(experiment_id)
unit = experiment.work_units()[work_unit_id]
print(unit.get_status())
print(unit.get_logs(tail=100))
print(unit.get_links())
```

`unit.job` contains the recorded concrete job request; `unit.source` identifies
its retained source when available. `unit.get_script()` reads the submitted
script. These describe what was requested, not a hermetic snapshot of data and
runtime dependencies.

`unit.stop()` cancels owned Slurm execution. Managed continuation is stopped
as a whole; it cannot schedule another attempt afterward. Local cancellation
is not implemented. `unit.wait_until_complete()` waits for a terminal outcome;
failed, stopped, and paused outcomes raise. Budget-exhausted continuation remains
paused rather than being reported as completed.

Reopening an experiment also permits adding new independent jobs with `add()`.
This creates new WorkUnits and new allocations; it does not extend an old Slurm
allocation. Existing WorkUnits do not accept another payload. Repeated titles
do not deduplicate experiments or submissions.
