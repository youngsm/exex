# Reopen and inspect experiments

## API and boundary

```python
get_experiment(experiment_id: int, *, config: Config | None = None) -> ClusterExperiment
list_experiments(*, project: str | None = None, config: Config | None = None) -> Sequence[ClusterExperiment]
experiment.work_units() -> Mapping[int, ClusterWorkUnit]
unit.get_status() -> WorkUnitStatus
unit.get_logs(*, task: int | None = None, tail: int = 200) -> str
unit.job -> xm.Job | ArrayJob | None
unit.source -> FrozenSource | None
unit.get_script() -> str
```

`get_experiment()` and `get_logs()` are additions. `work_units()` now uses actual
WorkUnit IDs rather than list offsets, and `get_status()` implements the existing
XManager interface. `WorkUnitStatus` exposes `state`, `message`, `is_active`,
`is_completed` and `is_failed`. These are values, not background subscriptions.

The original inspection patch implemented **reopening and inspection only**;
[WorkUnit control](control.md) now also supports Slurm cancellation and Local/Slurm
completion waiting. Reopening
does not replay Python, contact a scheduler, allocate IDs or submit work. Missing
experiment IDs raise `xm.NotFoundError` without creating storage. Retrieved
experiments now support [adding new independent WorkUnits](source-capture.md#retrieve-and-add-another-run)
inside their submission context. Loaded WorkUnits themselves cannot be resubmitted.
Sync/async contexts and awaitable `add()` remain.

`list_experiments()` returns retrieved handles newest first, ordered by experiment
ID. `project=None` includes every project in the configured author catalog, even
when `LXM_PROJECT` is set; a supplied project matches exactly. Listing never polls,
imports a launcher, captures source or submits. A missing catalog returns an empty
list without creating its database or directories. The returned handles support
the same inspection and explicit append operations as `get_experiment()`.

## Actual usage

Launch the existing source example with a durable local staging path in the TOML:

```bash
lxm3 launch examples/source/launch.py -- \
  --lxm_config=/path/to/lxm.toml --target=local \
  --output_dir=/absolute/path/to/new-output
```

After that command exits, use the experiment ID it printed in a new Python process:

```python
from lxm3 import xm_cluster as xc

config = xc.Config.from_file("/path/to/lxm.toml")
experiment = xc.get_experiment(experiment_id, config=config)
for work_unit_id, unit in experiment.work_units().items():
    status = unit.get_status()
    print(work_unit_id, status.state, status.message)

print(experiment.work_units()[1].get_logs(tail=100))
```

The standalone [inspection example](../examples/inspection/reopen.py) remains a
runnable Python example of this workflow:

```bash
python examples/inspection/reopen.py EXPERIMENT_ID \
  --config=/path/to/lxm.toml --logs --tail=100
```

For an array, choose `--work-unit=1 --task=0`; task indices are zero-based regardless
of Slurm's array offset. Status queries do not require logs to exist yet. Asking
for a missing log raises the ordinary file/command error rather than returning
an empty success. `tail=0` returns no lines; negative counts are invalid.

## Command line

The CLI uses the same Python methods and configured author catalog:

```bash
export LXM_CONFIG=/path/to/lxm.toml
lxm3 experiments
lxm3 experiments --project demo
lxm3 status 101
lxm3 status 101 3
lxm3 logs 101 3 --tail 100
lxm3 logs 101 3 --task 0 --tail 100
lxm3 script 101 3
lxm3 stop 101 3
```

Use the experiment and WorkUnit IDs printed by discovery/status. The config can
also be selected with `--lxm_config=/path/to/lxm.toml` before or after the management
subcommand. Otherwise the existing environment, working-directory `lxm.toml`, and
user-config lookup applies. Use the same durable `[local.storage].staging` that
was used to create the experiment; this is not discovery across other catalogs.

- `experiments` shows saved IDs, projects and titles. An empty catalog prints
  column headers only. It has no implicit status query or project filter.
- `status` queries all recorded WorkUnits, or just the supplied actual ID.
  A successful read exits zero even when the reported job is failed or unknown;
  this is an inspection command, not a completion wait.
- `logs` prints the returned text without decorations or an added newline.
  The default tail is 200 lines. Multi-task arrays require a zero-based `--task`.
- `stop` delegates to [WorkUnit.stop()](control.md), supports Slurm, and targets
  the whole WorkUnit/array. It returns silently without waiting for termination.
  With no accepted execution handle it is a no-op, just like the Python method.
- `script` prints the saved submission script without decorations or an added
  newline. An array has one script; this command takes no task index.
- Missing IDs, invalid arguments, unsupported operations and native/SSH failures
  exit nonzero. Errors propagate; there is no retry, auto-submission or watch loop.

`launch`/`version` are unchanged. Launch-script arguments still follow `--`.

## Concrete job history

After the launcher exits, retrieve the request without importing or rerunning it:

```python
from lxm3 import xm, xm_cluster as xc

experiment = xc.get_experiment(
    101, config=xc.Config.from_file("/path/to/lxm.toml")
)
unit = experiment.work_units()[3]
job = unit.job                         # xm.Job, or xc.ArrayJob for an array
print(job.executable.entrypoint_command)
print(job.executor.resources)          # Slurm's original allocation request
print(unit.source.id if unit.source else None)
print(unit.get_script())

# For a non-array Job, inspect the effective arguments/environment:
print(xm.merge_args(job.executable.args, job.args).to_list())
print({**job.executable.env_vars, **job.env_vars})
```

For an array, `job.args` and `job.env_vars` are broadcast, per-task lists. Package
defaults remain separate. Positional/keyword ordering, keyword overrides, repeated
flags, booleans, omitted values and `ShellSafeArg` expansion retain XM's semantics.
Enum/path/custom argument objects are normalized to their command-line strings;
retrieval does not import their original Python classes.

The concrete Job is captured after generators and WorkUnit overrides, before
backend submission. It includes the AppBundle entrypoint, staged archive URI,
container reference, defaults, prepared destination, source association, invocation
and executor fields (including native resources and container options). Each read
reconstructs independent objects. Editing a retrieved Job cannot edit history.
`job` is `None` before a payload is recorded. `source` is `None` for a payload not
packaged from SourceTree/FrozenSource, or before a payload exists. Both properties
are metadata-only; even a missing source archive does not prevent inspection.

An accepted backend handle records the script path and resolved host/user beside
the existing native ID. `get_script()` reads that exact file through the saved
endpoint, independent of current cluster profiles or custom log directories. A
missing reference raises `xm.NotFoundError`; missing files and SSH errors propagate
once, without retries. Submission failures can retain a Job without an accepted
handle/script reference: saved intent is not evidence that a job ran.

This is execution provenance, not hermetic replay. Archives/scripts must still be
retained; image tags, external inputs, installed runtimes and ambient environment
can change. Only explicitly supplied environment variables are saved, and those
can contain secrets: protect the catalog and staging directory accordingly. No
launcher replay, executable registry or automatic rerun command is introduced.

History uses `work_units.job` (one JSON snapshot) and `work_units.script_path` in
the author catalog; no extra files, ORM, dependency or schema counter. This
development slice uses a fresh catalog. Existing catalogs are not migrated or
backfilled; keep their matching checkout and staging data for historical reads.

## Persistence and observations

- One author-side `experiments.sqlite3` lives in `[local.storage].staging`, beside
  the existing archives. Experiment and WorkUnit tables hold IDs, title,
  project, producing package version, execution endpoint, accepted native job ID,
  generated job name, log directory, array shape and Local outcomes. The subsequent
  source-reuse slice adds a source-membership table; see its separate contract.
  Concrete Job history now also retains explicit package/job environments and
  source association as described above. No launchers, closures, ambient
  environment dump or full site profiles are serialized.
- SQLite allocates WorkUnit IDs inside short transactions. No transaction spans
  SSH, packaging, execution or waiting. There is no ORM, schema counter, migration
  framework, daemon, submission retry or remote catalog. Workers do not open SQLite.
- Use durable storage with reliable SQLite/POSIX locking, owned by the author
  host. The default rollback journal avoids WAL's network-filesystem restriction;
  it does **not** make arbitrary network filesystems or concurrent multi-host
  catalog access safe. See SQLite's [WAL restrictions](https://www.sqlite.org/wal.html)
  and [network-filesystem caveats](https://sqlite.org/useovernet.html).
- Recorded endpoints and paths drive inspection, not today's cluster-profile
  contents. Current SSH configuration still supplies authentication and routing.
  Local/on-site submissions record their author hostname; inspection there uses
  local commands. Copying a database does not copy its logs or grant site access.
- Slurm status reads native [sacct allocation rows](https://slurm.schedmd.com/sacct.html)
  with arrays expanded. A matching numeric ID **and generated job name** are
  required. Accounting must be enabled and accessible. Missing rows, accounting
  delay, purged history, renamed jobs or unfamiliar states return `unknown`.
  Connection/command errors propagate, with no retries or discovery searches.
- Slurm status is **not cached** in this slice. It reports the accounting
  observation at that call, which may lag the scheduler. `COMPLETED` requires
  exit code `0:0`; a nonzero exit/signal is failure. Batch outcomes remain the
  application's responsibility: LXM3 does not reinterpret swallowed worker errors.
- An array completes only when every expected task has matching successful
  evidence. One failed task makes the unit failed even if other tasks are still
  active; this neither cancels siblings nor claims all processes have stopped.
  Missing tasks prevent successful completion. `paused`, `stopped` and `unknown`
  satisfy none of the active/completed/failed predicates.
- Local remains attached. Live handles report their futures' state; author-side
  completion callbacks record final outcomes, including array failures. Reopening
  an unfinished Local execution returns `unknown` until that outcome is recorded.
  If the author dies before recording it, it remains unknown: there is no PID
  discovery, detached runner or invented success from an absent process.
- Logs use the existing stdout destination, including `Slurm.log_directory`.
  Preserve LXM3's generated output/job-name directives for inspectable Slurm jobs;
  overriding them through raw directives is outside this adapter. The tail is
  bounded by lines, not bytes, and includes stderr only when the native job merged
  it into stdout. There is no implicit follow loop or cross-rank merge.

Existing pre-patch jobs are not retroactively imported. GridEngine submissions
and Job/script retrieval work, but its status/log/control adapter is not implemented.
Local cancellation,
keyed submission, persistent annotations, prepared-executable
lookup, outputs and continuation remain
separate work. This does not complete section 3 of the broader fork plan.

## Verification

### Job history qualification, 2026-09-21

**498 tests pass, 2 integration tests deselected**, including 19 new history/CLI
cases. Coverage includes render-identical Job/ArrayJob round trips, all three
executors and container-option types, package/job merge semantics, generators and
overrides, mutation isolation, failed submission, missing source/script files,
recorded endpoint routing, ordinary SSH errors, and fresh-process reads after
deleting a disposable launcher. Ruff and formatting checks pass.

Live probes used a fresh, isolated author catalog at
`/sdf/group/neutrino/youngsam/representations/lxm3-history-qualification.DxfVT0/author`.
The adjacent `lxm.toml` selects S3DF and NERSC staging directories.

| Site | Experiment / WorkUnit | Native job | Outcome |
| --- | --- | --- | --- |
| S3DF | 1790024563051216718 / 1 | 38742940 (two-task array) | Both COMPLETED 0:0; 1/2 seconds; one A100 per task |
| S3DF → NERSC | 1790024563227876806 / 1 | 58708625 | COMPLETED 0:0; 6 seconds; one GPU node |

S3DF used `neutrino:default@ampere` / `preemptable`; NERSC used `m5238_g` / `debug`.
Both are terminal with empty queue results; these were shell probes, not GPU
computations. Fresh Python processes reopened both without any cluster profiles,
retrieved source membership and complete requests, read all task logs, and compared
saved scripts against scripts regenerated from the saved Jobs: exact matches.
Fresh `lxm3 script` processes independently produced matching SHA-256 hashes:

- S3DF: `e832c84414c84e320686d0f584b08f271309300340148f40c56534bbe013e689`
- NERSC: `68aeffae8a5ad0161a97aaee7b8a90bee7e400d73dc35f43859cfc4145b59772`

Both retained source ID `9ddc3620116b50612b561e2d15f23de5997c3e4f743842c95b9eafcbc9ccdc2c`.
Earlier catalogs and active pimm workflows were not modified.

### Discovery/CLI regressions

The subsequent discovery/CLI slice passes **479 tests, 2 integration tests
deselected**, including 26 new cases in `tests/cli_test.py`. These cover read-only
discovery, project filtering/order, actual-ID selection, literal output, zero-based
array logs, cancellation delegation, SSH failures without retry, error exits,
fresh CLI processes, config precedence and existing launch argument forwarding.

Read-only CLI qualification on 2026-09-21 used the existing S3DF author config at
`/sdf/group/neutrino/youngsam/representations/lxm3-reuse.wctFcV/lxm.toml`.
`experiments` found `1790014763042766562`; `status` reported its Local WorkUnit 1
and NERSC WorkUnit 2 completed, with native job `58703111` at `COMPLETED 0:0`.
`logs ... 2 --tail 20` retrieved the retained stdout from `nid008329`, including
literal quoted/dollar-sign/newline content. The catalog hash was unchanged.
No new jobs were submitted and no live cancellation was issued for that initial check;
CLI cancellation delegates to the previously qualified method and is tested with
a mocked native transport. Fresh CLI qualification is recorded below.

### Fresh Slurm CLI qualification, 2026-09-21

The existing `examples/control/launch.py` submits a shell start marker and a
300-second sleep with a 360-second walltime. The author-side config/catalog is
isolated under
`/sdf/group/neutrino/youngsam/representations/lxm3-cli-qualification.XxHs4S/`.
Commands run as independent CLI processes from S3DF after the launcher exits.

| Site | Experiment ID | Native job | Execution host | CLI result |
| --- | --- | --- | --- | --- |
| S3DF | 1790020578840469011 | 38740189 | sdfampere010 | Running status, stdout, explicit stop, then stopped status and retained logs |
| S3DF → NERSC | 1790020574307581110 | 58706742 | nid002240 | Queued then running status, remote stdout, explicit stop, then stopped status and retained logs |

The S3DF probe used `neutrino:default@ampere`, QoS `preemptable`, one A100,
one requested CPU and 1 GiB host memory. Native accounting confirmed allocation
cancellation after 75 seconds, batch cancellation with exit `0:15` after 76 seconds,
and extern completion after 77 seconds.

The NERSC probe used `m5238_g`, debug QoS, `constraint=gpu` and one node (the
site allocated four A100s). Native accounting confirmed allocation cancellation
after 91 seconds, batch cancellation with exit `0:15` after 92 seconds, and extern
completion after 93 seconds. One independent post-cancellation SSH check timed
out; separate explicit read-only checks then confirmed accounting and an empty
queue. Cancellation was not resent, and no library retry logic was added.

On both sites the launcher had exited before the CLI inspected or stopped work.
`experiments --project qualification` discovered both entries. `status` observed
native running state before `stop`, and `stopped` afterward; `logs --tail 20`
returned both the start marker and, after cancellation, Slurm's termination
message. The native queues contain neither test job, and all recorded steps are
terminal. Scripts, logs and source archives remain in the isolated staging paths;
NERSC staging is `/pscratch/sd/y/youngsam/lxm3-cli-qualification-XxHs4S/staging`.

No implementation changes or pimm changes were needed. These are single-WorkUnit
scheduler/log/control checks, not GPU-computation, array-cancellation, distributed
training or checkpoint-continuation qualification.

### Original inspection regressions

`tests/inspection_test.py` launches and reopens Local successes/failures in separate
Python processes. It covers read-only retrieval, missing IDs, ID allocation,
Local arrays, async contexts, native Slurm status parsing, per-task evidence,
recorded endpoint isolation, recycled IDs, custom log paths, literal paths,
federated IDs and native errors without resubmission. Existing launch, packaging,
container and source-capture regressions remain part of the gate.

Regression result: **408 passed, 2 integration tests deselected**, including 42
inspection tests. The remaining warnings are existing upstream deprecations.
Ruff and formatting checks, plus `git diff --check`, pass.

### Live qualification, 2026-09-21 UTC

Both launchers exited before the standalone inspection example ran in separate
Python processes. The catalog remained on S3DF; no LXM3 installation or database
access was needed inside either workload.

| Execution | Experiment ID | Native job | Outcome |
| --- | --- | --- | --- |
| Local on S3DF | 1789976397701942973 | — | completed; status and logs reopened |
| S3DF → NERSC | 1789976056632035902 | 58682890 | COMPLETED, exit 0:0, 6 seconds, nid001580 |

The NERSC reader retrieved the matching native status and actual remote stdout,
including the literal quoted/dollar-sign/newline argument. Native accounting and
filesystem checks independently confirmed completion and removal of only the
temporary source directory `source-work/lxm3.y95h3QBGKF`; the parent and application
output remain. This was a tiny standard-library workload, not a GPU-compute test.
The allocation is terminal.

The first Local probe exposed different fully qualified names under restricted
DNS. Locally recorded endpoints now use the stable machine hostname and submitter
identity. The final Local reader succeeded in that restricted environment without
SSH; regressions also ensure another recorded user does not bypass SSH identity.

Author catalog and local evidence are under
`/sdf/group/neutrino/youngsam/representations/lxm3-qualification.jmeYsG/`:
`local/experiments.sqlite3`, `local/projects/qualification/{jobs,logs}/`, and
`inspection-local-002/result.json`. NERSC evidence is under
`/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/`:
`staging/projects/qualification/{jobs,logs}/` and `inspection-nersc-001/result.json`.

For the recorded NERSC run, the actual read-only invocation from S3DF is:

```bash
python examples/inspection/reopen.py 1789976056632035902 \
  --config=/sdf/group/neutrino/youngsam/representations/lxm3-qualification.jmeYsG/lxm.toml \
  --logs --tail=20
```
