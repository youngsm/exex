# Fork API proposal — for review

Status: approved API baseline, 2026-09-16. Based on upstream `30f55855` and
the [fork plan](fork-plan.md). Approval does not imply every interface below is
implemented; implementation progress is recorded separately in the fork plan.

Review sections 1–6 as the first API patch. Section 7 specifies the next artifact/
continuation extension separately. Runtime/provider additions remain separate
reviews, not implicit changes hidden in this patch.

Signatures below describe the public contract, not a choice of implementation
decorators. `PathLike` means `str | os.PathLike[str]`.

## 1. Keep the authoring model

Keep `xm.Job`, `xm.Packageable`, `ArrayJob`, Python launchers, sync/async experiment
contexts and awaitable submission. Do not introduce `Project`, `Run`, a replacement
Job class, a universal Resources class or a second launcher framework.

One WorkUnit contains one concrete Job, one ArrayJob, or a singleton JobGroup.
JobGenerators, including the W&B wrapper, can produce that payload. Multi-job
JobGroups remain unsupported. This implements the existing one-payload WorkUnit
contract; it is not a general workflow engine.

## 2. Selected sites: change methods, not the configuration system

In `lxm3/xm_cluster/config.py`:

```python
# CHANGED: an explicit name bypasses the ambient/default cluster selection.
Config.cluster_settings(self, name: str | None = None) -> ClusterSettings
```

Keep the existing TOML file and `--lxm_config` flag. `server` is the OpenSSH alias
for remote Slurm; no server means on-site execution. Account/resource defaults
remain ordinary launcher/site configuration. No profile registry/version fields.

In `lxm3/xm_cluster/executors.py`:

```python
# ADDED keyword-only field; every existing constructor field remains.
Slurm(..., *, cluster: str | None = None)
GridEngine(..., *, cluster: str | None = None)

# CHANGED: instance methods, carrying the selected instance's target.
Slurm.Spec(self) -> SlurmSpec
GridEngine.Spec(self) -> GridEngineSpec

# UNCHANGED: Local's spec has no instance-specific target to carry.
Local.Spec() -> LocalSpec

# ADDED field on the existing specs.
SlurmSpec(cluster: str | None = None)
GridEngineSpec(cluster: str | None = None)
```

`executor.Spec()` continues to work. Class calls such as `Slurm.Spec()` become
invalid deliberately; no compatibility descriptor supporting two meanings.
Both `Local.Spec()` and `local.Spec()` keep working through the existing classmethod.
Local has no new target/runtime argument. Existing Docker/Singularity options stay.

Keep native Slurm `resources` as the allocation description; do not also introduce
duplicated `nodes`, `gpus` or `cpus` constructor fields. For example:

```python
executor = xm_cluster.Slurm(
    cluster="nersc",
    resources={"nodes": 2, "ntasks-per-node": 1, "gpus-per-node": 4},
    walltime=3600,  # Existing seconds/timedelta convention; not a new string parser.
)
spec = executor.Spec()  # Carries cluster="nersc".
```

An Experiment resolves its configuration once. Packaging resolves the spec's site;
submission resolves the executor's site from the same configuration. A prepared
bundle records its resolved target, so a changed profile cannot silently redirect
it. Explicit site selection never mutates environment variables or global config.

## 3. Experiment creation, reopening and packaging

In `lxm3/xm_cluster/experiment.py`, exported from `xm_cluster`:

```python
# CHANGED: add only the keyword-only configuration argument.
def create_experiment(
    experiment_title: str,
    project: str | None = None,
    *,
    config: Config | None = None,
) -> ClusterExperiment: ...

# NEW backend implementation of XManager's existing retrieval concept.
def get_experiment(
    experiment_id: int,
    *,
    config: Config | None = None,
) -> ClusterExperiment: ...

# NEW discovery; returns handles, not scheduler observations.
def list_experiments(
    *,
    project: str | None = None,
    config: Config | None = None,
) -> Sequence[ClusterExperiment]: ...
```

`config=None` loads the current LXM3 config once. The author catalog and retained
local source live under `[local.storage].staging`; use an absolute durable path
for normal research. No separate public Catalog/Project object or competing store
argument. Project is a grouping/namespace, not an authorization or placement rule.

Creation always creates a new experiment, even with the same title. Retrieval
does not replay launchers/generators, submit, poll or enter a context. Missing IDs
raise the existing `xm.NotFoundError`. Enter a retrieved experiment to add work;
read/status/stop calls do not require entering its submission context.

```python
class ClusterExperiment(xm.Experiment):
    # CHANGED from classmethods to instance methods; signatures otherwise preserved.
    def package(
        self, packageables: Sequence[xm.Packageable] = (),
    ) -> Sequence[xm.Executable]: ...

    def package_async(self, packageable: xm.Packageable) -> Awaitable[xm.Executable]: ...

    # EXISTING method, now reading durable records keyed by the actual WorkUnit ID.
    def work_units(self) -> Mapping[int, ClusterWorkUnit]: ...

    # EXISTING signature/awaitable behavior; implement persistent identity semantics.
    def add(
        self, job, args=None, *, role=xm.WorkUnitRole(), identity: str = "",
    ) -> asyncio.Future[ClusterWorkUnit]: ...
```

The packager belongs to the Experiment instance and captures its resolved config,
project and author store; no global async packager/client selects the destination.
Calling `ClusterExperiment.package(...)` without an instance is deliberately removed.
`package_async()` still queues work; `package()` flushes it. Packaging is allowed
without entering the submission context and can build/upload; it is not a dry run.

`add()` resolves when submission is accepted, not when computation finishes. Keep
Local's default wait-on-context-exit behavior; Slurm jobs survive launcher exit.
The existing title/tags/notes APIs under `context.annotations` become persistent.

Keyed additions are scoped to their experiment. Identical concrete intent returns
the existing WorkUnit; changed intent raises `ValueError` without cancelling it.
Intent comparison strengthens XM's current add-if-present behavior deliberately.
No identity means an intentional new unit. A submission error is recorded and
raised; re-adding its identity raises that recorded failure, not another submission.
No lost-response searches, retries or callback serialization.

## 4. Frozen source: two values, no new executable hierarchy

In `lxm3/xm_cluster/executable_specs.py`, exported from `xm_cluster`:

```python
class SourceTree(xm.ExecutableSpec):
    def __init__(
        self, entrypoint: ModuleName | CommandList,
        path: PathLike = ".", *, files: Sequence[str] | None = None,
    ): ...

class FrozenSource(xm.ExecutableSpec):
    # Returned by freeze()/sources(); no public construction from an ID alone.
    @property
    def id(self) -> str: ...

class ClusterExperiment:
    def freeze(self, source: SourceTree) -> FrozenSource: ...
    def sources(self) -> Mapping[str, FrozenSource]: ...
    def executables(self) -> Mapping[str, AppBundle]: ...

class AppBundle(xm.Executable):
    # ADD read-only metadata to the existing prepared object, not constructor knobs.
    id: str
    source_id: str | None
```

Both source specs implement the existing `ExecutableSpec.name` property. The frozen
value is immutable; its retained location is private backend data.

`freeze()` copies and hashes selected bytes synchronously without building,
uploading or submitting. Its identity includes the selected file manifest and
entrypoint, not absolute checkout path, timestamp or Git HEAD. `files` is an
allowlist; its default is Git-aware working-tree selection, including uncommitted
changes and untracked nonignored files. No commit/stash/index mutation.

`package(SourceTree)` captures at packaging time; `package(FrozenSource)` never
rereads the original checkout. Existing container wrappers accept both specs.
Prepared values still are AppBundles. `sources()` and `executables()` retrieve
recorded experiment members, including packages not yet submitted. Reuse a recorded
AppBundle on its prepared target; prepare its FrozenSource again for another target.

## 5. WorkUnit control: implement existing methods first

In `lxm3/xm_cluster/experiment.py`:

```python
class ClusterWorkUnit(xm.WorkUnit):
    # EXISTING interfaces, currently unimplemented by the cluster backend.
    def get_status(self) -> WorkUnitStatus: ...

    def stop(
        self, *, mark_as_failed: bool = False,
        mark_as_completed: bool = False, message: str | None = None,
    ) -> None: ...

    def wait_until_complete(self) -> xm.WorkUnitCompletedAwaitable: ...

    @property
    def launched_jobs(self) -> list[xm.LaunchedJob]: ...

    # NEW read-only identity and bounded log retrieval.
    @property
    def identity(self) -> str: ...

    def get_logs(
        self, *, task: int | None = None, attempt: int | None = None,
        tail: int = 200,
    ) -> str: ...
```

`get_status()` makes a current native observation and returns a value; it does not
start a poller. Connection errors raise. Missing native evidence yields `unknown`,
not a cached-success fallback. Previously confirmed terminal outcomes remain durable.

`WorkUnitStatus` implements `xm.ExperimentUnitStatus`'s existing `is_active`,
`is_completed`, `is_failed` and `message` properties. Add only `state`, with values
`created`, `queued`, `running`, `paused`, `completed`, `failed`, `stopped`, `unknown`.
`is_active` means created/queued/running; `is_completed` means completed;
`is_failed` means failed. Paused/stopped/unknown satisfy none of these predicates.
Unknown is not evidence of completion or safe resubmission. Array status aggregates
required tasks; one failed task fails the unit, and completion requires all tasks.

`stop()` records stop intent and sends native cancellation only to the recorded
execution. It returns after delivery, not after termination. Failed delivery raises;
calling again explicitly redelivers. `mark_as_failed=True` supports XM's launch-error
path; `mark_as_completed=True` remains explicitly unsupported in the initial patch,
rather than making cancelled/missing-output jobs look successful.

`wait_until_complete()` keeps XM's awaitable; implement `_wait_until_complete()`
under it rather than replacing the wrapper. Return the WorkUnit on success, raise
existing XM failure/not-completed errors for failure/stop/pause. Waiting is read-only
and never resumes work. Use `asyncio.wait_for(..., timeout=...)` for a timeout;
timing out the wait does not cancel the job. No second `wait()` method is added.

`launched_jobs` exposes the existing name/address/log-location view, including native
job/task references. `get_logs()` returns a bounded text tail; no implicit follow loop.
For arrays, `task` is the zero-based Python array index, independent of Slurm's
index offset; require an explicit task when there are multiple. `attempt=None`
selects the current attempt. No merged cross-rank log ordering is promised.

## 6. Thin CLI surface

Keep `lxm3 launch launcher.py -- ...` and `version`. Add only these initial commands:

```sh
lxm3 experiments --project demo
lxm3 status 101                 # WorkUnits in experiment 101.
lxm3 status 101 3               # One WorkUnit.
lxm3 logs 101 3 --tail 100
lxm3 stop 101 3                 # Explicit mutation, not an inspection command.
```

These are proposed syntax, not currently available commands. They use the existing
config-loading path and Python methods above. No launch replay, tracking server,
background watch, automatic target selection or second management database.

## 7. Next review: artifact and continuation extension

These are exact proposed signatures for the next slice, not part of the first patch.
Mappings are copied into frozen intent.

```python
# Extend the existing Experiment.add with optional WorkUnit-level execution policy.
def add(
    self, job, args=None, *, role=xm.WorkUnitRole(), identity: str = "",
    inputs: Mapping[str, Artifact] | None = None,
    outputs: Mapping[str, str] | None = None,
    max_attempts: int = 1,
) -> asyncio.Future[ClusterWorkUnit]: ...

class ClusterWorkUnit:
    def artifacts(
        self, *, task: int | None = None, attempt: int | None = None,
    ) -> Mapping[str, Artifact]: ...

    def attempts(self) -> Sequence[Attempt]: ...
    def resume(self) -> Awaitable[None]: ...

@dataclass(frozen=True)
class Attempt:
    number: int
    status: WorkUnitStatus
    jobs: tuple[xm.LaunchedJob, ...]
    links: Mapping[str, str]

class Artifact:
    @property
    def id(self) -> str: ...
    def fetch(self, into: PathLike) -> Path: ...
    def publish(self, to: Publisher) -> Publication: ...

class Publisher(Protocol):
    def publish(self, artifact: Artifact) -> Publication: ...

@dataclass(frozen=True)
class Publication:
    uri: str
    revision: str | None = None

class ClusterContextAnnotations:
    # Available on both Experiment and WorkUnit alongside title/tags/notes.
    def set_link(self, name: str, url: str) -> None: ...
    @property
    def links(self) -> Mapping[str, str]: ...
```

Keep `xm.Job` and `ClusterWorkUnit.add()` unchanged: the outer Experiment.add freezes
policy on the WorkUnit before invoking a generator. The one payload inherits it.
Arrays receive the same input bindings and isolated per-task outputs; array
continuation is unsupported initially. The W&B wrapper must preserve the payload's
name and fields rather than reconstruct a reduced Job.

Output mapping values are paths relative to an attempt/task output root, not shell
templates. All declared outputs are required; optional outputs are not another flag
in this slice. Normal output retention is automatic. `artifacts()` only exposes
completed content; task selection follows `get_logs()`. `fetch()` never overwrites
an existing destination. Publishers transfer content; applications own export format.
`Artifact.publish(to)` delegates once to `to.publish(artifact)` and records the returned
receipt. `uri` identifies the published object; `revision`, when supplied by a provider,
identifies that publication's immutable revision. No implicit upload or retry.

`attempts()` reads recorded snapshots; it does not poll every historical scheduler
job. `resume()` requires entering the owning Experiment context, submits exactly one
new attempt and resolves at acceptance, not completion. The attempt budget includes
the first attempt. Only a verified cooperative pause permits resume. It accepts no
configuration overrides; changed science is a new WorkUnit with an input artifact.

The accompanying application-facing helper, `lxm3.execution`, has this small surface:

```python
def context() -> ExecutionContext: ...

class ExecutionContext:
    inputs: Mapping[str, Path]
    output_dir: Path
    resume_from: Path | None

    def started(self) -> None: ...
    def checkpoint(self, path: PathLike) -> Artifact: ...
    def pause(self, checkpoint: Artifact) -> None: ...
    def link(self, name: str, url: str) -> None: ...
```

The context uses only its attempt request and execution-site storage, not the author
database. `started()` acknowledges application startup/restoration, not framework
initialization performed by LXM3. `checkpoint()` retains a complete stable generation;
`pause()` records eligibility but does not exit the process. Success requires a clean
exit after writers finish. `link()` reports the actual W&B/other URL; retain per-attempt
links and expose the latest reported named links through WorkUnit annotations.
No training framework, W&B initialization or automatic scheduler recovery lives here.

## 8. Internal method changes required by this surface

| Location | Required change |
| --- | --- |
| `config.py` / `create_experiment()` | Add named cluster lookup; snapshot settings per Experiment. Stop mutating the global project's config. Fix the existing uninitialized `vcs` variable when an explicit project is passed. |
| `experiment.py` | Per-instance AsyncPackager; durable WorkUnit allocation/retrieval; reconstruct handles without launcher replay; implement the existing lifecycle methods. |
| `metadata.py` | Back existing annotation methods with the author store; no new metadata framework. |
| `packaging/router.py` | `package(packageables, *, config, project, store)`; `packaging_router(packageable, *, config, project, store)`; route SourceTree/FrozenSource and record prepared bundles. Here `store` is a private author-store handle, not public configuration. |
| `execution/job_script_builder.py` | `create_artifact_store(*, project, settings)` must use its arguments; keyword-only calls remove today's reversed-argument bug. |
| `execution/{slurm,gridengine,local}.py` | Replace zero-argument client factories with `client(*, settings, project)`. Remove global caching initially; packaging/dispatch pass the same resolved values. |
| `clusters/slurm.py` | Implement selected-site submission/observation/cancellation/log reads with system OpenSSH or direct subprocesses. No remote service. |
| Vendored `xm/core.py` integration | Narrowly prevent an identity-conflict exception from running stop() on a pre-existing WorkUnit. Do not copy the whole base class or redesign its event loop. |

Implementation tests must cover keyed conflicts without cancellation, failed-key
re-add without submission, generators/arrays, sync/async contexts, two-site/two-project
isolation, client exit/reopen and ownership-safe stop. Artifact tests add producer/
consumer retention and singleton pause/resume; optional capabilities are never
silently accepted without their implementation.

## 9. Concrete usage of the proposed first patch

Save this launcher beside the existing `examples/basic/pyproject.toml`. It runs the
existing `py_package.main`, which prints its arguments. No hypothetical training
module or new pimm integration is needed for this example.

```python
# launch_review.py — requires the proposed API, not runnable on the baseline yet.
from pathlib import Path
from absl import app, flags
from lxm3 import xm, xm_cluster as xc

TARGET = flags.DEFINE_enum("target", "local", ["local", "s3df", "nersc"], "Execution site")
ACCOUNT = flags.DEFINE_string("account", None, "Slurm allocation account")
PARTITION = flags.DEFINE_string("partition", None, "Slurm partition")
CONSTRAINT = flags.DEFINE_string("constraint", None, "Slurm node constraint")

def main(_):
    resources = {
        name: value for name, value in
        (("account", ACCOUNT.value), ("constraint", CONSTRAINT.value))
        if value is not None
    }
    executor = (
        xc.Local()
        if TARGET.value == "local"
        else xc.Slurm(
            cluster=TARGET.value, walltime=60,
            resources=resources, partition=PARTITION.value,
        )
    )
    with xc.create_experiment("fork-api-demo", project="demo") as experiment:
        source = experiment.freeze(xc.SourceTree(
            entrypoint=xc.ModuleName("py_package.main"),
            path=Path(__file__).parent,
            files=["py_package/**"],
        ))
        [executable] = experiment.package([
            xm.Packageable(source, executor_spec=executor.Spec()),
        ])
        submissions = [
            experiment.add(
                xm.Job(executable, executor, args={"seed": seed}),
                identity=f"seed-{seed}",
            )
            for seed in (0, 1)
        ]
    # Context exit has flushed submission futures (and waited for attached Local).
    print("experiment:", experiment.experiment_id)
    print("executable:", executable.id)
    print("work units:", [future.result().work_unit_id for future in submissions])

if __name__ == "__main__":
    app.run(main)
```

With real `s3df`/`nersc` connection and staging entries in the selected TOML file,
the same launcher is used for all targets. Supply the account and partition/constraint
accepted by the chosen site; these values are not inferred from the cluster name:

```sh
lxm3 launch examples/basic/launch_review.py -- --target=local
lxm3 launch examples/basic/launch_review.py -- --target=s3df --account=neutrino
lxm3 launch examples/basic/launch_review.py -- --target=nersc --account="$NERSC_ACCOUNT" --constraint=cpu
```

The NERSC example requires setting `NERSC_ACCOUNT` to an account authorized for CPU
jobs. This is a source/dispatch demonstration, not a GPU training launcher.

Reopen later; substitute the printed IDs and use the same author storage configuration:

```python
import asyncio
from lxm3 import xm, xm_cluster as xc

experiment = xc.get_experiment(101)
unit = experiment.work_units()[1]
print(unit.get_status().state)
print(unit.get_logs(tail=100))

async def wait():
    await asyncio.wait_for(unit.wait_until_complete(), timeout=60)

asyncio.run(wait())  # A timeout only stops waiting.

with experiment:
    executable = experiment.executables()["<printed-executable-id>"]
    experiment.add(
        xm.Job(executable, xc.Local(), args={"seed": 2}),
        identity="seed-2",
    )  # This example reuses a package prepared for Local.
```

No new source capture occurs when adding that seed. To send the same source to
another site, retrieve its FrozenSource from `experiment.sources()` and package it
for that executor; do not recapture the mutable checkout.

## Review boundaries

Required decisions in this proposal: instance `Spec()` and packaging methods;
configuration-scoped author store; durable existing WorkUnit methods; raw frozen
source; and optional WorkUnit-level I/O rather than a second Job class.

Not proposed here: exact Shifter/image-builder, borrowed-allocation/salloc or Vertex
constructor signatures; detached Local flags; an aggregate status service; source-diff
CLI syntax; HF-specific publisher options. Those remain in the fork plan and receive
their own small method-level proposals before implementation. This is an explicit
review boundary, not permission to invent those interfaces during coding.
