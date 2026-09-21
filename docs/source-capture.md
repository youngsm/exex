# Raw source capture

## API and scope

```python
SourceTree(entrypoint: ModuleName | CommandList, path=".", *, files=None)
experiment.freeze(source: SourceTree) -> FrozenSource
experiment.sources() -> Mapping[str, FrozenSource]
frozen.id: str
```

Both specifications work with the existing `experiment.package()` and
`xm.Packageable` interfaces, directly or inside `SingularityContainer`,
`DockerContainer` and `ShifterContainer`. There is no new execution API.

`SourceTree` is a recipe; `FrozenSource` is the captured value. This distinction
keeps ordinary launchers short while giving callers an explicit capture boundary
when they need to edit their checkout before packaging another target. A retained
tar archive fits capture: archive headers carry relative names and permissions,
with no build system or parallel manifest. The experiment catalog records source
membership so another process can retrieve the retained value.

## Use it

For an application with a `train.py` module in the launcher's directory:

```python
from lxm3 import xm
from lxm3 import xm_cluster as xc

executor = xc.Slurm(
    cluster="nersc",
    resources={"account": "m5238_g", "qos": "debug", "constraint": "gpu"},
    walltime=120,
    workdir_root="/pscratch/sd/y/youngsam/lxm-work",
)
with xc.create_experiment("training") as experiment:
    frozen = experiment.freeze(xc.SourceTree(xc.ModuleName("train")))
    print(frozen.id)
    [executable] = experiment.package([xm.Packageable(frozen, executor.Spec())])
    experiment.add(xm.Job(executable, executor, args=["--steps=10"]))
```

The account and paths are examples, not defaults. `xc.Local()` selects the current
host instead. An appropriate Python/dependency environment must already be present
on the target, or wrap `frozen` in an existing container specification. For a
particular interpreter, use `CommandList(["/path/to/python -m train"])`.
Execution starts at the captured root, with no installation or Python-path
injection. For a `src/` layout, supply `env_vars={"PYTHONPATH": "src"}` on the job
or choose a command that sets up the application's import path.

An ordinary one-shot launcher can package `SourceTree` directly. Explicit freeze
is useful when packaging the **same captured source** for a second site later:

```python
[local] = experiment.package([xm.Packageable(frozen, xc.Local.Spec())])
# Still valid after the original checkout is changed or removed.
[remote] = experiment.package([xm.Packageable(frozen, executor.Spec())])
```

See the runnable [disposable-source probe](../examples/source/launch.py). It copies
the existing standard-library worker into a temporary source tree, freezes it,
removes that tree, and only then packages and launches the retained value:

```bash
lxm3 launch examples/source/launch.py -- \
  --lxm_config=/path/to/lxm.toml --target=local \
  --output_dir=/absolute/path/to/new-output
```

Use a fresh output directory; the worker refuses to overwrite `result.json`.
For NERSC, change `--target=nersc`, choose site-local output/workdir paths, and add
`--resource=account=m5238_g --resource=qos=debug --resource=constraint=gpu`.
This is a tiny CPU-only workload, not a GPU-compute qualification.

## Retrieve and add another run

After the source launcher exits, use its printed experiment/source IDs in a new
process. The disposable source tree is already gone:

```python
from lxm3 import xm, xm_cluster as xc

config = xc.Config.from_file("/path/to/lxm.toml")
with xc.get_experiment(experiment_id, config=config) as experiment:
    frozen = experiment.sources()[source_id]
    executor = xc.Slurm(
        cluster="nersc", walltime=120,
        resources={"account": "m5238_g", "qos": "debug", "constraint": "gpu", "nodes": 1},
    )
    [executable] = experiment.package([xm.Packageable(frozen, executor.Spec())])
    experiment.add(xm.Job(executable, executor, args=[
        "--output-dir=/absolute/new/nersc-output",
        "--message=another run from retained source",
    ]))
```

The runnable [extension launcher](../examples/source/extend.py) implements this:

```bash
lxm3 launch examples/source/extend.py -- \
  --lxm_config=/path/to/lxm.toml \
  --experiment_id=EXPERIMENT_ID --source_id=SOURCE_ID \
  --target=nersc --output_dir=/absolute/new/nersc-output \
  --resource=account=m5238_g --resource=qos=debug \
  --resource=constraint=gpu --resource=nodes=1
```

`--target=local` uses the same retained source locally. New additions get distinct
WorkUnit IDs; each Slurm addition is an independent `sbatch` submission, not a
step inside an old allocation. Retrieval itself submits nothing. Loaded WorkUnits
cannot be resubmitted, and entering/exiting the experiment leaves old work alone.
New Local jobs retain attached/wait-on-exit behavior; Slurm jobs outlive the launcher.

This slice is append-only: adding the same job twice intentionally creates two
WorkUnits. Nonempty `identity` raises `NotImplementedError`, rather than silently
pretending to deduplicate. This applies to direct and generator-produced payloads.
There is no automatic retry or discovery after a submission/SSH error.

`sources()` returns a fresh mapping of immutable values from the author catalog.
It does not read/hash archives, upload, replay a launcher or query a scheduler.
Missing IDs use ordinary mapping `KeyError`; missing/corrupt archives fail during
packaging before source transfer. Source lookup restores neither a container nor
dependencies, arguments, environment or resource settings: choose them explicitly.

`freeze()` records membership after successful capture; successful source packaging
records its captured source too, including captures inside existing container
wrappers. `package_async()` records at flush, not enqueue. A failed packaging batch
does not register its implicit captures; an earlier explicit freeze remains saved.
Reusing a FrozenSource in another experiment also records membership there. Equal
content IDs share archive bytes; the first membership record retains its cosmetic
name/location. New captures with different bytes/entrypoints remain separate values.

One `sources` table stores experiment ID, source ID, display name, retained archive
path and rendered entrypoint. It uses the existing author-side SQLite catalog;
there is no schema counter, pickle, migration framework, worker database or new
dependency. Keep the catalog and retained archives accessible. Copying the catalog
alone does not move its archives. Use a fresh catalog for this development version;
pre-feature archives are not scanned or retroactively associated with experiments.
The existing author-host/storage boundary remains; this is not cross-host database
coordination. Concurrent author-host additions allocate IDs in short transactions,
outside packaging, network operations and job execution.

## Capture contract

- Default selection uses [Git's file listing](https://git-scm.com/docs/git-ls-files):
  tracked files and nonignored untracked files beneath `path`. Contents come from
  the working tree, **not** the index or commit; deleted files are omitted. Git
  HEAD/index are untouched. Nonignored secrets are included like any other file.
- `files=["train.py", "configs/small.yaml"]` selects exactly those relative files,
  including ignored files if explicitly named. It also works outside Git.
  It does not expand directories or globs; `files=[]` captures an empty tree.
- Paths resolve relative to the launcher. Files must remain within the source
  root. This slice captures regular files, not symlinks or submodules. LXM3's
  existing runtime owns the root paths `job-param.sh` and `.environment`; selecting
  either, or files beneath them, raises instead of allowing source overwrite.
- Freeze copies synchronously, without installation, builds, upload or submission.
  It is **not** an atomic filesystem snapshot: do not edit files during that call.
  Changes after it returns cannot change the retained source.
- Identity covers selected relative paths, bytes, executable bits and the rendered
  entrypoint. Checkout location, timestamps, Git metadata and selection order do
  not affect it. Other permission/ownership metadata is normalized. This does not
  identify dependencies, inputs, job arguments, environment or container images.
- Retained archives live in `[local.storage].staging/sources/<id>.tar`, independently
  of the execution site. The existing atomic staging operation publishes the
  complete archive. Packaging verifies retained bytes against the source ID before
  copying them into the selected site's existing archive store.
- `package_async()` still queues specifications; raw-source capture occurs when
  packaging runs. Call `freeze()` first when the queue must use earlier bytes.
- Entry commands keep existing LXM3 semantics: `ModuleName` runs `python3 -m`;
  `CommandList` runs Bash commands with failure propagation and job arguments
  appended to its final command. Framework choice remains application-owned.

## Verification and deliberate exclusions

`tests/source_test.py` checks working-tree/index differences, deleted/new files,
stable identity and executable bits, capture timing, container composition, corrupt
retained bytes, and actual local execution after checkout removal. It also prepares
one frozen value for two independent staging destinations without its checkout.

Source lookup and append-only submission after retrieval are now implemented.
Prepared-executable lookup, keyed submission, retained outputs, publication,
continuation and pimm migration remain separate work. No dependency build,
allocation-sharing or scheduler-placement behavior was added.

Original capture regression result: **366 passed, 2 integration tests deselected**, including 32 new
source-capture cases. Ruff checks, formatting checks and `git diff --check` pass.

### Live qualification

The disposable-source example completed locally on S3DF on 2026-09-21 UTC,
experiment `1789973982392307385`, host `sdfiana008`. The result preserves the literal
argument (including quotes, dollar sign and newline). Its source directory was
removed before packaging; the extracted execution directory was cleaned after
completion. No real checkout was deleted or changed.

Using implementation commit `19aa704`, the same example completed from S3DF to
NERSC as job `58681142`, experiment `1789974034772354506`, on `nid001549`:
`COMPLETED`, exit `0:0`, 14 seconds elapsed with a two-minute limit. The application
result preserved the same literal argument. This was a CPU-only standard-library
workload, not a GPU calculation or training test.
Direct filesystem checks also confirmed removal of the unique extracted directory
`source-work/lxm3.YWhqaTTldf`, while the shared parent and result remained. The
allocation is terminal; there are no remaining jobs from this slice.
Both independently captured trees produced source ID
`25e75a57235748b2434e6cf244952b05b572c39a31b860b9f685a6ccebc08668`.
Direct hashing of the retained local archive and both staged copies agreed:
`638da22089c7777c4075e2f3c6bbbf79a6c22f88a19faa3ded4e9a09d7d6c18e`.
The source ID includes the entrypoint; the staged archive hash identifies only
archive bytes. This distinction is intentional.

Local evidence: `/sdf/group/neutrino/youngsam/representations/lxm3-qualification.jmeYsG/`,
under `source-local-001/result.json` and `local/projects/qualification/`.
NERSC staging/logs: `/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/staging/projects/qualification/`;
the job writes `source-nersc-001/result.json` under that qualification root.

### Source reuse and append qualification, 2026-09-21

The extended suite passes **453 tests, 2 integration tests deselected**, with 15 new
reuse cases and expanded capture/inspection coverage. Separate Python processes
retrieve retained sources after the disposable checkout has been deleted, append
concurrently to one experiment, and get distinct WorkUnit IDs. Tests cover sync/
async contexts, arrays, singleton groups, generators, keyed-add rejection, old-unit
resubmission rejection, corrupt/missing archives, container/queue capture membership
and new-submission failure isolation. Existing upstream async deprecation warnings
remain. Ruff, formatting and `git diff --check` pass.

The runnable source launcher created experiment `1790014763042766562` on S3DF,
removed its disposable checkout before packaging, completed Local WorkUnit 1 and
exited. A separate `examples/source/extend.py` process retrieved source ID
`25e75a57235748b2434e6cf244952b05b572c39a31b860b9f685a6ccebc08668` from the new
catalog and submitted WorkUnit 2 to NERSC as independent job `58703111`.

The remote job and its batch/extern steps completed with exit `0:0` in 6 seconds
on `nid008329`. A third process awaited completion and retrieved native status/logs.
Direct hashing confirmed that the remote archive matched the retained/local archive:
`638da22089c7777c4075e2f3c6bbbf79a6c22f88a19faa3ded4e9a09d7d6c18e`.
The original WorkUnit record, retained archive and Local application-output bytes
were unchanged after extension. Remote output preserved literal quotes, dollar
signs and a newline. The queue has no remaining entry for the new job.

This used one GPU-partition node with a 120-second walltime for a standard-library
CPU probe, not training or GPU computation. No real checkout was removed, no old
qualification catalog was migrated, and pimm was untouched.

Author evidence is under
`/sdf/group/neutrino/youngsam/representations/lxm3-reuse.wctFcV/`:
`lxm.toml`, `local/experiments.sqlite3`, retained sources/logs and
`first-local/result.json`. Remote evidence is under
`/pscratch/sd/y/youngsam/lxm3-reuse-wctFcV/`: staging/logs and
`second-nersc/result.json`. The generated source-directory cleanup message names
`source-work/lxm3.b9XmF6sNwH`.
Read-only filesystem checks also confirmed that directory was removed while its
parent and the application result remained.
