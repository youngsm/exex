# Raw source capture

## API and scope

```python
SourceTree(entrypoint: ModuleName | CommandList, path=".", *, files=None)
experiment.freeze(source: SourceTree) -> FrozenSource
frozen.id: str
```

Both specifications work with the existing `experiment.package()` and
`xm.Packageable` interfaces, directly or inside `SingularityContainer`,
`DockerContainer` and `ShifterContainer`. There is no new execution API.

`SourceTree` is a recipe; `FrozenSource` is the captured value. This distinction
keeps ordinary launchers short while giving callers an explicit capture boundary
when they need to edit their checkout before packaging another target. A retained
tar archive fits this operation: no source catalog, database, build system or
parallel manifest is needed. Archive headers carry relative names and permissions.

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

The archive persists, but this slice provides **no ID-based lookup/reopen API**.
Keep the returned value for re-preparation within the launcher. Durable Experiment/
WorkUnit retrieval, source discovery, retained outputs, publication, continuation
and pimm migration remain separate work. No dependency build or scheduler behavior
was added.

Regression result: **366 passed, 2 integration tests deselected**, including 32 new
source-capture cases. Ruff checks, formatting checks and `git diff --check` pass.

### Live qualification

The disposable-source example completed locally on S3DF on 2026-09-21 UTC,
experiment `1789973982392307385`, host `sdfiana008`. The result preserves the literal
argument (including quotes, dollar sign and newline). Its source directory was
removed before packaging; the extracted execution directory was cleaned after
completion. No real checkout was deleted or changed.

The same example was submitted from S3DF to NERSC as job `58681142`, experiment
`1789974034772354506`, with a two-minute limit. At submission it was pending
priority; acceptance and verified staging alone are not execution success.
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
