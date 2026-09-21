# Retained outputs

Declare the files or directories a successful job must produce. LXM3 retains them
at the execution site before removing the temporary working directory. Fetching
is explicit; it does not publish anything to an external service.

## API

```python
experiment.add(job, ..., outputs={"metrics": "metrics.json", "checkpoint": "checkpoint"})
unit.artifacts(*, task: int | None = None) -> Mapping[str, Artifact]
artifact.id: str
artifact.fetch(into: PathLike) -> Path
```

`outputs` belongs to the WorkUnit, not the executable. It works with ordinary Jobs,
ArrayJobs, singleton JobGroups and async job generators. The mapping is copied
at `add()`; invocation defaults and overrides retain their existing semantics.

The runnable [launcher](../examples/outputs/launch.py) packages the accompanying
[standard-library worker](../examples/outputs/worker.py). For local execution:

```bash
lxm3 launch examples/outputs/launch.py --target=local
```

The worker writes to `Path(os.environ["LXM_OUTPUT_DIR"])`. It does not import LXM3.
After the launcher exits, use its printed experiment ID in a new Python process
with the same author catalog:

```python
from pathlib import Path
from lxm3 import xm_cluster as xc

experiment = xc.get_experiment(1790027988366987346)
unit = experiment.work_units()[1]
metrics = unit.artifacts()["metrics"].fetch(Path("download/metrics.json"))
checkpoint = unit.artifacts()["checkpoint"].fetch(Path("download/checkpoint"))
print(metrics.read_text())
```

For arrays, select the zero-based task explicitly: `unit.artifacts(task=0)`.
Completed tasks remain available even when another task fails. A one-task job
also accepts `task=0`.

## Contract and boundaries

- Paths are literal, relative to `LXM_OUTPUT_DIR`; no absolute paths, `..` or shell
  expansion. Outputs are regular files or directories; links and special files
  are not supported.
- The workload must finish writing before its entrypoint exits successfully.
  Capture then runs within the same allocation and walltime. Every declared output
  is required. A missing output or capture error fails the job and publishes none
  of that task's result set. A nonzero payload exit skips capture entirely.
- A single completion manifest exposes each task's complete result set. Before
  publication, `artifacts()` returns `{}`. Jobs without declarations and older
  catalog entries also return `{}`. This is not a scheduler status query.
- Snapshots are normalized, uncompressed tar archives with SHA-256 identities.
  Modification times and ownership are normalized; executable file bits survive.
  Capture copies bytes, so later changes to the working tree cannot alter a result.
  IDs identify archive bytes, not semantic file contents: different Python tar
  writers can encode equivalent headers differently and produce different IDs.
  This is not a cross-job deduplication or garbage-collection service.
- SQLite retains the declarations and site artifact-directory reference; workers
  write only archives and a small manifest at the site. They never open SQLite.
  Existing catalogs gain nullable columns on the next submission, as with Job
  history; reads alone do not migrate anything.
- Retrieval uses the recorded endpoint, even if cluster profiles change. It reads
  the manifest but does not download artifacts until `fetch()`. SSH/transfer errors
  propagate directly without retries. Fetch verifies the archive hash and uses
  Python's standard data extraction filter before exposing results.
- `into` is the exact new destination, not a parent directory. Existing files,
  directories and symlinks are never overwritten. A failure during the final copy
  of a directory can leave a partial **new** destination for the caller to remove.
- Retention lasts only as long as the execution site's staging storage. Purge/deletion
  is not repaired: missing manifests yield `{}`, missing archives fail at fetch.
  Fetch important results elsewhere before scratch expiry.

The host needs `python3` for the embedded standard-library capture helper; no LXM3
installation is needed on workers, inside or outside the container. Author-side
fetch uses Python's [data extraction filter](https://docs.python.org/3.10/library/tarfile.html#extraction-filters),
available in 3.10.12+, 3.11.4+ and 3.12; package metadata excludes older patch
releases. Use maintained security patch releases.

The output directory is inside the existing work-directory mount. S3DF's SIF
sees its container path; NERSC Shifter sees its shared site path. No extra output
bind is needed. Multi-process applications must coordinate their writes themselves;
this API creates one result set per job/array task, not per distributed rank.

For another job to consume retained results, see [input bindings](inputs.md).
This API does not add publication, retries, continuation, per-attempt
history, a CLI fetch command, automatic uploads, or ML-framework dependencies.
Re-executing a successful native job against the same result location fails capture
instead of replacing its retained content; requeue/continuation is separate work.

## Qualification

The focused suite executes real Local/generated scheduler scripts and verifies
cleanup, failure propagation, all-or-nothing publication, array isolation,
source-independent identities, non-overwriting fetch, unsafe/corrupt archives,
streaming SSH errors, argument forwarding and fresh-process retrieval.

Live qualification uses only S3DF/Singularity and NERSC/Shifter, not the opposite
site/runtime pairings. The worker is standard-library-only; these probes test output
capture and retrieval, not new GPU-compute or collective behavior.

### Live evidence, 2026-09-21

| Target/runtime | Experiment | Slurm job | Outcome |
| --- | --- | --- | --- |
| Local/host | `1790027988366987346` | — | Completed; file and directory fetched |
| S3DF/Singularity | `1790028032684299143` | `38745384` | Completed, `0:0`, 2 seconds |
| S3DF/Singularity, missing checkpoint | `1790028163195267124` | `38745552` | Expected failure, `1:0`, 3 seconds; no artifacts |
| NERSC/Shifter | `1790028085846652899` | `58710267` | Completed, `0:0`, 13 seconds |
| NERSC/Shifter, missing checkpoint | `1790028170707694947` | `58710289` | Expected failure, `1:0`, 14 seconds; no artifacts |

Both sites used two-minute limits. S3DF requested one A100 using
`neutrino:default@ampere`, partition `ampere`, QoS `preemptable`, and the existing
SIF `d09506953e64575bde7ab7163751c1fc6be7cccd9457672c5957068252f5f27c.sif`
with `--cleanenv`. NERSC used account `m5238_g`, debug GPU nodes and installed image
`id:2f83f19de6816d2c7c0ef24c7d51a7f65362c6f5d706cf5b5c797c7cfb563180`
with the GPU module and `--clearenv`. All launches originated on S3DF.

Fresh author processes, with no cluster profiles, fetched metrics and checkpoint
directories from Local, S3DF and NERSC and verified their contents. The missing-output
probes left no completion manifests. Native accounting confirmed all four Slurm
jobs terminal, empty queues and cleaned temporary working directories. No site
image builds/imports, worker LXM3 installs or pimm changes were needed.

Identical checkpoint files had different archive IDs on NERSC and S3DF because
the host Python tar writers encoded unused device-number fields differently
(ASCII zeroes versus NUL padding). Inspection confirmed identical file contents
and equivalent normalized metadata. Retrieval verifies each archive's own ID.

Evidence/catalog and fetched files are retained under
`/sdf/group/neutrino/youngsam/representations/lxm3-outputs-qualification.9jk3dt`.
NERSC staging is under `/pscratch/sd/y/youngsam/lxm3-outputs-qualification-9jk3dt`.
The S3DF probes reused the cached SIF in the existing qualification staging root;
their unique experiment/job paths did not replace earlier results.

Final regression suite: **525 passed, 2 integration tests deselected**, including
22 new output cases. The deselected upstream tests are not substituted for the
live checks above. Changed Python files pass Ruff lint/format checks.
