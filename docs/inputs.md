# Retained artifacts as inputs

Bind a completed producer's artifacts to another WorkUnit on the same execution
endpoint. The consumer gets private extracted copies before its entrypoint starts.
No artifact download through the author machine, cross-site transfer, or worker
LXM3 installation is involved.

## API

The only public addition is `inputs` on the existing `Experiment.add`:

```python
experiment.add(
    job,
    inputs={"checkpoint": checkpoint_artifact},
    outputs={"result": "result.json"},
)
```

The workload reads `Path(os.environ["LXM_INPUT_DIR"]) / "checkpoint"`.
The path is a file or directory according to the artifact, regardless of its
original filename. `LXM_INPUT_DIR` is reserved and supplied by LXM3, like
`LXM_OUTPUT_DIR`; caller environment overrides cannot redirect it.

## Runnable producer and consumer

With a configured author catalog, first run the [output producer](../examples/outputs/launch.py):

```bash
lxm3 launch examples/outputs/launch.py --target=local
```

Then use its printed experiment ID in a **new process**:

```bash
lxm3 launch examples/inputs/launch.py --target=local --producer=EXPERIMENT_ID
```

The [consumer launcher](../examples/inputs/launch.py) loads WorkUnit 1 from that
experiment, binds its `metrics` file and `checkpoint` directory, and launches the
[standard-library worker](../examples/inputs/worker.py). The worker deliberately
changes its extracted checkpoint, then emits a retained result containing the
original bytes and producer/consumer hosts. The producer archive is unchanged.
`--producer_config=PATH` reads a different author catalog; this selects metadata,
not a transfer between execution sites. `--producer_task=0` selects an array task.

The same launcher accepts `--target=s3df --runtime=singularity --image=PATH` or
`--target=nersc --runtime=shifter --image=id:...`, along with repeated Slurm
`--resource=key=value` and `--workdir_root=PATH`. The producer must be on the same
endpoint, and the existing image must contain Python for this example worker.
No ML framework is required.

## Contract

- Input keys are single literal path components, not paths or shell templates.
  The mapping is frozen at `add()`, before a generator runs. Ordinary Jobs,
  ArrayJobs, singleton JobGroups and generators retain their invocation semantics.
- “Same-site” means the same recorded/configured host and user endpoint. The
  current author host and its FQDN normalize to local execution; on-site Slurm
  uses the current user. Remote aliases and usernames must match explicitly.
  There is no alias/DNS-equivalence inference or shared-filesystem detection.
- A cross-site binding fails before scheduler submission or input transfer.
  The referenced archive must be visible at the same absolute path on execution
  nodes. Local producer output in node-local scratch is not automatically staged
  to Slurm nodes, even when submission originates on the same author host.
- Immediately before execution, the host verifies each archive's SHA-256 identity
  and extracts it into a fresh private directory under the task's working directory.
  Only regular files/directories beneath the retained `data/` root are accepted;
  path traversal, links and special files are rejected. The constrained extraction
  works with older host Pythons that lack standard extraction filters.
- A missing, corrupt or invalid archive fails the job **without starting the
  payload/container**. Inputs are verified at execution time, so deleting or
  damaging an archive while its consumer queues is not silently accepted.
- Each array task gets its own extracted copies. Writes cannot alter the producer
  archive or another task's inputs. Distributed ranks within a single task share
  that task's directory; rank coordination remains application-owned.
- Container paths are derived from the existing work-directory mount, including
  Singularity `--cleanenv` and Shifter `--clearenv`. No additional input bind is
  needed. The host helper uses only the standard library; workers never open the
  author catalog or initiate SSH transfers.
- Input preparation uses the job's allocation, walltime and work-directory disk
  space. No worker transfer service or background dependency manager is introduced.

## Saved history and boundaries

`work_units.inputs` is one nullable JSON column in the existing SQLite catalog:
each input name records `id`, `archive_path`, `hostname` and `username`. These are
frozen references, not copied artifacts. The generated submission script retains
the same preparation instructions and can be inspected with `unit.get_script()`.
`unit.job` remains the concrete XM Job/ArrayJob, not a new wrapper containing
WorkUnit-level input/output policy. No new history accessor is added in this slice.

Existing catalogs acquire the nullable column when a new WorkUnit is created;
read-only producer lookup does not migrate old catalogs. Entries without input
declarations keep their previous behavior. Site scratch expiry/deletion still
applies: an input reference does not extend the producer's retention lifetime.

This is a new consumer WorkUnit, not continuation of its producer. There is no
automatic waiting for unfinished producers, retry, scheduler dependency, publication,
cross-site transfer, or pimm change. Checkpoint interpretation stays with the
application: Torch, TensorFlow, JAX and non-ML programs receive ordinary paths.

## Qualification

Regression coverage includes same-endpoint comparison, file/directory inputs,
literal names, arrays and generators, argument precedence, immutable declarations,
private-copy mutation, missing/corrupt archives, unsafe tar members, executable
permissions and preparation with site-packages disabled. Separate processes launch
a producer, reopen its artifacts, and launch a consumer after the producer exits.

Live qualification covers only Local, S3DF/Singularity and NERSC/Shifter. It reuses
the prior [output qualification](outputs.md#live-evidence-2026-09-21) artifacts,
not new producer submissions. Consumer metadata is kept in a separate author
catalog; no production training or pimm files are involved.

### Live evidence, 2026-09-21

| Target/runtime | Producer experiment | Consumer experiment | Slurm job / outcome |
| --- | --- | --- | --- |
| Local/host | `1790027988366987346` | `1790029902933463552` | Completed |
| S3DF/Singularity | `1790028032684299143` | `1790029937979419393` | `38747526`, `COMPLETED`, `0:0`, 4 seconds |
| NERSC/Shifter | `1790028085846652899` | `1790029948492685831` | `58711717`, `COMPLETED`, `0:0`, 15 seconds |

Both Slurm consumers had two-minute limits. The S3DF account/QoS was verified as
`neutrino:default@ampere` / `preemptable`, with one A100 and the previously qualified
SIF under `--cleanenv`. NERSC used `m5238_g`, debug GPU resources and the previously
qualified pinned Shifter image with `--clearenv`. No images were built or imported.
The NERSC host helper ran with its existing Python 3.6; no package installation was
needed there. This is input/output qualification, not a GPU-compute benchmark.

Each result was fetched and checked from a separate author process with an
author-only configuration. The metrics score and checkpoint contents matched the
producer. Source archive hashes were rechecked **after** consumer mutation and
remained unchanged. S3DF's producer/consumer ran on `sdfampere042`/`sdfampere010`;
NERSC's ran on `nid002736`/`nid001540`, demonstrating shared-site reuse across
compute nodes. Both native jobs are terminal, their queues empty, and their
temporary working directories removed.

Consumer catalog/results/config:
`/sdf/group/neutrino/youngsam/representations/lxm3-inputs-qualification.oDPMVL`.
NERSC staging:
`/pscratch/sd/y/youngsam/lxm3-inputs-qualification-oDPMVL`.
S3DF reused the existing SIF staging cache with unique new job paths.

Final regressions: **559 passed, 2 upstream integration tests deselected**,
including 34 new input cases. Changed Python files pass Ruff lint/format checks.
