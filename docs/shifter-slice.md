# Native Shifter qualification

Scope: one preinstalled image, one Slurm task, one GPU calculation, ordinary and
intentional-failure exits. This qualifies native command composition, not a new
`ShifterContainer` implementation or the entire runtime roadmap.

## Exact API

No public methods or fields are added. The complete runnable launcher is
[`examples/shifter/launch.py`](../examples/shifter/launch.py). Its existing objects are:

```python
executor = xc.Slurm(
    cluster="nersc",
    resources={
        "account": "m5238_g", "qos": "debug", "constraint": "gpu",
        "nodes": 1, "ntasks": 1, "cpus-per-task": 1, "gpus-per-node": 1,
        "image": "id:" + image_id, "module": "gpu",
    },
    walltime=120,
    workdir_root="/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/shifter-work",
)
package = xc.UniversalPackage(
    path=str(Path(__file__).parent),
    build_script="build.sh",
    entrypoint=["bash", "driver.sh"],
)
# Ordinary experiment.package([xm.Packageable(package, executor.Spec())])
# and experiment.add(xm.Job(executable, executor, args=..., env_vars=...)).
```

`image` and `module` are native Slurm/Shifter plugin options emitted as `#SBATCH`
directives. `Slurm.modules` remains ordinary environment-module loading; it is not
used to choose Shifter modules. The image is selected explicitly, not discovered
by an execution-site resolver.

The batch driver runs once, creates a new output directory, and executes:

```sh
exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
  bash task.sh "$input_dir" "$output_dir" "$fail" "$literal_argument"
```

`task.sh` enters Shifter after Slurm sets the task's GPU mask. It explicitly
forwards that mask and the example's payload environment values with `--env`,
binds the input read-only at `/mnt` and the output at `/media`, and runs the worker.
The batch mask is never forced onto a differently constrained task. The worker
checks the native image ID, task/runtime mask equality and one visible GPU before
computing a sum of squares with CuPy. CuPy is an example dependency, not a library
requirement or a framework-specific launch policy.

## Paths and image preparation

Source and `build.sh` are author-side files. Input, output, unpacking and volume
source paths are NERSC paths; LXM3 does not resolve them against the S3DF checkout.
The observed NERSC Shifter configuration binds `/pscratch` at the same path, so
the worker can use the shared extracted source directly. Both container mount
destinations were inspected in the preinstalled image before submission.

The image was already READY; no image pull, conversion or build is needed here.
`shifterimg lookup docker:cupy/cupy:v12.0.0` returned:

```text
2f83f19de6816d2c7c0ef24c7d51a7f65362c6f5d706cf5b5c797c7cfb563180
```

The job uses `id:` plus that complete native ID, not the mutable tag. This is a
Shifter identity, not an assumed OCI manifest digest. Importing images is an
explicit site preparation operation outside this example.

## Run from S3DF

Use the existing `nersc` TOML cluster with `server="nersc"` and NERSC staging.
Authentication must already work. From the fork checkout, stage the small example
fixture into a new directory:

```sh
ssh nersc 'mkdir /pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/shifter-input-002'
scp examples/shifter/input.txt \
  nersc:/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/shifter-input-002/input.txt

LXM_CONFIG=/sdf/group/neutrino/youngsam/representations/lxm3-qualification.jmeYsG/lxm.toml \
lxm3 launch examples/shifter/launch.py -- \
  --cluster=nersc \
  --image_id=2f83f19de6816d2c7c0ef24c7d51a7f65362c6f5d706cf5b5c797c7cfb563180 \
  --input_dir=/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/shifter-input-002 \
  --output_dir=/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/shifter-success-002 \
  --workdir_root=/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/shifter-work \
  --resource=account=m5238_g --resource=qos=debug --resource=constraint=gpu
```

Use a different new output directory and `--fail_worker` for the negative probe.
The negative worker completes the checks and writes evidence, then exits 7.
The launcher fixes one node/task, one requested GPU and a 120-second walltime.
NERSC debug reserves the whole GPU node; this probe computes on only one GPU.
Permissions of existing scratch/home ancestors must not be changed to make a
probe work. Select another authorized location if site mount access is blocked.

## API evaluation and limits

Native hooks are sufficient to test the real execution contract now. They are not
automatically the best final user-facing API: two scripts expose image/runtime
placement, environment precedence and site mounts to every application author.
A small first-class wrapper can remove that repetition, but it needs a separate
API review. A wrapper around the whole batch entrypoint executes the driver inside
the image; it does not produce `srun ... shifter ...` per task. This distinction
must remain explicit rather than be hidden behind resource-based auto-launching.

In particular, the native example explicitly forwards its own environment values;
it does not implement generic container injection of every `Job.env_vars` entry.
The existing `PythonPackage`/`UniversalPackage` source packagers need no redesign.
`PythonContainer` image building, registry publication, automatic import, runtime
caches, new step APIs and containerized multi-node collectives are not in this slice.

`Local` means the process's current execution host. Native Shifter commands can
run through it only where Shifter and the referenced site image already exist;
`Local` on S3DF does not become NERSC execution. Remote execution remains explicit
`Slurm(cluster="nersc")` over OpenSSH.

NERSC documents image selection in Slurm directives for multi-node jobs and the
outer `srun ... shifter ...` order. A later multi-node probe can reuse this order
and a shared source directory; the current single-node evidence is not that test.
[NERSC Shifter usage](https://docs.nersc.gov/development/containers/shifter/how-to-use/)

Podman-hpc is also available on NERSC and supports direct image builds, user-owned
image stores and explicit GPU/MPI options. Those capabilities do not replace the
need to qualify Shifter's already-installed images, and adding a second runtime
would widen this slice. No Podman adapter or fallback is introduced.
[NERSC Podman-hpc usage](https://docs.nersc.gov/development/containers/podman-hpc/overview/)

## Evidence

The focused tests execute generated batch scripts with a command recorder. They
verify literal arguments/environment, a different batch/task GPU mask, native
image/module directives, exit propagation and cleanup. They are not proof of GPU
or Shifter behavior. Both focused tests pass.

Native runs submitted from S3DF on 2026-09-20 (Pacific time):

| Probe | Job | Native result | Elapsed | Node |
| --- | --- | --- | --- | --- |
| Successful calculation | `58676072` | `COMPLETED`, `0:0` | 15 s | `nid008412` |
| Intentional failure after calculation | `58676522` | `FAILED`, `7:0` | 20 s | `nid001548` |

Both ran CuPy 12.0.0, saw the pinned full Shifter ID, preserved the quoted/newline
payloads and read-only input, and computed `sum(i*i for i in range(1024))` as
`357389824.0` on an A100. Each batch mask was `0,1,2,3`, while its native task and
container masks were both `0`; the runtime reported exactly one visible GPU.
The positive node had 80 GB A100s, the negative node 40 GB A100s. Both scheduler
steps returned the intended exit status. No retries, image pulls or builds occurred.

Results are retained beneath
`/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/` in
`shifter-success-001` and `shifter-failure-001`; the input is `shifter-input-001`.
Each output includes `result.json`, `driver-workdir.txt`, `batch-mask.txt` and
`task-mask.txt`. Extracted child directories were removed; the parent and outputs
remain. Worker source SHA-256 for both runs:

```text
ed39339475da09cb024878349480678a3f98d2c60b00b8d4369cc3fa171dba32
```

Saved job scripts and logs are under `staging/projects/qualification/{jobs,logs}/`
with respective prefixes `shifter_probe_1789968519032283202_1` and
`shifter_probe_1789968815999347720_1`. This evidence does not establish multi-node
container collectives, a generic Shifter package wrapper or image-builder support.
