# Minimal HPC launcher: API and scope

Scope agreed 2026-09-20. This supersedes the broad roadmap's initial release gate,
not its eventual feature list. No catalog, artifact API, continuation or new backend.

## Exact public API

No public constructor or method is added. Retain:

```python
xc.create_experiment(title, project=None, *, config=None)
xc.Local()
xc.Slurm(cluster=None, resources={}, walltime=None, ...)
executor.Spec()
xm.Packageable(package_spec, executor_spec=executor.Spec())
experiment.package(packageables)
experiment.add(xm.Job(executable, executor, args=None, env_vars=None))
```

`PythonPackage`/`UniversalPackage` remain the packaging interfaces. Applications
receive explicit durable output paths through ordinary job arguments. Native Slurm
tools and the saved job ID/script are the initial inspection/cancellation interface.
No new output environment variable, result type or execution-context helper.

## Evaluation

These interfaces already describe the required job. The missing work is execution
correctness and qualification, not API design. Freeze only the files selected by
the existing packager; do not add `SourceTree`, `freeze()` or a source catalog here.
The runnable example must demonstrate the actual APIs above, not future methods.

The only new internal adapter is a small OpenSSH command/file boundary for Slurm.
It uses the same host/user and system SSH configuration for both operations:

```python
run(argv, *, hostname=None, username=None, **subprocess_options)
OpenSSHFileSystem(hostname, username=None)
# Only the file operations needed by the existing ArtifactStore:
# abspath, makedirs, exists, info, put_file, pipe_file, mv.
```

No remote Python installation or resident service is required by this adapter.
GridEngine keeps its existing transport. No connection retry or lost-job discovery.
The private store factory gains `use_openssh=False`; only the Slurm caller enables it.

The example launcher accepts exactly `--target`, `--output_dir`, repeatable
`--resource=key=value`, `--python` and `--check_gpu`. These are example flags, not
package API additions. Its 120-second Slurm request is a launch/driver probe, not
a training or distributed-compute qualification.

## Implementation boundary

1. Literal argument/environment handling, correct mounts and GPU flags, local
   nonzero-exit propagation and context cleanup, durable local logs.
2. Collision-safe staged package paths and complete-file exposure. Keep existing
   packaging, not an artifact lifecycle or deduplication service.
3. OpenSSH Slurm command/staging path, parsable submission IDs and useful locations
   in launch output. Preserve the existing saved job ID and script.
4. One minimal launcher with an application-owned output directory. Qualify local,
   S3DF and S3DF-to-NERSC paths where existing authorization permits; report blocked
   authentication or untested paths honestly. No changes to pimm or exex.

Each implementation claim needs regression evidence. Live jobs are bounded probes,
not training experiments. Record per-file added/deleted line counts at handoff.

## Running the example

Install this checkout into a Python 3.10–3.12 environment with `pip install -e .`.
Set `LXM_CONFIG` to your TOML file. For example, with paths replaced by directories
you own that compute nodes can access:

```toml
[local.storage]
staging = "/absolute/local/staging"

[[clusters]]
name = "s3df"
[clusters.storage]
staging = "/absolute/s3df/shared/staging"

[[clusters]]
name = "nersc"
server = "nersc"  # Your ~/.ssh/config alias; authentication must already work.
[clusters.storage]
staging = "/absolute/nersc/shared/staging"
```

Omitting `server` means on-site submission. Slurm uses OpenSSH configuration for
authentication, identities and jump hosts, not the TOML Paramiko-specific options.
The execution site needs Bash, standard Unix tools and the workload's runtime;
this example additionally needs Python 3 and unzip. It does not install LXM3 or
Python dependencies on the execution site. Supply `--python=/path/to/python` if
the workload needs an existing environment.

From the fork checkout, launch locally with an unused absolute output directory:

```sh
lxm3 launch examples/hpc/launch.py -- \
  --target=local --output_dir=/absolute/local/probe-output
```

Changing the executor does not change the package or workload. These are the native
resource requests used for qualification; account membership is user-specific:

```sh
lxm3 launch examples/hpc/launch.py -- \
  --target=s3df --output_dir=/absolute/s3df/shared/probe-output \
  --resource=account=neutrino:default@ampere --resource=partition=ampere \
  --resource=qos=preemptable --resource=gpus=1 \
  --resource=cpus-per-task=1 --resource=mem=2G --check_gpu

lxm3 launch examples/hpc/launch.py -- \
  --target=nersc --output_dir=/absolute/nersc/shared/probe-output \
  --resource=account=m5238_g --resource=qos=debug --resource=constraint=gpu \
  --resource=nodes=1 --resource=cpus-per-task=1 --resource=gpus=1 \
  --python=/usr/bin/python3 --check_gpu
```

The NERSC request follows its [documented GPU debug example](https://docs.nersc.gov/jobs/examples/#command-line-submission-of-common-jobs).
Debug is for short tests, not production training; it reserves a whole node even
with a one-GPU request. Site policy remains launcher configuration, not package logic.

Local context exit waits for completion and raises on nonzero exit. Combined
stdout/stderr are saved as `task-0.log` (zero-based for arrays), not streamed to the
terminal. Slurm context exit waits for submission only. Use the printed job ID with
native `squeue`, `sacct` and `scancel`; use SSH for a remote site. The script and job
ID are also saved beneath `projects/qualification/jobs/<job-name>/` in staging.

`result.json` is written by the workload in `--output_dir`; reruns need a new
directory because the probe deliberately refuses to overwrite its result. Files
left only in the temporary working directory are deleted at job exit. There is no
automatic output collection or remote log fetching in this slice.

## Qualification and remaining limits

- Regression tests cover actual shell execution, literal argument/environment
  round trips, mount rendering, GPU flags, local failures, SSH failure propagation
  without retries, content-derived package names and interrupted uploads.
  Final run: **239 passed, 2 deselected** in 10.48 seconds on Python 3.12.14, with
  41 upstream deprecation warnings. Ruff and `git diff --check` pass.
- Live local execution completed and preserved the probe's literal message.
- S3DF job `38679691` completed with exit `0:0` on `sdfampere020`, exposing one
  NVIDIA A100-SXM4-40GB under `neutrino:default@ampere`, QoS `preemptable`.
- **S3DF-to-NERSC completed end to end:** job `58660405`, submitted from
  `sdfiana008`, completed with exit `0:0` on `nid001928` in eight seconds. The
  staged workload preserved the literal message, reported A100 GPUs, and retained
  its result and log after temporary-directory cleanup. Debug exposed four GPUs.
  Result: `debug-output-20260921T0028/result.json` under the NERSC evidence root.
  Experiment: `1789950517415496656`; native job ID is also saved in staging.
- Earlier evidence remains: the first internal-policy request was rejected;
  shared job `58660105` stayed queued and was cancelled. Debug job `58660351`
  reached a compute node but exposed Python 3.6.15's lack of `subprocess(text=...)`.
  Replacing that example-only argument with `encoding="utf-8"` fixed the probe;
  the fix was checked on NERSC's interpreter before the successful rerun. No
  library/API change or automatic retry was added.

These are archive/launch/output and GPU-driver probes, not CUDA computation,
container execution, multi-node training, preemption or continuation qualification.
No cloud jobs, image builds/pulls or pimm/exex changes are included. Package identity
is the prepared file's SHA-256, not a deterministic source snapshot or an output
catalog. A failed upload may leave a temporary file; there is no cleanup service.

Evidence is retained outside the source trees:
`/sdf/group/neutrino/youngsam/representations/lxm3-qualification.jmeYsG` on S3DF,
and `/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG` on NERSC. These contain the
staged packages/scripts/logs and the completed local/S3DF/NERSC probe results. The existing
Slurm-account skill guided the association/parent-limit check before the S3DF probe.

## Per-file diff sizes

Initial implementation commit `1f67053`, relative to `c08a336`; counts include
tests, examples and docs, before the NERSC qualification follow-up.
The earlier explicit-site routing patch is not counted again.

| File | Added | Deleted |
| --- | ---: | ---: |
| `README.md` | 13 | 8 |
| `docs/fork-plan.md` | 17 | 18 |
| `docs/minimal-hpc-slice.md` | 187 | 0 |
| `lxm3/_vendor/xmanager/xm/core.py` | 6 | 3 |
| `lxm3/clusters/slurm.py` | 11 | 30 |
| `lxm3/clusters/ssh.py` | 76 | 0 |
| `lxm3/xm_cluster/artifacts.py` | 10 | 18 |
| `lxm3/xm_cluster/execution/job_script_builder.py` | 60 | 72 |
| `lxm3/xm_cluster/execution/local.py` | 9 | 4 |
| `lxm3/xm_cluster/execution/slurm.py` | 21 | 7 |
| `lxm3/xm_cluster/experiment.py` | 19 | 12 |
| `lxm3/xm_cluster/packaging/create_archive.py` | 3 | 2 |
| `lxm3/xm_cluster/packaging/router.py` | 25 | 50 |
| `tests/clusters/slurm_test.py` | 11 | 10 |
| `tests/clusters/ssh_test.py` | 113 | 0 |
| `tests/execution_test.py` | 31 | 53 |
| `tests/launch_safety_test.py` | 208 | 0 |
| `examples/hpc/build.sh` | 3 | 0 |
| `examples/hpc/launch.py` | 73 | 0 |
| `examples/hpc/workload.py` | 33 | 0 |
