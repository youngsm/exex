# Native Slurm steps through an ordinary entrypoint

This is a two-node, standard-library-only qualification example. It uses the
existing `Slurm`, `UniversalPackage`, `Packageable`, and `Job` APIs. There is no
new step object, worker manager, distributed bootstrap, or framework dependency.

## Execution contract

1. LXM3 submits one two-node batch allocation, with a two-minute limit.
2. It unpacks once beneath `workdir_root`, which must be shared across the nodes.
3. The packaged `driver.sh` runs once. It creates a new output directory and a
   token in the unpacked workspace, then executes:

   ```sh
   srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
     "$python" worker.py "$output_dir" "$fail_rank"
   ```

4. Each worker records its native Slurm rank, host, working directory, source
   checksum, driver token, and an environment value containing quotes/newlines.
   Reading the token tests shared access to a file created *after* unpacking.
5. `srun` waits for its workers; its exit code reaches the batch script. LXM3
   cleans up the unpacked child directory, leaving the output directory intact.

Allocation and task launch are separate: requesting two nodes does not duplicate
the driver. Site settings remain in LXM3's TOML configuration and Slurm resources.
This example fixes two nodes, one task per node, and one CPU per task; other
resource flags select the account, queue, node type, and optional GPUs.

## Run from S3DF to NERSC

Use the fork's authoring environment and an existing `nersc` SSH/site definition.
The execution host needs only Bash, Slurm commands, unzip, and Python 3.6+.
It does not need LXM3, a training framework, or a copied virtual environment.

```sh
export LXM_CONFIG=/sdf/group/neutrino/youngsam/representations/lxm3-qualification.jmeYsG/lxm.toml
lxm3 launch examples/slurm_step/launch.py -- \
  --cluster=nersc --python=/usr/bin/python3 \
  --output_dir=/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/slurm-step-success-002 \
  --workdir_root=/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/slurm-step-work \
  --resource=account=m5238_g --resource=qos=debug \
  --resource=constraint=gpu --resource=gpus-per-node=1
```

GPU nodes are a site/account choice, not a workload requirement: this probe does
no GPU computation. With suitable CPU account access, use CPU nodes instead.
Use a new output directory every time; the driver refuses an existing directory.

Repeat the command with a new output path and `--fail_worker` for the negative
test. Rank 1 waits until rank 0 has recorded startup, then exits with code 7.
Rank 0 waits 30 seconds before writing its completion file. Native Slurm must
terminate it before that; neither completion file should exist after failure.

## What this does not claim

This tests host execution, not containers or framework collectives. Putting
`srun` inside a container entrypoint is not equivalent to launching a container
per task; that runtime placement needs separate qualification. There is no
automatic retry, requeue, continuation, borrowed allocation, or `salloc` API.

The local regression tests capture the driver's `srun` command and execute the
worker directly. They do not simulate or prove Slurm's scheduling or cancellation.

## Qualification evidence

Submitted from S3DF to NERSC on 2026-09-20 (Pacific time), using host Python and
shared Perlmutter scratch. No library implementation changes were needed.

- Success job **58674695**, experiment `1789966063218116775`: `COMPLETED`, exit
  `0:0`, six seconds. Native step `58674695.0` ran two ranks on `nid008560` and
  `nid008581`. Both recorded the same working directory and driver-created token,
  the exact environment value, and the packaged worker's source checksum. Both
  completion files survived; the unpacked child was removed and its parent kept.
- A first negative-test invocation, experiment `1789966170939300304`, failed
  with SSH exit 255 while creating the remote job-script directory, **before
  `sbatch`**. The error propagated; no job was submitted or automatically retried.
- A fresh explicit submission, job **58674935**, experiment
  `1789966551948301399`: `FAILED` as intended after six seconds on `nid002153`
  and `nid002156`. Rank 1 exited 7; Slurm terminated rank 0 before its 30-second
  sleep completed. Step `.0` recorded `7:0`; the batch recorded `143:0` because
  `srun` also reports signalled tasks (`128 + SIGTERM`). This follows its
  [documented return behavior](https://slurm.schedmd.com/srun.html#SECTION_RETURN-VALUE),
  not a promise that the batch status equals the first worker's exit code.
  Both worker records survived, neither completion file existed, and the shared
  source/token/environment checks passed. The unpacked child was removed.
  Neither live job restarted or requeued; both allocations have ended.

Evidence root on NERSC:
`/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG`. Success output is in
`slurm-step-success-001/`; scripts and logs are beneath
`staging/projects/qualification/{jobs,logs}/slurm_step_probe_<experiment>_1/`.
The negative-test output is in `slurm-step-failure-002/`.
The worker SHA-256 used for verification is
`c6dc96e71167d4ede4c7605384ad70504f79d2ba8b8e6fa7403c781b7fcae5d8`.

Regression suite: **265 passed, 2 integration tests deselected**, including four
new example tests. Ruff and shell syntax checks passed. These tests use no ML
framework and do not establish distributed training or container correctness.
