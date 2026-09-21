# Prebuilt SIF + separately packaged source

This example uses existing APIs: `SingularityContainer(UniversalPackage(...),
image_path=...)` and `Slurm(singularity_options=SingularityOptions(...))`.
The image supplies Python/PyTorch/CUDA dependencies; the source archive supplies
`probe.py`. Packaging does not build or pull an image.

The optional workload uses PyTorch to perform an actual CUDA matrix product.
Neither LXM3 nor the launcher imports PyTorch. Other workloads can use JAX,
TensorFlow, or no ML framework.

## Run

Use a cluster configuration whose staging directory and input/output paths are
shared by the authoring host and execution host. This example creates the output
directory on the authoring host; it is not a remote-directory provisioning API.
The SIF must already exist and be readable on the authoring host. LXM3 stages it
under its content hash, separately from the source archive.

Prepare an input directory containing `value.txt` with the number `3`, then:

```bash
LXM_CONFIG=/path/to/lxm.toml lxm3 launch examples/sif/launch.py -- \
  --cluster=s3df \
  --image=/path/to/existing-pytorch-cuda.sif \
  --input_dir=/shared/probe/input \
  --output_dir=/shared/probe/success-001 \
  --workdir_root=/shared/probe/work \
  --resource=account=neutrino:default@ampere \
  --resource=partition=ampere \
  --resource=qos=preemptable
```

The example fixes one node, one GPU, two CPUs, 8 GiB host memory, and a three-minute
walltime. Replace site-specific resource flags for another cluster. Choose a new
output directory each time: existing output is deliberately not overwritten.

The input and output directories are bound at `/probe-input` and `/probe-output`;
packaged source runs at `/run/lxm3/workdir`. `--cleanenv` prevents arbitrary host
environment inheritance. LXM3 forwards Slurm variables and the allocated
`CUDA_VISIBLE_DEVICES` mask explicitly, then applies the job's literal arguments
and environment inside the container. GPU allocation adds the runtime's `--nv`.

Add `--fail` with a new output directory to record GPU results and deliberately
exit 7. The Slurm job must fail, its temporary extraction directory must disappear,
and its mounted `result.json` must remain.

To change code without rebuilding dependencies, copy `build.sh` and `probe.py` to
a new source directory, change `REVISION` in the copied probe, and pass
`--source_dir=/path/to/edited-source` with the same `--image` and a new output path.
The source archive changes; the staged SIF is reused.

## Qualification

The initial regression exposed one concrete library defect: `--cleanenv` removed
Slurm's GPU mask because LXM3 forwarded only `SLURM_*`. The fix adds
`CUDA_VISIBLE_DEVICES` to the existing environment-file filter. Tests failed
before that one-line fix and pass after it. The probe checks the mask and visible
device count before computing; it does not rely on device cgroups for correctness.

S3DF live evidence, 2026-09-20 Pacific:

| Case | Native job | Result | Elapsed |
| --- | --- | --- | --- |
| Original source | `38700292` | `COMPLETED`, `0:0` | 15 s |
| Intentional failure | `38700408` | `FAILED`, `7:0` | 14 s |
| Source-only edit | `38700480` | `COMPLETED`, `0:0` | 12 s |

All jobs ran on `sdfampere001` with exactly one allocated A100-SXM4-40GB.
Apptainer 1.5.3 was invoked through the `singularity` command. The existing shared
SIF provided Python 3.8 and PyTorch `1.13.1+cu116`; its original file was not changed.
Host and container both reported `CUDA_VISIBLE_DEVICES=0`; each probe saw one GPU.
The mounted input was `3`, the CUDA product's checksum was `12288`, and literal
quotes, dollar signs, spaces, and newlines survived in arguments/environment.

Evidence root:
`/sdf/group/neutrino/youngsam/representations/lxm3-qualification.jmeYsG`.
The live invocations used the command above with these flag values:

| Flag | Value |
| --- | --- |
| `LXM_CONFIG` | `<evidence root>/lxm.toml` |
| `--image` | `/sdf/group/neutrino/junjie/img-CIDeR-ML/larcv2_ub20.04-cuda11.6-pytorch1.13-larndsim-2023-11-07.sif` |
| `--input_dir` | `<evidence root>/sif-input` |
| `--workdir_root` | `<evidence root>/sif-work` |
| `--output_dir` | `<evidence root>/sif-success-001 output`, `sif-failure-001`, or `sif-source-edit-001` |

The second invocation added `--fail`; the third added
`--source_dir=/lscratch/youngsam/tmp/lxm3-sif-source.wq79ZC`, containing only the
copied build script and probe with `REVISION = "source-edit"`. Each output contains
`result.json`. Native scripts/logs are under
`s3df/projects/qualification/{jobs,logs}/sif_probe_<experiment>_1`, with experiments
`1789968272063169346`, `1789968379690975364`, and `1789968436995078809`, respectively.

Recorded source SHA-256 changed from
`02bdd8533373cbc840bf92cd5173b9ecdfbf555075f65f50eba637417fe22eec` to
`8d13681ce04e6562362fcd262e89b3c1511180435dd395371899f9789ee78b68`.
The original and edited code archives are
`s3df/projects/qualification/archives/ab8e8521090a278b713fbfea8d53a111829b224508c81f3912525b517e5b0ff0.zip`
and `s3df/projects/qualification/archives/1ae6128bd35ee683098576ed24d204f81f3ba2180a2b973c385e341836a79f37.zip`.

The staged image is
`s3df/projects/qualification/containers/d09506953e64575bde7ab7163751c1fc6be7cccd9457672c5957068252f5f27c.sif`
(9,175,400,448 bytes), unchanged at mtime `1789968311` across all three launches.
All temporary children under `sif-work` were removed; the parent, image, source
archives, and result records remain. None of these jobs remains queued or running.

Local tests exercise command construction, explicit mask forwarding, unrelated
environment exclusion, exit/cleanup behavior, and source-only image reuse.
They do not simulate native GPU isolation or replace these live checks.
This slice does not qualify NERSC runtimes, image building, multi-node containers,
framework collectives, checkpoint continuation, or scheduler preemption recovery.
