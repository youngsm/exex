# ShifterContainer: installed-image execution

## Public API

```python
ShifterContainer(entrypoint, image)
ShifterOptions(bind={}, modules=[], extra_options=[])
Local(..., container_options=None)
Slurm(..., container_options=None)
GridEngine(..., container_options=None)
```

The collections above are per-instance defaults, not shared mutable objects.
`container_options` accepts `SingularityOptions`, `DockerOptions`, `ShifterOptions`
or `None`. The executable selects the runtime; `None` uses its defaults. One type
check rejects mismatched options (including options on a host-only executable)
before submission. The separate runtime-specific executor keywords are removed,
not retained as aliases. No live GridEngine/Shifter qualification is claimed.

This matches LXM3's existing Singularity/Docker package wrappers: the source
package is independent of its prepared dependency image. A private `_ContainerSpec`
shares only `entrypoint` and the delegated `name`; it already implements the
existing `xm.ExecutableSpec` contract. There is no new public base class or
runtime lifecycle. `PythonContainer` remains a separate image-build recipe.

## Runnable example

See [container_launch.py](../examples/shifter/container_launch.py). From the
authoring host, with the fork installed and a `nersc` cluster in the TOML config:

```bash
lxm3 launch examples/shifter/container_launch.py -- \
  --lxm_config=/path/to/lxm.toml \
  --cluster=nersc \
  --image=id:2f83f19de6816d2c7c0ef24c7d51a7f65362c6f5d706cf5b5c797c7cfb563180 \
  --input_dir=/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/shifter-input-001 \
  --output_dir=/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/NEW-EMPTY-OUTPUT \
  --workdir_root=/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/shifter-work \
  --resource=account=m5238_g --resource=qos=debug --resource=constraint=gpu
```

Create a new empty output directory on the execution site first. The example
requests one whole GPU node for at most two minutes and runs one Python process.
It performs a small CuPy calculation, checks a read-only input mount, and writes
`result.json`. `--fail` exits 7 after writing the result. It does not require pimm,
PyTorch or the LXM3 package inside the image. These account/path values are site
qualification fixtures, not library defaults.

## Contract and boundaries

- `image` is passed unchanged to native Shifter and the Slurm `--image` directive.
  Use `id:<native-Shifter-id>` to pin it. Tags remain tags; LXM3 does not freeze,
  import, build or look them up. The image must already exist at the selected site.
- `bind` maps execution-host **directories** to existing container directories;
  `"/site/input": "/mnt:ro"` is a read-only mount. Native mount errors propagate.
- Choose a `workdir_root` visible at the **same path inside the image**. NERSC's
  `/pscratch` site mount satisfies this. A host temporary directory is not
  automatically visible merely because it exists. Unlike Singularity, Shifter
  cannot use LXM3's separate parameter-file bind. Both source and parameters stay
  together in the chosen directory; only its unique temporary child is cleaned.
- `modules=["gpu"]` selects Shifter modules, not shell `module load` commands.
  They appear in both the Slurm header and runtime invocation. Empty means native
  defaults, not "disable all"; use `["none"]` to disable them. GPU modules are not
  inferred from requested resources.
- Native scheduler environment and GPU visibility use the existing environment
  file. Job arguments and environment are sourced inside the image, preserving
  literal quotes/newlines and overriding image defaults. `--clearenv` is optional,
  not a new default environment policy.
- LXM3 owns image/workdir and, when supplied, module selection. Use these typed
  fields rather than competing `resources`, `extra_directives` or `extra_options`.
  Header generation copies resources; it does not mutate the caller's executor.
- This wraps **one entrypoint**. It adds no `srun`, ranks, requeue policy or worker
  replication. A batch process sees its batch GPU mask, not a hypothetical task
  mask. For a host driver launching multiple container tasks, retain the explicit
  [native srun/Shifter example](shifter-slice.md). This slice does not containerize
  that driver or claim multi-node container collective support.
- `Local` runs on the current host; Shifter and the image must exist there. An
  S3DF `Local` executor does not turn into a NERSC submission.

This boundary follows Shifter's native
[directory-mount and environment behavior](https://docs.nersc.gov/development/containers/shifter/how-to-use/).
It requires no custom image layout, launcher-side image service or runtime ABC.

## Verification

`tests/shifter_container_test.py` covers wrapper constructors and source packaging,
header/runtime agreement without mutation, literal arguments/environment, Local
and Slurm array selection, application exit codes and temporary-directory cleanup.
The runtime stand-in executes the generated shell but does not prove container
isolation or GPU behavior. `tests/executor_test.py` checks the full matrix of
host/Singularity/Docker/Shifter with all option types on Local/Slurm/GridEngine,
and verifies that a mismatch never reaches scheduler submission.

Regression result: **334 passed, 2 integration tests deselected**, with 41 existing
deprecation warnings. Ruff and `git diff --check` pass.

### Live typed-API qualification

Qualified from S3DF to NERSC on 2026-09-20 PDT (2026-09-21 UTC), using commit
`75a3eca`. The initial expired-certificate preflight was resolved by certificate
renewal; no runtime or library fix was needed. Both jobs used the public launcher
above, the same staged source archive and the preinstalled pinned CuPy image.

| Probe | Job | Node | State | Exit code | Elapsed |
| --- | --- | --- | --- | --- | --- |
| Success | 58678901 | nid003201 | COMPLETED | 0:0 | 21 s |
| Intentional failure | 58678913 | nid003468 | FAILED | 7:0 | 21 s |

Both ran a real CuPy sum-of-squares calculation on an A100-SXM4-40GB, producing
`357389824.0`. Both preserved literal argument/environment values, read the input
through `/mnt:ro`, verified that writing there failed with `EROFS`, and retained
`result.json` through the writable `/media` mount, including after exit 7.
`--clearenv` was enabled. The recorded full image ID matched the pinned reference.

Each batch process saw `CUDA_VISIBLE_DEVICES=0,1,2,3` and four devices, matching
the four-GPU allocation. The small calculation used device 0; this is not a
multi-GPU computation or a per-task GPU-isolation qualification. No implicit
`srun` or worker replication was introduced.

Source SHA-256: `7f774b9656334bebbb2720ec2c29d13c52b71f7732f017fa528207f96e52bb08`.
Archive SHA-256: `2de084b30d5bd3417f9d4ca725cba3bbd90986c4f4d12e586e4f4dfad371136f`.
Both results report the same source hash as the committed worker. The staged
archive was independently hashed after execution and matched its filename.

Evidence remains on NERSC beneath
`/pscratch/sd/y/youngsam/lxm3-qualification-jmeYsG/`:

- Results: `shifter-typed-success-001/result.json` and
  `shifter-typed-failure-001/result.json`.
- Generated scripts/logs: `staging/projects/qualification/{jobs,logs}/`, under
  `shifter_container_probe_1789971830265033542_1` and
  `shifter_container_probe_1789971831313451936_1`, respectively.

Direct filesystem checks confirmed that only the temporary source children
`shifter-work/lxm3.L2zjbZ7Iw2` and `shifter-work/lxm3.kHh1dnGdjt` were removed;
the shared parent, input and result files remain. Both allocations are terminal.
The committed-code regression rerun remained **334 passed, 2 deselected**.
