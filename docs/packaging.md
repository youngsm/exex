# Source and containers

Package application code separately from its runtime dependencies. Reuse a
prepared Python environment, a SIF image, or an installed Shifter image across
experiments. Changing a Python file does not require rebuilding the runtime.

`SourceTree` captures raw files without building or installing them:

```python
source = xc.SourceTree(xc.ModuleName("worker"), path=".", files=["worker.py"])
```

Paths are relative to the launcher. With `files=None`, Git selects tracked and
nonignored untracked files; their current bytes are captured, including dirty
changes. An explicit allowlist is safer when a checkout contains datasets,
credentials or experiment output. Packaging never commits or resets your work.

`experiment.freeze(source)` returns a retained `FrozenSource`. It can be packaged
again for another target without consulting the original checkout. Retrieved
experiments expose retained sources through `experiment.sources()`.

Wrap that same source in an existing container:

```python
package = xc.SingularityContainer(source, image_path="/shared/runtime.sif")
executor = xc.Slurm(
    cluster="mycluster", resources={"gpus": 1}, walltime=300,
    container_options=xc.SingularityOptions(bind={"/data": "/data"}),
)
```

Or use Shifter on a cluster where the image is already available:

```python
package = xc.ShifterContainer(source, image="docker:example/runtime:tag")
executor = xc.Slurm(
    cluster="mycluster", resources={"gpus-per-node": 4}, walltime=300,
    container_options=xc.ShifterOptions(modules=["gpu"]),
)
```

Pass `package` to `xm.Packageable(package, executor.Spec())`. There is one
`container_options` argument; its type must match the executable's runtime.
Runtime binds and modules are explicit. Shifter is a Slurm-only execution path.
Use an immutable image identifier when reproducibility requires it.

`PythonPackage`, `PexBinary`, and `UniversalPackage` provide inherited packaging
alternatives. The inherited `PythonContainer` builds a dependency image with
Docker from a base image and requirements file, converts it to SIF, and packages
application code separately. It requires local Docker and Singularity tooling;
use `SingularityContainer` or `ShifterContainer` for an already-prepared runtime.
Exex does not import Shifter images. Singularity/Apptainer must be available as
`singularity` for the current SIF execution path.
