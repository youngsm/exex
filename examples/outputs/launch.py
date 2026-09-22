"""Retain a file and directory, independently of the chosen executor/runtime."""

from pathlib import Path

from absl import app
from absl import flags

from exex import xm
from exex import xm_cluster as xc

TARGET = flags.DEFINE_string("target", "local", "local or an explicit TOML cluster")
RUNTIME = flags.DEFINE_enum(
    "runtime", "host", ["host", "singularity", "shifter"], "Runtime for this example"
)
IMAGE = flags.DEFINE_string(
    "image", None, "Existing SIF path or installed Shifter image"
)
RESOURCE = flags.DEFINE_multi_string("resource", [], "Slurm key=value")
WORKDIR = flags.DEFINE_string("workdir_root", None, "Execution-site unpacking parent")
MISSING = flags.DEFINE_bool(
    "missing", False, "Omit the required checkpoint to test capture failure"
)


def main(_):
    source = xc.SourceTree(
        xc.ModuleName("worker"), Path(__file__).parent, files=["worker.py"]
    )
    package, options = source, None
    if RUNTIME.value == "singularity":
        package = xc.SingularityContainer(source, image_path=IMAGE.value)
        options = xc.SingularityOptions(extra_options=["--cleanenv"])
    elif RUNTIME.value == "shifter":
        package = xc.ShifterContainer(source, image=IMAGE.value)
        options = xc.ShifterOptions(modules=["gpu"], extra_options=["--clearenv"])
    executor = (
        xc.Local(workdir_root=WORKDIR.value, container_options=options)
        if TARGET.value == "local"
        else xc.Slurm(
            cluster=TARGET.value,
            walltime=120,
            resources=dict(item.split("=", 1) for item in RESOURCE.value),
            workdir_root=WORKDIR.value,
            container_options=options,
        )
    )
    with xc.create_experiment(
        "retained-outputs", project="qualification"
    ) as experiment:
        [executable] = experiment.package([xm.Packageable(package, executor.Spec())])
        experiment.add(
            xm.Job(executable, executor, args=["--missing"] if MISSING.value else []),
            outputs={"metrics": "metrics.json", "checkpoint": "checkpoint"},
        )
    print(f"Experiment: {experiment.experiment_id}")


if __name__ == "__main__":
    app.run(main)
