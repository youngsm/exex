"""Bounded Slurm continuation using an existing application runtime."""

from pathlib import Path

from absl import app
from absl import flags

from exex import xm
from exex import xm_cluster as xc

TARGET = flags.DEFINE_string("target", None, "Configured Slurm cluster", required=True)
MODE = flags.DEFINE_enum("mode", "sbatch", ["sbatch", "salloc"], "Allocation mechanism")
RESOURCE = flags.DEFINE_multi_string("resource", [], "Native Slurm key=value")
WORKDIR = flags.DEFINE_string(
    "workdir_root", None, "Shared execution-site unpacking parent"
)
RUNTIME = flags.DEFINE_enum(
    "runtime", "host", ["host", "singularity", "shifter"], "Existing runtime"
)
IMAGE = flags.DEFINE_string("image", None, "Existing SIF path or Shifter image")
WALLTIME = flags.DEFINE_integer("walltime", 120, "Allocation walltime in seconds")
PAUSE = flags.DEFINE_integer("pause_before", 30, "Checkpoint margin in seconds")
ATTEMPTS = flags.DEFINE_integer("max_attempts", 3, "Total attempt budget")
STEPS = flags.DEFINE_integer("steps", 150, "Counter target")


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
    executor = xc.Slurm(
        cluster=TARGET.value,
        mode=MODE.value,
        walltime=WALLTIME.value,
        resources=dict(item.split("=", 1) for item in RESOURCE.value),
        workdir_root=WORKDIR.value,
        container_options=options,
    )
    with xc.create_experiment("cooperative-counter") as experiment:
        [executable] = experiment.package([xm.Packageable(package, executor.Spec())])
        experiment.add(
            xm.Job(executable, executor, args={"steps": STEPS.value}),
            outputs={"checkpoint": "checkpoint", "metrics": "metrics.json"},
            continuation=xc.Continuation(
                checkpoint="checkpoint",
                max_attempts=ATTEMPTS.value,
                pause_before=PAUSE.value,
            ),
        )
    print(f"Experiment: {experiment.experiment_id}")


if __name__ == "__main__":
    app.run(main)
