"""Consume the metrics file and checkpoint directory from the outputs example."""

from pathlib import Path

from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster as xc

PRODUCER = flags.DEFINE_integer(
    "producer", None, "Completed producer experiment ID", required=True
)
PRODUCER_CONFIG = flags.DEFINE_string(
    "producer_config",
    None,
    "Producer catalog config; defaults to this launcher's config",
)
TASK = flags.DEFINE_integer(
    "producer_task", None, "Zero-based producer array task, if needed"
)
TARGET = flags.DEFINE_string("target", "local", "local or an explicit TOML cluster")
RUNTIME = flags.DEFINE_enum(
    "runtime", "host", ["host", "singularity", "shifter"], "Runtime for this example"
)
IMAGE = flags.DEFINE_string(
    "image", None, "Existing SIF path or installed Shifter image"
)
RESOURCE = flags.DEFINE_multi_string("resource", [], "Slurm key=value")
WORKDIR = flags.DEFINE_string("workdir_root", None, "Execution-site unpacking parent")


def main(_):
    producer = xc.get_experiment(
        PRODUCER.value,
        config=xc.Config.from_file(PRODUCER_CONFIG.value)
        if PRODUCER_CONFIG.value
        else None,
    )
    artifacts = producer.work_units()[1].artifacts(task=TASK.value)
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
        "artifact-consumer", project="qualification"
    ) as experiment:
        [executable] = experiment.package([xm.Packageable(package, executor.Spec())])
        experiment.add(
            xm.Job(executable, executor),
            inputs={
                "checkpoint": artifacts["checkpoint"],
                "metrics": artifacts["metrics"],
            },
            outputs={"result": "result.json"},
        )
    print(f"Experiment: {experiment.experiment_id}")


if __name__ == "__main__":
    app.run(main)
