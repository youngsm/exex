"""Add an independent run of a retained source probe; no original checkout needed."""

from pathlib import Path

from absl import app
from absl import flags

from exex import xm
from exex import xm_cluster as xc

EXPERIMENT = flags.DEFINE_integer(
    "experiment_id", None, "Existing experiment ID", required=True
)
SOURCE = flags.DEFINE_string("source_id", None, "Retained source ID", required=True)
TARGET = flags.DEFINE_string("target", "local", "local or a TOML cluster name")
OUTPUT = flags.DEFINE_string(
    "output_dir",
    None,
    "New absolute output directory at the execution site",
    required=True,
)
RESOURCE = flags.DEFINE_multi_string(
    "resource", [], "Slurm key=value; repeat as needed"
)
WORKDIR = flags.DEFINE_string("workdir_root", None, "Execution-site unpacking parent")


def main(_):
    if not Path(OUTPUT.value).is_absolute():
        raise ValueError(
            "output_dir must be absolute; temporary source is deleted on exit"
        )
    executor = (
        xc.Local(workdir_root=WORKDIR.value)
        if TARGET.value == "local"
        else xc.Slurm(
            cluster=TARGET.value,
            walltime=120,
            resources=dict(item.split("=", 1) for item in RESOURCE.value),
            workdir_root=WORKDIR.value,
        )
    )
    with xc.get_experiment(EXPERIMENT.value) as experiment:
        source = experiment.sources()[SOURCE.value]
        [executable] = experiment.package([xm.Packageable(source, executor.Spec())])
        experiment.add(
            xm.Job(
                executable,
                executor,
                args=[
                    f"--output-dir={OUTPUT.value}",
                    "--message=another execution of retained source: literal $value ' and spaces\nand a newline",
                ],
            )
        )
    print(f"Experiment: {experiment.experiment_id}; source: {source.id}")
    print(f"Packaged code: {executable.resource_uri}")
    print(f"Application output: {OUTPUT.value}/result.json")


if __name__ == "__main__":
    app.run(main)
