"""Run separately packaged source in an existing, locally readable GPU SIF."""

from pathlib import Path

from absl import app
from absl import flags

from exex import xm
from exex import xm_cluster as xc

CLUSTER = flags.DEFINE_string("cluster", None, "TOML cluster name", required=True)
IMAGE = flags.DEFINE_string("image", None, "Author-local SIF", required=True)
INPUT = flags.DEFINE_string("input_dir", None, "Shared input directory", required=True)
OUTPUT = flags.DEFINE_string(
    "output_dir", None, "New shared output directory", required=True
)
WORKDIR = flags.DEFINE_string(
    "workdir_root", None, "Shared unpacking parent", required=True
)
SOURCE = flags.DEFINE_string(
    "source_dir", str(Path(__file__).parent), "Packaged source"
)
RESOURCE = flags.DEFINE_multi_string("resource", [], "Slurm key=value")
FAIL = flags.DEFINE_bool("fail", False, "Exit 7 after recording GPU results")
MESSAGE = "literal $value ' with spaces\nand a newline"


def main(_):
    output = Path(OUTPUT.value).resolve()
    output.mkdir(parents=True)
    executor = xc.Slurm(
        cluster=CLUSTER.value,
        resources={
            **dict(item.split("=", 1) for item in RESOURCE.value),
            "nodes": 1,
            "gpus-per-node": 1,
            "cpus-per-task": 2,
            "mem": "8G",
        },
        walltime=180,
        workdir_root=WORKDIR.value,
        container_options=xc.SingularityOptions(
            bind={
                str(Path(INPUT.value).resolve()): "/probe-input",
                str(output): "/probe-output",
            },
            extra_options=["--cleanenv"],
        ),
    )
    package = xc.UniversalPackage(
        path=str(Path(SOURCE.value).resolve()),
        build_script="build.sh",
        entrypoint=["python3", "probe.py"],
    )
    with xc.create_experiment("sif-probe", project="qualification") as experiment:
        [executable] = experiment.package(
            [
                xm.Packageable(
                    xc.SingularityContainer(package, image_path=IMAGE.value),
                    executor.Spec(),
                )
            ]
        )
        experiment.add(
            xm.Job(
                executable,
                executor,
                args=[MESSAGE, "7" if FAIL.value else "0"],
                env_vars={"PROBE_MESSAGE": MESSAGE},
            )
        )
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Packaged code: {executable.resource_uri}")
    print(f"Staged image: {executable.container_image.name}")
    print(f"Output: {output}")


if __name__ == "__main__":
    app.run(main)
