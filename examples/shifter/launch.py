"""Qualify native Shifter composition without adding a runtime abstraction."""

from pathlib import Path

from absl import app
from absl import flags

from exex import xm
from exex import xm_cluster as xc

CLUSTER = flags.DEFINE_string("cluster", None, "TOML cluster name", required=True)
IMAGE = flags.DEFINE_string(
    "image_id", None, "Installed Shifter image ID", required=True
)
INPUT = flags.DEFINE_string(
    "input_dir", None, "Site directory with input.txt", required=True
)
OUTPUT = flags.DEFINE_string(
    "output_dir", None, "New site output directory", required=True
)
WORKDIR = flags.DEFINE_string(
    "workdir_root", None, "Shared unpacking parent", required=True
)
RESOURCE = flags.DEFINE_multi_string("resource", [], "Slurm key=value")
FAIL = flags.DEFINE_bool("fail_worker", False, "Exit 7 after writing evidence")


def main(_):
    executor = xc.Slurm(
        cluster=CLUSTER.value,
        resources={
            **dict(item.split("=", 1) for item in RESOURCE.value),
            "nodes": 1,
            "ntasks": 1,
            "cpus-per-task": 1,
            "gpus-per-node": 1,
            "image": "id:" + IMAGE.value,
            "module": "gpu",
        },
        walltime=120,
        workdir_root=WORKDIR.value,
    )
    package = xc.UniversalPackage(
        path=str(Path(__file__).parent),
        build_script="build.sh",
        entrypoint=["bash", "driver.sh"],
    )
    with xc.create_experiment("shifter-probe", project="qualification") as experiment:
        [executable] = experiment.package([xm.Packageable(package, executor.Spec())])
        experiment.add(
            xm.Job(
                executable,
                executor,
                args=[
                    INPUT.value,
                    OUTPUT.value,
                    "1" if FAIL.value else "0",
                    "literal argument $value ' with spaces\nand a newline",
                ],
                env_vars={
                    "PROBE_MESSAGE": "literal env $value ' with spaces\nand a newline",
                    "PROBE_IMAGE_ID": IMAGE.value,
                },
            )
        )
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Packaged code: {executable.resource_uri}")
    print(f"Shifter image: id:{IMAGE.value}")
    print(f"Output: {OUTPUT.value}")


if __name__ == "__main__":
    app.run(main)
