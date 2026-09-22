"""Launch one entrypoint through ShifterContainer; no implicit Slurm step."""

from pathlib import Path

from absl import app
from absl import flags

from exex import xm
from exex import xm_cluster as xc

CLUSTER = flags.DEFINE_string("cluster", "nersc", "Explicit TOML cluster name")
IMAGE = flags.DEFINE_string("image", None, "Installed Shifter reference", required=True)
INPUT = flags.DEFINE_string(
    "input_dir", None, "Site directory with input.txt", required=True
)
OUTPUT = flags.DEFINE_string(
    "output_dir", None, "Existing, empty site directory", required=True
)
WORKDIR = flags.DEFINE_string(
    "workdir_root", None, "Parent visible inside Shifter", required=True
)
RESOURCE = flags.DEFINE_multi_string("resource", [], "Slurm key=value")
FAIL = flags.DEFINE_bool("fail", False, "Exit 7 after recording the GPU result")
MESSAGE = "literal $value ' with spaces\nand a newline"


def main(_):
    executor = xc.Slurm(
        cluster=CLUSTER.value,
        resources={
            **dict(item.split("=", 1) for item in RESOURCE.value),
            "nodes": 1,
            "ntasks": 1,
            "gpus-per-node": 4,
        },
        walltime=120,
        workdir_root=WORKDIR.value,
        container_options=xc.ShifterOptions(
            modules=["gpu"],
            bind={INPUT.value: "/mnt:ro", OUTPUT.value: "/media"},
            extra_options=["--clearenv"],
        ),
    )
    package = xc.ShifterContainer(
        entrypoint=xc.UniversalPackage(
            path=str(Path(__file__).parent),
            build_script="build.sh",
            entrypoint=["python3", "container_probe.py"],
        ),
        image=IMAGE.value,
    )
    with xc.create_experiment(
        "shifter-container-probe", project="qualification"
    ) as experiment:
        [executable] = experiment.package([xm.Packageable(package, executor.Spec())])
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
    print(f"Output: {OUTPUT.value}")


if __name__ == "__main__":
    app.run(main)
