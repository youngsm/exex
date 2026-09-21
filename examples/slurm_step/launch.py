"""Two-node qualification using a normal packaged entrypoint, not a step API."""

from pathlib import Path

from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster as xc

CLUSTER = flags.DEFINE_string("cluster", None, "TOML cluster name", required=True)
OUTPUT = flags.DEFINE_string(
    "output_dir", None, "New shared output path", required=True
)
WORKDIR = flags.DEFINE_string(
    "workdir_root", None, "Shared unpacking parent", required=True
)
PYTHON = flags.DEFINE_string("python", "python3", "Execution-host Python")
RESOURCE = flags.DEFINE_multi_string("resource", [], "Slurm key=value")
FAIL = flags.DEFINE_bool("fail_worker", False, "Intentionally fail rank 1")


def main(_):
    executor = xc.Slurm(
        cluster=CLUSTER.value,
        resources={
            **dict(item.split("=", 1) for item in RESOURCE.value),
            "nodes": 2,
            "ntasks-per-node": 1,
            "cpus-per-task": 1,
        },
        walltime=120,
        workdir_root=WORKDIR.value,
    )
    package = xc.UniversalPackage(
        path=str(Path(__file__).parent),
        build_script="build.sh",
        entrypoint=["bash", "driver.sh"],
    )
    with xc.create_experiment(
        "slurm-step-probe", project="qualification"
    ) as experiment:
        [executable] = experiment.package([xm.Packageable(package, executor.Spec())])
        experiment.add(
            xm.Job(
                executable,
                executor,
                args=[OUTPUT.value, PYTHON.value, "1" if FAIL.value else "-1"],
                env_vars={
                    "PROBE_MESSAGE": "literal $value ' with spaces\nand a newline"
                },
            )
        )
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Packaged code: {executable.resource_uri}")
    print(f"Output: {OUTPUT.value}")


if __name__ == "__main__":
    app.run(main)
