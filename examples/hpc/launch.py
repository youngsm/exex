"""A bounded launch probe using only the existing LXM3 public API."""

from pathlib import Path

from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster as xc

TARGET = flags.DEFINE_string(
    "target", "local", "local or an explicit TOML cluster name"
)
OUTPUT = flags.DEFINE_string(
    "output_dir",
    None,
    "Absolute durable directory on the execution site",
    required=True,
)
RESOURCE = flags.DEFINE_multi_string(
    "resource", [], "Slurm key=value; repeat for each resource"
)
PYTHON = flags.DEFINE_string(
    "python", "python3", "Python executable on the execution site"
)
GPU = flags.DEFINE_bool(
    "check_gpu", False, "Require nvidia-smi to succeed inside the job"
)
WORKDIR = flags.DEFINE_string(
    "workdir_root", None, "Execution-host parent for temporary unpacked source"
)


def main(_):
    if not Path(OUTPUT.value).is_absolute():
        raise ValueError(
            "output_dir must be absolute: the job's temporary directory is deleted on exit"
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
    package_spec = xc.UniversalPackage(
        path=str(Path(__file__).parent),
        build_script="build.sh",
        entrypoint=[PYTHON.value, "workload.py"],
    )
    with xc.create_experiment(
        "minimal-hpc-probe", project="qualification"
    ) as experiment:
        [executable] = experiment.package(
            [xm.Packageable(package_spec, executor.Spec())]
        )
        experiment.add(
            xm.Job(
                executable,
                executor,
                args=[
                    f"--output-dir={OUTPUT.value}",
                    "--message=literal $value ' with spaces\nand a newline",
                    *(["--check-gpu"] if GPU.value else []),
                ],
            )
        )
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Packaged code: {executable.resource_uri}")
    print(f"Application output directory: {OUTPUT.value}")


if __name__ == "__main__":
    app.run(main)
