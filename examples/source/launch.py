"""Freeze a disposable source tree, remove it, then launch the retained copy."""

import shutil
import tempfile
from pathlib import Path

from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster as xc

TARGET = flags.DEFINE_string("target", "local", "local or a TOML cluster name")
OUTPUT = flags.DEFINE_string(
    "output_dir",
    None,
    "Absolute durable output directory at the execution site",
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
    with xc.create_experiment(
        "frozen-source-probe", project="qualification"
    ) as experiment:
        with tempfile.TemporaryDirectory(prefix="lxm3-disposable-source-") as checkout:
            shutil.copyfile(
                Path(__file__).parents[1] / "hpc/workload.py",
                Path(checkout) / "workload.py",
            )
            frozen = experiment.freeze(
                xc.SourceTree(
                    xc.ModuleName("workload"), checkout, files=["workload.py"]
                )
            )
        # Only the disposable copy above was removed, never the real repository.
        print(
            f"Source ID: {frozen.id}; original tree removed: {not Path(checkout).exists()}"
        )
        [executable] = experiment.package([xm.Packageable(frozen, executor.Spec())])
        experiment.add(
            xm.Job(
                executable,
                executor,
                args=[
                    f"--output-dir={OUTPUT.value}",
                    "--message=retained source: literal $value ' and spaces\nand a newline",
                ],
            )
        )
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Packaged code: {executable.resource_uri}")
    print(f"Application output: {OUTPUT.value}/result.json")


if __name__ == "__main__":
    app.run(main)
