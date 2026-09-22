"""One standard-library worker, locally or on an explicit Slurm cluster."""

from pathlib import Path

from absl import flags

from exex import xm
from exex import xm_cluster as xc

CLUSTER = flags.DEFINE_string("cluster", None, "Configured Slurm cluster; omit for local")
RESOURCE = flags.DEFINE_multi_string("resource", [], "Slurm key=value")
WALLTIME = flags.DEFINE_integer("walltime", 300, "Slurm allocation seconds")


def main(_):
    executor = (
        xc.Slurm(
            cluster=CLUSTER.value,
            resources=dict(item.split("=", 1) for item in RESOURCE.value),
            walltime=WALLTIME.value,
        )
        if CLUSTER.value
        else xc.Local()
    )
    source = xc.SourceTree(
        xc.ModuleName("worker"), Path(__file__).parent, files=["worker.py"]
    )
    with xc.create_experiment("hello") as experiment:
        [executable] = experiment.package([xm.Packageable(source, executor.Spec())])
        experiment.add(
            xm.Job(executable, executor), outputs={"result": "result.txt"}
        )
        print(f"Experiment: {experiment.experiment_id}", flush=True)
