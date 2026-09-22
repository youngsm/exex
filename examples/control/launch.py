"""Submit a bounded shell probe, then exit; reopen it with control.py."""

from pathlib import Path

from absl import app
from absl import flags

from exex import xm
from exex import xm_cluster as xc

TARGET = flags.DEFINE_string("target", "local", "local or a TOML cluster name")
DURATION = flags.DEFINE_integer("duration", 30, "Probe sleep in seconds", lower_bound=0)
EXIT_CODE = flags.DEFINE_integer(
    "exit_code", 0, "Probe exit code", lower_bound=0, upper_bound=255
)
TASKS = flags.DEFINE_integer("tasks", 1, "Number of array tasks", lower_bound=1)
RESOURCE = flags.DEFINE_multi_string(
    "resource", [], "Slurm key=value; repeat as needed"
)


def main(_):
    executor = (
        xc.Local()
        if TARGET.value == "local"
        else xc.Slurm(
            cluster=TARGET.value,
            walltime=DURATION.value + 60,
            resources=dict(item.split("=", 1) for item in RESOURCE.value),
        )
    )
    source = xc.SourceTree(
        xc.CommandList(
            [
                "echo control-probe-started",
                f"sleep {DURATION.value}",
                f"exit {EXIT_CODE.value}",
            ]
        ),
        Path(__file__).parent,
        files=[],
    )
    with xc.create_experiment("control-probe", project="qualification") as experiment:
        [executable] = experiment.package([xm.Packageable(source, executor.Spec())])
        job = (
            xm.Job(executable, executor)
            if TASKS.value == 1
            else xc.ArrayJob(
                executable, executor, args=[[] for _ in range(TASKS.value)]
            )
        )
        experiment.add(job)
    print(f"Experiment: {experiment.experiment_id}")


if __name__ == "__main__":
    app.run(main)
