"""Run a small native-W&B worker locally; offline unless explicitly selected."""

import shlex
import sys
from pathlib import Path

from absl import app
from absl import flags

from exex import xm
from exex import xm_cluster as xc
from exex.contrib.wandb import configure_wandb

MODE = flags.DEFINE_enum("wandb_mode", "offline", ["offline", "online"], "W&B mode")
ENTITY = flags.DEFINE_string("wandb_entity", "example", "W&B team or username")
PROJECT = flags.DEFINE_string("wandb_project", "exex-example", "W&B project")


def main(_):
    track = configure_wandb(PROJECT.value, ENTITY.value, mode=MODE.value)
    source = xc.SourceTree(
        xc.CommandList([f"{shlex.quote(sys.executable)} worker.py"]),
        Path(__file__).parent,
        files=["worker.py"],
    )
    with xc.create_experiment("wandb-example") as experiment:
        [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
        experiment.add(
            track(xm.Job(executable, xc.Local())),
            outputs={
                "tracking_state": "tracking-state.json",
                "wandb_history": "history.wandb",
            },
        )
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Links: {experiment.work_units()[1].get_links()}")


if __name__ == "__main__":
    app.run(main)
