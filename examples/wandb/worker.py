"""Tracking only: a real trainer saves this state alongside its model/optimizer."""

import json
import os
import shutil
from pathlib import Path

from exex.contrib import wandb as tracking

output_dir = Path(os.environ["EXEX_OUTPUT_DIR"])
with tracking.init(config={"example": True}) as run:
    run.define_metric("train/global_step")
    run.define_metric("train/*", step_metric="train/global_step")
    for step in range(3):
        run.log({"train/global_step": step, "train/loss": 1 / (step + 1)})
    output = output_dir / "tracking-state.json"
    output.write_text(json.dumps(tracking.checkpoint_state(run)))

# This scalar-only example retains the completed SDK history, not its symlinks.
shutil.copyfile(run.settings.sync_file, output_dir / "history.wandb")
