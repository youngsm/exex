"""A non-ML checkpoint: one increment per second until the requested total."""

import argparse
import json
import os
import time
from pathlib import Path

from exex import execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    source = Path(os.environ["EXEX_INPUT_DIR"]) / "checkpoint"
    step = int(source.read_text()) if source.exists() else 0
    print(f"Restored step {step}", flush=True)
    while step < args.steps and not execution.pause_requested():
        time.sleep(1)
        step += 1
    output = Path(os.environ["EXEX_OUTPUT_DIR"])
    (output / "checkpoint").write_text(str(step))
    print(f"Saved step {step}", flush=True)
    if step < args.steps:
        execution.mark_paused()
    else:
        (output / "metrics.json").write_text(json.dumps({"steps": step}))


if __name__ == "__main__":
    main()
