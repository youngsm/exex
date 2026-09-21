"""Standard-library probe; the application, not LXM3, owns its output format."""

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--message", required=True)
parser.add_argument("--check-gpu", action="store_true")
args = parser.parse_args()

result = {
    "host": socket.gethostname(),
    "python": sys.version,
    "message": args.message,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
if args.check_gpu:
    result["gpu_info"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        encoding="utf-8",
    )

args.output_dir.mkdir(parents=True, exist_ok=True)
with (args.output_dir / "result.json").open("x") as output:
    json.dump(result, output, indent=2)
print(json.dumps(result, indent=2))
