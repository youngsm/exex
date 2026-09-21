"""Optional PyTorch workload; the launcher and LXM3 do not depend on PyTorch."""

import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import torch

REVISION = "base"

mask = os.environ["CUDA_VISIBLE_DEVICES"]
assert mask and len(mask.split(",")) == 1, mask
assert Path("/.singularity.d").is_dir(), "Must run inside a SIF"

assert torch.cuda.device_count() == 1, torch.cuda.device_count()
value = float(Path("/probe-input/value.txt").read_text())
matrix = torch.full((64, 64), value, device="cuda")
result = matrix @ torch.eye(64, device="cuda")
torch.cuda.synchronize()
assert torch.equal(result, matrix)
record = {
    "revision": REVISION,
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "host": socket.gethostname(),
    "job_id": os.environ["SLURM_JOB_ID"],
    "workdir": str(Path.cwd()),
    "image": os.environ.get(
        "APPTAINER_CONTAINER", os.environ.get("SINGULARITY_CONTAINER")
    ),
    "cuda_visible_devices": mask,
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "torch_path": torch.__file__,
    "input": value,
    "checksum": result.sum().item(),
    "argument": sys.argv[1],
    "environment": os.environ["PROBE_MESSAGE"],
    "exit_code": int(sys.argv[2]),
}
assert record["argument"] == record["environment"]
with open("/probe-output/result.json", "x") as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record), flush=True)
sys.exit(record["exit_code"])
