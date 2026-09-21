"""Small GPU calculation; CuPy is supplied by the prebuilt dependency image."""

import errno
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import cupy as cp

COUNT = 1024

input_path = Path(sys.argv[1])
output = Path(sys.argv[2])
mask = os.environ["CUDA_VISIBLE_DEVICES"]
image = os.environ["SHIFTER_IMAGE"]
assert mask == (output / "task-mask.txt").read_text().strip()
assert image == os.environ["PROBE_IMAGE_ID"]
assert cp.cuda.runtime.getDeviceCount() == 1

try:
    with (input_path.parent / "write-probe").open("x"):
        pass
except OSError as error:
    assert error.errno == errno.EROFS
else:
    raise RuntimeError("Input mount was writable")

values = cp.arange(COUNT, dtype=cp.float64)
observed = float(cp.sum(values * values).get())
expected = COUNT * (COUNT - 1) * (2 * COUNT - 1) // 6
assert observed == expected
record = {
    "image_id": image,
    "host": socket.gethostname(),
    "job_id": os.environ["SLURM_JOB_ID"],
    "step_id": os.environ["SLURM_STEP_ID"],
    "workdir": os.getcwd(),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "message": os.environ["PROBE_MESSAGE"],
    "argument": sys.argv[4],
    "input": input_path.read_text(),
    "input_read_only": True,
    "cuda_visible_devices": mask,
    "device_count": cp.cuda.runtime.getDeviceCount(),
    "device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    "cupy_version": cp.__version__,
    "count": COUNT,
    "sum_of_squares": observed,
    "intentional_failure": sys.argv[3] == "1",
}
with (output / "result.json").open("x") as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record), flush=True)
sys.exit(7 if record["intentional_failure"] else 0)
