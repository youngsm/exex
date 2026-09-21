"""Exercise typed Shifter launch, real GPU work and directory mounts."""

import errno
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

# Shifter images are read-only; keep CuPy's generated cache in the job directory.
os.environ["CUPY_CACHE_DIR"] = str(Path.cwd() / ".cupy")
import cupy as cp

try:
    with Path("/mnt/write-probe").open("x"):
        pass
except OSError as error:
    assert error.errno == errno.EROFS
else:
    raise RuntimeError("Input mount was writable")

values = cp.arange(1024, dtype=cp.float64)
observed = float(cp.sum(values * values).get())
assert observed == 357389824.0
assert os.environ["PROBE_MESSAGE"] == sys.argv[1]
record = {
    "image_id": os.environ["SHIFTER_IMAGE"],
    "job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "workdir": os.getcwd(),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "message": os.environ["PROBE_MESSAGE"],
    "argument": sys.argv[1],
    "input": Path("/mnt/input.txt").read_text(),
    "input_read_only": True,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "device_count": cp.cuda.runtime.getDeviceCount(),
    "device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    "sum_of_squares": observed,
    "exit_code": int(sys.argv[2]),
}
with Path("/media/result.json").open("x") as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record), flush=True)
sys.exit(record["exit_code"])
