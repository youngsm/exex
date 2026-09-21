"""Observe native Slurm placement and shared files; optionally fail one worker."""

import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

output = Path(sys.argv[1])
fail_rank = int(sys.argv[2])
rank = int(os.environ["SLURM_PROCID"])
record = {
    "rank": rank,
    "local_rank": int(os.environ["SLURM_LOCALID"]),
    "world_size": int(os.environ["SLURM_NTASKS"]),
    "host": socket.gethostname(),
    "pid": os.getpid(),
    "job_id": os.environ["SLURM_JOB_ID"],
    "step_id": os.environ["SLURM_STEP_ID"],
    "workdir": os.getcwd(),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "driver_token": Path("driver-token.txt").read_text(),
    "message": os.environ["PROBE_MESSAGE"],
    "started_at": time.time(),
}
with (output / f"rank-{rank}.json").open("x") as stream:
    json.dump(record, stream, indent=2)
print(f"Rank {rank} started on {record['host']}", flush=True)

if fail_rank >= 0:
    if rank == fail_rank:
        # Fail only after the other worker has started, so peer termination is tested.
        deadline = time.monotonic() + 20
        while not (output / f"rank-{1 - rank}.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("The other worker did not start")
            time.sleep(0.05)
        print(f"Intentional rank {rank} failure (exit 7)", flush=True)
        raise SystemExit(7)
    time.sleep(30)  # A failed peer must terminate this before it can finish.

with (output / f"rank-{rank}.complete").open("x") as stream:
    stream.write(str(time.time()))
print(f"Rank {rank} completed", flush=True)
