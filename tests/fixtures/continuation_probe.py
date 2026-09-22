"""Live allocation probe; import the shipped, unmodified stdlib worker helper."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import execution

source = Path(os.environ["EXEX_INPUT_DIR"]) / "checkpoint"
checkpoint = (
    json.loads(source.read_text()) if source.exists() else {"step": 0, "history": []}
)
restored = checkpoint["step"]
checkpoint["step"] += 1
checkpoint["history"].append(
    {
        "native_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(),
        "restored": restored,
        "gpus": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).splitlines(),
    }
)
print(json.dumps(checkpoint), flush=True)
execution.link("probe", "https://example.org/continuation/" + str(checkpoint["step"]))
if os.environ.get("PROBE_WAIT"):
    time.sleep(180)  # Cancellation probe; native allocation remains capped at 120s.
if checkpoint["step"] < 2:
    deadline = time.monotonic() + 150
    while not execution.pause_requested():
        if time.monotonic() >= deadline:
            raise TimeoutError("No cooperative pause arrived")
        time.sleep(0.1)
output = Path(os.environ["EXEX_OUTPUT_DIR"])
(output / "checkpoint").write_text(json.dumps(checkpoint))
if checkpoint["step"] < 2:
    execution.mark_paused()
else:
    (output / "metrics.json").write_text(json.dumps({"completed": True, **checkpoint}))
