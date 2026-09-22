"""Read retained inputs, mutate the private copy, and emit a result; stdlib only."""

import json
import os
import socket
from pathlib import Path

inputs = Path(os.environ["EXEX_INPUT_DIR"])
weights = inputs / "checkpoint/weights.txt"
original = weights.read_text()
metrics = json.loads((inputs / "metrics").read_text())
assert original == "example checkpoint\n"
assert metrics["score"] == 0.75
weights.write_text("changed only in the consumer's private copy\n")
result = {
    "checkpoint": original,
    "score": metrics["score"],
    "producer_host": metrics["hostname"],
    "consumer_host": socket.gethostname(),
    "working_directory": str(Path.cwd()),
    "input_directory": str(inputs),
}
(Path(os.environ["EXEX_OUTPUT_DIR"]) / "result.json").write_text(json.dumps(result))
print(json.dumps(result), flush=True)
