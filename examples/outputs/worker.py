"""Standard-library workload: no LXM3 or ML framework installation required."""

import argparse
import json
import os
import socket
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--missing", action="store_true")
args = parser.parse_args()
output = Path(os.environ["LXM_OUTPUT_DIR"])
metrics = {
    "hostname": socket.gethostname(),
    "working_directory": str(Path.cwd()),
    "output_directory": str(output),
    "score": 0.75,
}
(output / "metrics.json").write_text(json.dumps(metrics))
if not args.missing:
    (output / "checkpoint").mkdir()
    (output / "checkpoint/weights.txt").write_text("example checkpoint\n")
print(json.dumps(metrics), flush=True)
