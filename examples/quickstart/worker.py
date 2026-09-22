"""Write a retained result without installing any application dependencies."""

import os
from pathlib import Path

(Path(os.environ["EXEX_OUTPUT_DIR"]) / "result.txt").write_text("hello\n")
print("Finished", flush=True)
