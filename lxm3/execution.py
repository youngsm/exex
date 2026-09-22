"""Optional, standard-library-only helpers for the running application."""

import json
import os
import uuid
from pathlib import Path


def link(name: str, url: str) -> None:
    """Record a named URL for this task; do nothing outside an LXM3 job.

    Call from one reporting process per task (e.g. rank zero). Repeated names
    replace their URL. Publication is independent of payload success and never
    connects to the author's database.
    """
    destination = os.environ.get("LXM_LINKS_FILE")
    if not destination:
        return
    path = Path(destination)
    links = json.loads(path.read_text()) if path.exists() else {}
    links[name] = url
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        # Normal umask permissions, not mkstemp's owner-only mode: a container
        # may write as a different UID from the process reading the receipt.
        with temporary.open("x") as file:
            json.dump(links, file)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
