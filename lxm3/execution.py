"""Optional, standard-library-only helpers for the running application."""

import json
import os
import uuid
from pathlib import Path


def pause_requested() -> bool:
    """Check for a cooperative pause at an application-defined safe point.

    False outside a continuation-enabled job. Distributed applications must
    agree on this decision before entering collective checkpoint operations.
    """
    path = os.environ.get("LXM_PAUSE_REQUEST")
    return bool(path and Path(path).exists())


def mark_paused() -> None:
    """Report a completed checkpoint, then let the application exit cleanly.

    Call once from the reporting rank, after all writers finish. This neither
    exits nor requeues; LXM3 must still observe success and retain the checkpoint.
    """
    Path(os.environ["LXM_PAUSE_READY"]).touch()


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
