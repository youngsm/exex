"""Post-success output capture. Embedded in job scripts; standard library only."""

import hashlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(info):
    if not (info.isfile() or info.isdir()):
        raise ValueError(
            "Outputs support regular files and directories, not links or special files"
        )
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644 | (info.mode & 0o111)
    info.pax_headers = {}
    return info


def capture(root, destination, outputs):
    root, destination = Path(root).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish the entire task's result set only after every required output exists.
    with tempfile.TemporaryDirectory(
        prefix=".capture-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        manifest = {}
        for name, relative in outputs.items():
            source = root / relative
            source.resolve(strict=True).relative_to(root)
            archive_path = staging / "output.tar"
            with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
                archive.add(source, arcname="data", filter=_member)
            identity = digest_file(archive_path)
            archive_path.replace(staging / (identity + ".tar"))
            manifest[name] = identity
        (staging / "manifest.json").write_text(json.dumps(manifest))
        # A previously published, nonempty task directory cannot be overwritten.
        os.rename(staging, destination)


if __name__ == "__main__":
    capture(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
