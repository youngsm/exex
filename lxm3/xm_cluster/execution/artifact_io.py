"""Input preparation and output capture; embedded, standard-library-only helpers."""

import hashlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from pathlib import PurePosixPath


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root, inputs):
    root = Path(root)
    for name, binding in inputs.items():
        archive_path = binding["archive_path"]
        if digest_file(archive_path) != binding["id"]:
            raise ValueError(
                "Input archive does not match its content identity: " + name
            )
        with tempfile.TemporaryDirectory(prefix=".input-", dir=root) as temporary:
            with tarfile.open(archive_path) as archive:
                members = archive.getmembers()
                # Older HPC host Pythons lack extraction filters. Accept only our
                # file/directory format beneath data/, into a fresh private tree.
                for member in members:
                    path = PurePosixPath(member.name)
                    if (
                        path.parts[:1] != ("data",)
                        or ".." in path.parts
                        or not (member.isfile() or member.isdir())
                    ):
                        raise ValueError("Invalid input archive member: " + member.name)
                archive.extractall(temporary, members=members)
            (Path(temporary) / "data").rename(root / name)


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
    if sys.argv[1] == "prepare":
        prepare(sys.argv[2], json.loads(sys.argv[3]))
    else:
        capture(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
