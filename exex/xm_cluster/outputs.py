"""Declared result paths and read-only handles to retained execution-site content."""

import json
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Optional

import fsspec

from exex.clusters import ssh
from exex.xm_cluster import inspection
from exex.xm_cluster.execution.artifact_io import digest_file


def declarations(outputs):
    result = {name: os.fspath(path) for name, path in (outputs or {}).items()}
    for path in result.values():
        relative = PurePosixPath(path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"Output must be a path beneath EXEX_OUTPUT_DIR: {path!r}")
    return result


@dataclass(frozen=True)
class Artifact:
    """Retained content; fetching copies it without changing the original."""

    id: str
    _archive_path: str
    _hostname: Optional[str] = None
    _username: Optional[str] = None

    def fetch(self, into: os.PathLike | str) -> Path:
        destination = Path(into).absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        filesystem = (
            ssh.OpenSSHFileSystem(self._hostname, self._username)
            if self._hostname
            else fsspec.filesystem("file")
        )
        with tempfile.TemporaryDirectory(
            prefix=".exex-fetch-", dir=destination.parent
        ) as temporary:
            archive_path = Path(temporary) / "artifact.tar"
            filesystem.get_file(self._archive_path, str(archive_path))
            if digest_file(archive_path) != self.id:
                raise ValueError(
                    f"Retained artifact does not match its content identity: {self.id}"
                )
            unpacked = Path(temporary) / "unpacked"
            with tarfile.open(archive_path) as archive:
                archive.extractall(unpacked, filter="data")
            data = unpacked / "data"
            if data.is_dir():
                shutil.copytree(data, destination)
            else:
                # Atomic, and fails if the destination appeared.
                os.link(data, destination)
        return destination


def artifacts(record, *, task=None):
    record = inspection.execution_record(record)
    if task is None and record["task_count"] > 1:
        raise ValueError("Choose a zero-based task index for array artifacts")
    task = 0 if task is None else task
    if not 0 <= task < record["task_count"]:
        raise ValueError("task must be in range")
    root = record.get("artifact_directory")
    if root is None:
        return {}
    directory = os.path.join(root, str(task))
    hostname, username = inspection._hostname(record), record["username"]
    receipt = ssh.run(
        [
            "sh",
            "-c",
            'if test -e "$1"; then cat -- "$1"; fi',
            "sh",
            os.path.join(directory, "manifest.json"),
        ],
        hostname=hostname,
        username=username,
    ).stdout
    result = {}
    for name, identity in json.loads(receipt or "{}").items():
        if (
            not isinstance(identity, str)
            or re.fullmatch(r"[0-9a-f]{64}", identity) is None
        ):
            raise ValueError("Invalid artifact content identity")
        result[name] = Artifact(
            identity, os.path.join(directory, identity + ".tar"), hostname, username
        )
    return result
