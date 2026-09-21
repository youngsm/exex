"""Capture raw source into a deterministic, retained archive; no catalog or build."""

import hashlib
import os
import shlex
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

import fsspec

from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster import executable_specs as specs
from lxm3.xm_cluster.packaging.create_archive import _create_entrypoint_cmds


def _selected_files(source: specs.SourceTree):
    root = Path(source.path).resolve()
    names = source.files
    if names is None:
        listing = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                ".",
            ]
        )
        names = [os.fsdecode(name) for name in listing.split(b"\0") if name]
    for name in sorted({Path(name) for name in names}):
        path = root / name
        if (
            name.is_absolute()
            or ".." in name.parts
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError(f"Source file must stay within {root}: {name}")
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            if source.files is not None:
                raise
            continue  # Tracked file deleted from the working tree.
        if not stat.S_ISREG(mode):
            raise ValueError(
                f"SourceTree supports regular files, not symlinks or directories: {name}"
            )
        if name.parts[0] in {"job-param.sh", ".environment"}:
            raise ValueError(f"Source path is reserved for LXM3 runtime files: {name}")
        yield name.as_posix(), path


def _identity(archive_path: str, entrypoint: str) -> str:
    content_hash = hashlib.sha256(entrypoint.encode() + b"\0")
    with open(archive_path, "rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            content_hash.update(block)
    return content_hash.hexdigest()


def freeze(source: specs.SourceTree, local_storage_root: str) -> specs.FrozenSource:
    # Reuse local storage, including its ignore file and atomic file exposure.
    store = artifacts.ArtifactStore(fsspec.filesystem("file"), local_storage_root)
    entrypoint = "bash -e -c " + shlex.quote(_create_entrypoint_cmds(source)) + " lxm3"
    with tempfile.TemporaryDirectory(prefix="lxm3-source-") as temporary:
        archive_path = os.path.join(temporary, "source.tar")
        with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
            for name, path in _selected_files(source):
                with path.open("rb") as stream:
                    metadata = os.fstat(stream.fileno())
                    member = tarfile.TarInfo(name)
                    member.size = metadata.st_size
                    member.mode = 0o644 | (metadata.st_mode & 0o111)
                    archive.addfile(member, stream)
        identity = _identity(archive_path, entrypoint)
        retained = store.put_file(archive_path, f"sources/{identity}.tar")
    return specs.FrozenSource(identity, source.name, retained, entrypoint)


def verify(source: specs.FrozenSource) -> None:
    if _identity(source._archive_path, source._entrypoint_command) != source.id:
        raise ValueError(
            f"Retained source does not match its content identity: {source.id}"
        )
