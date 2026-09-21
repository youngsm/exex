import datetime
import os
import shutil
import tempfile
import uuid
from typing import Optional, Tuple

import attr
import fsspec


@attr.s(auto_attribs=True)
class FileInfo:
    path: str
    size: int
    time_modified: datetime.datetime


class ArtifactStore:
    def __init__(
        self,
        filesystem: fsspec.AbstractFileSystem,
        staging_directory: str,
        project: Optional[str] = None,
    ):
        if project:
            self._storage_root = os.path.join(staging_directory, "projects", project)
        else:
            self._storage_root = staging_directory
        self._filesystem = filesystem
        self._initialize()

    @property
    def filesystem(self):
        return self._filesystem

    @property
    def storage_root(self):
        return self._storage_root

    def _initialize(self):
        self.ensure_dir(".")
        if not self.exists(".gitignore"):
            self.put_text("*\n", ".gitignore")

    def get_file_info(self, path: str) -> FileInfo:
        path = self.normalize_path(path)
        info = self._filesystem.info(path)
        size = info["size"]
        mod_time = info["mtime"]
        if not isinstance(mod_time, datetime.datetime):
            mod_time = datetime.datetime.fromtimestamp(
                info["mtime"], tz=datetime.timezone.utc
            )
        return FileInfo(path, size, mod_time)

    def normalize_path(self, path: str) -> str:
        assert not os.path.isabs(path)
        return os.path.join(self._storage_root, path)

    def exists(self, name: str) -> bool:
        return self._filesystem.exists(self.normalize_path(name))

    def ensure_dir(self, directory: str):
        directory = self.normalize_path(directory)
        return self._filesystem.makedirs(directory, exist_ok=True)

    def put_file(self, lpath: str, rpath: str) -> str:
        path = self.normalize_path(rpath)

        self._filesystem.makedirs(os.path.dirname(path), exist_ok=True)
        temporary_path = f"{path}.{uuid.uuid4().hex}.tmp"
        self._filesystem.put_file(lpath, temporary_path)
        self._filesystem.mv(temporary_path, path)

        return path

    def put_text(self, text: str, rpath: str) -> str:
        path = self.normalize_path(rpath)

        self._filesystem.makedirs(os.path.dirname(path), exist_ok=True)
        self._filesystem.pipe_file(path, text.encode())

        return path

    def put_fileobj(self, fileobj, rpath: str) -> str:
        with tempfile.NamedTemporaryFile() as source:
            shutil.copyfileobj(fileobj, source)
            source.flush()
            return self.put_file(source.name, rpath)

    def should_update(self, lpath: str, rpath: str) -> Tuple[bool, str]:
        if not self.exists(rpath):
            return True, "new file"

        local_stat = os.stat(lpath)
        local_size = local_stat.st_size
        local_mtime = datetime.datetime.fromtimestamp(
            local_stat.st_mtime, tz=datetime.timezone.utc
        )
        remote_file_info = self.get_file_info(rpath)

        if local_size != remote_file_info.size:
            return True, "file size mismatch"

        if local_mtime > remote_file_info.time_modified:
            return True, "local file is newer"

        return False, ""
