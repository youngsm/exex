"""System OpenSSH for Slurm commands and the existing staging-store operations."""

import shlex
import subprocess

import fsspec


def run(argv, *, hostname=None, username=None, **options):
    if hostname is not None:
        user = ["-l", username] if username else []
        argv = [
            "ssh",
            "-T",
            "-x",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            *user,
            "--",
            hostname,
            shlex.join(argv),
        ]
    return subprocess.run(
        argv, **{"check": True, "stdout": subprocess.PIPE, "text": True, **options}
    )


class OpenSSHFileSystem(fsspec.AbstractFileSystem):
    """Only the file operations used by ArtifactStore; no remote Python required."""

    cachable = False

    def __init__(self, hostname, username=None):
        super().__init__()
        self.hostname = hostname
        self.username = username

    def _run(self, argv, **options):
        return run(argv, hostname=self.hostname, username=self.username, **options)

    def abspath(self, path):
        script = 'case "$1" in "~") set -- "$HOME";; "~/"*) set -- "$HOME/${1#??}";; esac; realpath -m -- "$1"'
        return self._run(["sh", "-c", script, "sh", path]).stdout.strip()

    def makedirs(self, path, exist_ok=False):
        self._run(["mkdir", *(["-p"] if exist_ok else []), "--", path])

    def exists(self, path, **kwargs):
        result = self._run(["test", "-e", path], check=False)
        if result.returncode == 1:
            return False
        result.check_returncode()
        return True

    def info(self, path, **kwargs):
        size, mtime, kind = self._run(
            ["stat", "--printf=%s %Y %F", "--", path]
        ).stdout.split(maxsplit=2)
        return {
            "name": path,
            "size": int(size),
            "mtime": int(mtime),
            "type": "directory" if kind == "directory" else "file",
        }

    def put_file(self, lpath, rpath, **kwargs):
        with open(lpath, "rb") as source:
            self._run(["sh", "-c", 'cat > "$1"', "sh", rpath], stdin=source)

    def pipe_file(self, path, value, **kwargs):
        self._run(["sh", "-c", 'cat > "$1"', "sh", path], input=value, text=False)

    def mv(self, path1, path2, **kwargs):
        self._run(["mv", "--", path1, path2])
