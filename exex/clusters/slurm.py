import re
from typing import Optional

from exex.clusters import ssh


def parse_job_id(output: str) -> int:
    match = re.fullmatch(r"(?P<id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?", output.strip())
    if match is None:
        raise ValueError(f"Unable to parse job-id from:\n{output}")
    return int(match.group("id"))


class SlurmCluster:
    def __init__(
        self, hostname: Optional[str] = None, username: Optional[str] = None
    ) -> None:
        self._hostname = hostname
        self._username = username

    def launch(self, script_path) -> str:
        output = ssh.run(
            ["sbatch", "--parsable", "--", script_path],
            hostname=self._hostname,
            username=self._username,
        ).stdout.strip()
        parse_job_id(output)
        return output

    def cancel(self, job_id: str, job_name: str) -> None:
        """Cancel only the recorded ID/name pair, including all array elements."""
        numeric_id, _, cluster = job_id.partition(";")
        ssh.run(
            [
                "scancel",
                "--ctld",
                f"--name={job_name}",
                *([f"--clusters={cluster}"] if cluster else []),
                numeric_id,
            ],
            hostname=self._hostname,
            username=self._username,
        )

    def __repr__(self):
        if self._hostname is not None:
            return f'Client(hostname="{self._hostname}", user="{self._username}")'
        else:
            return "Client()"

    def accounting(self, job_id: str, job_name: str):
        """Return allocation rows only, excluding recycled IDs with another name."""
        numeric_id, _, cluster = job_id.partition(";")
        output = ssh.run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--array",
                f"--jobs={numeric_id}",
                f"--format=JobID%64,JobName%{len(job_name) + 1},State%32,ExitCode",
                *([f"--clusters={cluster}"] if cluster else []),
            ],
            hostname=self._hostname,
            username=self._username,
        ).stdout
        rows = (line.split("|") for line in output.splitlines() if line.strip())
        return {
            native_id.strip(): (state.strip(), code.strip())
            for native_id, name, state, code in rows
            if name.strip() == job_name
        }
