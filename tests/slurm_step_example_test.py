"""Test the example's command boundary; native placement is qualified on Slurm."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.xm_cluster.execution.slurm import SlurmJobScriptBuilder

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/slurm_step"
MESSAGE = "literal $value ' with spaces\nand a newline"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_driver_srun_arguments_exit_and_cleanup(tmp_path, exit_code):
    build = tmp_path / "build"
    build.mkdir()
    subprocess.run(
        [str(EXAMPLE / "build.sh")],
        cwd=EXAMPLE,
        env={**os.environ, "BUILDDIR": str(build)},
        check=True,
    )
    archive = shutil.make_archive(str(tmp_path / "package"), "zip", build)
    capture = tmp_path / "srun.json"
    fake = tmp_path / "srun"
    fake.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"with open({str(capture)!r}, 'x') as stream:\n"
        "    json.dump([sys.argv[1:], os.getcwd(), os.environ['PROBE_MESSAGE']], stream)\n"
        f"sys.exit({exit_code})\n"
    )
    fake.chmod(0o755)
    output = tmp_path / "output ' $literal"
    root = tmp_path / "shared work"
    executable = xc.AppBundle("probe", "bash driver.sh", archive)
    job = xm.Job(
        executable,
        xc.Slurm(workdir_root=str(root)),
        args=[str(output), sys.executable, "1"],
        env_vars={"PROBE_MESSAGE": MESSAGE},
    )
    script = SlurmJobScriptBuilder().build(job, "probe", str(tmp_path / "logs"))
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "123",
    }
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True)
    assert result.returncode == exit_code, result.stderr.decode()
    argv, workdir, message = json.loads(capture.read_text())
    assert argv == [
        "--nodes=2",
        "--ntasks=2",
        "--ntasks-per-node=1",
        "--kill-on-bad-exit=1",
        sys.executable,
        "worker.py",
        str(output),
        "1",
    ]
    assert message == MESSAGE
    assert Path(workdir).parent == root
    assert (output / "driver-workdir.txt").read_text().strip() == workdir
    assert (output / "driver-token.txt").read_text().startswith("123-")
    assert list(root.iterdir()) == []
    repeated = subprocess.run(["bash", "-c", script], env=env, capture_output=True)
    assert repeated.returncode != 0
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("fail", [False, True])
def test_worker_evidence_and_intentional_failure(tmp_path, fail):
    (tmp_path / "driver-token.txt").write_text("shared driver token")
    output = tmp_path / "output"
    output.mkdir()
    rank = 1 if fail else 0
    if fail:
        (output / "rank-0.json").write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "worker.py"),
            str(output),
            "1" if fail else "-1",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "SLURM_PROCID": str(rank),
            "SLURM_LOCALID": "0",
            "SLURM_NTASKS": "2",
            "SLURM_JOB_ID": "123",
            "SLURM_STEP_ID": "0",
            "PROBE_MESSAGE": MESSAGE,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == (7 if fail else 0), result.stderr
    record = json.loads((output / f"rank-{rank}.json").read_text())
    assert record["rank"] == rank
    assert record["world_size"] == 2
    assert record["message"] == MESSAGE
    assert record["driver_token"] == "shared driver token"
    assert (output / f"rank-{rank}.complete").exists() is not fail
