"""Exercise native command composition, not a simulated GPU/container runtime."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from exex import xm
from exex import xm_cluster as xc
from exex.xm_cluster.execution.slurm import SlurmJobScriptBuilder

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/shifter"
MESSAGE = "literal env $value ' with spaces\nand a newline"
ARGUMENT = "literal argument $value ' with spaces\nand a newline"
IMAGE = "2f83f19de6816d2c7c0ef24c7d51a7f65362c6f5d706cf5b5c797c7cfb563180"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_native_step_mask_arguments_failure_and_cleanup(tmp_path, exit_code):
    build = tmp_path / "build"
    build.mkdir()
    subprocess.run(
        [str(EXAMPLE / "build.sh")],
        cwd=EXAMPLE,
        env={**os.environ, "BUILDDIR": str(build)},
        check=True,
    )
    archive = shutil.make_archive(str(tmp_path / "package"), "zip", build)
    capture = tmp_path / "shifter.json"
    step_capture = tmp_path / "srun.json"
    fake_step = tmp_path / "srun"
    fake_step.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"with open({str(step_capture)!r}, 'x') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = '2'\n"
        "os.execvp(sys.argv[5], sys.argv[5:])\n"
    )
    fake_runtime = tmp_path / "shifter"
    fake_runtime.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"with open({str(capture)!r}, 'x') as stream:\n"
        "    json.dump([sys.argv[1:], os.getcwd()], stream)\n"
        f"sys.exit({exit_code})\n"
    )
    fake_step.chmod(0o755)
    fake_runtime.chmod(0o755)
    input_dir = tmp_path / "input ' $literal"
    output = tmp_path / "output ' $literal"
    root = tmp_path / "shared work"
    job = xm.Job(
        xc.AppBundle("probe", "bash driver.sh", archive),
        xc.Slurm(
            resources={"image": "id:" + IMAGE, "module": "gpu"}, workdir_root=str(root)
        ),
        args=[str(input_dir), str(output), "1", ARGUMENT],
        env_vars={"PROBE_MESSAGE": MESSAGE, "PROBE_IMAGE_ID": IMAGE},
    )
    script = SlurmJobScriptBuilder().build(job, "probe", str(tmp_path / "logs"))
    assert f"#SBATCH --image=id:{IMAGE}" in script
    assert "#SBATCH --module=gpu" in script
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        },
        capture_output=True,
    )
    assert result.returncode == exit_code, result.stderr.decode()
    step_argv = json.loads(step_capture.read_text())
    assert step_argv == [
        "--nodes=1",
        "--ntasks=1",
        "--gpus-per-task=1",
        "--kill-on-bad-exit=1",
        "bash",
        "task.sh",
        str(input_dir),
        str(output),
        "1",
        ARGUMENT,
    ]
    runtime_argv, workdir = json.loads(capture.read_text())
    assert runtime_argv == [
        "--module=gpu",
        f"--workdir={workdir}",
        f"--volume={input_dir}:/mnt:ro",
        f"--volume={output}:/media",
        "--env=CUDA_VISIBLE_DEVICES=2",
        f"--env=PROBE_MESSAGE={MESSAGE}",
        f"--env=PROBE_IMAGE_ID={IMAGE}",
        f"--env=CUPY_CACHE_DIR={workdir}/.cupy",
        "--",
        "python3",
        "worker.py",
        "/mnt/input.txt",
        "/media",
        "1",
        ARGUMENT,
    ]
    assert (output / "batch-mask.txt").read_text() == "0,1,2,3\n"
    assert (output / "task-mask.txt").read_text() == "2\n"
    assert (output / "driver-workdir.txt").read_text().strip() == workdir
    assert Path(workdir).parent == root
    assert list(root.iterdir()) == []
