"""Command-boundary regressions; a real SIF/GPU is qualified separately."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from exex import xm
from exex import xm_cluster as xc
from exex.xm_cluster import executables
from exex.xm_cluster.execution.slurm import SlurmJobScriptBuilder
from exex.xm_cluster.packaging import router

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/sif"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_singularity_gpu_mask_mounts_and_cleanup(tmp_path, exit_code):
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("probe.py", "# Container workload\n")
    capture = tmp_path / "runtime.json"
    runtime = tmp_path / "singularity"
    runtime.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "from pathlib import Path\n"
        "envfile = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--env-file='))\n"
        f"Path({str(capture)!r}).write_text(json.dumps([sys.argv[1:], os.getcwd(), Path(envfile).read_text()]))\n"
        f"sys.exit({exit_code})\n"
    )
    runtime.chmod(0o755)
    image = str(tmp_path / "image ' $literal.sif")
    executable = xc.AppBundle(
        "probe",
        "python3 probe.py",
        str(archive),
        container_image=executables.ContainerImage(
            image, executables.ContainerImageType.SINGULARITY
        ),
    )
    inputs, outputs = tmp_path / "input ' $literal", tmp_path / "output dir"
    root = tmp_path / "work"
    executor = xc.Slurm(
        resources={"gpus-per-node": 1},
        workdir_root=str(root),
        container_options=xc.SingularityOptions(
            bind={str(inputs): "/probe-input", str(outputs): "/probe-output"},
            extra_options=["--cleanenv"],
        ),
    )
    script = SlurmJobScriptBuilder().build(
        xm.Job(executable, executor), "probe", str(tmp_path / "logs")
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "123",
            "CUDA_VISIBLE_DEVICES": "2",
            "UNRELATED_SECRET": "must-not-forward",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stderr
    args, workdir, environment = json.loads(capture.read_text())
    assert "CUDA_VISIBLE_DEVICES=2" in environment.splitlines()
    assert "SLURM_JOB_ID=123" in environment.splitlines()
    assert "UNRELATED_SECRET" not in environment
    assert args[0] == "exec"
    assert "--nv" in args and "--cleanenv" in args
    assert f"--bind={inputs}:/probe-input" in args
    assert f"--bind={outputs}:/probe-output" in args
    assert image in args
    assert Path(workdir).parent == root
    assert list(root.iterdir()) == []


def test_source_change_reuses_staged_image(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("build.sh", "probe.py"):
        shutil.copy(EXAMPLE / name, source / name)
    image = tmp_path / "dependencies.sif"
    image.write_bytes(
        b"Image bytes; this test exercises staging, not a container runtime"
    )
    config = xc.Config({"local": {"storage": {"staging": str(tmp_path / "staging")}}})
    spec = xc.SingularityContainer(
        xc.UniversalPackage(
            path=str(source),
            build_script="build.sh",
            entrypoint=["python3", "probe.py"],
        ),
        image_path=str(image),
    )
    packageable = xm.Packageable(spec, xc.Local.Spec())
    first = router.packaging_router(packageable, config=config, project="probe")
    staged_image = Path(first.container_image.name)
    first_stat = staged_image.stat()
    probe = source / "probe.py"
    probe.write_text(
        probe.read_text().replace('REVISION = "base"', 'REVISION = "edited"')
    )
    second = router.packaging_router(packageable, config=config, project="probe")
    assert first.resource_uri != second.resource_uri
    assert second.container_image.name == first.container_image.name
    assert staged_image.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert staged_image.read_bytes() == image.read_bytes()
    with zipfile.ZipFile(second.resource_uri) as archive:
        assert b'REVISION = "edited"' in archive.read("probe.py")
