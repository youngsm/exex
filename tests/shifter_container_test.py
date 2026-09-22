"""Typed container packaging and generated-shell contracts; no simulated GPUs."""

import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from exex import xm
from exex import xm_cluster as xc
from exex.xm_cluster import executables
from exex.xm_cluster.execution.local import LocalJobScriptBuilder
from exex.xm_cluster.execution.slurm import SlurmJobScriptBuilder
from exex.xm_cluster.packaging import router

IMAGE = "id:" + "f" * 64
LITERAL = "literal $value ' $(touch DO_NOT_CREATE) with spaces\nand a newline"


@pytest.mark.parametrize(
    "wrapper,image,kind",
    [
        (xc.SingularityContainer, "docker://python:3.12", "singularity"),
        (xc.DockerContainer, "python:3.12", "docker"),
        (xc.ShifterContainer, IMAGE, "shifter"),
        (xc.ShifterContainer, "docker:python:3.12", "shifter"),
    ],
)
def test_wrappers_keep_constructor_name_and_source_packaging(
    tmp_path, wrapper, image, kind
):
    source = xc.UniversalPackage(
        ["python3", "main.py"],
        "build.sh",
        path=str(Path(__file__).parent / "testdata/test_universal"),
    )
    spec = wrapper(source, image)
    assert isinstance(spec, xm.ExecutableSpec)
    assert spec.entrypoint is source
    assert spec.name == source.name
    config = xc.Config({"local": {"storage": {"staging": str(tmp_path)}}})
    executable = router.packaging_router(
        xm.Packageable(
            spec, xc.Local.Spec(), args=[LITERAL], env_vars={"MESSAGE": LITERAL}
        ),
        config=config,
        project="test",
    )
    assert executable.container_image == executables.ContainerImage(
        image, executables.ContainerImageType(kind)
    )
    assert executable.args.to_list() == [shlex.quote(LITERAL)]
    assert executable.env_vars == {"MESSAGE": LITERAL}
    assert executable._target == ("local", str(tmp_path), "test")
    with zipfile.ZipFile(executable.resource_uri) as archive:
        assert archive.namelist() == ["main.py"]
    assert not list(tmp_path.rglob("*.sif"))


@pytest.mark.parametrize("array", [False, True])
def test_slurm_header_matches_runtime_without_mutating_executor(array):
    options = xc.ShifterOptions(modules=["gpu", "nccl-plugin"])
    resources = {"nodes": 2, "ntasks-per-node": 4, "image": "old", "module": "old"}
    executor = xc.Slurm(resources=resources, container_options=options)
    bundle = xc.AppBundle(
        "test",
        "python3 main.py",
        "/shared/source.zip",
        container_image=executables.ContainerImage(
            IMAGE, executables.ContainerImageType.SHIFTER
        ),
    )
    job = (
        xc.ArrayJob(bundle, executor, args=[[], []])
        if array
        else xm.Job(bundle, executor)
    )
    script = SlurmJobScriptBuilder().build(job, "test", "/logs")
    assert script.count(f"#SBATCH --image={IMAGE}") == 1
    assert "#SBATCH --module=gpu,nccl-plugin" in script
    assert "--module=gpu,nccl-plugin --workdir=" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks-per-node=4" in script
    assert "srun " not in script
    assert script.index("#SBATCH --image=") < script.index("set -e")
    assert ("#SBATCH --array=1-2" in script) is array
    assert executor.resources is resources
    assert resources["image"] == resources["module"] == "old"
    assert options.modules == ["gpu", "nccl-plugin"]


def test_shifter_modules_are_not_inferred_from_gpu_requests():
    bundle = xc.AppBundle(
        "test",
        "true",
        "/shared/source.zip",
        container_image=executables.ContainerImage(
            IMAGE, executables.ContainerImageType.SHIFTER
        ),
    )
    script = SlurmJobScriptBuilder().build(
        xm.Job(bundle, xc.Slurm(resources={"gpus-per-node": 1})), "test", "/logs"
    )
    assert "--module" not in script


@pytest.mark.parametrize(
    "executor_type,builder",
    [(xc.Local, LocalJobScriptBuilder), (xc.Slurm, SlurmJobScriptBuilder)],
)
@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_generated_shell_arguments_env_arrays_failure_cleanup(
    tmp_path, executor_type, builder, array, exit_code
):
    capture = tmp_path / "runtime.json"
    output = tmp_path / "result.json"
    runtime = tmp_path / "shifter"
    runtime.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "args = sys.argv[1:]\n"
        "opts = dict(arg.split('=', 1) for arg in args[:args.index('--')] if '=' in arg)\n"
        "env = dict(os.environ, MESSAGE='image default', CUDA_VISIBLE_DEVICES='all')\n"
        "with open(opts['--env-file']) as stream:\n"
        "    env.update(line.rstrip('\\n').split('=', 1) for line in stream)\n"
        f"with open({str(capture)!r}, 'x') as stream:\n"
        "    json.dump([args, opts['--workdir']], stream)\n"
        "os.chdir(opts['--workdir'])\n"
        "command = args[args.index('--') + 1:]\n"
        "os.execvpe(command[0], command, env)\n"
    )
    runtime.chmod(0o755)
    probe = (
        "import json, os, sys\n"
        f"with open({str(output)!r}, 'x') as stream:\n"
        "    json.dump([sys.argv[1:], os.environ['MESSAGE'], os.environ['CUDA_VISIBLE_DEVICES']], stream)\n"
        f"sys.exit({exit_code})\n"
    )
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("probe.py", probe)
    root = tmp_path / "work ' $literal"
    bind_source = str(tmp_path / "input ' $literal")
    options = xc.ShifterOptions(bind={bind_source: "/mnt:ro"}, modules=["gpu"])
    executor = executor_type(workdir_root=str(root), container_options=options)
    bundle = xc.AppBundle(
        "test",
        f"{shlex.quote(sys.executable)} probe.py",
        str(archive),
        env_vars={"MESSAGE": "packaged default"},
        container_image=executables.ContainerImage(
            IMAGE, executables.ContainerImageType.SHIFTER
        ),
    )
    job = (
        xc.ArrayJob(
            bundle,
            executor,
            args=[["first"], [LITERAL]],
            env_vars=[{"MESSAGE": "first"}, {"MESSAGE": LITERAL}],
        )
        if array
        else xm.Job(bundle, executor, args=[LITERAL], env_vars={"MESSAGE": LITERAL})
    )
    script = builder().build(job, "test", str(tmp_path / "logs"))
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CUDA_VISIBLE_DEVICES": "2",
            "SLURM_ARRAY_TASK_ID": "2",
            "LOCAL_TASK_ID": "2",
        },
    )
    assert result.returncode == exit_code, result.stderr
    assert json.loads(output.read_text()) == [[LITERAL], LITERAL, "2"]
    argv, workdir = json.loads(capture.read_text())
    assert argv[:3] == [
        f"--image={IMAGE}",
        "--module=gpu",
        f"--volume={bind_source}:/mnt:ro",
    ]
    assert Path(workdir).parent == root
    assert not Path(workdir).exists()
    assert list(root.iterdir()) == []
    assert not (tmp_path / "DO_NOT_CREATE").exists()
