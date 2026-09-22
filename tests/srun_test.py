"""Exercise generated scripts; native placement is qualified on real Slurm."""

import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

from exex import xm
from exex import xm_cluster as xc
from exex.xm_cluster import job_snapshot
from exex.xm_cluster.executables import ContainerImage
from exex.xm_cluster.executables import ContainerImageType
from exex.xm_cluster.execution.slurm import SlurmJobScriptBuilder


@pytest.mark.parametrize("runtime", [None, "singularity", "shifter"])
@pytest.mark.parametrize("fail", [False, True])
def test_srun_refreshes_task_environment_and_captures_once(tmp_path, runtime, fail):
    binary = tmp_path / "bin"
    binary.mkdir()
    srun = binary / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "assert args[:3] == ['--ntasks=2', '--ntasks-per-node=1', '--kill-on-bad-exit=1']\n"
        "processes = [subprocess.Popen(args[3:], env=dict(os.environ, SLURM_PROCID=str(rank), "
        "SLURM_NODEID=str(rank), CUDA_VISIBLE_DEVICES=str(rank + 3))) for rank in range(2)]\n"
        "sys.exit(max(p.wait() for p in processes))\n"
    )
    srun.chmod(0o755)
    if runtime:
        container = binary / runtime
        container.write_text(
            f"#!{sys.executable}\n"
            "import json, os, shlex, subprocess, sys\n"
            "from pathlib import Path\n"
            "envfile = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--env-file='))\n"
            "env = dict(line.split('=', 1) for line in Path(envfile).read_text().splitlines())\n"
            + (
                "env = json.loads(subprocess.check_output(['bash', '-ec', "
                '\'set -a; source "$1"; exec "$2" -c "import os,json; print(json.dumps(dict(os.environ)))"\', '
                "'env-test', envfile, sys.executable], env={}))\n"
                if runtime == "singularity"
                else ""
            )
            + "assert env['SLURM_TASKS_PER_NODE'] == '1(x2)'\n"
            "rank = env['SLURM_PROCID']\n"
            "assert env['CUDA_VISIBLE_DEVICES'] == str(int(rank) + 3), env\n"
            "args = sys.argv[sys.argv.index('sh'):]\n"
            "args[-1] = args[-1].replace('/tmp/job-param.sh', shlex.quote(os.getcwd() + '/job-param.sh'))\n"
            "sys.exit(subprocess.call(args, env=dict(env, PATH=os.environ['PATH'])))\n"
        )
        container.chmod(0o755)
    source = tmp_path / "source"
    source.mkdir()
    (source / "worker.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "rank = os.environ['SLURM_PROCID']\n"
        "assert os.environ['SLURM_NODEID'] == rank\n"
        "out = Path(os.environ['EXEX_OUTPUT_DIR']) / 'result'\nout.mkdir(exist_ok=True)\n"
        "(out / (rank + '.json')).write_text(json.dumps([rank, os.environ['CUDA_VISIBLE_DEVICES'], sys.argv[1:]]))\n"
        f"sys.exit(7 if {fail!r} and rank == '1' else 0)\n"
    )
    archive = shutil.make_archive(str(tmp_path / "source"), "zip", source)
    image = (
        ContainerImage("image.sif", ContainerImageType(runtime)) if runtime else None
    )
    bundle = xc.AppBundle(
        "probe",
        shlex.join([sys.executable, "worker.py"]),
        archive,
        container_image=image,
    )
    executor = xc.Slurm(
        resources={"nodes": 2, "ntasks-per-node": 1, "gpus-per-node": 1},
        workdir_root=str(tmp_path / "shared ' work"),
        srun_options=["--ntasks=2", "--ntasks-per-node=1", "--kill-on-bad-exit=1"],
    )
    argument = "literal $value ' with spaces\nand newline"
    job = xm.Job(bundle, executor, args=[argument])
    restored = job_snapshot.loads(job_snapshot.dumps(job)).executor
    assert restored.srun_options == executor.srun_options
    assert restored.resources == executor.resources
    logs = tmp_path / "logs"
    script = SlurmJobScriptBuilder().build(
        job, "probe", str(logs), outputs={"result": "result"}
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "SLURM_PROCID": "99",
            "SLURM_TASKS_PER_NODE": "1(x2)",
            "CUDA_VISIBLE_DEVICES": "wrong",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == (7 if fail else 0), result.stderr
    manifests = list(logs.rglob("manifest.json"))
    assert len(manifests) == (0 if fail else 1)
    if not fail:
        import tarfile

        [artifact] = logs.rglob("*.tar")
        with tarfile.open(artifact) as stream:
            for rank in range(2):
                [member] = [
                    m for m in stream.getmembers() if m.name.endswith(f"/{rank}.json")
                ]
                assert json.load(stream.extractfile(member)) == [
                    str(rank),
                    str(rank + 3),
                    [argument],
                ]
    assert list((tmp_path / "shared ' work").iterdir()) == []


def test_srun_is_opt_in():
    job = xm.Job(
        xc.AppBundle("probe", "echo hello", "source.zip"),
        xc.Slurm(resources={"nodes": 2}),
    )
    script = SlurmJobScriptBuilder().build(job, "probe", "/logs")
    assert "srun " not in script
