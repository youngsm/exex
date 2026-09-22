"""Concrete request history is data, not launcher replay or live mutable state."""

import asyncio
import datetime
import enum
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import attr
import pytest

from exex import xm
from exex import xm_cluster as xc
from exex.clusters import slurm
from exex.xm_cluster import executables
from exex.xm_cluster import experiment as experiment_lib
from exex.xm_cluster import job_snapshot
from exex.xm_cluster.execution import gridengine
from exex.xm_cluster.execution import local
from exex.xm_cluster.execution.slurm import SlurmJobScriptBuilder


class Choice(enum.Enum):
    NAMED = "different-from-name"


def bundle(image=None):
    executable = executables.AppBundle(
        "payload",
        "python -m train",
        "/staging/source.zip",
        args=xm.merge_args(["positional", "positional"], {"seed": 1, "flag": True}),
        env_vars={"OVERRIDE": "package", "PACKAGE": "literal $ ' value"},
        container_image=image,
    )
    executable._target = ("SlurmSpec", "saved-host", "saved-user", "/staging", None)
    executable._source = xc.FrozenSource(
        "source-id", "source", "/source.tar", "python -m train"
    )
    return executable


def invocation():
    return xm.merge_args(
        {"seed": 2, "flag": False, "omit": None},
        ["positional", "--", xm.ShellSafeArg('"$EXPLICIT"'), Choice.NAMED, Path("a b")],
        {
            "repeat": (1, "two words", xm.ShellSafeArg('"$EXPLICIT"')),
            "empty": [],
            "nested": [[1, 2], [3]],
            "mapping": {"type": "Slurm"},
        },
    )


def assert_executor_equal(before, after):
    for field in attr.fields(type(before)):
        if field.name != "requirements":
            assert getattr(before, field.name) == getattr(after, field.name)
    assert before.requirements.task_requirements == after.requirements.task_requirements
    assert before.requirements.location == after.requirements.location


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("runtime", [None, "SINGULARITY", "SHIFTER"])
def test_job_roundtrip_preserves_rendering_merging_and_executor_fields(array, runtime):
    image = (
        executables.ContainerImage(
            "registry/image@sha256:abc", executables.ContainerImageType[runtime]
        )
        if runtime
        else None
    )
    options = {
        None: None,
        "SINGULARITY": xc.SingularityOptions(
            bind={"/a b": "/input:ro"}, extra_options=["--cleanenv"]
        ),
        "SHIFTER": xc.ShifterOptions(
            bind={"/a b": "/input:ro"}, modules=["gpu"], extra_options=["--clearenv"]
        ),
    }
    executor = xc.Slurm(
        cluster="original-site",
        resources={
            "account": "science",
            "qos": "preempt",
            "gpus": 2,
            "cpus-per-task": 8,
            "mem": "32G",
            "nodes": 2,
        },
        requirements=xc.JobRequirements(cpu=8, gpu=2, location="location"),
        walltime=datetime.timedelta(days=1, seconds=123, microseconds=456),
        partition="gpu",
        exclusive=True,
        modules=("python", "cuda"),
        workdir_root="/shared/work",
        log_directory="/separate/logs",
        extra_directives=["--requeue"],
        skip_directives=["--ntasks="],
        container_options=options[runtime],
    )
    executable = bundle(image)
    job = (
        xc.ArrayJob(
            executable,
            executor,
            args=[invocation()],
            env_vars=[{"OVERRIDE": "one"}, {"OVERRIDE": "two"}],
        )
        if array
        else xm.Job(
            executable, executor, args=invocation(), env_vars={"OVERRIDE": "job"}
        )
    )
    restored = job_snapshot.loads(job_snapshot.dumps(job))
    assert type(restored) is type(job)
    assert_executor_equal(job.executor, restored.executor)
    assert restored.executable._source == executable._source
    assert restored.executable._target == executable._target
    assert restored.executable.container_image == image
    assert restored.executable.env_vars == executable.env_vars
    assert restored.env_vars == job.env_vars
    original_args = job.args if array else [job.args]
    restored_args = restored.args if array else [restored.args]
    assert len(restored_args) == len(original_args)
    for before, after in zip(original_args, restored_args):
        assert before.to_list() == after.to_list()
        assert (
            xm.merge_args(executable.args, before, {"seed": 42}).to_list()
            == xm.merge_args(restored.executable.args, after, {"seed": 42}).to_list()
        )
        assert after.to_dict(kwargs_only=True)["flag"] is False
        assert isinstance(after.to_dict(kwargs_only=True)["repeat"], tuple)
    builder = SlurmJobScriptBuilder()
    assert builder.build(job, "name", "/logs") == builder.build(
        restored, "name", "/logs"
    )


@pytest.mark.parametrize(
    "executor,builder",
    [
        (
            xc.Local(
                container_options=xc.DockerOptions(
                    volumes={"/a": "/b"}, extra_options=["--network=none"]
                )
            ),
            local.LocalJobScriptBuilder(),
        ),
        (
            xc.GridEngine(
                resources={"mem": "2G"},
                parallel_environments={"mpi": 2},
                walltime=60,
                queue="queue",
                reserved=True,
                merge_output=False,
                shell="/bin/sh",
                project="p",
                account="a",
                modules=["python"],
                max_parallel_tasks=2,
                extra_directives=["-notify"],
                skip_directives=["-cwd"],
                cluster="site",
            ),
            gridengine.GridEngineJobScriptBuilder(),
        ),
    ],
)
def test_other_executors_remain_serializable(executor, builder):
    image = (
        executables.ContainerImage("image:tag", executables.ContainerImageType.DOCKER)
        if isinstance(executor, xc.Local)
        else None
    )
    job = xm.Job(bundle(image), executor)
    restored = job_snapshot.loads(job_snapshot.dumps(job))
    assert_executor_equal(executor, restored.executor)
    assert builder.build(restored, "name", "/logs") == builder.build(
        job, "name", "/logs"
    )


@pytest.fixture(autouse=True)
def isolated_loop_policy(monkeypatch):
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    monkeypatch.delenv("EXEX_PROJECT", raising=False)
    monkeypatch.delenv("EXEX_CLUSTER", raising=False)
    yield
    asyncio.set_event_loop_policy(previous)


@pytest.fixture
def config(tmp_path):
    return xc.Config(
        {
            "local": {"storage": {"staging": str(tmp_path / "author")}},
            "clusters": [
                {"name": "site", "storage": {"staging": str(tmp_path / "site")}}
            ],
        }
    )


@pytest.mark.parametrize("generator", [False, True])
def test_snapshot_taken_after_overrides_before_submission_and_is_independent(
    config, tmp_path, generator, monkeypatch
):
    monkeypatch.setenv("NOT_JOB_ENV", "do-not-save-this-ambient-value")
    experiment = xc.create_experiment("history", config=config)
    executor = xc.Slurm(cluster="site", log_directory=str(tmp_path / "other logs"))
    [executable] = experiment.package(
        [
            xm.Packageable(
                xc.SourceTree(xc.CommandList(["true"]), tmp_path, files=[]),
                executor.Spec(),
                args={"seed": 1},
                env_vars={"DEFAULT": "package"},
            )
        ]
    )
    source = executable._source
    unused = experiment.freeze(
        xc.SourceTree(xc.CommandList(["false"]), tmp_path, files=[])
    )
    job = xm.Job(executable, executor, args={"seed": 2}, env_vars={"EXPLICIT": "job"})
    overrides = {"args": {"seed": 3}, "env_vars": {"EXPLICIT": "override"}}

    async def generate(unit):
        await unit.add(job, args=overrides)

    def accepted(script_path):
        unit = experiment.work_units()[1]
        assert unit.job.args.to_list() == ["--seed=3"]
        assert unit.job.executable.args.to_list() == ["--seed=1"]
        assert unit.job.env_vars == {"EXPLICIT": "override"}
        assert unit.source == source
        assert unit._record["native_id"] is None
        return "123"

    with mock.patch.object(slurm.SlurmCluster, "launch", side_effect=accepted):
        with experiment:
            experiment.add(generate) if generator else experiment.add(
                job, args=overrides
            )
    unit = experiment.work_units()[1]
    before = experiment._catalog.path.read_bytes()
    executor.resources["gpus"] = 100
    executable.env_vars["DEFAULT"] = "changed"
    retrieved = unit.job
    retrieved.executor.resources["gpus"] = 200
    retrieved.executable.env_vars.clear()
    retrieved.args.to_dict(kwargs_only=True)["seed"] = 999
    Path(source._archive_path).unlink()
    with mock.patch.object(
        subprocess,
        "Popen",
        side_effect=AssertionError("Metadata must not spawn a process"),
    ):
        assert unit.job.executor.resources == {}
        assert unit.job.args.to_list() == ["--seed=3"]
        assert unit.job.executable.env_vars == {"DEFAULT": "package"}
        assert unit.source == source != unused
    assert experiment._catalog.path.read_bytes() == before
    assert "do-not-save-this-ambient-value" not in unit._record["job"]
    config._data["clusters"] = []
    reopened = xc.get_experiment(experiment.experiment_id, config=config).work_units()[
        1
    ]
    assert reopened.get_script() == Path(unit._record["script_path"]).read_text()
    assert Path(unit._record["script_path"]).parent != Path(
        unit._record["log_directory"]
    )


def test_unsubmitted_and_failed_submission_history(config, tmp_path):
    experiment = xc.create_experiment("failed", config=config)
    unit_id = experiment._catalog.create_work_unit(experiment.experiment_id)
    empty = experiment.work_units()[unit_id]
    assert empty.job is empty.source is None
    with pytest.raises(xm.NotFoundError, match="No submission script"):
        empty.get_script()
    executable = bundle()
    executable._target = executable._source = None
    with mock.patch.object(
        slurm.SlurmCluster, "launch", side_effect=OSError("disconnect")
    ) as launch:
        with pytest.raises(OSError, match="disconnect"):
            with experiment:
                experiment.add(
                    xm.Job(executable, xc.Slurm(cluster="site"), args={"seed": 4})
                )
    assert launch.call_count == 1
    failed = experiment.work_units()[2]
    assert failed.job.args.to_list() == ["--seed=4"]
    assert failed.source is None
    assert failed._record["native_id"] is None
    assert failed.get_status().state == "unknown"
    with pytest.raises(xm.NotFoundError):
        failed.get_script()


def test_fresh_process_without_launcher_reads_local_job_source_and_script(tmp_path):
    launcher = tmp_path / "launcher.py"
    launcher.write_text("""
import sys
from exex import xm, xm_cluster as xc
config = xc.Config({"local": {"storage": {"staging": sys.argv[1]}}})
with xc.create_experiment("fresh history", config=config) as experiment:
    source = xc.SourceTree(xc.CommandList(["true"]), sys.argv[1], files=[])
    [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec(), args={"seed": 1}, env_vars={"PACKAGE": "yes"})])
    experiment.add(xc.ArrayJob(executable, xc.Local(), args=[{"seed": 2}, {"seed": 3}], env_vars=[{"JOB": "yes"}]))
print(experiment.experiment_id)
""")
    root = str(tmp_path / "author")
    result = subprocess.run(
        [sys.executable, str(launcher), root],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    experiment_id = result.stdout.splitlines()[-1]
    launcher.unlink()
    read = """
import json, sys
from exex import xm, xm_cluster as xc
config = xc.Config({"local": {"storage": {"staging": sys.argv[1]}}})
experiment = xc.get_experiment(int(sys.argv[2]), config=config)
unit = experiment.work_units()[1]
assert isinstance(unit.job, xc.ArrayJob)
assert unit.job.executable.env_vars == {"PACKAGE": "yes"}
assert unit.job.env_vars == [{"JOB": "yes"}, {"JOB": "yes"}]
assert unit.source == experiment.sources()[unit.source.id]
assert unit.get_status().is_completed
print(json.dumps({"args": [xm.merge_args(unit.job.executable.args, args).to_list() for args in unit.job.args], "script": unit.get_script()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", read, root, experiment_id],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    retained = json.loads(result.stdout)
    assert retained["args"] == [["--seed=2"], ["--seed=3"]]
    assert "LOCAL_TASK_ID" in retained["script"]
