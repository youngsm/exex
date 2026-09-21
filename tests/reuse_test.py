"""Retained source lookup and append-only submission after launcher exit."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.clusters import slurm
from lxm3.xm_cluster import experiment as experiment_lib
from lxm3.xm_cluster.packaging import router
from lxm3.xm_cluster.packaging import source as source_capture


@pytest.fixture(autouse=True)
def isolated_loop_policy(monkeypatch):
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    monkeypatch.delenv("LXM_PROJECT", raising=False)
    monkeypatch.delenv("LXM_CLUSTER", raising=False)
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


def packaged(experiment, tmp_path, executor):
    source = xc.SourceTree(xc.CommandList(["true"]), tmp_path, files=[])
    [bundle] = experiment.package([xm.Packageable(source, executor.Spec())])
    return xm.Job(bundle, executor)


def test_sources_are_read_only_experiment_members_and_can_be_reused(config, tmp_path):
    first = xc.create_experiment("first", config=config)
    second = xc.create_experiment("second", config=config)
    assert first.sources() == second.sources() == {}
    source = first.freeze(xc.SourceTree(xc.CommandList(["true"]), tmp_path, files=[]))
    database = tmp_path / "author/experiments.sqlite3"
    before = database.read_bytes()
    with mock.patch.object(
        source_capture, "verify", side_effect=AssertionError("Unexpected content read")
    ):
        with mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("Unexpected process")
        ):
            retrieved = xc.get_experiment(first.experiment_id, config=config).sources()
    assert retrieved == {source.id: source}
    assert database.read_bytes() == before
    retrieved.clear()
    assert first.sources() == {source.id: source}
    assert second.sources() == {}
    second.package([xm.Packageable(source, xc.Local.Spec())])
    assert second.sources() == first.sources()


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_retrieval_does_not_read_archive_but_packaging_verifies_it(
    config, tmp_path, damage
):
    experiment = xc.create_experiment("damaged", config=config)
    source = experiment.freeze(
        xc.SourceTree(xc.CommandList(["true"]), tmp_path, files=[])
    )
    archive = Path(source._archive_path)
    if damage == "corrupt":
        archive.write_bytes(b"not the frozen content")
    else:
        archive.unlink()
    reopened = xc.get_experiment(experiment.experiment_id, config=config)
    assert reopened.sources() == {source.id: source}
    with mock.patch.object(router, "_transfer_file") as transfer:
        with pytest.raises(ValueError if damage == "corrupt" else FileNotFoundError):
            reopened.package(
                [xm.Packageable(reopened.sources()[source.id], xc.Local.Spec())]
            )
    transfer.assert_not_called()
    assert reopened.work_units() == {}


def test_failed_implicit_packaging_does_not_record_successful_membership(
    config, tmp_path
):
    experiment = xc.create_experiment("failed package", config=config)
    with mock.patch.object(
        router, "_transfer_file", side_effect=OSError("transfer failed")
    ):
        with pytest.raises(OSError, match="transfer failed"):
            packaged(experiment, tmp_path, xc.Local())
    assert experiment.sources() == {}


@pytest.mark.parametrize("reopened", [False, True])
def test_keyed_add_rejected_before_id_allocation_or_job_changes(
    config, tmp_path, reopened
):
    experiment = xc.create_experiment("keys", config=config)
    if reopened:
        experiment = xc.get_experiment(experiment.experiment_id, config=config)
    job = packaged(experiment, tmp_path, xc.Slurm(cluster="site"))
    with mock.patch.object(experiment_lib, "_launch") as launch:
        with mock.patch.object(xc.ClusterWorkUnit, "stop") as stop:
            with pytest.raises(NotImplementedError, match="Keyed submission"):
                experiment.add(job, identity="seed-1")
    assert experiment.work_units() == {}
    launch.assert_not_called()
    stop.assert_not_called()


@pytest.mark.parametrize("payload", ["job", "array", "generator", "group"])
def test_reopened_add_creates_independent_native_jobs_and_ids(
    config, tmp_path, payload
):
    with mock.patch.object(
        slurm.SlurmCluster, "launch", side_effect=["101", "102", "103"]
    ) as launch:
        with xc.create_experiment("append", config=config) as experiment:
            job = packaged(experiment, tmp_path, xc.Slurm(cluster="site"))
            experiment.add(job)
        original = experiment.work_units()[1]._record
        reopened = xc.get_experiment(experiment.experiment_id, config=config)

        async def generator(unit):
            await unit.add(job)

        choices = {
            "job": job,
            "array": xc.ArrayJob(job.executable, job.executor, args=[["a"], ["b"]]),
            "generator": generator,
            "group": xm.JobGroup(worker=job),
        }

        async def append():
            async with reopened:
                assert list(reopened.work_units()) == [1]
                for expected_id in (2, 3):
                    unit = await reopened.add(choices[payload])
                    assert unit.work_unit_id == expected_id
                    assert not unit._local_handles
            return unit

        asyncio.run(append())
    assert launch.call_count == 3
    units = xc.get_experiment(experiment.experiment_id, config=config).work_units()
    assert list(units) == [1, 2, 3]
    assert [unit._record["native_id"] for unit in units.values()] == [
        "101",
        "102",
        "103",
    ]
    assert len({unit._record["job_name"] for unit in units.values()}) == 3
    assert units[1]._record == original


@pytest.mark.parametrize("identity", ["", "unsupported-key"])
def test_loaded_work_unit_cannot_be_resubmitted(config, tmp_path, identity):
    with mock.patch.object(slurm.SlurmCluster, "launch", return_value="101"):
        with xc.create_experiment("one payload", config=config) as experiment:
            job = packaged(experiment, tmp_path, xc.Slurm(cluster="site"))
            experiment.add(job)
    reopened = xc.get_experiment(experiment.experiment_id, config=config)
    before = reopened.work_units()[1]._record

    async def resubmit():
        async with reopened:
            await reopened.work_units()[1].add(job, identity=identity)

    with mock.patch.object(experiment_lib, "_launch") as launch:
        with mock.patch.object(slurm.SlurmCluster, "cancel") as cancel:
            with pytest.raises(NotImplementedError if identity else ValueError):
                asyncio.run(resubmit())
    launch.assert_not_called()
    cancel.assert_not_called()
    assert reopened.work_units()[1]._record == before


def test_keyed_generator_payload_is_also_rejected(config, tmp_path):
    experiment = xc.create_experiment("generator key", config=config)
    job = packaged(experiment, tmp_path, xc.Slurm(cluster="site"))

    async def generator(unit):
        await unit.add(
            xc.ArrayJob(job.executable, job.executor, args=[[]]), identity="key"
        )

    with mock.patch.object(slurm.SlurmCluster, "launch") as launch:
        with pytest.raises(NotImplementedError, match="Keyed submission"):
            with experiment:
                experiment.add(generator)
    launch.assert_not_called()


def test_failed_new_submission_cannot_cancel_or_modify_old_work(config, tmp_path):
    with mock.patch.object(slurm.SlurmCluster, "launch", return_value="101"):
        with xc.create_experiment("failure isolation", config=config) as experiment:
            job = packaged(experiment, tmp_path, xc.Slurm(cluster="site"))
            experiment.add(job)
    before = experiment.work_units()[1]._record
    reopened = xc.get_experiment(experiment.experiment_id, config=config)
    with mock.patch.object(
        slurm.SlurmCluster,
        "launch",
        side_effect=subprocess.CalledProcessError(255, "ssh"),
    ) as launch:
        with mock.patch.object(slurm.SlurmCluster, "cancel") as cancel:
            with pytest.raises(subprocess.CalledProcessError):
                with reopened:
                    reopened.add(job)
    assert launch.call_count == 1
    cancel.assert_not_called()
    units = reopened.work_units()
    assert units[1]._record == before
    assert units[2].get_status().state == "unknown"
    assert "Submission error" in units[2].get_status().message


def test_deleted_checkout_reused_by_concurrent_fresh_processes(config, tmp_path):
    launcher = """
import json, sys, tempfile
from pathlib import Path
from lxm3 import xm, xm_cluster as xc
config = xc.Config(json.loads(sys.argv[1]))
with xc.create_experiment('process boundary', project='test', config=config) as experiment:
    with tempfile.TemporaryDirectory() as checkout:
        Path(checkout, 'worker.py').write_text("import sys\\nprint('frozen-original:' + sys.argv[1])\\n")
        source = experiment.freeze(xc.SourceTree(xc.ModuleName('worker'), checkout, files=['worker.py']))
    assert not Path(checkout).exists()
    [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
    experiment.add(xm.Job(executable, xc.Local(), args=['first']))
print(json.dumps([experiment.experiment_id, source.id]))
"""
    config_json = json.dumps(config._data)
    result = subprocess.run(
        [sys.executable, "-c", launcher, config_json],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    experiment_id, source_id = json.loads(result.stdout.splitlines()[-1])
    original = xc.get_experiment(experiment_id, config=config).work_units()[1]._record
    reader = """
import asyncio, json, sys
from lxm3 import xm, xm_cluster as xc
config = xc.Config(json.loads(sys.argv[1]))
async def append():
    async with xc.get_experiment(int(sys.argv[2]), config=config) as experiment:
        source = experiment.sources()[sys.argv[3]]
        [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
        unit = await experiment.add(xm.Job(executable, xc.Local(), args=[sys.argv[4]]))
        assert await unit.wait_until_complete() is unit
    print(json.dumps([unit.work_unit_id, source.id, unit.get_logs()]))
asyncio.run(append())
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                reader,
                config_json,
                str(experiment_id),
                source_id,
                label,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
        )
        for label in ("second", "third")
    ]
    records = []
    for process, label in zip(processes, ("second", "third")):
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        record = json.loads(stdout.splitlines()[-1])
        assert record[1] == source_id
        assert f"frozen-original:{label}" in record[2]
        records.append(record)
    assert {record[0] for record in records} == {2, 3}
    units = xc.get_experiment(experiment_id, config=config).work_units()
    assert units[1]._record == original
    assert all(unit.get_status().is_completed for unit in units.values())
