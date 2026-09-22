"""Additive job-history upgrade without changing reads or existing records."""

import asyncio
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest import mock

import pytest

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.xm_cluster import catalog
from lxm3.xm_cluster import experiment as experiment_lib


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    # Schema immediately before concrete Job history was introduced.
    store = catalog.Catalog(tmp_path)
    with store.connect() as db:
        db.executescript("""
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY, title TEXT, project TEXT, package_version TEXT
            );
            CREATE TABLE work_units (
                experiment_id INTEGER, id INTEGER,
                backend TEXT, hostname TEXT, username TEXT,
                native_id TEXT, job_name TEXT, log_directory TEXT,
                task_count INTEGER DEFAULT 1, is_array INTEGER DEFAULT 0,
                state TEXT DEFAULT 'unknown', message TEXT DEFAULT '',
                PRIMARY KEY (experiment_id, id)
            );
            CREATE TABLE sources (
                experiment_id INTEGER, id TEXT, name TEXT,
                archive_path TEXT, entrypoint_command TEXT,
                PRIMARY KEY (experiment_id, id)
            );
            INSERT INTO experiments VALUES (101, 'original', 'project', 'original-version');
            INSERT INTO work_units
                (experiment_id, id, backend, state, message, log_directory)
                VALUES (101, 1, 'local', 'completed', 'original outcome', '/retained/logs');
            INSERT INTO sources VALUES (101, 'source-id', 'source', '/retained/source.tar', 'true');
        """)
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    monkeypatch.delenv("LXM_PROJECT", raising=False)
    monkeypatch.delenv("LXM_CLUSTER", raising=False)
    config = xc.Config({"local": {"storage": {"staging": str(tmp_path)}}})
    yield store, config
    asyncio.set_event_loop_policy(previous)


def test_old_catalog_reads_remain_read_only_and_do_not_invent_history(legacy):
    store, config = legacy
    before = store.path.read_bytes()
    with mock.patch.object(
        subprocess, "Popen", side_effect=AssertionError("Unexpected process")
    ):
        [experiment] = xc.list_experiments(config=config)
        unit = experiment.work_units()[1]
        assert experiment.experiment_id == 101
        assert unit.job is unit.source is None
        assert unit.artifacts() == {}
        assert unit.get_links() == {}
        assert unit.get_status().is_completed
        assert list(experiment.sources()) == ["source-id"]
        with pytest.raises(xm.NotFoundError, match="No submission script"):
            unit.get_script()
    assert store.path.read_bytes() == before


@pytest.mark.parametrize("reopen", [False, True])
def test_first_submission_upgrades_and_preserves_existing_records(
    legacy, tmp_path, reopen
):
    store, config = legacy
    original = store.work_unit(101, 1)
    original_experiment = store.experiment(101)
    original_sources = store.sources(101)
    experiment = (
        xc.get_experiment(101, config=config)
        if reopen
        else xc.create_experiment("new", config=config)
    )
    with experiment:
        source = xc.SourceTree(xc.CommandList(["true"]), tmp_path, files=[])
        [executable] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
        experiment.add(xm.Job(executable, xc.Local(), args={"seed": 3}))
    loaded = xc.get_experiment(experiment.experiment_id, config=config)
    unit = loaded.work_units()[2 if reopen else 1]
    assert unit.job.args.to_list() == ["--seed=3"]
    assert unit.source == executable._source
    assert unit.get_script().startswith("#!/usr/bin/env bash")
    assert unit.get_status().is_completed
    assert store.work_unit(101, 1) == {
        **original,
        "job": None,
        "script_path": None,
        "outputs": None,
        "artifact_directory": None,
        "inputs": None,
        "links_directory": None,
        "continuation": None,
        "execution_directory": None,
    }
    assert store.experiment(101) == original_experiment
    assert store.sources(101).items() >= original_sources.items()
    old_unit = xc.get_experiment(101, config=config).work_units()[1]
    assert old_unit.job is old_unit.source is None
    with pytest.raises(xm.NotFoundError):
        old_unit.get_script()


def test_concurrent_writers_upgrade_once_and_allocate_distinct_ids(legacy):
    store, _ = legacy
    barrier = Barrier(2)

    def create(_):
        barrier.wait(timeout=10)
        return store.create_work_unit(101)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(create, range(2))) == [2, 3]
    assert list(store.work_units(101)) == [1, 2, 3]
    assert store.work_unit(101, 1)["job"] is None
    assert store.work_unit(101, 1)["script_path"] is None


def test_failed_creation_rolls_back_column_upgrade_too(legacy):
    store, _ = legacy
    with store.connect() as db:
        db.execute("""
            CREATE TRIGGER reject_new_work BEFORE INSERT ON work_units
            BEGIN SELECT RAISE(ABORT, 'intentional failure'); END
        """)
    with pytest.raises(sqlite3.IntegrityError, match="intentional failure"):
        store.create_work_unit(101)
    assert "job" not in store.work_unit(101, 1)
    assert "script_path" not in store.work_unit(101, 1)
    assert list(store.work_units(101)) == [1]
