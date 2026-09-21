"""Author-side experiment metadata. Workers never open this database."""

import contextlib
import sqlite3
import time
from pathlib import Path

import fsspec

from lxm3 import __version__
from lxm3 import xm
from lxm3.xm_cluster import artifacts
from lxm3.xm_cluster.executable_specs import FrozenSource


class Catalog:
    def __init__(self, storage_root):
        self.path = Path(storage_root) / "experiments.sqlite3"

    @contextlib.contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_experiment(self, title, project):
        artifacts.ArtifactStore(fsspec.filesystem("file"), str(self.path.parent))
        experiment_id = time.time_ns()
        with self.connect() as db:
            # Use SQLite's rollback journal, not WAL on HPC shared storage.
            db.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY, title TEXT, project TEXT,
                    package_version TEXT
                );
                CREATE TABLE IF NOT EXISTS work_units (
                    experiment_id INTEGER, id INTEGER,
                    backend TEXT, hostname TEXT, username TEXT,
                    native_id TEXT, job_name TEXT, log_directory TEXT,
                    task_count INTEGER DEFAULT 1, is_array INTEGER DEFAULT 0,
                    state TEXT DEFAULT 'unknown', message TEXT DEFAULT '',
                    PRIMARY KEY (experiment_id, id)
                );
                CREATE TABLE IF NOT EXISTS sources (
                    experiment_id INTEGER, id TEXT, name TEXT,
                    archive_path TEXT, entrypoint_command TEXT,
                    PRIMARY KEY (experiment_id, id)
                );
            """)
            db.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?)",
                (experiment_id, title, project, __version__),
            )
        return experiment_id

    def experiment(self, experiment_id):
        if not self.path.exists():
            raise xm.NotFoundError(experiment_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise xm.NotFoundError(experiment_id)
        return dict(row)

    def create_work_unit(self, experiment_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            unit_id = db.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM work_units WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO work_units (experiment_id, id) VALUES (?, ?)",
                (experiment_id, unit_id),
            )
        return unit_id

    def record_source(self, experiment_id, source):
        with self.connect() as db:
            db.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (experiment_id, id) DO NOTHING",
                (
                    experiment_id,
                    source.id,
                    source.name,
                    source._archive_path,
                    source._entrypoint_command,
                ),
            )

    def sources(self, experiment_id):
        with self.connect() as db:
            return {
                row["id"]: FrozenSource(
                    row["id"],
                    row["name"],
                    row["archive_path"],
                    row["entrypoint_command"],
                )
                for row in db.execute(
                    "SELECT * FROM sources WHERE experiment_id = ? ORDER BY id",
                    (experiment_id,),
                )
            }

    def work_units(self, experiment_id):
        with self.connect() as db:
            return {
                row["id"]: dict(row)
                for row in db.execute(
                    "SELECT * FROM work_units WHERE experiment_id = ? ORDER BY id",
                    (experiment_id,),
                )
            }

    def work_unit(self, experiment_id, unit_id):
        with self.connect() as db:
            return dict(
                db.execute(
                    "SELECT * FROM work_units WHERE experiment_id = ? AND id = ?",
                    (experiment_id, unit_id),
                ).fetchone()
            )

    def update_work_unit(self, experiment_id, unit_id, **fields):
        # Column names are internal call-site constants; values are parameters.
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self.connect() as db:
            db.execute(
                f"UPDATE work_units SET {assignments} WHERE experiment_id = ? AND id = ?",
                (*fields.values(), experiment_id, unit_id),
            )
