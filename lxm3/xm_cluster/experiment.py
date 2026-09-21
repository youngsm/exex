import asyncio
import functools
import subprocess
from typing import Any, Awaitable, Mapping, Optional, Sequence, Union

import vcsinfo
from absl import logging

from lxm3._vendor.xmanager import xm
from lxm3._vendor.xmanager.xm import async_packager
from lxm3._vendor.xmanager.xm import core
from lxm3.clusters import slurm
from lxm3.xm_cluster import array_job as array_job_lib
from lxm3.xm_cluster import catalog
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import console
from lxm3.xm_cluster import executable_specs
from lxm3.xm_cluster import inspection
from lxm3.xm_cluster import metadata
from lxm3.xm_cluster import packaging
from lxm3.xm_cluster.execution import gridengine as gridengine_execution
from lxm3.xm_cluster.execution import job_script_builder
from lxm3.xm_cluster.execution import local as local_execution
from lxm3.xm_cluster.execution import slurm as slurm_execution
from lxm3.xm_cluster.packaging import source as source_capture
from lxm3.xm_cluster.packaging.router import _package_target

_STATUS_POLL_INTERVAL = 10


class _LaunchResult:
    def __init__(self, local_handles, non_local_handles):
        self.local_handles = local_handles
        self.non_local_handles = non_local_handles


async def _launch(
    experiment_title: str,
    work_unit_name: str,
    job: Union[xm.JobGroup, array_job_lib.ArrayJob],
    *,
    config: config_lib.Config,
    project: Optional[str],
):
    local_handles = []
    non_local_handles = []

    job_name = f"{experiment_title}_{work_unit_name}"

    for payload in job_script_builder.flatten_job(job):
        target = payload.executable._target
        if target is not None and target != _package_target(
            payload.executor.Spec(), config=config, project=project
        ):
            raise ValueError(
                "Executable was packaged for a different destination; package it for this executor and experiment."
            )

    local_handles.extend(
        await local_execution.launch(job_name, job, config=config, project=project)
    )
    non_local_handles.extend(
        await slurm_execution.launch(job_name, job, config=config, project=project)
    )
    non_local_handles.extend(
        await gridengine_execution.launch(job_name, job, config=config, project=project)
    )

    return _LaunchResult(local_handles, non_local_handles)


class ClusterWorkUnit(xm.WorkUnit):
    """A submitted payload with an author-side execution reference."""

    def __init__(
        self,
        experiment: "ClusterExperiment",
        work_unit_id: int,
        args: Optional[Mapping[str, Any]] = None,
        role: xm.ExperimentUnitRole = xm.WorkUnitRole(),
    ) -> None:
        super().__init__(experiment, experiment._create_task, args, role)
        self._work_unit_id = work_unit_id
        self._submitted = False
        self._local_handles = []
        self._non_local_handles = []

    async def _launch_job_group(
        self,
        job_group: xm.JobGroup,
        args_view: Optional[Mapping[str, Any]],
        identity: str,
    ) -> None:
        await self._submit_job_for_execution(job_group, identity)

    async def _launch_job_config(self, job_config, args_view, identity):
        assert not args_view
        assert isinstance(job_config, array_job_lib.ArrayJob)
        await self._submit_job_for_execution(job_config, identity)

    async def _submit_job_for_execution(
        self, job: Union[xm.JobGroup, array_job_lib.ArrayJob], identity
    ):
        if identity:
            raise NotImplementedError("Keyed submission is not implemented")
        if self._submitted:
            raise ValueError(
                "A WorkUnit accepts one payload; reopened units cannot submit"
            )
        self._submitted = True
        try:
            (payload,) = job_script_builder.flatten_job(job)
            is_array = isinstance(payload, array_job_lib.ArrayJob)
            self._save(
                task_count=len(payload.args) if is_array else 1, is_array=is_array
            )
            launch_result = await _launch(
                self.experiment._experiment_title,
                self.experiment_unit_name,
                job,
                config=self.experiment._config,
                project=self.experiment._project,
            )
            self._ingest_handles(launch_result)
        except Exception as error:
            self._save(message=f"Submission error (acceptance may be unknown): {error}")
            raise

    def _ingest_handles(self, launch_result):
        self._local_handles.extend(launch_result.local_handles)
        self._non_local_handles.extend(launch_result.non_local_handles)
        handles = self._local_handles + self._non_local_handles
        if handles:
            self._save(**handles[0].record)
        for handle in self._local_handles:
            handle.future.add_done_callback(self._record_local_outcome)

    def _save(self, **fields):
        self.experiment._catalog.update_work_unit(
            self.experiment_id, self.work_unit_id, **fields
        )

    def _record_local_outcome(self, _):
        if all(handle.future.done() for handle in self._local_handles):
            status = inspection.local_status(self._local_handles)
            self._save(state=status.state, message=status.message)

    def get_status(self) -> inspection.WorkUnitStatus:
        if self._local_handles:
            return inspection.local_status(self._local_handles)
        return inspection.get_status(self._record)

    def get_logs(self, *, task: Optional[int] = None, tail: int = 200) -> str:
        return inspection.get_logs(self._record, task=task, tail=tail)

    def stop(
        self,
        *,
        mark_as_failed: bool = False,
        mark_as_completed: bool = False,
        message: Optional[str] = None,
    ) -> None:
        """Request native Slurm cancellation, without waiting for termination."""
        if mark_as_completed:
            raise NotImplementedError("Cancellation cannot mark a WorkUnit completed")
        record = self._record
        if record["backend"] is None:
            # XM calls stop after submission errors. No accepted handle is owned.
            return
        if record["backend"] != "slurm":
            raise NotImplementedError("Cancellation is supported only for Slurm")
        # For Slurm these fields record intent, never evidence of termination.
        self._save(
            state="failed" if mark_as_failed else "stopped", message=message or ""
        )
        slurm.SlurmCluster(inspection._hostname(record), record["username"]).cancel(
            record["native_id"], record["job_name"]
        )

    async def _wait_until_complete(self) -> None:
        if self._local_handles:
            # Cancelling a waiter must not cancel queued/running Local futures.
            await asyncio.gather(
                *(
                    asyncio.shield(asyncio.wrap_future(handle.future))
                    for handle in self._local_handles
                ),
                return_exceptions=True,
            )
        while True:
            status = await asyncio.to_thread(self.get_status)
            if status.is_completed:
                return
            if status.is_failed:
                raise xm.ExperimentUnitFailedError(status.message, work_unit=self)
            if status.state in {"stopped", "paused"}:
                raise xm.ExperimentUnitNotCompletedError(
                    status.message or status.state, work_unit=self
                )
            await asyncio.sleep(_STATUS_POLL_INTERVAL)

    @property
    def _record(self):
        return self.experiment._catalog.work_unit(self.experiment_id, self.work_unit_id)

    async def wait_for_local_jobs(self, is_exit_abrupt: bool):
        if self._local_handles and not is_exit_abrupt:
            results = await asyncio.gather(
                *[handle.wait() for handle in self._local_handles],
                return_exceptions=True,
            )
            self._record_local_outcome(None)
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    @property
    def work_unit_id(self) -> int:
        return self._work_unit_id

    @property
    def experiment_unit_name(self) -> str:
        return f"{self.experiment_id}_{self._work_unit_id}"

    @property
    def context(self) -> metadata.ClusterMetadataContext:
        return metadata.ClusterMetadataContext()


class ClusterExperiment(xm.Experiment):
    """An experiment with its own configuration and packaging queue."""

    def __init__(
        self,
        experiment_title: str,
        vcs: Optional[vcsinfo.VCS] = None,
        *,
        config: Optional[config_lib.Config] = None,
        project: Optional[str] = None,
        _experiment_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._work_units = {}
        self._experiment_title = experiment_title
        self._vcs = vcs
        self._config = (
            config if config is not None else config_lib.default()
        )._snapshot()
        self._project = project if project is not None else self._config.project()
        self._catalog = catalog.Catalog(self._config.local_settings().storage_root)
        self._reopened = _experiment_id is not None
        if self._reopened:
            record = self._catalog.experiment(_experiment_id)
            self._experiment_id = record["id"]
            self._experiment_title = record["title"]
            self._project = record["project"]
        else:
            self._experiment_id = self._catalog.create_experiment(
                experiment_title, self._project
            )
        self._async_packager = async_packager.AsyncPackager(self._package)

    def _package(self, packageables):
        executables = packaging.package(
            packageables, config=self._config, project=self._project
        )
        for executable in executables:
            if executable._source is not None:
                self._catalog.record_source(self.experiment_id, executable._source)
        return executables

    def package(
        self, packageables: Sequence[xm.Packageable] = ()
    ) -> Sequence[xm.Executable]:
        """Package for this experiment and flush its queued packaging requests."""
        return self._async_packager.package(packageables)

    def package_async(self, packageable: xm.Packageable) -> Awaitable[xm.Executable]:
        """Queue a specification; package() performs the build and transfer."""
        return self._async_packager.add(packageable)

    def freeze(
        self, source: executable_specs.SourceTree
    ) -> executable_specs.FrozenSource:
        """Capture and record source without building, uploading or submitting."""
        frozen = source_capture.freeze(
            source, self._config.local_settings().storage_root
        )
        self._catalog.record_source(self.experiment_id, frozen)
        return frozen

    def sources(self) -> Mapping[str, executable_specs.FrozenSource]:
        """Retrieve retained source values without inspecting or staging their bytes."""
        return self._catalog.sources(self.experiment_id)

    def _create_experiment_unit(
        self,
        args: Optional[Mapping[str, Any]],
        role: xm.ExperimentUnitRole = xm.WorkUnitRole(),
        identity: str = "",
    ) -> Awaitable[ClusterWorkUnit]:
        """Creates a new WorkUnit instance for the experiment."""
        if identity:
            raise NotImplementedError("Keyed submission is not implemented")
        if not isinstance(role, xm.WorkUnitRole):
            raise NotImplementedError("Auxiliary units are not supported")
        future = asyncio.Future(loop=self._event_loop)
        experiment_unit = ClusterWorkUnit(
            self,
            self._catalog.create_work_unit(self.experiment_id),
            args,
            role,
        )
        self._work_units[experiment_unit.work_unit_id] = experiment_unit

        future.set_result(experiment_unit)
        return future

    def _wait_for_local_jobs(self, is_exit_abrupt: bool):
        if self._work_units:
            if any(wu._local_handles for wu in self._work_units.values()):
                console.info("Waiting for local jobs to complete.")
        for unit in self._work_units.values():
            self._create_task(unit.wait_for_local_jobs(is_exit_abrupt))

    def __exit__(self, exc_type, exc_value, traceback):
        # Flush `.add` calls.
        try:
            self._wait_for_tasks()
            self._wait_for_local_jobs(exc_value is not None)
        finally:
            super().__exit__(exc_type, exc_value, traceback)

    async def __aexit__(self, exc_type, exc_value, traceback):
        # Flush `.add` calls.
        try:
            await self._await_for_tasks()
            self._wait_for_local_jobs(exc_value is not None)
        finally:
            await super().__aexit__(exc_type, exc_value, traceback)

    @property
    def work_unit_count(self) -> int:
        return len(self.work_units())

    def work_units(self):
        for unit_id in self._catalog.work_units(self.experiment_id):
            if unit_id not in self._work_units:
                unit = ClusterWorkUnit(self, unit_id)
                unit._submitted = True
                self._work_units[unit_id] = unit
        return dict(self._work_units)

    @property
    def experiment_id(self) -> int:
        return self._experiment_id

    @property
    def context(self) -> metadata.ClusterMetadataContext:
        return metadata.ClusterMetadataContext()


@functools.lru_cache()
def _load_vcsinfo() -> Optional[vcsinfo.VCS]:
    vcs = None

    try:
        vcs_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        vcs = vcsinfo.detect_vcs(vcs_root)
    except subprocess.SubprocessError:
        logging.debug("Failed to detect VCS info")

    return vcs


def create_experiment(
    experiment_title: str,
    project: Optional[str] = None,
    *,
    config: Optional[config_lib.Config] = None,
) -> ClusterExperiment:
    """Create a LXM3 experiment backed by the xm_cluster backend.
    Args:
        experiment_title: Title of the experiment.
        project: project that the experiment is launched in.
            If not set, a project name will be automatically deduced
            from the environment.
        config: Site and storage configuration. Defaults to the current LXM3 config.

    """
    config = config if config is not None else config_lib.default()
    vcs = _load_vcsinfo()
    if project is None:
        project = config.project() or (vcs.name if vcs is not None else None)
    return ClusterExperiment(experiment_title, vcs=vcs, config=config, project=project)


def get_experiment(
    experiment_id: int, *, config: Optional[config_lib.Config] = None
) -> ClusterExperiment:
    """Reopen recorded metadata without submitting, polling or replaying a launcher."""
    return ClusterExperiment("", config=config, _experiment_id=experiment_id)


def get_current_experiment():
    try:
        return core._current_experiment.get()
    except LookupError as e:
        raise RuntimeError(
            "get_current_experiment requires an experiment context"
        ) from e
