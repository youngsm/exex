import datetime
import typing
from typing import Any, Dict, Optional, Protocol, Sequence, Union

import attr

from exex import xm
from exex.xm_cluster.requirements import JobRequirements


def _convert_time(
    time: Optional[Union[str, datetime.datetime, int]],
) -> Optional[datetime.timedelta]:
    if time is None:
        return None
    if isinstance(time, int):
        return datetime.timedelta(seconds=time)
    elif isinstance(time, datetime.timedelta):
        return time
    else:
        raise TypeError(
            f"Expect walltime to be type int (seconds) or datetime.timedelta, got {type(time)}"
        )


@attr.s(auto_attribs=True)
class SingularityOptions:
    """Options for singularity container.

    Args:
        bind: Mapping of the form ``{src: dst}``.
        User-bind path specification of the form ``-B <src>:<dst>``.
        extra_options: Extra commandline options to pass to singularity.

          When using the ``extra_options``, be aware of the following:
            1. You should not set a working directory as Exex configures those
            for the executable.
            2. You don't have to pass ``--nv`` for GPU jobs,
               exex will do it for you.
    """

    bind: Dict[str, str] = attr.Factory(dict)
    extra_options: Sequence[str] = attr.Factory(list)


@attr.s(auto_attribs=True)
class DockerOptions:
    """Options for a Docker container.

    Args:
        volumes: Bind-mounted volumes used by the docker runtime.
        extra_options: Extra commandline options to pass to Docker.
    """

    volumes: Dict[str, str] = attr.Factory(dict)
    extra_options: Sequence[str] = attr.Factory(list)


@attr.s(auto_attribs=True)
class ShifterOptions:
    """Native Shifter options, separate from environment-module loading.

    bind maps execution-host directories to existing container directories;
    append ``:ro`` to a destination for a read-only mount. modules selects
    Shifter modules, e.g. ["gpu"]. An empty list leaves the site's defaults.
    extra_options contains native flags such as --clearenv. Exex owns image
    selection and the working directory; do not override them here.
    """

    bind: Dict[str, str] = attr.Factory(dict)
    modules: Sequence[str] = attr.Factory(list)
    extra_options: Sequence[str] = attr.Factory(list)


_ContainerOptions = Union[SingularityOptions, DockerOptions, ShifterOptions]


@typing.runtime_checkable
class SupportsContainer(Protocol):
    container_options: Optional[_ContainerOptions]


@attr.s(auto_attribs=True)
class LocalSpec(xm.ExecutorSpec):
    """Spec for local execution."""


@attr.s(auto_attribs=True)
class Local(xm.Executor, SupportsContainer):
    """Local executor.

    Args:
        requirements: placeholder, no effect right now
        container_options: Options matching the executable's container runtime.
            None uses runtime defaults; a host executable accepts only None.
        workdir_root: Execution-host parent for a temporary unpacked directory.
            None uses the system temporary directory. Only the child is cleaned up.
    """

    requirements: JobRequirements = attr.Factory(JobRequirements)

    container_options: Optional[_ContainerOptions] = attr.field(
        default=None, kw_only=True
    )

    workdir_root: Optional[str] = attr.field(default=None, kw_only=True)

    @classmethod
    def Spec(cls) -> LocalSpec:
        return LocalSpec()


@attr.s(auto_attribs=True)
class GridEngineSpec(xm.ExecutorSpec):
    """Spec for SGE execution."""

    cluster: Optional[str] = None


@attr.s(auto_attribs=True)
class GridEngine(xm.Executor, SupportsContainer):
    """SGE executor.

    Attributes:
        requirements: placeholder, no effect right now.
        resources: Resources passed to qsub as `-l key=value`.
        parallel_environments: Parallel environments in the form of ``--pe <name> <slots>``.
        walltime: Maximum running time, ``-l h_rt=time``.
            When an ``int`` is used, this is interpreted as seconds.
            A ``datetime.timedelta`` can also be used.
        queue: queue to submit the job to: ``-q``.
        reserved: If set, use ``-R y``.
        log_directory: Log directory for stdout/stderr.
        merge_output: If False, log to separate files.
        shell: Shell to use, default ``/bin/bash``.
        project: ``-P``.
        account: ``-A``.
        modules: Modules to load before running the job.
            See https://modules.readthedocs.io/en/latest/
        max_parallel_tasks: ``-tc``.
        extra_directives: Extra directives to pass to ``qsub``.
        skip_directives: Directives to skip.
        container_options: Options matching the executable's container runtime.

    """

    # WARNING:
    # requirements are currently ignored as different SGE clusters
    # use different approaches for configurting these resources.
    # To configure, use resources for -l directives.
    # For UCL clusters, use the auto-configuration package from exex.contrib.
    requirements: JobRequirements = attr.Factory(JobRequirements)
    # Resources passed to qsub as -l
    resources: Dict[str, Any] = attr.Factory(dict)
    # Parallel environments in the form of --pe <name> <slots>
    parallel_environments: Dict[str, int] = attr.Factory(dict)
    # Maximum running time, -l h_rt
    walltime: Optional[datetime.timedelta] = attr.field(
        default=None, converter=_convert_time
    )

    # queue to submit the job to: -q
    queue: Optional[str] = None
    # If set, use -R y
    reserved: Optional[bool] = None
    # Log directory for stdout/stderr
    log_directory: Optional[str] = None
    # If False, log to separate files
    merge_output: bool = True
    shell: str = "/bin/bash"

    # -P
    project: Optional[str] = None
    # -A
    account: Optional[str] = None

    # Modules to load before running the job
    modules: Sequence[str] = attr.Factory(list)

    # -tc
    max_parallel_tasks: Optional[int] = None
    extra_directives: Sequence[str] = attr.Factory(list)
    skip_directives: Sequence[str] = attr.Factory(list)

    container_options: Optional[_ContainerOptions] = attr.field(
        default=None, kw_only=True
    )

    cluster: Optional[str] = attr.field(default=None, kw_only=True)

    def Spec(self) -> GridEngineSpec:
        return GridEngineSpec(cluster=self.cluster)


@attr.s(auto_attribs=True)
class SlurmSpec(xm.ExecutorSpec):
    """Spec for Slurm execution."""

    cluster: Optional[str] = None


@attr.s(auto_attribs=True)
class Slurm(xm.Executor, SupportsContainer):
    """Slurm executor.

    workdir_root selects the execution-host parent for a temporary unpacked
    directory. None uses the system temporary directory. Only the child is
    cleaned up; use a shared parent when workers on other nodes need the files.
    container_options configures the runtime selected by the executable's image;
    None uses runtime defaults. Host executables accept only None.
    mode selects detached sbatch submission or an attached owned salloc chain.
    It does not select QoS or borrow an existing allocation.
    srun_options=None runs the entrypoint once. A sequence launches the existing
    host/container command through srun with those native options. Preparation
    and artifact capture still run once; workdir_root must be shared by tasks.
    """

    requirements: JobRequirements = attr.Factory(JobRequirements)
    resources: Dict[str, Any] = attr.Factory(dict)
    walltime: Optional[datetime.timedelta] = attr.field(
        default=None, converter=_convert_time
    )

    container_options: Optional[_ContainerOptions] = attr.field(
        default=None, kw_only=True
    )

    log_directory: Optional[str] = None
    # Modules to load before running the job
    modules: Sequence[str] = attr.Factory(list)

    exclusive: bool = False
    partition: Optional[str] = None

    extra_directives: Sequence[str] = attr.Factory(list)
    skip_directives: Sequence[str] = attr.Factory(list)

    cluster: Optional[str] = attr.field(default=None, kw_only=True)
    workdir_root: Optional[str] = attr.field(default=None, kw_only=True)

    mode: str = attr.field(
        default="sbatch",
        kw_only=True,
        validator=attr.validators.in_(("sbatch", "salloc")),
    )

    srun_options: Optional[Sequence[str]] = attr.field(default=None, kw_only=True)

    def Spec(self) -> SlurmSpec:
        return SlurmSpec(cluster=self.cluster)
