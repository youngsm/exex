import concurrent.futures
import datetime
import getpass
import json
import os
import re
import shlex
import socket
import threading
from pathlib import Path
from typing import List, Optional

import attr

from exex import xm
from exex.clusters import slurm
from exex.clusters import ssh
from exex.xm_cluster import array_job
from exex.xm_cluster import artifacts
from exex.xm_cluster import config as config_lib
from exex.xm_cluster import console
from exex.xm_cluster import executables
from exex.xm_cluster import executors
from exex.xm_cluster.execution import continuation as driver
from exex.xm_cluster.execution import job_script_builder


class SlurmJobScriptBuilder(job_script_builder.JobScriptBuilder[executors.Slurm]):
    ARRAY_TASK_ID = "SLURM_ARRAY_TASK_ID"
    ARRAY_TASK_OFFSET = 1
    JOB_SCRIPT_SHEBANG = "#!/usr/bin/bash -l"
    JOB_ENV_PATTERN = "^(SLURM_|CUDA_VISIBLE_DEVICES=)"

    def _create_entrypoint_commands(self, job, install_dir):
        options = job.executor.srun_options
        if options is None:
            return super()._create_entrypoint_commands(job, install_dir)
        # Capture each task's environment after srun assigns its rank and GPUs.
        # The shared controller environment must not overwrite those values.
        worker = (
            "EXEX_WORKDIR=$1\nEXEX_LINK_DIR=$2\n"
            f'EXEX_ENV_FILE=$(mktemp "{install_dir}/.environment.XXXXXXXXXX")\n'
            + self._create_environment_command(job, '"$EXEX_ENV_FILE"')
            + "\n"
            + super()._create_entrypoint_commands(
                job, install_dir, env_file='"$EXEX_ENV_FILE"'
            )
        )
        return (
            shlex.join(["srun", *options, "bash", "-e", "-c", worker, "exex-worker"])
            + f' "{install_dir}" "$EXEX_LINK_DIR"'
        )

    @classmethod
    def _is_gpu_requested(cls, executor: executors.Slurm) -> bool:
        return any(
            value and str(value) != "0"
            for key, value in executor.resources.items()
            if key.startswith("gpus")
        ) or any(
            item.split(":", 1)[0] == "gpu" and item.rsplit(":", 1)[-1] != "0"
            for item in str(executor.resources.get("gres", "")).split(",")
        )

    @classmethod
    def _create_job_script_prologue(cls, executable, executor: executors.Slurm) -> str:
        cmds = ['echo >&2 "INFO[$(basename "$0")]: Running on host $(hostname)"']

        for module in executor.modules:
            cmds.append(f"module load {shlex.quote(module)}")
        if cls._is_gpu_requested(executor):
            cmds.append(
                'echo >&2 "INFO[$(basename "$0")]: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
            )

        return "\n".join(cmds)

    @classmethod
    def _create_job_script_header(
        cls,
        executable: executables.AppBundle,
        executor: executors.Slurm,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        image = executable.container_image
        if (
            image is not None
            and image.image_type == executables.ContainerImageType.SHIFTER
        ):
            resources = {**executor.resources, "image": image.name}
            options = executor.container_options or executors.ShifterOptions()
            if options.modules:
                resources["module"] = ",".join(options.modules)
            executor = attr.evolve(executor, resources=resources)
        job_header = header_from_executor(
            job_name, executor, num_array_tasks, job_log_dir
        )
        return job_header

    def build(
        self,
        job: job_script_builder.JobType,
        job_name: str,
        job_log_dir: str,
        *,
        outputs=None,
        inputs=None,
        continuation=None,
    ) -> str:
        assert isinstance(job.executor, executors.Slurm)
        assert isinstance(job.executable, executables.AppBundle)
        return super().build(
            job,
            job_name,
            job_log_dir,
            outputs=outputs,
            inputs=inputs,
            continuation=continuation,
            attached=job.executor.mode == "salloc",
        )


class SlurmHandle:
    def __init__(self, job_id: str, **record) -> None:
        self.job_id = job_id
        self.record = dict(backend="slurm", native_id=job_id, **record)


class AttachedHandle(SlurmHandle):
    def __init__(self, command, **record):
        super().__init__("", **record)
        self.future = concurrent.futures.Future()

        def run():
            try:
                ssh.run(
                    command,
                    hostname=record["hostname"],
                    username=record["username"],
                    stdout=None,
                )
            except BaseException as error:
                self.future.set_exception(error)
            else:
                self.future.set_result(None)

        threading.Thread(target=run, name="exex-salloc", daemon=True).start()


class SlurmClient:
    builder_cls: type[SlurmJobScriptBuilder] = SlurmJobScriptBuilder

    def __init__(
        self,
        settings: config_lib.ClusterSettings,
        artifact_store: artifacts.ArtifactStore,
    ) -> None:
        self._settings = settings
        self._artifact_store = artifact_store

        self._cluster = slurm.SlurmCluster(
            hostname=self._settings.hostname, username=self._settings.user
        )

    @property
    def artifact_store(self):
        return self._artifact_store

    def launch(
        self,
        job_name: str,
        job: job_script_builder.JobType,
        *,
        outputs=None,
        inputs=None,
        continuation=None,
    ):
        job_name = re.sub("\\W", "_", job_name)
        job_log_dir = job_script_builder.job_log_path(job_name)
        self._artifact_store.ensure_dir(job_log_dir)
        job_log_dir = self._artifact_store.normalize_path(job_log_dir)
        builder = self.builder_cls()
        job_script_content = builder.build(
            job,
            job_name,
            job_log_dir,
            outputs=outputs,
            inputs=inputs,
            continuation=continuation,
        )

        if isinstance(job, array_job.ArrayJob):
            num_jobs = len(job.env_vars)
        else:
            num_jobs = 1
        log_directory = job.executor.log_directory or job_log_dir
        if not os.path.isabs(log_directory):
            log_directory = (
                self._artifact_store.filesystem.abspath(log_directory)
                if self._settings.hostname
                else os.path.abspath(log_directory)
            )
        managed = continuation is not None or job.executor.mode == "salloc"
        if managed:
            job_script_content = self._managed_script(
                job,
                job_name,
                job_log_dir,
                log_directory,
                job_script_content,
                inputs=inputs,
                continuation=continuation,
            )
        job_script_path = self._artifact_store.put_text(
            job_script_content, job_script_builder.job_script_path(job_name)
        )
        record = dict(
            hostname=self._settings.hostname or socket.gethostname(),
            username=self._settings.user
            if self._settings.hostname
            else getpass.getuser(),
            job_name=job_name,
            log_directory=log_directory,
            links_directory=os.path.join(job_log_dir, "links"),
            script_path=job_script_path,
            artifact_directory=os.path.join(job_log_dir, "artifacts")
            if outputs
            else None,
            execution_directory=job_log_dir if managed else None,
        )
        console.info(f"Launching {num_jobs} job on {self._settings.hostname}")
        if job.executor.mode == "salloc":
            # None denotes an on-site process, not an SSH back into this host.
            handle = AttachedHandle(
                ["bash", job_script_path],
                **{**record, "hostname": self._settings.hostname},
            )
            handle.record.update(record)
        else:
            job_id = self._cluster.launch(job_script_path)
            console.info(f"Successfully launched job {job_id}")
            self._artifact_store.put_text(str(job_id), f"jobs/{job_name}/job_id")
            handle = SlurmHandle(job_id, **record)
        console.info(f"Logs: {job_log_dir}; script: {job_script_path}")
        return [handle]

    def _managed_script(
        self, job, name, directory, logs, payload, *, inputs, continuation
    ):
        store = self._artifact_store
        root = job_script_builder.job_path(name)
        payload_path = store.put_text(payload, root + "/payload.sh")
        driver_path = store.put_text(
            Path(driver.__file__).read_text(), root + "/driver.py"
        )
        header = self.builder_cls._create_job_script_header(
            job.executable, job.executor, None, directory, name
        )
        options = [
            arg
            for line in header.splitlines()
            for arg in shlex.split(line.removeprefix("#SBATCH "))
        ]
        salloc_options = [
            arg
            for arg in options
            if not arg.startswith(
                (
                    "--output=",
                    "--error=",
                    "--signal=",
                    "--requeue",
                    "--no-requeue",
                    "--open-mode=",
                )
            )
        ]
        step_options = [
            arg
            for arg in options
            if arg.startswith(
                ("--gpus", "--gres=", "--cpus-per-task=", "--cpus-per-gpu=")
            )
        ]
        config_path = store.put_text(
            json.dumps(
                dict(
                    directory=directory,
                    log_directory=logs,
                    payload=payload_path,
                    job_name=name,
                    continuation=continuation,
                    inputs=inputs or {},
                    salloc_options=salloc_options,
                    step_options=step_options,
                )
            ),
            root + "/driver.json",
        )
        mode = "batch" if job.executor.mode == "sbatch" else "attached"
        if mode == "batch":
            header += f"\n#SBATCH --requeue\n#SBATCH --signal=B:USR1@{continuation['pause_before']}\n#SBATCH --open-mode=append"
        else:
            header = ""
        return (
            self.builder_cls.JOB_SCRIPT_SHEBANG
            + "\n"
            + header
            + "\nset -e\n"
            + "exec python3 "
            + shlex.join([driver_path, mode, config_path])
            + "\n"
        )


def client(
    *, settings: config_lib.ClusterSettings, project: Optional[str]
) -> SlurmClient:
    artifact_store = job_script_builder.create_artifact_store(
        settings=settings,
        project=project,
        use_openssh=True,
    )
    return SlurmClient(settings, artifact_store)


def _slurm_job_predicate(job):
    if isinstance(job, xm.Job):
        return isinstance(job.executor, executors.Slurm)
    elif isinstance(job, array_job.ArrayJob):
        return isinstance(job.executor, executors.Slurm)
    else:
        raise ValueError(f"Unexpected job type: {type(job)}")


async def launch(
    job_name: str,
    job,
    *,
    config: config_lib.Config,
    project: Optional[str],
    outputs=None,
    inputs=None,
    continuation=None,
) -> List[SlurmHandle]:
    jobs = job_script_builder.flatten_job(job)
    jobs = [job for job in jobs if _slurm_job_predicate(job)]

    if not jobs:
        return []

    if len(jobs) > 1:
        raise ValueError(
            "Cannot launch a job group with multiple jobs as a single job."
        )

    settings = config.cluster_settings(jobs[0].executor.cluster)
    return client(settings=settings, project=project).launch(
        job_name, jobs[0], outputs=outputs, inputs=inputs, continuation=continuation
    )


def _format_slurm_time(duration: datetime.timedelta) -> str:
    # See
    # https://github.com/SchedMD/slurm/blob/master/src/common/parse_time.c#L786
    days = duration.days
    seconds = int(duration.seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return "{:02}-{:02}:{:02}:{:02}".format(duration.days, hours, minutes, seconds)
    else:
        return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)


def header_from_executor(
    job_name: str,
    executor: executors.Slurm,
    num_array_tasks: Optional[int],
    job_log_dir: str,
) -> str:
    header = []

    header.append(f"#SBATCH --job-name={job_name}")
    # TODO(yl): Only one task is supported for now.
    if (
        "ntasks" not in executor.resources
        and "ntasks-per-node" not in executor.resources
    ):
        header.append("#SBATCH --ntasks=1")

    for resource, value in executor.resources.items():
        if value:
            header.append(f"#SBATCH --{resource}={shlex.quote(str(value))}")

    if executor.walltime is not None:
        duration = executor.walltime
        header.append(f"#SBATCH --time={_format_slurm_time(duration)}")

    log_directory = executor.log_directory or job_log_dir
    if num_array_tasks is not None:
        stdout = os.path.join(log_directory, "%x-%A_%a.out")
    else:
        stdout = os.path.join(log_directory, "%x-%j.out")

    header.append(f"#SBATCH --output={shlex.quote(stdout)}")

    if executor.exclusive:
        header.append("#SBATCH --exclusive")

    if executor.partition:
        header.append(f"#SBATCH --partition={executor.partition}")

    if num_array_tasks is not None:
        array_spec = f"1-{num_array_tasks}"
        header.append(f"#SBATCH --array={array_spec}")

    # Skip requested header directives
    header = list(
        filter(
            lambda line: not any(skip in line for skip in executor.skip_directives),
            header,
        )
    )

    for line in executor.extra_directives:
        if not line.startswith("#SBATCH"):
            line = "#SBATCH " + line
        header.append(line)

    return "\n".join(header)
