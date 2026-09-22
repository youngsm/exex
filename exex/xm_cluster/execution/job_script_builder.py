import abc
import json
import os
import shlex
import textwrap
from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar, Union, cast

import attr
import fsspec
from fsspec.implementations import sftp

from exex import xm
from exex._vendor.xmanager.xm.utils import ARG_ESCAPER
from exex.clusters.ssh import OpenSSHFileSystem
from exex.xm_cluster import array_job
from exex.xm_cluster import artifacts
from exex.xm_cluster import config as config_lib
from exex.xm_cluster import executables
from exex.xm_cluster import executors
from exex.xm_cluster.execution import artifact_io

JobType = Union[xm.Job, array_job.ArrayJob]
ExecutorType = TypeVar("ExecutorType", bound=executors.SupportsContainer)


class JobScriptBuilder(abc.ABC, Generic[ExecutorType]):
    ARRAY_TASK_ID: str
    ARRAY_TASK_OFFSET: int
    JOB_SCRIPT_SHEBANG: str = "#!/usr/bin/env bash"
    JOB_ENV_PATTERN = None

    CONTAINER_WORKDIR: str = "/run/exex/workdir"
    JOB_PARAM_NAME: str = "job-param.sh"
    CONTAINER_JOB_PARAM_PATH: str = f"/tmp/{JOB_PARAM_NAME}"

    @classmethod
    @abc.abstractmethod
    def _create_job_script_header(
        cls,
        executable: executables.AppBundle,
        executor: ExecutorType,
        num_array_tasks: Optional[int],
        job_log_dir: str,
        job_name: str,
    ) -> str:
        """Create a job header"""

    @classmethod
    @abc.abstractmethod
    def _is_gpu_requested(cls, executor: ExecutorType) -> bool:
        """Infer GPU resource from executor"""

    @classmethod
    @abc.abstractmethod
    def _create_job_script_prologue(
        cls,
        executable: executables.AppBundle,
        executor: ExecutorType,
    ) -> str:
        """Generate backend specific setup commands"""

    def _create_job_args_script(self, job: JobType) -> str:
        executable = job.executable
        assert isinstance(executable, executables.AppBundle)
        if isinstance(job, xm.Job):
            args = xm.merge_args(executable.args, job.args).to_list()
            env = {**executable.env_vars, **job.env_vars}
            return _create_job_args_script(args, env)
        elif isinstance(job, array_job.ArrayJob):
            args = [
                xm.merge_args(executable.args, per_task_args).to_list()
                for per_task_args in job.args
            ]
            env = [
                {**executable.env_vars, **per_task_envs}
                for per_task_envs in job.env_vars
            ]

            return _create_array_job_args_script(
                args, env, self.ARRAY_TASK_ID, self.ARRAY_TASK_OFFSET
            )
        else:
            raise ValueError(f"{type(job)} is not supported")

    def _create_install_commands(self, job: JobType, install_dir: str) -> str:
        executable = job.executable
        assert isinstance(executable, executables.AppBundle)
        job_args_script = self._create_job_args_script(job)
        if self.JOB_ENV_PATTERN:
            export_env_file_cmds = (
                f"printenv | {{ grep -E '{self.JOB_ENV_PATTERN}' || :; }} > "
                f'"{install_dir}/.environment"'
            )
        else:
            export_env_file_cmds = f'touch "{install_dir}/.environment"'
        extract_pkg_cmds = _get_extract_command(executable.resource_uri, install_dir)
        save_job_args_cmds = f"printf '%s\\n' {shlex.quote(job_args_script)} > \"{install_dir}/{self.JOB_PARAM_NAME}\""
        return "\n".join(
            [
                extract_pkg_cmds,
                save_job_args_cmds,
                export_env_file_cmds,
            ]
        )

    def _create_entrypoint_commands(self, job: JobType, install_dir: str) -> str:
        executable = job.executable
        if not isinstance(executable, executables.AppBundle):
            raise ValueError("Only Command executable is supported")
        executor = job.executor
        if not isinstance(executor, executors.SupportsContainer):
            raise TypeError("Executor should support container configuration")

        if (
            executable.container_image is not None
            and executable.container_image.image_type
            == executables.ContainerImageType.SHIFTER
        ):
            return " ".join(
                create_shifter_command(
                    image=executable.container_image.name,
                    options=executor.container_options or executors.ShifterOptions(),
                    install_dir=install_dir,
                    args=_rewrite_array_job_command(
                        f"./{self.JOB_PARAM_NAME}", executable.entrypoint_command
                    ),
                )
            )

        if executable.container_image is not None:
            image = executable.container_image.name
            image_type = executable.container_image.image_type

            if image_type == executables.ContainerImageType.SINGULARITY:
                get_container_cmd = create_singularity_command
                singularity_options = (
                    executor.container_options or executors.SingularityOptions()
                )
                bind_mounts = [
                    BindMount(src, dst) for src, dst in singularity_options.bind.items()
                ]
                runtime_options = [*singularity_options.extra_options]
            elif image_type == executables.ContainerImageType.DOCKER:
                get_container_cmd = create_docker_command
                docker_options = executor.container_options or executors.DockerOptions()
                bind_mounts = [
                    BindMount(src, dst) for src, dst in docker_options.volumes.items()
                ]
                runtime_options = [*docker_options.extra_options]
            else:
                assert False

            bind_mounts.extend(
                [
                    BindMount(
                        xm.ShellSafeArg(f'"{install_dir}"'), self.CONTAINER_WORKDIR
                    ),
                    BindMount(
                        xm.ShellSafeArg(f'"{install_dir}/{self.JOB_PARAM_NAME}"'),
                        self.CONTAINER_JOB_PARAM_PATH,
                        read_only=True,
                    ),
                    BindMount(xm.ShellSafeArg('"$EXEX_LINK_DIR"'), "/run/exex/links"),
                ]
            )

            entrypoint = get_container_cmd(
                image=image,
                args=_rewrite_array_job_command(
                    self.CONTAINER_JOB_PARAM_PATH, executable.entrypoint_command
                ),
                bind_mounts=bind_mounts,
                env_vars={},
                options=runtime_options,
                working_dir=self.CONTAINER_WORKDIR,
                use_gpu=self._is_gpu_requested(executor),
                env_file=xm.ShellSafeArg(f'"{install_dir}/.environment"'),
            )

        else:
            entrypoint = _rewrite_array_job_command(
                f"./{self.JOB_PARAM_NAME}", executable.entrypoint_command
            )

        return " ".join(entrypoint)

    def build(
        self,
        job: Union[xm.Job, array_job.ArrayJob],
        job_name: str,
        job_log_dir: str,
        *,
        outputs=None,
        inputs=None,
        continuation=None,
        attached=False,
    ) -> str:
        executable = job.executable
        if not isinstance(executable, executables.AppBundle):
            raise TypeError("Only AppBundle is supported")
        executor = cast(executors.SupportsContainer, job.executor)

        image_type = (
            executable.container_image.image_type
            if executable.container_image
            else None
        )
        options_type = {
            None: type(None),
            executables.ContainerImageType.SINGULARITY: executors.SingularityOptions,
            executables.ContainerImageType.DOCKER: executors.DockerOptions,
            executables.ContainerImageType.SHIFTER: executors.ShifterOptions,
        }[image_type]
        if executor.container_options is not None and not isinstance(
            executor.container_options, options_type
        ):
            raise TypeError(
                f"container_options={type(executor.container_options).__name__} "
                f"does not match the executable's {image_type.value if image_type else 'host'} runtime"
            )

        num_array_tasks = None
        if isinstance(job, array_job.ArrayJob):
            num_array_tasks = len(job.args)

        header = self._create_job_script_header(
            executable, executor, num_array_tasks, job_log_dir, job_name
        )
        prologue = self._create_job_script_prologue(executable, executor)
        install_dir = "$EXEX_WORKDIR"
        install_cmds = self._create_install_commands(job, install_dir)
        entrypoint_cmds = self._create_entrypoint_commands(job, install_dir)
        task = (
            f"$(({self.ARRAY_TASK_ID} - {self.ARRAY_TASK_OFFSET}))"
            if num_array_tasks is not None
            else "0"
        )
        managed = continuation is not None or attached
        link_directory = (
            '"$EXEX_ATTEMPT_DIR/links/0"'
            if managed
            else shlex.quote(os.path.join(job_log_dir, "links")) + f'/"{task}"'
        )
        link_path = (
            shlex.quote("/run/exex/links/links.json")
            if image_type
            in {
                executables.ContainerImageType.SINGULARITY,
                executables.ContainerImageType.DOCKER,
            }
            else link_directory + "/links.json"
        )
        install_cmds += (
            f'\nEXEX_LINK_DIR={link_directory}\nmkdir -p -- "$EXEX_LINK_DIR"\n'
        )
        if managed:
            # Shifter keeps host paths but may clear the inherited environment.
            # Freeze the attempt path into the parameter script after user env.
            install_cmds += """python3 - "$EXEX_WORKDIR/job-param.sh" "$EXEX_ATTEMPT_DIR" <<'EXEX_ATTEMPT_ENV'
import shlex, sys
with open(sys.argv[1], "a") as stream:
    stream.write("\\nexport EXEX_ATTEMPT_DIR=" + shlex.quote(sys.argv[2]) + "\\n")
EXEX_ATTEMPT_ENV
"""
        link_export = shlex.quote(f"export EXEX_LINKS_FILE={link_path}")
        install_cmds += (
            f'''printf '%s\\n' {link_export} >> "$EXEX_WORKDIR/job-param.sh"'''
        )
        if continuation is not None:
            for variable, name in (
                ("EXEX_PAUSE_REQUEST", "pause-request"),
                ("EXEX_PAUSE_READY", "paused"),
            ):
                path = link_path.rsplit("/", 1)[0] + "/" + name
                line = shlex.quote(f"export {variable}={path}")
                install_cmds += (
                    f'''\nprintf '%s\\n' {line} >> "$EXEX_WORKDIR/job-param.sh"'''
                )
        for kind, bindings in (("INPUT", inputs), ("OUTPUT", outputs)):
            if bindings or (kind == "INPUT" and continuation is not None):
                variable = f"EXEX_{kind}_DIR"
                install_cmds += f'\n{variable}="$(mktemp -d "$EXEX_WORKDIR/exex-{kind.lower()}.XXXXXXXXXX")"\n'
                install_cmds += f'''printf '\\nexport {variable}="$PWD/%s"\\n' "${{{variable}##*/}}" >> "$EXEX_WORKDIR/job-param.sh"'''
        if inputs or continuation is not None:
            install_cmds += _artifact_command(
                "prepare", '"$EXEX_INPUT_DIR"', None if managed else inputs
            )
        if outputs:
            destination = (
                '"$EXEX_ATTEMPT_DIR/artifacts/0"'
                if managed
                else shlex.quote(os.path.join(job_log_dir, "artifacts")) + f'/"{task}"'
            )
            capture = _artifact_command(
                "capture", f'"$EXEX_OUTPUT_DIR" {destination}', outputs
            )
            if continuation is not None:
                checkpoint = continuation["checkpoint"]
                entrypoint_cmds += '\nif test -e "$EXEX_LINK_DIR/paused"; then\n'
                entrypoint_cmds += _artifact_command(
                    "capture",
                    f'"$EXEX_OUTPUT_DIR" {destination}',
                    {checkpoint: outputs[checkpoint]},
                )
                entrypoint_cmds += "\nelse\n" + capture + "\nfi\n"
            else:
                entrypoint_cmds += capture
        workdir_cmds = 'EXEX_WORKDIR="$(mktemp -d)"'
        workdir_root = getattr(executor, "workdir_root", None)
        if workdir_root is not None:
            root = shlex.quote(workdir_root)
            workdir_cmds = (
                f"mkdir -p -- {root}\n"
                f'EXEX_WORKDIR="$(cd -- {root} && mktemp -d "$PWD/exex.XXXXXXXXXX")"'
            )
        return _JOB_SCRIPT_TEMPLATE % {
            "shebang": self.JOB_SCRIPT_SHEBANG,
            "header": header,
            "workdir": workdir_cmds,
            "install": install_cmds,
            "prologue": prologue,
            "entrypoint": entrypoint_cmds,
        }


def _artifact_command(operation, paths, bindings):
    helper = Path(artifact_io.__file__).read_text()
    bindings = (
        shlex.quote(json.dumps(bindings))
        if bindings is not None
        else '"$EXEX_ATTEMPT_INPUTS"'
    )
    return (
        f"\npython3 - {operation} {paths} {bindings} <<'EXEX_ARTIFACT_PY'\n"
        + helper
        + "\nEXEX_ARTIFACT_PY\n"
    )


_JOB_SCRIPT_TEMPLATE = """\
%(shebang)s
%(header)s
set -e

%(workdir)s
cleanup() {
  echo >& 2 "DEBUG[$(basename "$0")] Cleaning up $EXEX_WORKDIR"
  rm -rf "$EXEX_WORKDIR"
}
trap cleanup EXIT
cd "$EXEX_WORKDIR"
%(install)s

%(prologue)s

%(entrypoint)s
"""


@attr.s(auto_attribs=True)
class BindMount:
    path: Union[str, xm.ShellSafeArg]
    mount_path: str
    read_only: bool = False


def create_singularity_command(
    *,
    image: str,
    args: List[str],
    env_vars: Dict[str, str],
    options: List[str],
    bind_mounts: List[BindMount],
    use_gpu: bool,
    working_dir: str,
    env_file: Union[str, xm.ShellSafeArg],
) -> List[str]:
    cmd = ["singularity", "exec"]

    for mount in bind_mounts:
        bm = f"{ARG_ESCAPER(mount.path)}:{shlex.quote(mount.mount_path)}"
        if mount.read_only:
            bm += ":ro"
        cmd.append(f"--bind={bm}")

    for key, value in env_vars.items():
        cmd.append(f"--env={shlex.quote(key + '=' + value)}")

    cmd.extend(map(shlex.quote, options))

    if use_gpu:
        cmd.append("--nv")

    cmd.append(f"--pwd={shlex.quote(working_dir)}")
    cmd.append(f"--env-file={ARG_ESCAPER(env_file)}")

    cmd.extend([shlex.quote(image), *args])

    return cmd


def create_docker_command(
    *,
    image: str,
    args: List[str],
    env_vars: Dict[str, str],
    bind_mounts: List[BindMount],
    options: List[str],
    working_dir: str,
    use_gpu: bool,
    env_file: Union[str, xm.ShellSafeArg],
) -> List[str]:
    cmd = ["docker", "run", "--rm"]

    for mount in bind_mounts:
        mount_spec = f"type=bind,source={ARG_ESCAPER(mount.path)},target={shlex.quote(mount.mount_path)}"
        if mount.read_only:
            mount_spec += ",readonly"
        cmd.append(f"--mount={mount_spec}")

    for k, v in env_vars.items():
        cmd.append(f"--env={shlex.quote(k + '=' + v)}")

    if use_gpu:
        cmd.extend(
            [
                "--runtime=nvidia",
                '--env=NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"',
            ]
        )

    cmd.extend(map(shlex.quote, options))
    cmd.append(f"--workdir={shlex.quote(working_dir)}")
    cmd.append(f"--env-file={ARG_ESCAPER(env_file)}")
    cmd.extend([shlex.quote(image), *args])

    return cmd


def create_shifter_command(
    *, image: str, options: executors.ShifterOptions, install_dir: str, args: List[str]
) -> List[str]:
    # Shifter mounts directories, not the per-job parameter file used by the
    # other runtimes. Source and job-param.sh stay together on a visible path.
    cmd = ["shifter", f"--image={shlex.quote(image)}"]
    if options.modules:
        cmd.append(f"--module={shlex.quote(','.join(options.modules))}")
    cmd.extend(
        f"--volume={shlex.quote(src + ':' + dst)}" for src, dst in options.bind.items()
    )
    cmd.extend(map(shlex.quote, options.extra_options))
    cmd.extend(
        [
            f'--workdir="{install_dir}"',
            f'--env-file="{install_dir}/.environment"',
            "--",
            *args,
        ]
    )
    return cmd


def _rewrite_array_job_command(array_script_path: str, cmd: str) -> List[str]:
    return [
        "sh",
        "-e",
        "-c",
        shlex.quote(f'. {shlex.quote(array_script_path)}; {cmd} "$@"'),
    ]


def _get_extract_command(archive: str, directory: str) -> str:
    if archive.endswith(".zip"):
        return f'unzip -q -d "{directory}" {shlex.quote(archive)}'
    elif archive.endswith(".tar"):
        return f'tar -C "{directory}" -xf {shlex.quote(archive)}'
    elif archive.endswith((".tar.gz", ".tgz")):
        return f'tar -C "{directory}" -xzf {shlex.quote(archive)}'
    else:
        raise ValueError(archive)


def _create_job_args_script(args: List[str], env: Dict[str, str]) -> str:
    return "\n".join(
        [
            "export EXEX_TASK_ID=0",
            _create_env_vars([env], "EXEX_TASK_ID", 0),
            _create_args([args], "EXEX_TASK_ID", 0),
        ]
    )


def _create_array_job_args_script(
    args: List[List[str]],
    env: List[Dict[str, str]],
    index_name: str,
    index_offset: int,
) -> str:
    return "\n".join(
        [
            textwrap.dedent(
                f"""\
                if [ -z ${{{index_name}+x}} ];
                then
                echo >&2 "ERROR[$0]: \\${index_name} is not set."
                exit 2
                fi"""
            ),
            _create_env_vars(env, index_name, index_offset),
            _create_args(args, index_name, index_offset),
        ]
    )


def _create_env_vars(
    env_vars_list: List[Dict[str, str]], index_name: str, index_offset: int
) -> str:
    """Select a task's literal environment, without shell re-evaluation."""
    if not any(env_vars_list):
        return ""
    lines = [f'case "${{{index_name}}}" in']
    for task_id, env in enumerate(env_vars_list, start=index_offset):
        lines.append(f"{task_id})")
        lines.extend(
            f"export {shlex.quote(key + '=' + value)}" for key, value in env.items()
        )
        lines.append(";;")
    return "\n".join([*lines, "*) exit 2;;", "esac"])


def _create_args(args_list: List[List[str]], index_name: str, index_offset: int) -> str:
    """Create the args list."""
    if not args_list:
        return ""
    lines = [f'case "${{{index_name}}}" in']
    for task_id, args in enumerate(args_list, start=index_offset):
        # SequentialArgs.to_list() has already escaped each argument.
        lines.append(f"{task_id}) set -- {' '.join(args)};;")
    return "\n".join([*lines, "*) exit 2;;", "esac"])


def job_path(job_name: str):
    return os.path.join("jobs", job_name)


def job_script_path(job_name: str):
    return os.path.join(job_path(job_name), "job.sh")


def job_log_path(job_name: str):
    return os.path.join("logs", job_name)


def flatten_job(
    job: Union[xm.JobGroup, array_job.ArrayJob],
) -> List[Union[xm.Job, array_job.ArrayJob]]:
    if isinstance(job, array_job.ArrayJob):
        return [job]  # type: ignore
    elif isinstance(job, xm.JobGroup):
        jobs = xm.job_operators.flatten_jobs(job)
        if len(jobs) > 1:
            raise NotImplementedError("JobGroup is not supported.")
        return jobs  # type: ignore
    else:
        raise NotImplementedError()


def create_artifact_store(
    *,
    project: Optional[str],
    settings: config_lib.ClusterSettings,
    use_openssh: bool = False,
):
    hostname = settings.hostname
    storage_root = settings.storage_root
    user = settings.user

    if hostname is None:
        filesystem = fsspec.filesystem("file")
        storage_root = os.path.abspath(os.path.expanduser(storage_root))
    elif use_openssh:
        filesystem = OpenSSHFileSystem(hostname, username=user)
        storage_root = filesystem.abspath(storage_root)
    else:
        filesystem = sftp.SFTPFileSystem(
            host=hostname, username=user, **settings.ssh_config
        )
        storage_root = filesystem.ftp.normalize(storage_root)

    return artifacts.ArtifactStore(
        filesystem, staging_directory=storage_root, project=project
    )
