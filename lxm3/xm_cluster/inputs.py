"""Frozen input references and destination staging through the author host."""

import getpass
import socket
import tempfile
from pathlib import Path
from pathlib import PurePosixPath

from lxm3.clusters import ssh
from lxm3.xm_cluster import config as config_lib
from lxm3.xm_cluster import executors
from lxm3.xm_cluster import inspection
from lxm3.xm_cluster.execution import job_script_builder
from lxm3.xm_cluster.execution.artifact_io import digest_file


def _endpoint(hostname=None, username=None):
    hostname = hostname or socket.gethostname()
    if inspection._hostname(dict(hostname=hostname, username=username)) is None:
        return dict(hostname=socket.gethostname(), username=getpass.getuser())
    return dict(hostname=hostname, username=username)


def declarations(inputs):
    result = {}
    for name, artifact in (inputs or {}).items():
        if PurePosixPath(name).parts != (name,) or name == "..":
            raise ValueError("Input name must be a single path component: " + name)
        result[name] = dict(
            id=artifact.id,
            archive_path=artifact._archive_path,
            **_endpoint(artifact._hostname, artifact._username),
        )
    return result


def stage(inputs, executor, config, project):
    """Keep same-site references; copy foreign archives before job submission."""
    if not inputs:
        return inputs
    settings = (
        config_lib.ClusterSettings(
            {"storage": {"staging": config.local_settings().storage_root}}
        )
        if isinstance(executor, executors.Local)
        else config.cluster_settings(executor.cluster)
    )
    endpoint = _endpoint(
        settings.hostname, settings.user if settings.hostname else None
    )
    result = dict(inputs)
    store = None
    for name, binding in inputs.items():
        if {key: binding[key] for key in endpoint} == endpoint:
            continue
        if store is None:
            store = job_script_builder.create_artifact_store(
                project=project, settings=settings, use_openssh=True
            )
        relative = "inputs/" + binding["id"] + ".tar"
        if not store.exists(relative):
            with tempfile.TemporaryDirectory(prefix="lxm-input-") as temporary:
                hostname = inspection._hostname(binding)
                archive = Path(binding["archive_path"])
                if hostname:
                    archive = Path(temporary) / "artifact.tar"
                    ssh.OpenSSHFileSystem(hostname, binding["username"]).get_file(
                        binding["archive_path"], str(archive)
                    )
                if digest_file(archive) != binding["id"]:
                    raise ValueError(
                        "Input archive does not match its content identity"
                    )
                store.put_file(str(archive), relative)
        result[name] = dict(
            id=binding["id"],
            archive_path=store.normalize_path(relative),
            **endpoint,
        )
    return result
