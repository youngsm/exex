"""Frozen input references and explicit same-endpoint reuse, without transfers."""

import getpass
import socket
from pathlib import PurePosixPath

from lxm3.xm_cluster import executors
from lxm3.xm_cluster import inspection


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


def check_site(inputs, executor, config):
    if not inputs:
        return
    if isinstance(executor, executors.Local):
        endpoint = _endpoint()
    else:
        settings = config.cluster_settings(executor.cluster)
        endpoint = _endpoint(
            settings.hostname, settings.user if settings.hostname else None
        )
    if any(
        {key: binding[key] for key in endpoint} != endpoint
        for binding in inputs.values()
    ):
        raise ValueError(
            "Inputs must use the same host/user endpoint as the executor; "
            "cross-site artifact transfer is not implemented"
        )
