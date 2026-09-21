import enum
from typing import Dict, Optional

import attr

from lxm3 import xm


class ContainerImageType(enum.Enum):
    DOCKER = "docker"
    SINGULARITY = "singularity"
    SHIFTER = "shifter"


@attr.s(auto_attribs=True)
class ContainerImage:
    name: str
    image_type: ContainerImageType


@attr.s(auto_attribs=True)
class AppBundle(xm.Executable):
    entrypoint_command: str
    resource_uri: str
    args: xm.SequentialArgs = attr.Factory(xm.SequentialArgs)
    env_vars: Dict[str, str] = attr.Factory(dict)
    container_image: Optional[ContainerImage] = None
    # Set by packaging; handwritten bundles retain their existing behavior.
    _target: Optional[tuple] = attr.field(default=None, init=False, repr=False)
