"""JSON snapshots of concrete jobs, using only Exex's known value types.

No launcher imports or pickle. Argument values that XM renders as strings are
stored as those strings; ordering, keyword merging and shell expansion survive.
"""

import datetime
import enum
import json

import attr

from exex import xm
from exex.xm_cluster import executables
from exex.xm_cluster import executors
from exex.xm_cluster.array_job import ArrayJob
from exex.xm_cluster.executable_specs import FrozenSource
from exex.xm_cluster.requirements import JobRequirements

_TYPES = {
    cls.__name__: cls
    for cls in (
        xm.Job,
        ArrayJob,
        executables.AppBundle,
        executables.ContainerImage,
        executors.Local,
        executors.Slurm,
        executors.GridEngine,
        executors.SingularityOptions,
        executors.DockerOptions,
        executors.ShifterOptions,
        FrozenSource,
        xm.ShellSafeArg,
    )
}


def _argument(value):
    if isinstance(value, enum.Enum):
        return value.name
    if value is None or isinstance(value, (str, bool, int, float, xm.ShellSafeArg)):
        return value
    return str(value)


def _keyword(value):
    # Match SequentialArgs' repeated-flag rule, including empty sequences.
    if isinstance(value, (list, tuple)) and not any(
        type(item) in (list, tuple) for item in value
    ):
        return type(value)(_argument(item) for item in value)
    return _argument(value)


def _encode(value):
    if isinstance(value, xm.SequentialArgs):
        parts = [
            {item.name: _keyword(value._kwvalues[item.name])}
            if isinstance(item, xm.SequentialArgs._KeywordItem)
            else [_argument(item.value)]
            for item in value._items
        ]
        return {"type": "SequentialArgs", "parts": _encode(parts)}
    if isinstance(value, JobRequirements):
        return {
            "type": "JobRequirements",
            "resources": {
                key.name: amount for key, amount in value.task_requirements.items()
            },
            "location": value.location,
        }
    if isinstance(value, datetime.timedelta):
        return {
            "type": "timedelta",
            "parts": [value.days, value.seconds, value.microseconds],
        }
    if isinstance(value, executables.ContainerImageType):
        return {"type": "ContainerImageType", "value": value.value}
    if type(value) in _TYPES.values():
        return {
            "type": type(value).__name__,
            "fields": {
                field.name: _encode(getattr(value, field.name))
                for field in attr.fields(type(value))
            },
        }
    if isinstance(value, dict):
        # Tag mappings too, so user keys cannot be mistaken for type markers.
        return {"type": "dict", "items": [[k, _encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "dict":
        return {key: _decode(item) for key, item in value["items"]}
    if kind == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if kind == "SequentialArgs":
        return xm.merge_args(*_decode(value["parts"]))
    if kind == "JobRequirements":
        return JobRequirements(value["resources"], location=value["location"])
    if kind == "timedelta":
        return datetime.timedelta(*value["parts"])
    if kind == "ContainerImageType":
        return executables.ContainerImageType(value["value"])
    cls = _TYPES[kind]
    fields = {key: _decode(item) for key, item in value["fields"].items()}
    result = cls(
        **{
            field.alias: fields[field.name]
            for field in attr.fields(cls)
            if field.init and field.name in fields
        }
    )
    for field in attr.fields(cls):
        if not field.init:
            setattr(result, field.name, fields[field.name])
    return result


def dumps(job):
    return json.dumps(_encode(job), default=str)


def loads(record):
    return _decode(json.loads(record))
