#!/usr/bin/env python3
import argparse
import errno
import importlib
import os
import sys

from absl import app
from absl.flags import argparse_flags
from rich.console import Console
from rich.table import Table

import lxm3
from lxm3 import xm
from lxm3 import xm_cluster as xc


def version(_):
    print(f"lxm3 {lxm3.__version__}")


def register_version_parser(parsers: argparse._SubParsersAction):
    version_parser = parsers.add_parser(
        "version",
        help="Print version.",
        inherited_absl_flags=None,  # type: ignore
    )
    version_parser.set_defaults(command=version)


def launch(args):
    launch_script = args.launch_script
    if not os.path.exists(launch_script):
        raise OSError(errno.ENOENT, f"File not found: {launch_script}")
    sys.path.insert(0, os.path.abspath(os.path.dirname(launch_script)))
    launch_module, _ = os.path.splitext(os.path.basename(launch_script))
    m = importlib.import_module(launch_module)
    argv = [launch_script, "--xm_launch_script={}".format(launch_script)] + args.args
    app.run(m.main, argv=argv)
    sys.path.pop(0)


def register_launch_parser(parsers: argparse._SubParsersAction):
    launch_parser = parsers.add_parser(
        "launch",
        help="Launch experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        inherited_absl_flags=None,  # type: ignore
        epilog=r"""
examples:

  Launch experiment defined in "launcher.py"
  and pass extra args "--task 1" to "launcher.py".

    lxm3 launch launcher.py -- --task 1
""",
    )
    launch_parser.add_argument(
        "launch_script",
        metavar="LAUNCH_SCRIPT",
        # nargs=1,
        help="Path to launch script.",
    )
    launch_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="ARGS",
        help="Additional arguments to pass to launch script.",
    )
    launch_parser.set_defaults(command=launch)


def experiments(args):
    table = Table("EXPERIMENT", "PROJECT", "TITLE", box=None)
    table.columns[0].no_wrap = True
    for experiment in xc.list_experiments(project=args.project):
        table.add_row(
            str(experiment.experiment_id),
            experiment._project or "—",
            experiment._experiment_title,
        )
    Console(markup=False, highlight=False).print(table)


def _work_units(args):
    units = xc.get_experiment(args.experiment_id).work_units()
    if args.work_unit_id is None:
        return units.values()
    try:
        return (units[args.work_unit_id],)
    except KeyError:
        raise xm.NotFoundError(
            f"WorkUnit {args.work_unit_id} in experiment {args.experiment_id}"
        ) from None


def status(args):
    table = Table("WORK_UNIT", "STATE", "MESSAGE", box=None)
    for unit in _work_units(args):
        observation = unit.get_status()
        table.add_row(str(unit.work_unit_id), observation.state, observation.message)
    Console(markup=False, highlight=False).print(table)


def logs(args):
    (unit,) = _work_units(args)
    print(unit.get_logs(task=args.task, tail=args.tail), end="")


def stop(args):
    (unit,) = _work_units(args)
    unit.stop()


def register_management_parsers(parsers: argparse._SubParsersAction):
    experiments_parser = parsers.add_parser(
        "experiments", help="List saved experiments without contacting schedulers."
    )
    experiments_parser.add_argument("--project", help="Filter by exact project name.")
    experiments_parser.set_defaults(command=experiments)

    status_parser = parsers.add_parser("status", help="Query WorkUnit status.")
    status_parser.add_argument("experiment_id", type=int)
    status_parser.add_argument("work_unit_id", type=int, nargs="?")
    status_parser.set_defaults(command=status)

    logs_parser = parsers.add_parser("logs", help="Read a bounded tail of job output.")
    logs_parser.add_argument("experiment_id", type=int)
    logs_parser.add_argument("work_unit_id", type=int)
    logs_parser.add_argument("--task", type=int, help="Zero-based array task index.")
    logs_parser.add_argument("--tail", type=int, default=200, help="Number of lines.")
    logs_parser.set_defaults(command=logs)

    stop_parser = parsers.add_parser("stop", help="Request Slurm cancellation.")
    stop_parser.add_argument("experiment_id", type=int)
    stop_parser.add_argument("work_unit_id", type=int)
    stop_parser.set_defaults(command=stop)


def _parse_flags(argv):
    parser = argparse_flags.ArgumentParser(description="lxm3 experiment scheduler.")
    parser.set_defaults(command=lambda _: parser.print_help())

    subparsers = parser.add_subparsers()

    register_version_parser(subparsers)
    register_launch_parser(subparsers)
    register_management_parsers(subparsers)

    args = parser.parse_args(argv[1:])
    return args


def main(args):
    args.command(args)


def entrypoint():
    app.run(main, flags_parser=_parse_flags)


if __name__ == "__main__":
    entrypoint()
