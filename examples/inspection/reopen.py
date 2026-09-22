"""Inspect a recorded experiment in a separate process; no launcher is imported."""

import argparse

from exex import xm_cluster as xc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-unit", type=int)
    parser.add_argument("--task", type=int)
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    experiment = xc.get_experiment(
        args.experiment_id, config=xc.Config.from_file(args.config)
    )
    units = experiment.work_units()
    if args.work_unit is not None:
        units = {args.work_unit: units[args.work_unit]}
    for unit_id, unit in units.items():
        status = unit.get_status()
        print(f"WorkUnit {unit_id}: {status.state} ({status.message})")
        if args.logs:
            print(unit.get_logs(task=args.task, tail=args.tail), end="")


if __name__ == "__main__":
    main()
