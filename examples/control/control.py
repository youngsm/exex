"""Wait for a recorded WorkUnit, optionally requesting Slurm cancellation first."""

import argparse
import asyncio

from lxm3 import xm_cluster as xc


async def main(args):
    experiment = xc.get_experiment(
        args.experiment_id, config=xc.Config.from_file(args.config)
    )
    unit = experiment.work_units()[args.work_unit]
    if args.stop:
        unit.stop(message="Explicit cancellation from the control example")
    # Timeout cancels this wait, not the execution. XM errors retain their meaning.
    await asyncio.wait_for(unit.wait_until_complete(), timeout=args.timeout)
    print(f"WorkUnit {unit.work_unit_id}: completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-unit", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--stop", action="store_true")
    asyncio.run(main(parser.parse_args()))
