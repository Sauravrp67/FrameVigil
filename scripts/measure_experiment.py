"""Run a prepared GPU campaign with optional clock locking and power telemetry."""

import argparse
from _checkout import add_source_directory

add_source_directory()

from adaptive_video.measurement.stages import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument(
        '--power',
        choices=['none', 'vdd_in'],
        default='none',
        help='Energy backend; vdd_in requires a passing power-validation run',
    )
    parser.add_argument(
        '--lock-clocks',
        action='store_true',
        help='Lock clocks using Jetson utilities or nvidia-smi; clean up on exit',
    )
    args = parser.parse_args()
    measure(args.experiment, power_backend=args.power, lock_clocks=args.lock_clocks)


if __name__ == "__main__":
    main()
