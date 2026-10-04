"""Run a prepared CUDA GPU campaign with optional validated module-power telemetry."""
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
    args = parser.parse_args()
    measure(args.experiment, power_backend=args.power)


if __name__ == "__main__":
    main()
