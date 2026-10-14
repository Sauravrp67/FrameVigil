"""Validate current VDD_IN module-power telemetry in a prepared experiment."""
import argparse
from _checkout import add_source_directory

add_source_directory()

from src.measurement.stages import validate_power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()
    validate_power(args.experiment)


if __name__ == "__main__":
    main()
