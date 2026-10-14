"""Create an experiment directory containing the trial schedule."""

import argparse
from _checkout import add_source_directory

add_source_directory()

from src.measurement.stages import prepare
from src.project import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--dataset-manifest', required=True, help='New full-validation dataset inventory')
    parser.add_argument('--config', default=ROOT / 'configs/study.yaml')
    parser.add_argument('--repeats', type=int, help='Override experiment.repeats in YAML')
    args = parser.parse_args()
    prepare(args.experiment, args.dataset_manifest, args.repeats, config=args.config)


if __name__ == "__main__":
    main()
