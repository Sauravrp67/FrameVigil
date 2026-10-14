"""Generate a hashed inventory for all 200 validation videos"""

import argparse
from _checkout import add_source_directory

add_source_directory()

from src.evaluation.dataset import prepare_dataset_manifest
from src.project import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--config',
        default=ROOT / 'configs/study.yaml',
        help='YAML containing dataset image and annotation paths',
    )
    args = parser.parse_args()
    result = prepare_dataset_manifest(args.output, config=args.config)
    print(f"Prepared {len(result['videos'])} videos and {len(result['frames'])} frames")


if __name__ == '__main__':
    main()
