"""Evaluate a completed live campaign and summarize its measured accuracy, latency and energy."""

import argparse
from _checkout import add_source_directory

add_source_directory()

from adaptive_video.experiments.evaluate import run
from adaptive_video.project import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-manifest', help='Explicit generated evaluation population')
    args = parser.parse_args()
    bootstrap()
    run(args.experiment, args.output, live=True, dataset_manifest=args.dataset_manifest)


if __name__ == "__main__":
    main()
