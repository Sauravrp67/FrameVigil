"""Generate diagnostic figures from saved summary CSV files."""
import argparse
from pathlib import Path

from _checkout import add_source_directory

add_source_directory()

from src.experiments.figures import run
from src.experiments import prediction_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experiment', help='Saved experiment for image-based case studies; no inference')
    args = parser.parse_args()
    paths = run(args.input, args.output)

    if args.experiment:
        paths.extend(prediction_plots.run(args.experiment, Path(args.input).parent, Path(args.output)))

    print(f'Wrote {len(paths)} figure files to {args.output}')


if __name__ == "__main__":
    main()
