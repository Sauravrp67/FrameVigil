"""Plot an explicit accuracy/resource table; does not run any experiments."""
import argparse
from _checkout import add_source_directory

add_source_directory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from adaptive_video.experiments.figures import run
    run(args.input, args.output)


if __name__ == "__main__":
    main()
