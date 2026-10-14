"""Render a selected sequence from complete saved prediction files."""

import argparse

from _checkout import add_source_directory

add_source_directory()

from src.experiments.render import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', required=True)
    parser.add_argument('--video', required=True, help='Sequence name in the dataset manifest')
    parser.add_argument('--predictions', required=True, nargs='+',
                        help='One complete trial JSON or JSON.GZ file per variant')
    parser.add_argument('--output', required=True, help='New directory for video and provenance')
    parser.add_argument('--fps', type=float, default=5.0,
                        help='Playback speed; default 5 FPS matches BDD100K MOT sampling')
    args = parser.parse_args()

    run(args.dataset_manifest, args.predictions, args.video, args.output, args.fps)


if __name__ == '__main__':
    main()
