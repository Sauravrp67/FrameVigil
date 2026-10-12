"""Build the study's TensorRT engine on the GPU where measurement will run.

Requires compatible TensorRT Python bindings (10+) and ONNX. Use the NVIDIA TensorRT installation for your platform;
Jetson normally supplies its bindings through JetPack.
"""

import argparse

from _checkout import add_source_directory

add_source_directory()

import yaml

from adaptive_video.project import ROOT
from adaptive_video.detectors.build_engine import build_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=ROOT / 'configs/study.yaml')
    parser.add_argument('--device', default='cuda', help='CUDA device used to build the engine')
    args = parser.parse_args()

    with open(args.config) as stream:
        configuration = yaml.safe_load(stream) or {}

    # Optional build dependencies must not be needed for --help or eager usage.
    build_engine(configuration.get('inference'), device=args.device)


if __name__ == '__main__':
    main()
