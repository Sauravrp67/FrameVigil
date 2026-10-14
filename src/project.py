"""Checkout paths, package initialization and downloaded-input verification."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()


def read(path):
    path = Path(path)
    opener = gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()

    with opener as stream:
        return json.load(stream)


def sha(path):
    checksum = hashlib.sha256()

    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)

    return checksum.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bootstrap():
    """Select optional target-specific dependencies and runtime defaults."""
    sys.dont_write_bytecode = True

    directory = ROOT / '.runtime' / platform.machine()

    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs/yolo_config'))
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


def verify(dataset_manifest):
    """Verify an explicitly indexed dataset; model checks belong to downloads."""
    inventory = read(dataset_manifest)

    for name, expected in inventory['image_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Image changed: {name}')

    for video, name in inventory['labels'].items():
        if sha(ROOT / name) != inventory['label_sha256'][video]:
            raise ValueError(f'Annotation changed: {name}')

    return {
        'status': 'PASS',
        'videos': len(inventory['videos']),
        'images': len(inventory['frames']),
    }
