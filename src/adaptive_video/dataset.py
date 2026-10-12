"""Serialization helpers for dataset inventories and newly generated results.
"""

import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

from .project import read, sha


def now():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    """Convert NumPy values to JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]

    if isinstance(value, np.ndarray):
        return plain(value.tolist())

    if isinstance(value, np.generic):
        return plain(value.item())

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def save(path, value):
    """Write JSON, optionally gzip-compressed, via an atomic file replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    contents = json.dumps(plain(value), indent=2, allow_nan=False) + '\n'

    if path.suffix == '.gz':
        with gzip.open(temporary, 'wt') as stream:
            stream.write(contents)
    else:
        temporary.write_text(contents)

    temporary.replace(path)


def csvwrite(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))

    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
