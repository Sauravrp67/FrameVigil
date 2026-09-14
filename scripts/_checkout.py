"""Make the local package available when running scripts directly."""

from pathlib import Path
import sys


def add_source_directory():
    source_directory = str(Path(__file__).resolve().parents[1] / 'src')

    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)
