import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_video.project import bootstrap

bootstrap()
