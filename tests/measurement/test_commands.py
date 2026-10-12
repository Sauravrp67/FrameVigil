"""Stages must remain separately invokable and must not launch each other."""

import os
import subprocess
import sys

import pytest

from adaptive_video.measurement.stages import prepare
from adaptive_video.project import ROOT


STAGES = [
    'prepare_evaluation_dataset',
    'prepare_experiment',
    'build_detector_engine',
    'validate_power',
    'measure_experiment',
    'summarize_experiment',
    'render_demo',
    'plot_tradeoffs',
]


@pytest.mark.parametrize('name', STAGES)
def test_help_does_not_require_inputs_or_execute_stage(name, tmp_path):
    env = dict(
        os.environ,
        VIDEO_PROJECT_ROOT=str(tmp_path),
        PROJECT_ROOT=str(tmp_path),
        PYTHONDONTWRITEBYTECODE='1',
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / f'{name}.py'), '--help'],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_preparation_requires_an_explicit_dataset(tmp_path):
    with pytest.raises(TypeError):
        prepare(tmp_path / 'experiment')

    assert not list(tmp_path.iterdir())
