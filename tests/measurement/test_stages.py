"""Check target enforcement and explicit sampler selection without GPU execution."""

from unittest.mock import Mock

import pytest

from adaptive_video.dataset import save
from adaptive_video.measurement import power, stages


def test_direct_power_validation_requires_module_sensor_before_loading_workload(
    monkeypatch, tmp_path
):
    target_check = Mock(side_effect=RuntimeError('Module sensor required'))
    sampler = Mock()
    monkeypatch.setattr(power, 'require_module_power', target_check)

    with pytest.raises(RuntimeError, match='Module sensor required'):
        power.validate(tmp_path / 'power.json', sampler_factory=sampler)

    target_check.assert_called_once()
    sampler.assert_not_called()

    assert not list(tmp_path.iterdir())


def test_validation_selects_isolated_sampler_without_replacing_power_class(
    monkeypatch, tmp_path
):
    original_sampler = power.Power
    capture = Mock()
    monkeypatch.setattr(stages, 'checked_experiment', lambda path: tmp_path)
    monkeypatch.setattr(stages.guard, 'check_competing_benchmark', lambda: None)
    monkeypatch.setattr(stages.guard, 'device_snapshot', lambda: {})
    monkeypatch.setattr(stages.power, 'validate', capture)
    monkeypatch.setattr(
        stages.power_check,
        'main',
        lambda directory: save(
            tmp_path / 'results/power_check.json',
            {'status': 'PASS'},
        ),
    )

    stages.validate_power(tmp_path)

    capture.assert_called_once_with(
        tmp_path / 'results/power_validation.json',
        sampler_factory=stages.IsolatedPower,
    )

    assert power.Power is original_sampler
