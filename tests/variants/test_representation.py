"""Alternative box coordinates preserve physical geometry and covariance units."""

import numpy as np
import pytest

from .helpers import (
    assert_birth_covariance,
    assert_jacobian_and_process_noise,
    make_track,
)
from src.variants.motion import ConstantVelocity
from src.variants.representation import (
    AspectHeightCV,
    LogSizeCV,
    RepresentedCV,
)


@pytest.mark.parametrize('track_type', [RepresentedCV, LogSizeCV, AspectHeightCV])
def test_birth_uncertainty_preserves_original_box_units(track_type):
    assert_birth_covariance(track_type)


@pytest.mark.parametrize('track_type', [LogSizeCV, AspectHeightCV])
def test_full_jacobian_and_physical_process_noise(track_type):
    assert_jacobian_and_process_noise(track_type)


def test_linear_control_matches_cv8_trajectory():
    track, control = make_track(RepresentedCV), make_track(ConstantVelocity)

    for i in range(20):
        track.predict()
        control.predict()

        if i % 3 == 0:
            box = np.array([640.0 + 4 * i, 360.0 - 2 * i, 100.0 + i, 80.0])
            track.update(box, 2, 'car', i, 0.85)
            control.update(box, 2, 'car', i, 0.85)

        assert np.allclose(track.x, control.x, atol=1e-14)
        assert np.allclose(track.P, control.P, atol=1e-14)


def test_log_state_represents_proportional_size_growth():
    track = make_track(LogSizeCV)
    track.x[6:8] = np.log(1.1)
    track.predict()

    assert np.allclose(track.get_box_cxcywh()[2:], [110.0, 88.0])
