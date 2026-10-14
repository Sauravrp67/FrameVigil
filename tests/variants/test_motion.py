"""Check motion behavior with synthetic inputs."""

import numpy as np

from src.variants.motion import ConstantVelocity, DampedAcceleration


def test_cv_constant_motion_and_measurement_velocity_update():
    track = ConstantVelocity(
        np.array([640.0, 360.0, 100.0, 80.0]),
        2,
        'car',
        0,
        frame_width=1280,
        frame_height=720,
    )
    track.x[4] = 10 / 1280
    track.x[5] = -5 / 720
    track.predict()

    assert np.allclose(track.get_box_cxcywh()[:2], [650, 355])
    prior = track.x.copy()
    track.update(np.array([660.0, 355.0, 100.0, 80.0]), 2, 'car', 1, 0.9)

    assert 650 < track.get_box_cxcywh()[0] < 660 and track.x[4] > prior[4]
    assert track.x.shape == (8,) and track.P.shape == (8, 8)
    assert np.linalg.eigvalsh((track.P + track.P.T) / 2).min() > 0
    assert track.last_conf == 0.9 and track.frames_since_infer == 0 and track.hits == 2


def test_damping_propagates_mean_and_covariance_together():
    track = DampedAcceleration(
        np.array([640.0, 360.0, 100.0, 80.0]),
        2,
        'car',
        0,
        frame_width=1280,
        frame_height=720,
    )
    track.x[8] = 2 / 1280
    before = track.x.copy()
    previous_covariance = track.P.copy()
    track.predict()

    assert np.allclose(track.x, track.F @ before)
    assert np.allclose(track.P, track.F @ previous_covariance @ track.F.T + track.Q)
    assert np.isclose(track.x[8] * 1280, 1) and np.isclose(track.x[4] * 1280, 2)
