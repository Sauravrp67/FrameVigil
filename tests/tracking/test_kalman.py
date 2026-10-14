"""Kalman correction agrees with an independently calculable scalar example."""

import numpy as np

from src.tracking import scheduler
from src.tracking.kalman import BoxTrack, correct_distribution
from src.tracking.output import make
from src.variants.motion import ConstantVelocity


def test_scalar_measurement_weight_and_uncertainty():
    mean, covariance = correct_distribution(
        np.array([100.0]),
        np.array([[9.0]]),
        np.array([110.0]),
        np.eye(1),
        np.array([[1.0]]),
    )

    assert np.allclose(mean, [109.0])
    assert np.allclose(covariance, [[0.9]])


def test_track_factories_are_owned_by_each_controller(monkeypatch):
    monkeypatch.setattr(scheduler, 'load_scene_classifier', lambda device: None)
    ConstantVelocity.reset_id_counter()
    BoxTrack.reset_id_counter()
    cv = make(scheduler.AdaptiveDetectionTracker, ConstantVelocity)
    ca = make(scheduler.AdaptiveDetectionTracker, BoxTrack)
    observation = dict(
        box=np.array([640.0, 360.0, 100.0, 80.0]),
        class_id=2,
        class_name='car',
        conf=0.9,
    )
    cv._reconcile([observation])
    ca._reconcile([observation])

    assert cv.tracks[0].x.shape == (8,)
    assert ca.tracks[0].x.shape == (10,)
    assert cv.track_type is ConstantVelocity and ca.track_type is BoxTrack
