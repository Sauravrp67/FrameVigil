"""Check camera behavior with synthetic inputs."""

import cv2
import numpy as np

from .helpers import make_track
from adaptive_video.variants.camera import estimate, transform_track
from adaptive_video.variants.motion import DampedAcceleration


def test_sparse_flow_recovers_known_camera_translation():
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, (90, 160), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    expected = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    current = cv2.warpAffine(gray, expected, (160, 90))
    matrix, info = estimate(gray, current, np.full_like(gray, 255))

    assert info['valid'] and info['inliers'] >= 6
    assert np.allclose(matrix, expected, atol=0.15)


def test_textureless_or_absent_background_falls_back():
    gray = np.zeros((90, 160), np.uint8)

    for previous in [None, gray]:
        matrix, info = estimate(previous, gray, np.full_like(gray, 255))

        assert matrix is None and not info['valid']


def test_camera_translation_preserves_velocity_and_inflates_covariance():
    track = make_track(DampedAcceleration)
    previous_state, previous_covariance = track.x.copy(), track.P.copy()
    transform_track(track, np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]), 0.25)

    assert np.allclose(track.get_box_cxcywh(), [656, 368, 100, 80])
    assert np.array_equal(track.x[2:], previous_state[2:])
    assert np.all(np.diag(track.P)[:4] > np.diag(previous_covariance)[:4])
    assert np.linalg.eigvalsh(track.P).min() > 0


def test_rotation_transforms_box_envelope_and_correlated_covariance():
    track = make_track(DampedAcceleration)
    matrix = cv2.getRotationMatrix2D((80, 45), 5, 1.0)
    old = track.get_box_cxcywh()
    corners = np.array(
        [
            [old[0] + a * old[2] / 2, old[1] + b * old[3] / 2]
            for a in [-1, 1]
            for b in [-1, 1]
        ]
    ) / [8.0, 8.0]
    warped = corners @ matrix[:, :2].T + matrix[:, 2]
    transform_track(track, matrix, 0.25)
    expected = np.r_[warped.mean(0), np.ptp(warped, axis=0)] * 8

    assert np.allclose(track.get_box_cxcywh(), expected)
    assert np.allclose(track.P, track.P.T)
    assert np.linalg.eigvalsh(track.P).min() > 0
