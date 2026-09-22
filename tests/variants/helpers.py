"""Synthetic tracks and shared numerical assertions for representation tests."""

import numpy as np

from adaptive_video.variants.motion import ConstantVelocity


def make_track(track_type):
    return track_type(
        np.array([640.0, 360.0, 100.0, 80.0]),
        2,
        'car',
        0,
        frame_width=1280,
        frame_height=720,
        conf=0.9,
    )


def assert_birth_covariance(track_type):
    track, reference = make_track(track_type), make_track(ConstantVelocity)

    assert np.allclose(track.get_box_cxcywh(), reference.get_box_cxcywh())
    assert np.allclose(track.box_covariance(), reference.P[:4, :4], atol=1e-14)
    assert track.is_uncertain() == reference.is_uncertain()
    assert np.isclose(track.get_confidence(), reference.get_confidence())
    inverse = np.linalg.inv(track.full_jacobian())

    assert np.allclose(track.Q, inverse @ reference.Q @ inverse.T)


def decode_geometry(track, state):
    """Decode box geometry and its velocity for numerical differentiation."""
    track.x = state
    box, jacobian, _ = track.geometry()

    return np.r_[box, jacobian @ state[4:]]


def assert_jacobian_and_process_noise(track_type):
    track = make_track(track_type)
    track.x[4:] = [0.01, -0.02, 0.03, 0.04]
    original = track.x.copy()

    step_size = 1e-6
    numerical_jacobian = np.column_stack(
        [
            (
                decode_geometry(track, original + np.eye(8)[i] * step_size)
                - decode_geometry(track, original - np.eye(8)[i] * step_size)
            )
            / (2 * step_size)
            for i in range(8)
        ]
    )
    track.x = original

    assert np.allclose(numerical_jacobian, track.full_jacobian(), rtol=1e-7, atol=1e-9)
    track.predict()
    jacobian = track.full_jacobian()

    assert np.allclose(jacobian @ track.Q @ jacobian.T, track.box_process_q, atol=1e-12)
