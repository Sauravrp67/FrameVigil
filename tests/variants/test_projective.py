"""Check projective behavior with synthetic inputs."""

import numpy as np

from .helpers import (
    assert_birth_covariance,
    assert_jacobian_and_process_noise,
    make_track,
)
from src.variants.projective import ProjectiveCV


def test_projective_jacobian_and_covariance_units():
    assert_jacobian_and_process_noise(ProjectiveCV)
    assert_birth_covariance(ProjectiveCV)


def test_approaching_relative_depth_grows_box_non_linearly():
    track = make_track(ProjectiveCV)
    depth = track.x[2]
    track.x[6] = -0.1 * depth
    track.predict()

    assert np.allclose(track.get_box_cxcywh()[2:], [100 / 0.9, 80 / 0.9])
    track.predict()

    assert np.allclose(track.get_box_cxcywh()[2:], [100 / 0.8, 80 / 0.8])


def test_invalid_depth_demands_detector_and_retains_valid_box():
    track = make_track(ProjectiveCV)
    before = track.get_box_cxcywh()
    track.x[6] = -2 * track.x[2]
    track.predict()

    assert track.force_detector and track.depth_domain_failures == 1
    assert track.is_uncertain(1e99)
    assert np.allclose(track.get_box_cxcywh(), before)
    assert np.linalg.eigvalsh((track.P + track.P.T) / 2).min() > 0
    track.update(before, 2, 'car', 1, 0.9)

    assert not track.force_detector


def test_boundary_pruning_uses_image_projection_not_3d_coordinate():
    track = make_track(ProjectiveCV)
    track.x[0] = 2.0  # physical X=2, image cx=.5+2/9=.722, still inside.

    assert not track.is_out_of_frame()
    track.x[0] = 20.0

    assert track.is_out_of_frame()


def test_monocular_box_does_not_identify_metric_scale():
    xyz = np.array([1.0, 2.0, 10.0])
    height = 2.0
    project = lambda q, h: np.r_[q[:2] / q[2], h / q[2]]

    assert np.array_equal(project(xyz, height), project(3 * xyz, 3 * height))
