"""Check projective restart behavior with synthetic inputs."""

import numpy as np

from .helpers import make_track
from adaptive_video.variants.projective_restart import ProjectiveRestartCV


def test_invalid_correction_restarts_from_real_measurement_without_overconfidence():
    track = make_track(ProjectiveRestartCV)
    old_count = track.correction_restarts_total
    measurement = np.array([640.0, 360.0, 100.0, 5000.0])
    track.update(measurement, 2, 'car', 1, 0.8)

    assert track.correction_restarts_total == old_count + 1
    assert np.allclose(track.get_box_cxcywh(), measurement)
    assert np.allclose(track.x[4:], 0)
    assert np.allclose(np.diag(track.box_covariance()), [1e-4, 1e-4, 2.5e-5, 2.5e-5])
    assert track.hits == 2 and track.last_seen_frame == 1 and track.last_conf == 0.8
