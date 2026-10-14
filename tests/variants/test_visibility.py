"""Check visibility behavior with synthetic inputs."""

import numpy as np

from .helpers import make_track
from src.tracking.kalman import BoxTrack
from src.variants.visibility import (
    ExistenceTrack,
    VisibilityDetector,
    covered_fraction,
    missed_posterior,
)


def test_missing_visible_object_reduces_existence_more_than_occluded_object():
    assert missed_posterior(0.99, 0.9) < missed_posterior(0.99, 0.2) < 0.99
    assert missed_posterior(missed_posterior(0.99, 0.9), 0.9) < 0.5
    assert missed_posterior(missed_posterior(0.99, 0.2), 0.2) > 0.9


def test_existence_memory_leaves_matched_kalman_dynamics_unchanged():
    track, reference = make_track(ExistenceTrack), make_track(BoxTrack)

    for i in range(10):
        track.predict()
        reference.predict()

        if i % 3 == 0:
            box = np.array([640 + 2 * i, 360, 100, 80])
            track.update(box, 2, 'car', i, 0.8)
            reference.update(box, 2, 'car', i, 0.8)

            assert track.existence == 0.99

        assert np.array_equal(track.x, reference.x) and np.array_equal(
            track.P, reference.P
        )


def test_coverage_is_relative_to_missing_track_area():
    assert covered_fraction([10, 10, 10, 10], [10, 10, 20, 20]) == 1
    assert covered_fraction([10, 10, 20, 20], [10, 10, 10, 10]) == 0.25


def test_actual_unmatched_track_rescue_is_vetoed_when_existence_low():
    tracker = VisibilityDetector.__new__(VisibilityDetector)
    tracker.iou_threshold = 0.3
    tracker.mahal_threshold = 9.4877
    tracker.visibility_info = dict(
        missed_tracks=0, occlusion_proxy_tracks=0, rescue_vetoes=0, memory_update_ms=0.0
    )
    track = make_track(ExistenceTrack)
    track.state = 'confirmed'
    track.existence = 0.8
    matched_tracks, matched_detections = tracker._match([track], [])

    assert matched_tracks == [] and matched_detections == [] and track.state == 'lost'
    assert tracker.visibility_info['rescue_vetoes'] == 1 and track.existence < 0.5
