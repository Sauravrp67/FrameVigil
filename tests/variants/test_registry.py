"""Variant construction, independent track identities and array storage accounting."""

import numpy as np
import pytest

from adaptive_video.measurement.support import configure, extended_state
from adaptive_video.tracking import scheduler
from adaptive_video.tracking.kalman import BoxTrack
from adaptive_video.tracking.output import DetectorOutputTracker, digest
from adaptive_video.tracking.scheduler import AdaptiveDetectionTracker
from adaptive_video.variants.registry import VARIANTS, compare_outputs, payload


@pytest.fixture(autouse=True)
def stub_scene_classifier(monkeypatch):
    # Registry tests construct trackers but do not need a neural classifier.
    monkeypatch.setattr(scheduler, 'load_scene_classifier', lambda device: None)


def test_all_families_present_and_reset_does_not_leak_track_identity():
    for name in [*VARIANTS, 'always_on']:
        detector, track_type, emission = configure(name)

        if detector is None:
            continue

        a = track_type(np.array([640.0, 360.0, 100.0, 80.0]), 2, 'car', 0, 1280, 720)

        assert a.track_id == 0
        detector, track_type, emission = configure(name)
        b = track_type(np.array([640.0, 360.0, 100.0, 80.0]), 2, 'car', 0, 1280, 720)

        assert b.track_id == 0 and not detector.tracks

    configure('original')


def test_final_digest_covers_covariance_and_appearance_memory():
    detector, track_type, _ = configure('appearance_ca10')
    detector.tracks = [
        track_type(np.array([640.0, 360.0, 100.0, 80.0]), 2, 'car', 0, 1280, 720)
    ]
    detector.templates = {0: np.ones(64, np.float32) / 64}
    a = digest(extended_state(detector, track_type))
    detector.templates[0][0] = 0.5

    assert digest(extended_state(detector, track_type)) != a
    b = digest(extended_state(detector, track_type))
    detector.tracks[0].P[0, 0] *= 2

    assert digest(extended_state(detector, track_type)) != b
    configure('original')


def test_payload_does_not_double_count_shared_arrays():
    a = np.zeros((10, 10), np.float64)

    assert payload([{'P': a}, {'same_P': a}]) == 800


def test_output_parity_rejects_box_class_and_confidence_drift():
    row = dict(box_xyxy=[0.0, 1.0, 20.0, 30.0], class_id=2, class_name='car', conf=0.9)
    compare_outputs([[row]], [[dict(row)]])

    for change in [dict(box_xyxy=[0, 1, 21, 30]), dict(class_id=7), dict(conf=0.89)]:
        with pytest.raises(AssertionError):
            compare_outputs([[dict(row, **change)]], [[row]])


def test_registry_contains_expected_variants_and_output_modes():
    assert set(VARIANTS) == {
        'original',
        'raw_both',
        'cv8',
        'damped_ca10',
        'log_cv8',
        'aspect_cv8',
        'projective_restart_cv8',
        'camera_ca10',
        'appearance_ca10',
        'visibility_ca10',
    }
    assert VARIANTS['original'] == (AdaptiveDetectionTracker, BoxTrack, 'original')
    assert VARIANTS['raw_both'] == (DetectorOutputTracker, BoxTrack, 'raw_both')
