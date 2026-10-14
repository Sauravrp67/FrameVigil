"""Matched output handling preserves real associations and internal states."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_video import dataset as common
from ultralytics.engine.results import Results
from adaptive_video.detectors.nms import _pure_torch_nms
from adaptive_video.measurement.protocol import always_rows
from adaptive_video.tracking.kalman import BoxTrack
from adaptive_video.tracking.output import (
    DetectorOutputTracker,
    digest,
    emit,
    make,
    state,
)
import adaptive_video.tracking.scheduler as module


class StubDetector:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1

        return [
            Results(
                frame,
                path='',
                names={0: 'person', 2: 'car', 9: 'traffic light'},
                boxes=torch.tensor(self.rows, dtype=torch.float32).reshape(-1, 6),
            )
        ]


def make_tracker(monkeypatch):
    monkeypatch.setattr(module, 'load_scene_classifier', lambda device: None)
    monkeypatch.setattr(module, 'classify_scene', lambda *args: 1)
    BoxTrack.reset_id_counter()

    return make(DetectorOutputTracker)


def test_exact_association_and_no_state_mutation(monkeypatch):
    tracker_instance = make_tracker(monkeypatch)
    image = np.zeros((720, 1280, 3), np.uint8)
    tracker_instance.step(
        image, StubDetector([[100.25, 100.5, 180.75, 180.5, 0.9, 2]]), 'cpu'
    )

    assert (
        len(tracker_instance.birth_rows) == 1 and not tracker_instance.matched_outputs
    )
    born = emit(tracker_instance)

    assert emit(tracker_instance, 'detector_observations') == born
    tracker_instance.tracks[0].P *= 10
    tracker_instance.step(
        image, StubDetector([[104.125, 100.375, 184.875, 180.625, 0.85, 2]]), 'cpu'
    )

    assert len(tracker_instance.matched_outputs) == 1
    before = digest(state(tracker_instance))
    out = emit(tracker_instance, 'detector_observations')

    assert out[0]['box_xyxy'] == [104.125, 100.375, 184.875, 180.625]
    assert out[0]['conf'] == pytest.approx(0.85)
    assert digest(state(tracker_instance)) == before


def test_rescue_birth_and_filtered_detection_unchanged(monkeypatch):
    tracker_instance = make_tracker(monkeypatch)
    image = np.zeros((720, 1280, 3), np.uint8)
    tracker_instance.step(image, StubDetector([[10, 10, 50, 50, 0.9, 2]]), 'cpu')
    tracker_instance.tracks[0].state = 'confirmed'
    tracker_instance.tracks[0].P *= 10
    tracker_instance.step(
        image,
        StubDetector([[900, 500, 950, 550, 0.8, 2], [200, 200, 240, 240, 0.9, 9]]),
        'cpu',
    )

    assert (
        tracker_instance.filtered_out == 1
        and len(tracker_instance.birth_rows) == 1
        and not tracker_instance.matched_outputs
    )
    assert tracker_instance.tracks[0].consecutive_rescues == 1
    assert emit(tracker_instance, 'detector_observations') == emit(tracker_instance)


def test_prediction_only_clears_previous_match(monkeypatch):
    tracker_instance = make_tracker(monkeypatch)
    image = np.zeros((720, 1280, 3), np.uint8)

    for _ in range(2):
        tracker_instance.step(
            image, StubDetector([[100, 100, 180, 180, 0.9, 2]]), 'cpu'
        )

    for track in tracker_instance.tracks:
        track.P *= 0
        track.Q *= 0

    model = StubDetector([])
    result = tracker_instance.step(image, model, 'cpu')

    assert not result['ran_inference'] and model.calls == 0
    assert not tracker_instance.matched_outputs and emit(
        tracker_instance, 'detector_observations'
    ) == emit(tracker_instance)


def test_association_is_not_forced_to_match_class(monkeypatch):
    tracker_instance = make_tracker(monkeypatch)
    image = np.zeros((720, 1280, 3), np.uint8)
    tracker_instance.step(image, StubDetector([[100, 100, 180, 180, 0.9, 2]]), 'cpu')
    track_id = tracker_instance.tracks[0].track_id
    tracker_instance.tracks[0].P *= 10
    tracker_instance.step(image, StubDetector([[100, 100, 180, 180, 0.8, 0]]), 'cpu')

    assert (
        tracker_instance.tracks[0].track_id == track_id
        and tracker_instance.tracks[0].class_id == 0
    )
    assert (
        track_id in tracker_instance.matched_outputs
        and emit(tracker_instance, 'detector_observations')[0]['class_id'] == 0
    )


def test_adapter_never_reads_labels_or_caches(monkeypatch):
    tracker_instance = make_tracker(monkeypatch)
    image = np.zeros((720, 1280, 3), np.uint8)

    def forbidden(*args, **kwargs):
        raise AssertionError('forbidden data access')

    for name in ['read', 'save', 'csvwrite']:
        monkeypatch.setattr(common, name, forbidden)

    tracker_instance.step(image, StubDetector([[10, 10, 50, 50, 0.9, 2]]), 'cpu')
    emit(tracker_instance, 'detector_observations')


def test_detector_observations_preserve_matched_geometry_and_score(monkeypatch):
    tracker_instance = make_tracker(monkeypatch)
    image = np.zeros((720, 1280, 3), np.uint8)
    tracker_instance.step(image, StubDetector([[10, 10, 50, 50, 0.9, 2]]), 'cpu')
    tracker_instance.tracks[0].P *= 10
    tracker_instance.step(image, StubDetector([[12.5, 10, 52.5, 50, 0.8, 2]]), 'cpu')
    output = emit(tracker_instance, 'detector_observations')[0]

    assert output['box_xyxy'] == [12.5, 10, 52.5, 50]
    assert output['conf'] == pytest.approx(0.8)


def test_baseline_control_preserves_floating_point_boxes():
    boxes = torch.tensor(
        [[1.25, 2.5, 10.75, 20.125, 0.9, 2], [1.0, 2.0, 3.0, 4.0, 0.8, 9]]
    )
    result = [
        SimpleNamespace(
            boxes=SimpleNamespace(data=boxes), names={2: 'car', 9: 'traffic light'}
        )
    ]
    rows = always_rows(result)

    assert len(rows) == 1
    assert rows[0]['box_xyxy'] == [1.25, 2.5, 10.75, 20.125]
    assert 'track_id' not in rows[0]


def test_tensor_box_suppression_and_empty_input():
    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [10.0, 10.0, 20.0, 20.0]]
    )

    assert _pure_torch_nms(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.5).tolist() == [0, 2]
    assert _pure_torch_nms(boxes[:0], torch.empty(0), 0.5).shape == (0,)
