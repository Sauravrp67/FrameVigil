"""Check appearance behavior with synthetic inputs."""

from types import SimpleNamespace

import numpy as np

from src.variants.appearance import AppearanceDetector, descriptor, distance


def test_histogram_identity_distinction_and_invalid_crop():
    image = np.zeros((40, 80, 3), np.uint8)
    image[:, :40] = [0, 0, 255]
    image[:, 40:] = [255, 0, 0]
    red = descriptor(image, [20, 20, 40, 40])
    blue = descriptor(image, [60, 20, 40, 40])

    assert red.shape == (64,) and red.dtype == np.float32
    assert (
        np.isclose(red.sum(), 1)
        and distance(red, red) == 0
        and distance(red, blue) > 0.99
    )
    assert descriptor(image, [-100, -100, 1, 1]) is None


def test_actual_association_uses_stored_object_appearance():
    tracker = AppearanceDetector.__new__(AppearanceDetector)
    tracker.current_image = np.zeros((40, 80, 3), np.uint8)
    tracker.current_image[:, :40] = [0, 0, 255]
    tracker.current_image[:, 40:] = [255, 0, 0]
    boxes = [np.array([20, 20, 40, 40]), np.array([60, 20, 40, 40])]
    tracker.templates = {
        0: descriptor(tracker.current_image, boxes[1]),
        1: descriptor(tracker.current_image, boxes[0]),
    }
    tracker.iou_threshold = 0.3
    tracker.mahal_threshold = 9.4877
    tracks = [
        SimpleNamespace(
            track_id=i,
            frames_since_update=1,
            get_box_cxcywh=lambda: np.array([40, 20, 80, 40]),
            mahalanobis_sq=lambda box: 0.0,
        )
        for i in range(2)
    ]
    matched_tracks, matched_detections = tracker._match(
        tracks, [{'box': b} for b in boxes]
    )

    assert list(zip(matched_tracks, matched_detections)) == [(0, 1), (1, 0)]
    assert tracker.association_pairs == [(0, 1), (1, 0)]
