"""Visibility-conditioned existence memory; veto weak unmatched-track rescue."""

import time

import numpy as np

from ..tracking.kalman import BoxTrack
from ..tracking.output import DetectorOutputTracker


def covered_fraction(box, other):
    a = np.asarray(box)
    b = np.asarray(other)
    intersection_top_left = np.maximum(a[:2] - a[2:] / 2, b[:2] - b[2:] / 2)
    intersection_bottom_right = np.minimum(a[:2] + a[2:] / 2, b[:2] + b[2:] / 2)

    return float(
        np.prod(np.maximum(intersection_bottom_right - intersection_top_left, 0))
        / max(float(a[2] * a[3]), 1e-12)
    )


def missed_posterior(prior, detection_probability):
    return prior * (1 - detection_probability) / (1 - prior * detection_probability)


class ExistenceTrack(BoxTrack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.existence = 0.99
        self.visibility_proxy = 1.0
        self.missed_observations = 0

    def predict(self):
        self.existence *= 0.995

        return super().predict()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.existence = 0.99
        self.visibility_proxy = 1.0
        self.missed_observations = 0


class VisibilityDetector(DetectorOutputTracker):
    def _match(self, active, detections):
        matched_track_indices, matched_detection_indices = super()._match(
            active, detections
        )
        start = time.perf_counter()
        matched = set(matched_track_indices)

        for index, track in enumerate(active):
            if index in matched:
                continue

            occluded_fraction = max(
                [
                    covered_fraction(track.get_box_cxcywh(), d['box'])
                    for d in detections
                ],
                default=0.0,
            )
            track.visibility_proxy = 1 - occluded_fraction
            detection_probability = 0.2 if occluded_fraction >= 0.5 else 0.9
            track.existence = missed_posterior(track.existence, detection_probability)
            track.missed_observations += 1
            self.visibility_info['missed_tracks'] += 1
            self.visibility_info['occlusion_proxy_tracks'] += int(
                occluded_fraction >= 0.5
            )

            if track.existence < 0.5:
                # Constant-acceleration unmatched-track handling will then decline rescue.
                track.state = 'lost'
                self.visibility_info['rescue_vetoes'] += 1

        self.visibility_info['memory_update_ms'] += 1000 * (time.perf_counter() - start)

        return matched_track_indices, matched_detection_indices

    def step(self, frame, model, device):
        self.visibility_info = dict(
            missed_tracks=0,
            occlusion_proxy_tracks=0,
            rescue_vetoes=0,
            memory_update_ms=0.0,
            device_transfer_bytes=0,
        )

        result = super().step(frame, model, device)
        self.visibility_info['tracks'] = len(self.tracks)
        self.visibility_info['scalar_payload_bytes'] = len(self.tracks) * 24

        return result
