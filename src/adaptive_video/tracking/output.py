"""Separate published detections from the estimator's internal geometry."""

import hashlib
import json

import numpy as np

from .. import dataset as common
from .kalman import BoxTrack
from .scheduler import AdaptiveDetectionTracker


class DetectorOutputTracker(AdaptiveDetectionTracker):
    """Record real assignment indices."""

    def _observations(self, result):
        """Retain raw detector rows and their accepted-class indices."""
        rows = super()._observations(result)
        boxes = result.boxes
        self._raw_observations = (
            np.empty((0, 6), np.float32)
            if boxes is None
            else boxes.data.detach().cpu().numpy()
        )
        self._accepted_indices = [
            i
            for i, detection in enumerate(self._raw_observations)
            if self.class_filter is None or int(detection[5]) in self.class_filter
        ]
        self.filtered_out = len(self._raw_observations) - len(self._accepted_indices)

        return rows

    def _match(self, active, observations):
        """Record the actual detector-to-track assignments used for correction."""
        object_indices, observation_indices = super()._match(active, observations)
        self.association_pairs = [
            (active[i].track_id, int(j))
            for i, j in zip(object_indices, observation_indices)
        ]
        self.preexisting_ids = {track.track_id for track in active}
        self.filtered_detection_count = len(observations)

        return object_indices, observation_indices

    def _record_outputs(self):
        """Map matched tracks and newborn tracks back to detector rows."""
        assert len(self._accepted_indices) == self.filtered_detection_count
        used_detection_indices = set()

        for identity, filtered_index in self.association_pairs:
            raw_index = self._accepted_indices[filtered_index]
            detection = self._raw_observations[raw_index]
            self.matched_outputs[identity] = dict(
                raw_index=raw_index,
                filtered_index=filtered_index,
                box_xyxy=detection[:4].tolist(),
                conf=float(detection[4]),
                class_id=int(detection[5]),
            )
            used_detection_indices.add(filtered_index)

        unmatched_detection_indices = [
            i
            for i in range(len(self._accepted_indices))
            if i not in used_detection_indices
        ]
        new_tracks = [
            track for track in self.tracks if track.track_id not in self.preexisting_ids
        ]
        assert len(new_tracks) == len(unmatched_detection_indices)
        self.birth_rows = [
            dict(
                track_id=track.track_id,
                filtered_index=i,
                raw_index=self._accepted_indices[i],
            )
            for track, i in zip(new_tracks, unmatched_detection_indices)
        ]

    def step(self, frame, model, device):
        """Clear frame-local associations before processing the next image."""
        self.association_pairs = []
        self.birth_rows = []
        self.preexisting_ids = set()
        self.matched_outputs = {}
        self.filtered_detection_count = 0
        self.filtered_out = 0

        result = super().step(frame, model, device)

        if result['ran_inference']:
            self._record_outputs()

        return result


def emit(detector, variant='original'):
    """Publish track outputs"""
    choices = {
        'original': (False, False),
        'raw_boxes': (True, False),
        'raw_scores': (False, True),
        'raw_both': (True, True),
    }

    if variant not in choices:
        raise ValueError(variant)

    use_geometry, use_score = choices[variant]
    observations = getattr(detector, 'matched_outputs', {})
    published = []

    for track in detector.tracks:
        if track.state == 'lost':
            continue

        detection = dict(
            track_id=track.track_id,
            box_xyxy=track.get_box_xyxy().tolist(),
            conf=track.get_confidence(detector.uncertainty_threshold),
            class_id=track.class_id,
            class_name=track.class_name,
        )
        matched_detection = observations.get(track.track_id)

        if matched_detection is not None and (use_geometry or use_score):
            assert matched_detection['class_id'] == track.class_id

            if use_geometry:
                detection['box_xyxy'] = list(matched_detection['box_xyxy'])

            if use_score:
                detection['conf'] = matched_detection['conf']

        published.append(detection)

    return published


def state(detector):
    """Serialize controller state without changing it."""
    return dict(
        frame_idx=detector.frame_idx,
        tracks=[common.plain(track.__dict__) for track in detector.tracks],
        next_track_id=detector.track_type._id_counter,
    )


def digest(value):
    """Return a stable fingerprint for comparing serialized states or outputs."""
    payload = json.dumps(
        common.plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False
    )

    return hashlib.sha256(payload.encode()).hexdigest()


def make(controller_type, track_type=BoxTrack):
    """Construct a controller with the shared experiment settings."""
    return controller_type(
        frame_width=1280,
        frame_height=720,
        class_filter={0, 1, 2, 3, 5, 7},
        yolo_conf=0.5,
        uncertainty_threshold=0.0003,
        max_skip=15,
        min_infer_every=30,
        sigma_meas_px=5,
        max_lost_frames=3,
        rescue_conf=0.5,
        max_rescue=2,
        track_type=track_type,
    )
