"""Small per-object HSV memory used only in actual detector/track association."""

import time

from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np

from ..tracking import scheduler
from ..tracking.output import DetectorOutputTracker


def descriptor(image, box):
    """Extract a normalized HSV histogram from a clipped object crop."""
    center_x, center_y, width, height = box
    left = max(0, int(center_x - width / 2))
    top = max(0, int(center_y - height / 2))
    right = min(image.shape[1], int(np.ceil(center_x + width / 2)))
    bottom = min(image.shape[0], int(np.ceil(center_y + height / 2)))

    if right <= left or bottom <= top:
        return None

    patch = cv2.resize(
        image[top:bottom, left:right], (32, 32), interpolation=cv2.INTER_AREA
    )
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).reshape(-1)

    return (histogram / max(float(histogram.sum()), 1.0)).astype(np.float32)


def distance(a, b):
    """Measure histogram disagreement; missing descriptors contribute no penalty."""
    if a is None or b is None:
        return 0.0

    return float(np.sqrt(max(0.0, 1 - float(np.sqrt(a * b).sum()))))


class AppearanceDetector(DetectorOutputTracker):
    def _match(self, active_tracks, detections):
        start = time.perf_counter()
        self.new_descriptors = [
            descriptor(self.current_image, d['box']) for d in detections
        ]
        self.descriptor_failures = sum(d is None for d in self.new_descriptors)
        self.descriptor_ms = 1000 * (time.perf_counter() - start)
        start = time.perf_counter()
        matched_track_indices = []
        matched_detection_indices = []
        used_track_indices = set()
        used_detection_indices = set()
        recent = [i for i, t in enumerate(active_tracks) if t.frames_since_update <= 2]

        def visual(i, j):
            return distance(
                self.templates.get(active_tracks[i].track_id), self.new_descriptors[j]
            )

        if recent and detections:
            overlap = np.array(
                [
                    [
                        scheduler.overlap_ratio(
                            active_tracks[i].get_box_cxcywh(), d['box']
                        )
                        for d in detections
                    ]
                    for i in recent
                ]
            )
            costs = (
                1
                - overlap
                + np.array(
                    [
                        [0.25 * visual(i, j) for j in range(len(detections))]
                        for i in recent
                    ]
                )
            )
            rows, columns = linear_sum_assignment(costs)

            for row_index, column_index in zip(rows, columns):
                if overlap[row_index, column_index] >= self.iou_threshold:
                    matched_track_indices.append(recent[row_index])
                    matched_detection_indices.append(int(column_index))
                    used_track_indices.add(recent[row_index])
                    used_detection_indices.add(int(column_index))

        remaining_track_indices = [
            i for i in range(len(active_tracks)) if i not in used_track_indices
        ]
        remaining_detection_indices = [
            j for j in range(len(detections)) if j not in used_detection_indices
        ]

        if remaining_track_indices and remaining_detection_indices:
            mahalanobis_distances = np.array(
                [
                    [
                        active_tracks[i].mahalanobis_sq(detections[j]['box'])
                        for j in remaining_detection_indices
                    ]
                    for i in remaining_track_indices
                ]
            )
            costs = np.where(
                mahalanobis_distances < self.mahal_threshold,
                mahalanobis_distances / self.mahal_threshold
                + np.array(
                    [
                        [0.25 * visual(i, j) for j in remaining_detection_indices]
                        for i in remaining_track_indices
                    ]
                ),
                1e9,
            )
            rows, columns = linear_sum_assignment(costs)

            for row_index, column_index in zip(rows, columns):
                if (
                    mahalanobis_distances[row_index, column_index]
                    < self.mahal_threshold
                ):
                    matched_track_indices.append(remaining_track_indices[row_index])
                    matched_detection_indices.append(
                        remaining_detection_indices[column_index]
                    )
        # These are the actual pairs subsequently consumed by the constant-acceleration step.

        self.association_pairs = [
            (active_tracks[i].track_id, j)
            for i, j in zip(matched_track_indices, matched_detection_indices)
        ]
        self.preexisting_ids = {t.track_id for t in active_tracks}
        self.filtered_detection_count = len(detections)
        self.association_ms = 1000 * (time.perf_counter() - start)

        return matched_track_indices, matched_detection_indices

    def step(self, frame, model, device):
        if not hasattr(self, 'templates'):
            self.templates = {}

        self.current_image = frame
        self.descriptor_ms = 0.0
        self.association_ms = 0.0
        self.descriptor_failures = 0

        result = super().step(frame, model, device)
        start = time.perf_counter()

        if result['ran_inference']:
            pairs = list(self.association_pairs) + [
                (b['track_id'], b['filtered_index']) for b in self.birth_rows
            ]

            for track_id, index in pairs:
                value = self.new_descriptors[index]

                if value is not None:
                    self.templates[track_id] = value

        alive = {t.track_id for t in self.tracks}
        self.templates = {k: v for k, v in self.templates.items() if k in alive}
        self.appearance_info = dict(
            descriptor_ms=self.descriptor_ms,
            association_ms=self.association_ms,
            memory_update_ms=1000 * (time.perf_counter() - start),
            descriptor_failures=self.descriptor_failures,
            templates=len(self.templates),
            persistent_descriptor_bytes=sum(t.nbytes for t in self.templates.values()),
            device_transfer_bytes=0,
        )
        del self.current_image

        return result
