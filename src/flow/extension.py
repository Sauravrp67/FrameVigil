"""Correct track geometry with optical flow.
"""

import time

import numpy as np

from ..tracking.scheduler import AdaptiveDetectionTracker


def eligible(tracks, old):
    """Check predicted motion and track age before requesting selective flow."""
    motion_ratios = []

    for track in tracks:
        if track.track_id not in old:
            continue

        previous_box = old[track.track_id]
        predicted_box = track.get_box_cxcywh()
        box_diagonal = max(1.0, np.linalg.norm(previous_box[2:]))
        center_displacement = np.linalg.norm(predicted_box[:2] - previous_box[:2])
        motion_ratios.append(center_displacement / box_diagonal)

    return bool(
        motion_ratios
        and (
            np.median(motion_ratios) >= 0.05
            or any(track.frames_since_update >= 2 for track in tracks)
        )
    )


def observation(field, box):
    """Estimate a pixel-space [cx, cy, w, h] measurement from a four-pixel flow grid.

    field contains [dx, dy] pixel displacements, not raw fixed-point values.
    Return (measurement, sigma_pixels), or None when quality checks fail.
    """
    center_x, center_y, width, height = box
    left = center_x - width / 2
    top = center_y - height / 2
    right = center_x + width / 2
    bottom = center_y + height / 2

    # Flow sample centers are at (grid_index + 0.5) * 4 pixels.
    column_start = max(0, int(np.ceil(left / 4 - 0.5)))
    column_end = min(field.shape[1], int(np.ceil(right / 4 - 0.5)))
    row_start = max(0, int(np.ceil(top / 4 - 0.5)))
    row_end = min(field.shape[0], int(np.ceil(bottom / 4 - 0.5)))

    if column_end <= column_start or row_end <= row_start:
        return None

    vectors = field[row_start:row_end, column_start:column_end].reshape(-1, 2)
    finite_mask = np.isfinite(vectors).all(axis=1)

    if finite_mask.sum() < 4:
        return None

    vectors = vectors[finite_mask]
    median_motion = np.median(vectors, axis=0)
    distances = np.linalg.norm(vectors - median_motion, axis=1)
    dispersion = float(np.median(distances))
    inlier_mask = distances <= max(2.0, 3 * dispersion)
    normalized_dispersion = dispersion / max(1, np.hypot(width, height))

    if inlier_mask.mean() < 0.5 or normalized_dispersion > 0.25:
        return None

    # Allocate grid coordinates only inside the object's previous box.
    grid_rows, grid_columns = np.mgrid[row_start:row_end, column_start:column_end]
    relative_x = ((grid_columns + 0.5) * 4 - center_x).ravel()[finite_mask][inlier_mask]
    relative_y = ((grid_rows + 0.5) * 4 - center_y).ravel()[finite_mask][inlier_mask]
    horizontal_scale = 1.0
    vertical_scale = 1.0

    if inlier_mask.sum() >= 16:
        horizontal_residual = vectors[inlier_mask, 0] - median_motion[0]
        vertical_residual = vectors[inlier_mask, 1] - median_motion[1]

        horizontal_scale = float(np.clip(
            1 + np.dot(relative_x, horizontal_residual)
            / max(1e-6, np.dot(relative_x, relative_x)),
            0.8, 1.25,
        ))
        vertical_scale = float(np.clip(
            1 + np.dot(relative_y, vertical_residual)
            / max(1e-6, np.dot(relative_y, relative_y)),
            0.8, 1.25,
        ))

    measurement = np.array([
        center_x + median_motion[0],
        center_y + median_motion[1],
        max(1, width * horizontal_scale),
        max(1, height * vertical_scale),
    ])
    measurement_sigma = max(3.0, 1.4826 * dispersion)

    return measurement, measurement_sigma


def correct(track, measurement, sigma):
    """Update the state mean using flow; retain covariance and lifecycle fields."""
    normalization = np.array([track._sx, track._sy, track._sx, track._sy])
    normalized_measurement = measurement / normalization
    measurement_noise = np.diag((sigma / normalization) ** 2)

    innovation_covariance = track.H @ track.P @ track.H.T + measurement_noise
    kalman_gain = np.linalg.solve(
        innovation_covariance,
        (track.P @ track.H.T).T,
    ).T

    residual = normalized_measurement - track.H @ track.x
    track.x = track.x + kalman_gain @ residual

    # Keep width and height at least one pixel in normalized coordinates.
    track.x[2:4] = np.maximum(track.x[2:4], 1 / normalization[2:4])

    # Deliberately no covariance reduction or age/lifecycle/confidence reset.


def make_tracker(backend, mode='selective'):
    """Build a controller using the supplied optical-flow backend.

    Modes: off disables flow; every attempts it on each otherwise eligible
    skipped frame; selective additionally checks predicted movement and age.
    The controller inherits the adaptive tracker constructor without changing defaults.
    """
    assert mode in ['off', 'every', 'selective']

    class FlowAdaptive(AdaptiveDetectionTracker):
        def step(self, frame, model, device):
            previous_boxes = {
                track.track_id: track.get_box_cxcywh().copy()
                for track in self.tracks
            }
            previous_image = getattr(self, 'previous_image', None)
            frame_index = self.frame_idx

            # Let the adaptive scheduler decide and process the current frame.
            result = super().step(frame, model, device)
            result.update(
                ran_flow=False,
                flow_corrected=0,
                flow_rejected=0,
                flow_failed=False,
            )
            start = time.perf_counter()

            should_run_flow = (
                mode != 'off'
                and previous_image is not None
                and not result['ran_inference']
                and not result['ran_erd']
                and self.tracks
                and (mode == 'every' or eligible(self.tracks, previous_boxes))
            )

            if should_run_flow:
                result['ran_flow'] = True

                try:
                    field = backend.acquire(previous_image, frame, frame_index)
                except (RuntimeError, ValueError, OSError) as error:
                    result['flow_failed'] = True
                    result['flow_error'] = repr(error)
                    field = None

                if field is not None:
                    for track in self.tracks:
                        if track.track_id not in previous_boxes:
                            continue

                        flow_observation = observation(field, previous_boxes[track.track_id])

                        if flow_observation is None:
                            result['flow_rejected'] += 1
                            continue

                        correct(track, *flow_observation)
                        result['flow_corrected'] += 1

            result['flow_path_ms'] = (time.perf_counter() - start) * 1000
            self.previous_image = frame

            # Preserve the output schema, emitting only confirmed tracks.
            result['detections'] = [
                {
                    'track_id': track.track_id,
                    'class_name': track.class_name,
                    'box_xyxy': track.get_box_xyxy().tolist(),
                    'conf': track.get_confidence(self.uncertainty_threshold),
                    'state': track.state,
                }
                for track in self.tracks
                if track.state == 'confirmed'
            ]

            return result

    return FlowAdaptive
