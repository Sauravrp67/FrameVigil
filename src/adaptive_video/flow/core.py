"""Analyze computed optical flow without changing the live tracker.

This module does not load images, calculate optical flow, read a flow cache, or
use ground truth. A caller supplies the flow field. A live caller supplies newly computed fields.
"""

import copy
import time

import numpy as np

from .extension import observation, correct


def analyze(tracker):
    """Predict copied tracks and identify uncertainty-only detector triggers."""
    tracks = copy.deepcopy(tracker.tracks)

    for track in tracks:
        track.predict()

    tracks = [
        track for track in tracks
        if not track.is_out_of_frame(tracker.boundary_margin)
        and track.state != 'lost'
    ]

    uncertainty = any(
        track.is_uncertain(tracker.uncertainty_threshold) for track in tracks
    )
    periodic = bool(tracks and tracker.frame_idx % tracker.min_infer_every == 0)
    skip_limit = any(track.needs_forced_infer(tracker.max_skip) for track in tracks)
    age_limit = any(track.frames_since_update >= 15 for track in tracks)
    lifecycle_requires_detection = any(
        track.state != 'confirmed' or track.consecutive_rescues > 0
        for track in tracks
    )

    eligible = bool(
        tracks
        and uncertainty
        and not (periodic or skip_limit or age_limit or lifecycle_requires_detection)
    )

    # Preserve field names used by saved diagnostics and their consumers.
    reasons = {
        'no_tracks': not tracks,
        'uncertainty': uncertainty,
        'periodic': periodic,
        'cap': skip_limit,
        'real_cap': age_limit,
        'lifecycle': lifecycle_requires_detection,
        'eligible': eligible,
    }

    return tracks, reasons


def outputs(tracks, u=0.0003):
    """Convert tracks to detection rows; u is the confidence-decay threshold."""
    detections = []

    for track in tracks:
        detections.append({
            'box_xyxy': track.get_box_xyxy().tolist(),
            'conf': track.get_confidence(u),
            'class_id': track.class_id,
            'class_name': track.class_name,
            'track_id': track.track_id,
        })

    return detections


def measure(tracks, old, field):
    """Compare flow corrections on copies and summarize measurement quality.

    Inputs:
        tracks: Nonempty list of current-frame Kalman-predicted tracks.
        old: Previous boxes, indexed by track ID, each [cx, cy, w, h] in pixels.
        field: Flow array of shape (grid_height, grid_width, 2), containing
            [dx, dy] in pixels on a four-pixel grid. Raw fixed-point VPI values
            must already have been converted by the flow backend.

    Returns:
        Kalman-weighted tracks, direct-flow tracks, frame-level features,
        per-object quality rows, and elapsed computation milliseconds.

    Neither alternative modifies input tracks. The weighted correction preserves
    covariance, age and stored confidence. This function does not decide to skip
    a detector. Its timer excludes image decoding and optical-flow computation.
    """
    start = time.perf_counter()
    corrected_tracks = copy.deepcopy(tracks)
    direct_flow_tracks = copy.deepcopy(tracks)
    quality_rows = []

    for corrected_track, direct_track in zip(corrected_tracks, direct_flow_tracks):
        previous_box = old[corrected_track.track_id]
        center_x, center_y, width, height = previous_box
        box_diagonal = max(1.0, np.hypot(width, height))
        flow_observation = observation(field, previous_box)

        # Flow samples lie at pixel centers (grid_index + 0.5) * 4.
        # Select samples inside this object's previous bounding box.
        column_start = max(0, int(np.ceil((center_x - width / 2) / 4 - 0.5)))
        column_end = min(field.shape[1], int(np.ceil((center_x + width / 2) / 4 - 0.5)))
        row_start = max(0, int(np.ceil((center_y - height / 2) / 4 - 0.5)))
        row_end = min(field.shape[0], int(np.ceil((center_y + height / 2) / 4 - 0.5)))

        vectors = field[row_start:row_end, column_start:column_end].reshape(-1, 2)
        finite_mask = np.isfinite(vectors).all(axis=1)
        finite_vectors = vectors[finite_mask]

        # Retain the existing fallback values when no finite vectors exist.
        dispersion = 1.0
        disagreement = 1.0
        motion_spread = 1.0
        inlier_fraction = 0.0
        normalized_motion = np.zeros(2)

        if len(finite_vectors):
            median_motion = np.median(finite_vectors, axis=0)
            distances = np.linalg.norm(finite_vectors - median_motion, axis=1)
            median_distance = float(np.median(distances))
            dispersion = median_distance / box_diagonal
            inlier_threshold = max(2.0, 3 * median_distance)
            inlier_fraction = float((distances <= inlier_threshold).mean())
            normalized_motion = median_motion / box_diagonal

            predicted_motion = corrected_track.get_box_cxcywh()[:2] - previous_box[:2]
            disagreement = float(
                np.linalg.norm(predicted_motion - median_motion) / box_diagonal
            )
            motion_spread = float(np.quantile(distances, 0.9) / box_diagonal)

        finite_fraction = float(finite_mask.mean()) if len(vectors) else 0.0
        quality_rows.append({
            'track_id': corrected_track.track_id,
            'valid': flow_observation is not None,
            'n_vectors': len(vectors),
            'finite_fraction': finite_fraction,
            'dispersion': dispersion,
            'disagreement': disagreement,
            'inlier_fraction': inlier_fraction,
            'motion_spread': motion_spread,
            'motion': normalized_motion.tolist(),
        })

        if flow_observation is not None:
            measurement, measurement_sigma = flow_observation
            correct(corrected_track, measurement, measurement_sigma)

            normalization = np.array([
                direct_track._sx, direct_track._sy,
                direct_track._sx, direct_track._sy,
            ])
            direct_track.x[:4] = measurement / normalization

    # Track-context features refer to the original predicted tracks, before
    # either hypothetical correction. Flow-quality features summarize objects.
    features = {
        'n_tracks': len(tracks),
        'max_age': max(track.frames_since_update for track in tracks),
        'max_cov': max(float(np.trace(track.P[:4, :4])) for track in tracks),
        'min_diag': min(float(np.linalg.norm(track.get_box_cxcywh()[2:])) for track in tracks),
        'max_speed': max(float(np.linalg.norm(track.x[4:6])) for track in tracks),
        'invalid_fraction': float(np.mean([not row['valid'] for row in quality_rows])),
        'max_dispersion': max(row['dispersion'] for row in quality_rows),
        'max_disagreement': max(row['disagreement'] for row in quality_rows),
        'min_inlier_fraction': min(row['inlier_fraction'] for row in quality_rows),
        'max_motion_spread': max(row['motion_spread'] for row in quality_rows),
    }
    elapsed_ms = (time.perf_counter() - start) * 1000

    return corrected_tracks, direct_flow_tracks, features, quality_rows, elapsed_ms
