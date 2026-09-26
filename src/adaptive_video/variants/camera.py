"""Causal sparse-LK/RANSAC camera compensation for original CA10 tracks.

Sparse optical flow estimates background motion between consecutive frames.
An accepted affine transform adjusts each track mean and covariance.
"""

import time

import cv2
import numpy as np

from ..tracking.output import DetectorOutputTracker


def estimate(previous, current, mask):
    """Estimate background motion with forward/backward flow and RANSAC."""
    info = dict(valid=False, reason='no_previous', corners=0, survivors=0, inliers=0)

    if previous is None:
        return None, info

    points = cv2.goodFeaturesToTrack(previous, 100, 0.01, 5, mask=mask, blockSize=3)
    info['reason'] = 'insufficient_corners'

    if points is None:
        return None, info

    info['corners'] = len(points)
    flow_parameters = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    forward_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, **flow_parameters
    )

    if forward_points is None:
        info['reason'] = 'lk_forward_failed'

        return None, info

    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward_points, None, **flow_parameters
    )

    if backward_points is None:
        info['reason'] = 'lk_backward_failed'

        return None, info

    round_trip_error = np.linalg.norm(backward_points[:, 0] - points[:, 0], axis=1)
    valid_points = (
        (status[:, 0] > 0)
        & (backward_status[:, 0] > 0)
        & np.isfinite(round_trip_error)
        & (round_trip_error <= 1.5)
        & np.isfinite(forward_points[:, 0]).all(1)
    )
    valid_points &= (
        (forward_points[:, 0, 0] >= 0)
        & (forward_points[:, 0, 0] < 160)
        & (forward_points[:, 0, 1] >= 0)
        & (forward_points[:, 0, 1] < 90)
    )
    info['survivors'] = int(valid_points.sum())
    info['reason'] = 'insufficient_lk_survivors'

    if valid_points.sum() < 6:
        return None, info

    source_points = points[valid_points, 0]
    destination_points = forward_points[valid_points, 0]
    cv2.setRNGSeed(20261028)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        destination_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=500,
        confidence=0.99,
        refineIters=10,
    )
    info['reason'] = 'ransac_failed'

    if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
        return None, info

    inliers = inlier_mask[:, 0].astype(bool)
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    info.update(
        inliers=int(inliers.sum()), inlier_fraction=float(inliers.mean()), scale=scale
    )
    info['reason'] = 'ransac_quality_rejected'

    if inliers.sum() < 6 or inliers.mean() < 0.5 or not 0.8 <= scale <= 1.2:
        return None, info

    residual = source_points @ matrix[:, :2].T + matrix[:, 2] - destination_points
    variance = max(0.25, float(np.mean(residual[inliers] ** 2)))
    info.update(
        valid=True,
        reason='accepted',
        residual_variance_small_pixel2=variance,
        matrix=matrix.tolist(),
    )

    return matrix, info


def transform_track(track, matrix, residual_variance):
    """Apply camera motion to the state and inflate covariance for residual error."""
    scale = np.diag([160.0, 90.0])
    normalized_transform = np.linalg.inv(scale) @ matrix[:, :2] @ scale
    normalized_translation = np.linalg.inv(scale) @ matrix[:, 2]
    state_transform = np.zeros((10, 10))

    for start, block in [
        (0, normalized_transform),
        (2, np.abs(normalized_transform)),
        (4, normalized_transform),
        (6, np.abs(normalized_transform)),
        (8, normalized_transform),
    ]:
        state_transform[start : start + 2, start : start + 2] = block

    track.x = state_transform @ track.x
    track.x[:2] += normalized_translation
    track.P = state_transform @ track.P @ state_transform.T
    # Residual-based safety inflation. This is a heuristic, not calibrated
    # affine-parameter uncertainty. Never shrink P using visual correspondence.
    track.P[np.arange(4), np.arange(4)] += residual_variance * np.array(
        [1 / 160**2, 1 / 90**2, 4 / 160**2, 4 / 90**2]
    )


class CameraDetector(DetectorOutputTracker):
    def step(self, frame, model, device):
        start = time.perf_counter()
        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (160, 90),
            interpolation=cv2.INTER_AREA,
        )
        previous = getattr(self, 'previous_gray', None)
        mask = np.full((90, 160), 255, np.uint8)

        for track in self.tracks:
            x1, y1, x2, y2 = track.get_box_xyxy().astype(float) * [
                160 / self.frame_width,
                90 / self.frame_height,
                160 / self.frame_width,
                90 / self.frame_height,
            ]
            mask[
                max(0, int(y1)) : min(90, int(np.ceil(y2)) + 1),
                max(0, int(x1)) : min(160, int(np.ceil(x2)) + 1),
            ] = 0

        matrix, info = estimate(previous, gray, mask)
        extraction = time.perf_counter()

        if matrix is not None:
            for track in self.tracks:
                transform_track(track, matrix, info['residual_variance_small_pixel2'])

        finish = time.perf_counter()
        self.previous_gray = gray
        self.camera_info = dict(
            **info,
            extraction_ms=1000 * (extraction - start),
            transform_ms=1000 * (finish - extraction),
            persistent_image_bytes=gray.nbytes,
            device_transfer_bytes=0,
            transformed_tracks=len(self.tracks) if matrix is not None else 0,
        )

        return super().step(frame, model, device)
