"""Constant-velocity and damped-acceleration box motion models."""

import numpy as np

from ..tracking.kalman import BoxTrack


class ConstantVelocity(BoxTrack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = self.x[:8].copy()
        self.P = self.P[:8, :8].copy()
        self.Q = self.Q[:8, :8].copy()
        self.F = np.eye(8)
        self.F[np.arange(4), np.arange(4) + 4] = 1
        self.H = np.eye(4, 8)

    def update(self, box_cxcywh_px, class_id, class_name, frame_idx, conf=1.0):
        center_x, center_y, width, height = box_cxcywh_px
        measurement = np.array(
            [
                center_x / self._sx,
                center_y / self._sy,
                width / self._sx,
                height / self._sy,
            ],
            dtype=np.float64,
        )
        residual = measurement - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x + kalman_gain @ residual
        self.P = (np.eye(8) - kalman_gain @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1 / self._sx)
        self.x[3] = max(self.x[3], 1 / self._sy)

        self.hits += 1
        self.frames_since_update = 0
        self.frames_since_infer = 0
        self.last_seen_frame = frame_idx
        self.last_conf = conf
        self.class_id = class_id
        self.class_name = class_name

        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'


class DampedAcceleration(BoxTrack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Current-frame motion uses previous acceleration, next stored acceleration
        # decays by .5; F propagates covariance with the same transition.
        self.F[8, 8] = 0.5
        self.F[9, 9] = 0.5


TRACKS = {
    'original': BoxTrack,
    'raw_both': BoxTrack,
    'cv8': ConstantVelocity,
    'damped_ca10': DampedAcceleration,
}
