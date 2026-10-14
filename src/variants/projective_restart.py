"""Restart a relative-depth track when correction leaves its valid domain."""

import numpy as np

from .projective import ProjectiveCV


class ProjectiveRestartCV(ProjectiveCV):
    correction_restarts_total = 0

    def update(self, box_cxcywh_px, class_id, class_name, frame_idx, conf=1.0):
        residual, observation_matrix, innovation_covariance = self.measurement(
            box_cxcywh_px
        )
        kalman_gain = (
            self.P @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        )
        candidate = self.x + kalman_gain @ residual

        if candidate[2] > 0 and np.isfinite(candidate).all():
            return super().update(box_cxcywh_px, class_id, class_name, frame_idx, conf)
        # No new detector invocation: use the actual already-matched observation.
        # Restart with original birth uncertainty, not a clamped confident EKF.

        type(self).correction_restarts_total += 1
        normalized = np.asarray(box_cxcywh_px) / [
            self._sx,
            self._sy,
            self._sx,
            self._sy,
        ]
        self.x = np.r_[self.encode(normalized), np.zeros(4)]
        inverse = np.linalg.inv(self.full_jacobian())
        birth_covariance = np.diag(
            [1e-4, 1e-4, 2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5]
        )
        self.P = inverse @ birth_covariance @ inverse.T
        self.Q = inverse @ self.box_process_q @ inverse.T
        self.H = self.observation_jacobian()

        self.hits += 1
        self.frames_since_update = 0
        self.frames_since_infer = 0
        self.last_seen_frame = frame_idx
        self.last_conf = conf
        self.class_id = class_id
        self.class_name = class_name

        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'

        self.force_detector = False


TRACKS = {'projective_restart_cv8': ProjectiveRestartCV}
