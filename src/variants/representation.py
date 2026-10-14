"""CV8 in alternative box coordinates, measured in original normalized box units.
"""

import numpy as np

from ..tracking.kalman import BoxTrack


class RepresentedCV(BoxTrack):
    representation = 'linear'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        box = self.x[:4].copy()
        original_covariance = self.P[:8, :8].copy()
        self.box_process_q = self.Q[:8, :8].copy()
        self.x = np.r_[self.encode(box), np.zeros(4)]
        inverse = np.linalg.inv(self.full_jacobian())
        self.P = inverse @ original_covariance @ inverse.T
        self.Q = inverse @ self.box_process_q @ inverse.T
        self.F = np.eye(8)
        self.F[np.arange(4), np.arange(4) + 4] = 1.0
        self.H = self.observation_jacobian()

    def encode(self, box):
        """Convert normalized box geometry into the chosen state coordinates."""
        coordinates = np.array(box, dtype=np.float64, copy=True)

        if self.representation == 'log':
            coordinates[2:] = np.log(coordinates[2:])

        elif self.representation == 'aspect':
            coordinates[2] /= coordinates[3]

        return coordinates

    def geometry(self):
        """Return box geometry and the Jacobians mapping state position and velocity."""
        coordinates = self.x[:4]
        velocity = self.x[4:]
        box = coordinates.copy()
        geometry_jacobian = np.eye(4)
        velocity_position_jacobian = np.zeros((4, 4))

        if self.representation == 'log':
            box[2:] = np.exp(coordinates[2:])
            geometry_jacobian[2, 2] = box[2]
            geometry_jacobian[3, 3] = box[3]
            velocity_position_jacobian[2, 2] = box[2] * velocity[2]
            velocity_position_jacobian[3, 3] = box[3] * velocity[3]

        elif self.representation == 'aspect':
            box[2] = coordinates[2] * coordinates[3]
            geometry_jacobian[2, 2] = coordinates[3]
            geometry_jacobian[2, 3] = coordinates[2]
            velocity_position_jacobian[2, 2] = velocity[3]
            velocity_position_jacobian[2, 3] = velocity[2]

        return box, geometry_jacobian, velocity_position_jacobian

    def full_jacobian(self):
        _, geometry_jacobian, velocity_position_jacobian = self.geometry()

        return np.block(
            [
                [geometry_jacobian, np.zeros((4, 4))],
                [velocity_position_jacobian, geometry_jacobian],
            ]
        )

    def observation_jacobian(self):
        return np.c_[self.geometry()[1], np.zeros((4, 4))]

    def constrain(self):
        # Same one-pixel minimum as reference implementation, represented in chosen coordinates.
        box = self.geometry()[0]
        box[2:] = np.maximum(box[2:], [1 / self._sx, 1 / self._sy])
        self.x[:4] = self.encode(box)

        if not np.isfinite(self.x).all():
            raise FloatingPointError('Non-finite represented state')

    def predict(self):
        self.x = self.F @ self.x
        self.constrain()
        inverse = np.linalg.inv(self.full_jacobian())
        self.Q = inverse @ self.box_process_q @ inverse.T
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.H = self.observation_jacobian()

        self.age += 1
        self.frames_since_update += 1
        self.frames_since_infer += 1

        return self.get_box_cxcywh()

    def measurement(self, box_cxcywh_px):
        """Return the detector residual, observation Jacobian and innovation covariance."""
        measurement = np.asarray(box_cxcywh_px) / [
            self._sx,
            self._sy,
            self._sx,
            self._sy,
        ]
        observation_matrix = self.observation_jacobian()

        return (
            measurement - self.geometry()[0],
            observation_matrix,
            observation_matrix @ self.P @ observation_matrix.T + self.R,
        )

    def update(self, box_cxcywh_px, class_id, class_name, frame_idx, conf=1.0):
        residual, observation_matrix, innovation_covariance = self.measurement(
            box_cxcywh_px
        )
        kalman_gain = (
            self.P @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        )
        self.x = self.x + kalman_gain @ residual
        self.P = (np.eye(8) - kalman_gain @ observation_matrix) @ self.P
        self.constrain()
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

    def box_covariance(self):
        observation_matrix = self.observation_jacobian()

        return observation_matrix @ self.P @ observation_matrix.T

    def is_uncertain(self, threshold=0.0003):
        return float(np.trace(self.box_covariance())) > threshold

    def get_confidence(self, uncertainty_threshold=0.0003):
        return self.last_conf * max(
            0.1, 1 - np.trace(self.box_covariance()) / uncertainty_threshold
        )

    def get_box_cxcywh(self):
        return self.geometry()[0] * [self._sx, self._sy, self._sx, self._sy]

    def mahalanobis_sq(self, box_cxcywh_px):
        residual, _, innovation_covariance = self.measurement(box_cxcywh_px)

        try:
            return float(residual @ np.linalg.solve(innovation_covariance, residual))
        except np.linalg.LinAlgError:
            return 1e9


class LogSizeCV(RepresentedCV):
    representation = 'log'


class AspectHeightCV(RepresentedCV):
    representation = 'aspect'


TRACKS = {
    'linear_cv8_check': RepresentedCV,
    'log_cv8': LogSizeCV,
    'aspect_cv8': AspectHeightCV,
}
