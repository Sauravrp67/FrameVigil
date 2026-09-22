"""Unit-height relative-depth CV state. This is not measured metric 3D.

Virtual camera uses normalized focal lengths 1 and principal point (.5,.5).
u=[X,Y,Z,A], where Z=1/h, X=(cx-.5)/h, Y=(cy-.5)/h, A=w/h.
Object height is fixed to one arbitrary unit, an explicit testable assumption.
"""

import numpy as np

from .representation import RepresentedCV


class ProjectiveCV(RepresentedCV):
    domain_failures_total = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_domain_failures = 0
        self.force_detector = False

    def encode(self, box):
        center_x, center_y, width, height = box

        return np.array(
            [
                (center_x - 0.5) / height,
                (center_y - 0.5) / height,
                1 / height,
                width / height,
            ],
            dtype=np.float64,
        )

    def geometry(self):
        horizontal_position, vertical_position, depth, aspect_ratio = self.x[:4]
        horizontal_velocity, vertical_velocity, depth_velocity, aspect_velocity = (
            self.x[4:]
        )

        if depth <= 0 or not np.isfinite(self.x).all():
            raise FloatingPointError('Relative-depth state outside positive-Z domain')

        box = np.array(
            [
                0.5 + horizontal_position / depth,
                0.5 + vertical_position / depth,
                aspect_ratio / depth,
                1 / depth,
            ]
        )
        geometry_jacobian = np.array(
            [
                [1 / depth, 0, -horizontal_position / depth**2, 0],
                [0, 1 / depth, -vertical_position / depth**2, 0],
                [0, 0, -aspect_ratio / depth**2, 1 / depth],
                [0, 0, -1 / depth**2, 0],
            ]
        )
        velocity_position_jacobian = np.zeros((4, 4))
        velocity_position_jacobian[0, 0] = -depth_velocity / depth**2
        velocity_position_jacobian[0, 2] = (
            -horizontal_velocity / depth**2
            + 2 * horizontal_position * depth_velocity / depth**3
        )
        velocity_position_jacobian[1, 1] = -depth_velocity / depth**2
        velocity_position_jacobian[1, 2] = (
            -vertical_velocity / depth**2
            + 2 * vertical_position * depth_velocity / depth**3
        )
        velocity_position_jacobian[2, 3] = -depth_velocity / depth**2
        velocity_position_jacobian[2, 2] = (
            -aspect_velocity / depth**2 + 2 * aspect_ratio * depth_velocity / depth**3
        )
        velocity_position_jacobian[3, 2] = 2 * depth_velocity / depth**3

        return box, geometry_jacobian, velocity_position_jacobian

    def predict(self):
        proposed = self.F @ self.x

        if proposed[2] <= 0 or not np.isfinite(proposed).all():
            # Retain the last valid box, grow uncertainty, zero extrapolation,
            # and demand an actual detector observation on this frame.
            self.depth_domain_failures += 1
            type(self).domain_failures_total += 1
            self.force_detector = True
            self.P = self.F @ self.P @ self.F.T + self.Q
            self.x[4:] = 0.0

            self.age += 1
            self.frames_since_update += 1
            self.frames_since_infer += 1

            return self.get_box_cxcywh()

        return super().predict()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.force_detector = False

    def is_uncertain(self, threshold=0.0003):
        return self.force_detector or super().is_uncertain(threshold)

    def is_out_of_frame(self, margin=0.1):
        center_x, center_y = self.geometry()[0][:2]

        return (
            center_x < -margin
            or center_x > 1 + margin
            or center_y < -margin
            or center_y > 1 + margin
        )


TRACKS = {'projective_cv8': ProjectiveCV}
