"""Map experiment names to controller, track type and output mode."""

import numpy as np

from ..tracking.kalman import BoxTrack
from ..tracking.output import DetectorOutputTracker
from ..tracking.scheduler import AdaptiveDetectionTracker
from .appearance import AppearanceDetector
from .camera import CameraDetector
from .motion import ConstantVelocity, DampedAcceleration
from .projective_restart import ProjectiveRestartCV
from .representation import LogSizeCV, AspectHeightCV
from .visibility import VisibilityDetector, ExistenceTrack

VARIANTS = {
    'constant_acceleration': (AdaptiveDetectionTracker, BoxTrack, 'constant_acceleration'),
    'cv8': (DetectorOutputTracker, ConstantVelocity, 'detector_observations'),
    'damped_ca10': (DetectorOutputTracker, DampedAcceleration, 'detector_observations'),
    'log_cv8': (DetectorOutputTracker, LogSizeCV, 'detector_observations'),
    'aspect_cv8': (DetectorOutputTracker, AspectHeightCV, 'detector_observations'),
    'projective_restart_cv8': (
        DetectorOutputTracker,
        ProjectiveRestartCV,
        'detector_observations',
    ),
    'camera_ca10': (CameraDetector, BoxTrack, 'detector_observations'),
    'appearance_ca10': (AppearanceDetector, BoxTrack, 'detector_observations'),
    'visibility_ca10': (VisibilityDetector, ExistenceTrack, 'detector_observations'),
}


def payload(value, seen=None):
    """Count unique NumPy array storage reachable from a nested value."""
    seen = set() if seen is None else seen

    if id(value) in seen:
        return 0

    seen.add(id(value))

    if isinstance(value, np.ndarray):
        return value.nbytes

    if isinstance(value, dict):
        return sum(payload(v, seen) for v in value.values())

    if isinstance(value, (list, tuple)):
        return sum(payload(v, seen) for v in value)

    return 0


def compare_outputs(got, expected):
    """Compare classes, geometry and confidence for corresponding frames."""
    assert len(got) == len(expected)

    for actual_frame, expected_frame in zip(got, expected):
        assert len(actual_frame) == len(expected_frame)

        for actual_detection, expected_detection in zip(actual_frame, expected_frame):
            assert (
                actual_detection['class_id'] == expected_detection['class_id']
                and actual_detection['class_name'] == expected_detection['class_name']
            )
            assert np.allclose(
                actual_detection['box_xyxy'],
                expected_detection['box_xyxy'],
                rtol=0,
                atol=1e-9,
            )
            assert np.isclose(
                actual_detection['conf'], expected_detection['conf'], rtol=0, atol=1e-9
            )
