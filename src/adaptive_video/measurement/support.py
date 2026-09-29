"""Variant construction, state snapshots and repeat-consistency checks."""

import hashlib

from .. import dataset as common
from ..tracking.kalman import BoxTrack
from ..tracking.output import digest, emit, make, state
from ..variants.registry import VARIANTS
from . import guard


def configure(variant):
    """Reset identities and counters, then construct this variant's controller."""
    if variant == 'always_on':
        BoxTrack.reset_id_counter()
        return None, None, None

    controller_type, track_type, emission = VARIANTS[variant]
    track_type.reset_id_counter()

    for counter in ('domain_failures_total', 'correction_restarts_total'):
        if hasattr(track_type, counter):
            setattr(track_type, counter, 0)

    return make(controller_type, track_type), track_type, emission


def extended_state(detector, track_cls):
    """Capture track state and any variant-specific memory for exact comparisons."""
    if detector is None:
        return None

    snapshot = {
        'frame_idx': detector.frame_idx,
        'tracks': [common.plain(track.__dict__) for track in detector.tracks],
        'next_track_id': track_cls._id_counter,
    }

    if hasattr(detector, 'templates'):
        snapshot['templates'] = detector.templates

    if hasattr(detector, 'previous_gray'):
        snapshot['previous_gray_sha256'] = hashlib.sha256(detector.previous_gray.tobytes()).hexdigest()

    for field in ['domain_failures_total', 'correction_restarts_total']:
        if hasattr(track_cls, field):
            snapshot[field] = getattr(track_cls, field)

    return snapshot


def exact_repeat(a, b):
    """Require repeated measurements to use identical predictions and decisions."""
    fields = ['outputs', 'trace', 'detector_raw_hashes', 'calls', 'extended_final_state_sha256']

    for field in fields:
        assert a[field] == b[field], (a['variant'], a['video'], field, 'repeat drift')
