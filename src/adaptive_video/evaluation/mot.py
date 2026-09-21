"""Multi-object tracking metrics for ordered, per-video prediction sequences.

TrackEval performs the matching and metric calculations. This adapter uses the
same six mapped classes and continuous xyxy geometry as our COCO evaluator.
It is a project protocol, not the official eight-class BDD100K leaderboard.
"""

from importlib.metadata import version

import numpy as np

from .coco import IDS, category
from .diagnostics import iou

try:
    from trackeval.metrics import CLEAR, HOTA, Identity
except ModuleNotFoundError as error:
    if error.name != 'trackeval':
        raise

    CLEAR = HOTA = Identity = None


METRIC_NAMES = ('MOTA', 'IDF1', 'HOTA', 'DetA', 'AssA', 'LocA')


def validate_frames(frames, label):
    """Reject ambiguous identities or invalid geometry before matching."""

    for frame_index, objects in enumerate(frames):
        seen = set()

        for obj in objects:
            identity = obj.get('track_id')

            if isinstance(identity, bool) or not isinstance(identity, (str, int)):
                raise ValueError(f'{label} frame {frame_index}: track_id must be an integer or string')

            identity = str(identity)

            if not identity or identity in seen:
                raise ValueError(f'{label} frame {frame_index}: empty or duplicate track_id')

            seen.add(identity)

            if category(obj) not in IDS:
                raise ValueError(f'{label}: class is outside the evaluation protocol')

            box = np.asarray(obj['box_xyxy'], dtype=float)

            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError(f'{label} frame {frame_index}: invalid xyxy box')


def sequence_data(ground_truth_frames, prediction_frames, class_id):
    """Build TrackEval input for one class in one video.

    Each side gets its own compact identity mapping, maintained across frames
    and reset for the next video. Ground-truth IDs never assign tracker IDs.
    """

    gt_mapping = {}
    prediction_mapping = {}
    gt_ids = []
    prediction_ids = []
    similarities = []

    for gt_frame, prediction_frame in zip(ground_truth_frames, prediction_frames):
        gt_objects = [obj for obj in gt_frame if category(obj) == class_id]
        predicted_objects = [obj for obj in prediction_frame if category(obj) == class_id]

        for obj in gt_objects:
            gt_mapping.setdefault(str(obj['track_id']), len(gt_mapping))

        for obj in predicted_objects:
            prediction_mapping.setdefault(str(obj['track_id']), len(prediction_mapping))

        gt_ids.append(np.array([gt_mapping[str(obj['track_id'])] for obj in gt_objects], dtype=int))
        prediction_ids.append(np.array([
            prediction_mapping[str(obj['track_id'])] for obj in predicted_objects
        ], dtype=int))
        similarities.append(iou(
            [obj['box_xyxy'] for obj in gt_objects],
            [obj['box_xyxy'] for obj in predicted_objects],
        ))

    return {
        'num_timesteps': len(ground_truth_frames),
        'num_gt_ids': len(gt_mapping),
        'num_tracker_ids': len(prediction_mapping),
        'num_gt_dets': sum(len(ids) for ids in gt_ids),
        'num_tracker_dets': sum(len(ids) for ids in prediction_ids),
        'gt_ids': gt_ids,
        'tracker_ids': prediction_ids,
        'similarity_scores': similarities,
    }


def summary(results):
    """Return ratios (not percentages); average HOTA over its 19 IoU thresholds."""

    clear = results['CLEAR']
    identity = results['Identity']
    hota = results['HOTA']
    has_ground_truth = clear['CLR_TP'] + clear['CLR_FN'] > 0
    values = {
        'MOTA': float(clear['MOTA']),
        'IDF1': float(identity['IDF1']),
        **{name: float(np.mean(hota[name])) for name in ('HOTA', 'DetA', 'AssA', 'LocA')},
    }

    if not has_ground_truth:
        values = dict.fromkeys(METRIC_NAMES)

    values.update({
        'TP': int(clear['CLR_TP']),
        'FP': int(clear['CLR_FP']),
        'FN': int(clear['CLR_FN']),
        'ID_switches': int(clear['IDSW']),
        'IDTP': int(identity['IDTP']),
        'IDFP': int(identity['IDFP']),
        'IDFN': int(identity['IDFN']),
    })

    return values


def mot(ground_truth_sequences, prediction_sequences):
    """Compute MOTA, IDF1 and HOTA without merging identities across videos.

    Both arguments map video names to lists of frames. A frame is a list of
    dictionaries with box_xyxy, class_id (or class_name) and track_id. Empty
    frames must be retained. Callers must supply the same complete frame order.
    Returns (overall_metrics, details), like evaluation.coco.coco().
    """

    if not ground_truth_sequences or set(ground_truth_sequences) != set(prediction_sequences):
        raise ValueError('Ground truth and predictions must contain the same nonempty video set')

    if HOTA is None:
        raise ImportError('Install tracking metrics: python -m pip install --no-deps -r environments/requirements.txt')

    algorithms = {
        'CLEAR': CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}),
        'Identity': Identity({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}),
        'HOTA': HOTA(),
    }
    results = {name: {class_id: {} for class_id in IDS} for name in algorithms}
    per_video = {}

    for video, gt_frames in ground_truth_sequences.items():
        predicted_frames = prediction_sequences[video]

        if not gt_frames or len(gt_frames) != len(predicted_frames):
            raise ValueError(f'{video}: predictions must cover every ground-truth frame')

        validate_frames(gt_frames, f'{video} ground truth')
        validate_frames(predicted_frames, f'{video} predictions')

        for class_id in IDS:
            data = sequence_data(gt_frames, predicted_frames, class_id)

            for name, algorithm in algorithms.items():
                results[name][class_id][video] = algorithm.eval_sequence(data)

        per_video[video] = summary({
            name: algorithm.combine_classes_det_averaged({
                class_id: results[name][class_id][video] for class_id in IDS
            })
            for name, algorithm in algorithms.items()
        })

    # Combine sequence statistics first, then detection-weighted class statistics.
    # This is not an arithmetic average of per-video or per-class final scores.
    per_class_results = {
        name: {class_id: algorithm.combine_sequences(results[name][class_id]) for class_id in IDS}
        for name, algorithm in algorithms.items()
    }
    combined = {
        name: algorithm.combine_classes_det_averaged(per_class_results[name])
        for name, algorithm in algorithms.items()
    }
    details = {
        'status': 'computed',
        'backend': f'TrackEval {version("trackeval")}',
        'aggregation': 'combine_sequences_then_detection_averaged_classes',
        'class_ids': list(IDS),
        'clear_identity_iou_threshold': 0.5,
        'hota_iou_thresholds': algorithms['HOTA'].array_labels.tolist(),
        'prediction_policy': 'all emitted boxes; no additional confidence cutoff or maxDets cap',
        'annotation_policy': 'same mapped annotations as COCO; no extra crowd/ignore-region preprocessing',
        'per_video': per_video,
        'per_class': {
            str(class_id): summary({name: per_class_results[name][class_id] for name in algorithms})
            for class_id in IDS
        },
        'hota_by_threshold': combined['HOTA']['HOTA'].tolist(),
    }

    return summary(combined), details
