"""Collect causal flow alternatives, then label them offline against annotations."""

from pathlib import Path
import time

from ..dataset import csvwrite, read, save
from ..evaluation.coco import category, coco, ground_truth
from ..evaluation.dataset import EvaluationDataset
from ..evaluation.diagnostics import greedy, iou
from ..tracking.output import digest, emit, state
from .core import analyze, measure, outputs


def collect_video(source, video, tracker, model, backend, device='cuda', synchronize=None):
    """Observe a baseline run without feeding flow back into its live state.

    Dependencies are supplied by the caller. Flow features are computed before
    the current detector invocation. This function never reads annotations.
    """
    frames = source.frames(video)
    rows = []
    detector_samples = []

    class TimedDetector:
        def predict(self, *args, **kwargs):
            start = time.perf_counter()
            result = model.predict(*args, **kwargs)

            if synchronize is not None:
                synchronize()

            detector_samples.append((time.perf_counter() - start) * 1000)

            return result

    timed_model = TimedDetector()
    previous_image = None

    for index, frame in enumerate(frames):
        image = source.image(frame)
        previous_boxes = {
            track.track_id: track.get_box_cxcywh().copy()
            for track in tracker.tracks
        }
        before = digest(state(tracker))
        predicted, reasons = analyze(tracker)
        alternative = None
        failure = None

        if reasons['eligible'] and previous_image is not None:
            start = time.perf_counter()

            try:
                field = backend.acquire(previous_image, image, index)
                corrected, direct, features, objects, correction_ms = measure(
                    predicted, previous_boxes, field,
                )
                alternative = {
                    'predicted': outputs(predicted, tracker.uncertainty_threshold),
                    'weighted': outputs(corrected, tracker.uncertainty_threshold),
                    'direct': outputs(direct, tracker.uncertainty_threshold),
                    'features': features,
                    'objects': objects,
                    'correction_ms': correction_ms,
                    'flow_and_correction_ms': (time.perf_counter() - start) * 1000,
                }
            except (RuntimeError, ValueError, OSError) as error:
                failure = repr(error)

        if digest(state(tracker)) != before:
            raise RuntimeError('Shadow computation modified the baseline tracker')

        calls_before = len(detector_samples)
        result = tracker.step(image, timed_model, device)

        if reasons['eligible'] and not result['ran_inference']:
            raise RuntimeError('Candidate classification disagrees with baseline decision')

        rows.append({
            'video': video,
            'index': index,
            'image_id': frame['image_id'],
            'ran_detector': result['ran_inference'],
            'reasons': reasons,
            'baseline': emit(tracker, 'constant_acceleration'),
            'alternative': alternative,
            'failure': failure,
            'baseline_state_sha256': digest(state(tracker)),
            'detector_ms': detector_samples[calls_before:],
        })
        previous_image = image

    return rows


def collect(dataset_manifest, output):
    """Generate a new diagnostic capture using hardware flow, one file per complete video."""
    from ..measurement.support import configure
    from ..detectors.factory import load_detector
    from .flow_backend import HardwareFlow


    import torch

    from ..project import verify

    verify(dataset_manifest=dataset_manifest)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = EvaluationDataset(dataset_manifest)
    model = load_detector()
    save(destination / 'dataset.json', source.manifest)
    save(destination / 'capture.json', {
        'schema_version': 1,
        'videos': source.videos,
        'scope': 'Unchanged baseline with one-step shadow observations; not a resource benchmark',
    })

    detector_costs = []

    for video in source.videos:
        tracker, _, _ = configure('constant_acceleration')
        warmup_image = source.image(source.frames(video)[0])

        for _ in range(3):
            model.predict(warmup_image, imgsz=(736, 1280), conf=0.5, verbose=False)

        torch.cuda.synchronize()
        rows = collect_video(source, video, tracker, model, HardwareFlow(),
                             synchronize=torch.cuda.synchronize)
        save(destination / 'videos' / f'{video}.json.gz', rows)
        detector_costs.extend(
            {'video': video, 'index': row['index'], 'detector_ms': elapsed}
            for row in rows for elapsed in row['detector_ms']
        )

    csvwrite(destination / 'detector_cost.csv', detector_costs)


def label(detections, baseline, annotations):
    """Offline event label: preserve baseline matches without adding false positives."""
    result = {}
    safe = True

    for threshold, suffix in [(0.5, '50'), (0.75, '75')]:
        matches, matched_truth, _ = greedy(detections, annotations, threshold)
        baseline_matches, baseline_truth, _ = greedy(baseline, annotations, threshold)
        false_positives = len(detections) - len(matches)
        baseline_false_positives = len(baseline) - len(baseline_matches)
        result.update({
            f'TP{suffix}': len(matches),
            f'FP{suffix}': false_positives,
            f'base_TP{suffix}': len(baseline_matches),
            f'base_FP{suffix}': baseline_false_positives,
            f'lost_gt{suffix}': len(baseline_truth - matched_truth),
        })
        safe = safe and baseline_truth.issubset(matched_truth)
        safe = safe and false_positives <= baseline_false_positives

    result['safe'] = bool(safe and result['base_TP50'] > 0 and result['TP50'] > 0)

    return result


def prediction_rows(detections, image_id):
    rows = []

    for detection in detections:
        left, top, right, bottom = detection['box_xyxy']
        rows.append({
            'image_id': image_id,
            'category_id': category(detection),
            'bbox': [left, top, right - left, bottom - top],
            'score': detection['conf'],
        })

    return rows


def object_diagnostics(alternative, baseline, truth, previous_outputs, previous_truth):
    """Attribute geometry failures using previous matches, only during labeling."""
    previous_matches, _, _ = greedy(previous_outputs, previous_truth, 0.5)
    identities = {
        previous_outputs[index]['track_id']: previous_truth[matched]['track_id']
        for index, matched in previous_matches.items()
    }
    current_truth = {item['track_id']: item for item in truth}
    quality = {item['track_id']: item for item in alternative['objects']}
    objects = []

    for detection in alternative['weighted']:
        identity = identities.get(detection['track_id'])
        annotation = current_truth.get(identity)

        if annotation is None:
            continue

        flow_iou = float(iou([detection['box_xyxy']], [annotation['box_xyxy']])[0, 0])
        baseline_iou = max((
            float(iou([item['box_xyxy']], [annotation['box_xyxy']])[0, 0])
            for item in baseline if category(item) == annotation['class_id']
        ), default=0.0)
        objects.append({
            **quality[detection['track_id']], 'gt_id': identity,
            'flow_iou': flow_iou, 'baseline_best_iou': baseline_iou,
            'catastrophic': flow_iou < 0.1 and baseline_iou >= 0.75,
        })

    _, represented_truth, _ = greedy(baseline, truth, 0.5)
    unsupported = sum(truth[index]['track_id'] not in identities.values()
                      for index in represented_truth)

    return objects, unsupported


def run(capture, output):
    """Evaluate complete sequences from an explicit newly generated capture."""
    capture = Path(capture).resolve()
    destination = Path(output).resolve()
    source = EvaluationDataset(capture / 'dataset.json')
    metadata = read(capture / 'capture.json')

    if metadata['schema_version'] != 1 or metadata['videos'] != source.videos:
        raise ValueError('Capture population does not match dataset inventory')

    destination.mkdir(parents=True, exist_ok=False)
    ground_truth_data, annotations = ground_truth(source)
    predictions = {name: [] for name in ['baseline', 'predicted', 'weighted', 'direct']}
    features = []
    labels = []
    actions = []
    objects = []

    for video in source.videos:
        rows = read(capture / 'videos' / f'{video}.json.gz')
        frames = source.frames(video)

        if len(rows) != len(frames):
            raise ValueError(f'Incomplete capture: {video}')

        for index, (row, frame) in enumerate(zip(rows, frames)):
            if (row['video'], row['index'], row['image_id']) != (video, index, frame['image_id']):
                raise ValueError(f'Frame identity mismatch: {video}, {index}')

            identity = {'video': video, 'index': index, 'image_id': frame['image_id']}
            alternative = row['alternative']
            actions.append({**identity, **row['reasons'],
                            'ran_detector': row['ran_detector'], 'failure': row['failure']})

            for name in predictions:
                detections = row['baseline']

                if alternative is not None and name != 'baseline':
                    detections = alternative[name]

                predictions[name].extend(prediction_rows(detections, frame['image_id']))

            if alternative is None:
                continue

            if not row['reasons']['eligible'] or not row['ran_detector']:
                raise ValueError('Shadow alternatives require an eligible detector invocation')

            features.append({**identity, **alternative['features'],
                             'flow_and_correction_ms': alternative['flow_and_correction_ms']})
            truth = annotations[frame['image_id']]
            weighted_label = label(alternative['weighted'], row['baseline'], truth)
            previous_truth = annotations[frames[index - 1]['image_id']] if index else []
            previous_outputs = rows[index - 1]['baseline'] if index else []
            object_rows, unsupported = object_diagnostics(
                alternative, row['baseline'], truth, previous_outputs, previous_truth,
            )
            catastrophic = sum(item['catastrophic'] for item in object_rows)
            weighted_label['safe'] = weighted_label['safe'] and catastrophic == 0
            weighted_label['catastrophic'] = catastrophic
            weighted_label['unsupported_gt'] = unsupported
            labels.append({
                **identity, **weighted_label,
                'raw_safe': label(alternative['direct'], row['baseline'], truth)['safe'],
                'kalman_safe': label(alternative['predicted'], row['baseline'], truth)['safe'],
            })
            objects.extend({**identity, **item} for item in object_rows)

    image_ids = [frame['image_id'] for frame in source.frames()]
    metrics = {name: coco(ground_truth_data, rows, image_ids)[0]
               for name, rows in predictions.items()}
    csvwrite(destination / 'features.csv', features)
    csvwrite(destination / 'labels.csv', labels)
    csvwrite(destination / 'actions.csv', actions)
    csvwrite(destination / 'objects.csv', objects)
    save(destination / 'shadow_AP.json', {
        'scope': 'One-step substitutions on unchanged baseline history; not a causal policy',
        'metrics': metrics,
    })
    save(destination / 'analysis.json', {
        'capture': str(capture), 'frames': len(image_ids), 'events': len(labels),
        'failed_flow_events': sum(row['failure'] is not None for row in actions),
    })
