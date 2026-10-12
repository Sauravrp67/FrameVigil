"""Evaluate predictions and recorded live measurements.
"""

from collections import defaultdict
import math
from pathlib import Path

from .. import dataset
from ..evaluation.dataset import EvaluationDataset
from ..evaluation.coco import coco, ground_truth, predrows, localization_curve, frame_errors
from .statistics import timing_summary, frame_rows, power_rows, write_report_tables
from ..evaluation.mot import METRIC_NAMES, mot
from ..measurement.power import integrate
from ..project import ROOT, read, save


class TrialRecords:
    """Read one video trial at a time instead of retaining an entire campaign."""

    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        for path in self.paths:
            yield read(path)

    def __len__(self):
        return len(self.paths)


def load_results(folder, live):
    """Group saved trial outputs by variant; reject duplicate live trials."""

    paths = (
        (folder / 'results/live').glob('*.json.gz')
        if live else folder.glob('*/*.json.gz')
    )
    groups = defaultdict(list)
    seen_trials = set()

    for path in sorted(paths):
        record = read(path)

        if live:
            if record['status'] != 'PASS':
                raise ValueError(f'Incomplete trial: {path}')

            identity = (record['repeat'], record['variant'], record['video'])

            if identity in seen_trials:
                raise ValueError(f'Duplicate trial: {identity}')

            seen_trials.add(identity)

        groups[record['variant']].append(path)

    if not groups:
        raise ValueError('No generated outputs')

    return {variant: TrialRecords(paths) for variant, paths in groups.items()}


def validate_schedule(groups, protocol):
    """Require all scheduled variants and trials, with no extra results."""

    if set(groups) != set(protocol['arms']):
        raise ValueError('Results do not match the scheduled variants')

    expected = {
        (trial['repeat'], trial['variant'], trial['video'])
        for trial in protocol['schedule']
    }
    actual = {
        (trial['repeat'], trial['variant'], trial['video'])
        for records in groups.values()
        for trial in records
    }

    if len(expected) != len(protocol['schedule']):
        raise ValueError('Schedule contains duplicate trials')

    if actual != expected:
        raise ValueError('Results do not match the scheduled trials')


def collect_predictions(records, live, source):
    """Use complete videos and count each image once for accuracy."""

    predictions = []
    image_ids = []

    for record in records:
        frames = source.frames(record['video'])

        if len(frames) != len(record['outputs']):
            raise ValueError('Complete clips required; partial runs are not AP results')

        if live and record['frames'] != len(frames):
            raise ValueError('Recorded frame count differs from the dataset')

        # Repeats add resource observations, not independent accuracy samples.
        if live and record['repeat'] != 0:
            continue

        for frame, detections in zip(frames, record['outputs']):
            image_id = frame['image_id']
            image_ids.append(image_id)
            predictions.extend(predrows(detections, image_id))

    if not image_ids:
        raise ValueError('No accuracy frames; live evaluation requires repeat 0')

    if len(set(image_ids)) != len(image_ids):
        raise ValueError('Duplicate frame identity')

    return predictions, image_ids


def summarize_resources(folder, records):
    """Aggregate complete workload intervals, including detector-skipped frames.

    Power integration consistency does not independently validate the sensor.
    Telemetry validation must precede trials that request energy measurement.
    """

    total_frames = 0
    total_seconds = 0.0
    total_joules = 0.0
    energy_available = True
    detector_calls = 0

    for record in records:
        frame_count = record['frames']
        actions = record['trace']

        if frame_count <= 0 or len(actions) != frame_count:
            raise ValueError('Each measured frame must have an invocation decision')
        
        if [action['frame_idx'] for action in actions] != list(range(frame_count)):
            raise ValueError('Frame decisions must be consecutive and ordered')
        
        if any(type(action['ran_inference']) is not bool for action in actions):
            raise ValueError('Invocation decisions must be boolean')
        
        calls = sum(action['ran_inference'] for action in actions)
        
        if calls != record['calls']:
            raise ValueError('Invocation count disagrees with frame decisions')

        elapsed = record['wall_s']
        energy = record['energy_joules']
        
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Invalid workload duration')
        
        if energy is None:
            if record.get('power_backend') != 'none':
                raise ValueError('Missing energy for a power-measured trial')

            energy_available = False
        else:
            if not math.isfinite(energy) or energy < 0:
                raise ValueError('Invalid module energy')
        
            filename = f"r{record['repeat']}_{record['variant']}_{record['video']}.json"
            power = read(folder / 'results/power' / filename)
            measured_energy = integrate(power['samples'], power['start'], power['end'])
        
            if abs(measured_energy - energy) >= 1e-9:
                raise ValueError(f'Energy disagrees with power samples: {filename}')
        
            if abs((power['end'] - power['start']) - elapsed) >= 1e-9:
                raise ValueError(f'Power interval disagrees with workload duration: {filename}')

        total_frames += frame_count
        total_seconds += elapsed
        if energy is not None:
            total_joules += energy
        detector_calls += calls

    skipped_frames = total_frames - detector_calls
    
    return {
        'measured_frames': total_frames,
        'trials': len(records),
        'latency_ms_per_frame': 1000 * total_seconds / total_frames,
        'module_joules_per_frame': total_joules / total_frames if energy_available else None,
        'module_average_W': total_joules / total_seconds if energy_available else None,
        'detector_calls': detector_calls,
        'skipped_frames': skipped_frames,
        'skipped_frames_percent': 100 * skipped_frames / total_frames,
    }


def percentage(value):
    """Preserve undefined COCO metrics instead of treating them as zero."""

    return None if value is None else 100 * value


def evaluate_tracking(records, annotations_by_image, live, source):
    """Arrange complete outputs by video and delegate tracking metrics to mot()."""

    ground_truth_sequences = {}
    prediction_sequences = {}

    for record in records:
        if live and record['repeat'] != 0:
            continue

        video = record['video']

        if video in prediction_sequences:
            raise ValueError(f'Duplicate tracking sequence: {video}')

        frames = source.frames(video)

        if len(frames) != len(record['outputs']):
            raise ValueError('Tracking predictions must cover complete sequences')

        ground_truth_sequences[video] = [annotations_by_image[frame['image_id']] for frame in frames]
        prediction_sequences[video] = record['outputs']

    objects = [obj for frames in prediction_sequences.values() for frame in frames for obj in frame]
    missing_ids = sum('track_id' not in obj for obj in objects)
    detector_only = all(record['variant'] == 'baseline' for record in records)

    if detector_only or (objects and missing_ids == len(objects)):
        reason = (
            'The baseline detector has no temporal identity association.'
            if detector_only else 'Saved predictions do not contain track IDs; rerun with ID-preserving outputs.'
        )

        return dict.fromkeys(METRIC_NAMES), {'status': 'unavailable', 'reason': reason}

    if missing_ids:
        raise ValueError('Only some predictions have track IDs; refusing partial tracking evaluation')

    return mot(ground_truth_sequences, prediction_sequences)


def run(folder, output, live=False, dataset_manifest=None):
    """Write detection and tracking metrics plus recorded live resource summaries."""

    folder = Path(folder).resolve()
    output = Path(output).resolve()
    
    data_root = ROOT / 'data'
    
    if output == data_root or data_root in output.parents:
        raise ValueError('Dataset inputs are read-only')

    groups = load_results(folder, live)
    
    if live:
        protocol = read(folder / 'manifests/freeze.json')
        validate_schedule(groups, protocol)

    if dataset_manifest is None and live and protocol.get('dataset_manifest'):
        dataset_manifest = folder / protocol['dataset_manifest']

    if dataset_manifest is None:
        raise ValueError('Provide the dataset manifest used to generate these predictions')

    source = EvaluationDataset(dataset_manifest)
    expected_videos = set(source.videos)

    for records in groups.values():
        observed = {record['video'] for record in records if not live or record['repeat'] == 0}

        if observed != expected_videos:
            raise ValueError('Results must cover every selected dataset video')

    annotations, annotations_by_image = ground_truth(source)
    rows = []
    details = {}
    curves = []
    videos = []
    frames_table = []
    timing = []
    power = []
    reference_population = None
    
    for variant, records in groups.items():
        predictions, image_ids = collect_predictions(records, live, source)
        population = set(image_ids)

        if reference_population is None:
            reference_population = population
        elif population != reference_population:
            raise ValueError('Variants must share the same frame population')

        print(f'Evaluating {variant}...', flush=True)
        metrics, evaluator = coco(annotations, predictions, image_ids)
        curves.extend({'variant': variant, **point} for point in localization_curve(evaluator))
        errors = frame_errors(evaluator)
        tracking_metrics, tracking_details = evaluate_tracking(records, annotations_by_image, live, source)
        metrics['tracking'] = {'metrics': tracking_metrics, **tracking_details}
        details[variant] = metrics
        row = {
            'variant': variant,
            'unique_frames': len(image_ids),
            'AP50_95_percent': percentage(metrics['AP50_95']),
            'AP50_percent': percentage(metrics['AP50']),
            'AP75_percent': percentage(metrics['AP75']),
            'AR300_percent': percentage(metrics['AR300']),
            'tracking_status': tracking_details['status'],
            'tracking_reason': tracking_details.get('reason', ''),
            'MOTA_percent': percentage(tracking_metrics['MOTA']),
            'IDF1_percent': percentage(tracking_metrics['IDF1']),
            'HOTA_percent': percentage(tracking_metrics['HOTA']),
            'ID_switches': tracking_metrics.get('ID_switches'),
        }

        if live:
            row.update(summarize_resources(folder, records))
            timing.append({'variant': variant, **timing_summary(records)})
            by_video = defaultdict(list)

            for record in records:
                by_video[record['video']].append(record)

                if record['repeat'] == 0:
                    frames_table.extend(frame_rows(record, source.frames(record['video']), errors))

                power.extend(power_rows(folder, record))

            for video, trials in sorted(by_video.items()):
                video_predictions, video_ids = collect_predictions(trials, live, source)
                video_metrics, _ = coco(annotations, video_predictions, video_ids)
                videos.append({
                    'variant': variant, 'video': video,
                    'AP50_95_percent': percentage(video_metrics['AP50_95']),
                    'AP50_percent': percentage(video_metrics['AP50']),
                    'AP75_percent': percentage(video_metrics['AP75']),
                    **summarize_resources(folder, trials),
                })

        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    dataset.csvwrite(output / 'summary.csv', rows)
    write_report_tables(rows, timing, output)
    save(output / 'metrics.json', details)
    dataset.csvwrite(output / 'localization.csv', curves)

    if live:
        dataset.csvwrite(output / 'per_video.csv', videos)
        dataset.csvwrite(output / 'frames.csv', frames_table)
        dataset.csvwrite(output / 'timing.csv', timing)

        if power:
            dataset.csvwrite(output / 'power.csv', power)

        save(output / 'analysis.json', {
            'experiment': str(folder),
            'dataset_manifest': str(dataset_manifest),
            'videos': len(expected_videos),
            'repeats': protocol['repeats'],
            'accuracy': 'Pooled COCO AP; repeat 0 only; maxDets=300. Per-video AP is diagnostic.',
            'timing': 'Controller/output includes emission. Other is measured wall time minus components.',
            'frame_errors': 'COCO TP/FP/FN at IoU 0.75, all sizes, original scores, maxDets=300.',
            'power': 'Module VDD_IN; interpolated workload boundaries; cumulative energy uses wall time.',
            'power_validation': read(folder / 'results/power_check.json') if power else None,
            'limitations': 'No per-frame power timestamps, separate ERD timing, or run-to-run uncertainty with one repeat.',
        })
    print(f'Evaluated {len(rows)} variants on {len(reference_population)} unique frames', flush=True)
