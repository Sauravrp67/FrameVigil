"""Render plots from saved predictions."""

from pathlib import Path

import matplotlib.patches as patches

from ..dataset import csvwrite
from ..evaluation.coco import ground_truth
from ..evaluation.dataset import EvaluationDataset
from ..project import read
from .figures import create_figure, read_table, save_figure, LABELS


def select_cases(rows, variant):
    """Select minimum, median and maximum error reduction."""
    reference = {row['image_id']: row for row in rows if row['variant'] == 'baseline'}
    candidates = []

    for row in rows:
        if row['variant'] != variant or row['image_id'] not in reference:
            continue

        baseline = reference[row['image_id']]
        baseline_errors = int(baseline['FP75']) + int(baseline['FN75'])
        variant_errors = int(row['FP75']) + int(row['FN75'])
        candidates.append({
            'video': row['video'], 'frame_index': int(row['frame_index']),
            'image_id': row['image_id'], 'variant': variant,
            'baseline_errors': baseline_errors, 'variant_errors': variant_errors,
            'error_reduction': baseline_errors - variant_errors,
        })

    candidates.sort(key=lambda row: (row['error_reduction'], row['video'], row['frame_index']))

    if not candidates:
        return []

    return [dict(candidates[index], selection=label) for index, label in [
        (0, 'Minimum error reduction'),
        (len(candidates) // 2, 'Median error reduction'),
        (len(candidates) - 1, 'Maximum error reduction'),
    ]]


def draw_boxes(axis, boxes, color):
    for detection in boxes:
        x1, y1, x2, y2 = detection['box_xyxy']
        axis.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=0.7,
        ))


def run(experiment, summary_directory, output):
    experiment = Path(experiment).resolve()
    protocol = read(experiment / 'manifests/freeze.json')
    source = EvaluationDataset(experiment / protocol['dataset_manifest'])
    frames_table = read_table(Path(summary_directory) / 'frames.csv')
    variants = {row['variant'] for row in frames_table}
    target = next((name for name in ('visibility_ca10', 'cv8', 'constant_acceleration') if name in variants), None)

    if target is None or 'baseline' not in variants:
        return []

    cases = select_cases(frames_table, target)
    _, truth = ground_truth(source)
    columns = list(dict.fromkeys(['baseline', *(['constant_acceleration'] if 'constant_acceleration' in variants else []), target]))
    figure, axes = create_figure(rows=len(cases), columns=len(columns) + 1, width=20, height=10)
    records = {}

    for row_index, case in enumerate(cases):
        video = case['video']
        index = case['frame_index']
        frame = source.frames(video)[index]
        image = source.image(frame)[:, :, ::-1]
        ground_truth_axis = axes[row_index, 0]
        ground_truth_axis.imshow(image)
        draw_boxes(ground_truth_axis, truth[frame['image_id']], 'lime')
        ground_truth_axis.set_title(f"Ground truth | {video}\nframe {index}", fontsize=8)
        ground_truth_axis.set_ylabel(f"{case['selection']}\nΔ(FP+FN) = {case['error_reduction']}", fontsize=9)

        for column_index, variant in enumerate(columns, start=1):
            key = (variant, video)

            if key not in records:
                records[key] = read(experiment / 'results/live' / f'r0_{variant}_{video}.json.gz')

            record = records[key]
            axis = axes[row_index, column_index]
            axis.imshow(image)
            draw_boxes(axis, record['outputs'][index], 'yellow')
            action = record['trace'][index]
            decision = 'detector ran' if action['ran_inference'] else 'detector skipped'
            axis.set_title(f'{LABELS[variant]} | {decision}', fontsize=9)

        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        f'{LABELS[target]} versus baseline detector: cases selected by COCO FP+FN at IoU 0.75\n'
        'Green: ground truth; yellow: every emitted prediction, without a new score cutoff', fontsize=12,
    )
    csvwrite(output / 'case_selection.csv', cases)
    return save_figure(figure, output, 'visual_cases')
