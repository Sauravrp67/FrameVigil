"""Known tracking outcomes, sequence boundaries, and malformed inputs."""

from pycocotools.coco import COCO
from types import SimpleNamespace
import json

import numpy as np
import pytest

from src.dataset import save
from src.evaluation.mot import mot
from src.experiments import evaluate
from src.tracking.output import emit


def box(identity=1, class_id=2, xyxy=(0, 0, 10, 10)):
    return dict(track_id=identity, class_id=class_id, box_xyxy=list(xyxy), conf=0.9)


def score(gt, pred):
    return mot({'video': gt}, {'video': pred})[0]


def test_perfect_trajectory_and_completely_missed_trajectory():
    gt = [[box()], [box()], [box()]]
    perfect = score(gt, [[box(99)], [box(99)], [box(99)]])

    assert [perfect[name] for name in ['MOTA', 'IDF1', 'HOTA']] == [1, 1, 1]
    missed = score(gt, [[], [], []])

    assert [missed[name] for name in ['MOTA', 'IDF1', 'HOTA']] == [0, 0, 0]
    assert missed['FN'] == 3


def test_identity_switch_with_unchanged_boxes():
    result = score([[box()], [box()]], [[box(10)], [box(11)]])

    assert result['TP'] == 2
    assert result['ID_switches'] == 1
    assert result['MOTA'] == 0.5
    assert result['IDF1'] == 0.5
    assert result['HOTA'] == pytest.approx(np.sqrt(0.5))


def test_video_boundaries_reset_identity_association():
    gt = {'one': [[box()]], 'two': [[box()]]}
    pred = {'one': [[box(10)]], 'two': [[box(11)]]}
    metrics, details = mot(gt, pred)

    assert metrics['IDF1'] == metrics['HOTA'] == metrics['MOTA'] == 1
    assert metrics['ID_switches'] == 0
    assert set(details['per_video']) == {'one', 'two'}


def test_scores_pool_counts_instead_of_averaging_video_scores():
    gt = {'short': [[box()]], 'long': [[box()] for _ in range(9)]}
    pred = {'short': [[box(10)]], 'long': [[] for _ in range(9)]}
    metrics, _ = mot(gt, pred)

    assert metrics['MOTA'] == 0.1
    assert metrics['IDF1'] == pytest.approx(2 / 11)
    assert metrics['HOTA'] == pytest.approx(np.sqrt(0.1))


def test_wrong_class_and_false_positives_are_penalized():
    result = score([[box()]], [[box(10, class_id=7)]])

    assert result['TP'] == 0 and result['FP'] == result['FN'] == 1
    assert result['MOTA'] == -1
    assert result['IDF1'] == result['HOTA'] == 0


def test_hota_averages_over_iou_thresholds():
    # IoU 0.6 matches at 12 of HOTA's 19 thresholds.
    result = score([[box()]], [[box(10, xyxy=(0, 0, 6, 10))]])

    assert result['MOTA'] == result['IDF1'] == 1
    assert result['HOTA'] == pytest.approx(12 / 19)


def test_empty_frame_between_observations_is_retained():
    result = score([[box()], [box()], [box()]], [[box(10)], [], [box(10)]])

    assert result['FN'] == 1
    assert result['MOTA'] == pytest.approx(2 / 3)
    assert result['IDF1'] == 0.8
    assert result['HOTA'] == pytest.approx(2 / 3)
    assert result['ID_switches'] == 0


def test_no_ground_truth_returns_undefined_scores_but_counts_false_positives():
    result = score([[]], [[box(10)]])

    assert result['MOTA'] is None and result['HOTA'] is None
    assert result['FP'] == 1


@pytest.mark.parametrize(
    'predictions, message',
    [
        ([[box(10), box(10)]], 'duplicate track_id'),
        ([[dict(class_id=2, box_xyxy=[0, 0, 10, 10])]], 'track_id'),
        ([[box(10, xyxy=(1, 0, 0, 10))]], 'invalid xyxy'),
        ([], 'every ground-truth frame'),
    ],
)
def test_invalid_tracking_input_rejected(predictions, message):
    with pytest.raises(ValueError, match=message):
        score([[box()]], predictions)


def test_tracking_output_preserves_real_id_and_geometry():
    track = SimpleNamespace(
        track_id=17,
        state='confirmed',
        class_id=2,
        class_name='car',
        get_box_xyxy=lambda: np.array([1.0, 2.0, 3.0, 4.0]),
        get_confidence=lambda threshold: 0.8,
    )
    controller = SimpleNamespace(tracks=[track], uncertainty_threshold=0.0003)
    output = emit(controller)[0]

    assert output == dict(
        track_id=17,
        class_id=2,
        class_name='car',
        box_xyxy=[1.0, 2.0, 3.0, 4.0],
        conf=0.8,
    )


def test_wrapper_does_not_invent_ids_for_old_predictions(monkeypatch):
    source = SimpleNamespace(
        videos=['clip'], frames=lambda video=None: [{'image_id': 1}]
    )
    prediction = box()
    del prediction['track_id']
    records = [dict(variant='constant_acceleration', video='clip', outputs=[[prediction]])]
    metrics, details = evaluate.evaluate_tracking(
        records, {1: [box()]}, live=False, source=source
    )

    assert metrics['MOTA'] is None and details['status'] == 'unavailable'
    records[0]['outputs'][0].append(box(2))

    with pytest.raises(ValueError, match='Only some'):
        evaluate.evaluate_tracking(records, {1: [box()]}, live=False, source=source)


def test_wrapper_uses_only_first_repeat(monkeypatch):
    source = SimpleNamespace(
        videos=['clip'], frames=lambda video=None: [{'image_id': 1}]
    )
    records = [
        dict(variant='constant_acceleration', video='clip', repeat=0, outputs=[[box(10)]]),
        dict(variant='constant_acceleration', video='clip', repeat=1, outputs=[[]]),
    ]
    metrics, details = evaluate.evaluate_tracking(
        records, {1: [box()]}, live=True, source=source
    )

    assert metrics['MOTA'] == 1 and details['status'] == 'computed'


def test_experiment_calls_detection_and_tracking_evaluators(monkeypatch, tmp_path):
    gt = COCO()
    gt.dataset = dict(
        info={},
        images=[dict(id=1, width=100, height=100)],
        categories=[dict(id=2, name='car')],
        annotations=[
            dict(
                id=1,
                image_id=1,
                category_id=2,
                bbox=[0, 0, 10, 10],
                area=100,
                iscrowd=0,
            )
        ],
    )
    gt.createIndex()
    monkeypatch.setattr(evaluate, 'ground_truth', lambda dataset: (gt, {1: [box()]}))
    source = SimpleNamespace(
        videos=['clip'], frames=lambda video=None: [{'image_id': 1}]
    )
    monkeypatch.setattr(evaluate, 'EvaluationDataset', lambda path: dataset)
    dataset = source
    source = tmp_path / 'experiment'
    save(
        source / 'constant_acceleration/clip.json.gz',
        dict(variant='constant_acceleration', video='clip', outputs=[[box(10)]]),
    )
    save(
        source / 'baseline/clip.json.gz',
        dict(variant='baseline', video='clip', outputs=[[box(10)]]),
    )
    destination = tmp_path / 'metrics'
    evaluate.run(source, destination, dataset_manifest='explicit-manifest.json')
    results = json.loads((destination / 'metrics.json').read_text())

    assert results['constant_acceleration']['AP50_95'] == pytest.approx(1)
    assert results['constant_acceleration']['tracking']['metrics']['HOTA'] == 1
    assert results['baseline']['tracking']['status'] == 'unavailable'
    assert results['baseline']['tracking']['metrics']['MOTA'] is None
    assert 'MOTA_percent' in (destination / 'summary.csv').read_text()


@pytest.mark.parametrize('coordinates', [(0, 0, 0, 10), (0, 0, 10, 0), (0, 0, 0, 0)])
def test_collapsed_predictions_remain_false_positives(coordinates):
    predictions = {'video': [[box(10), box(11, xyxy=coordinates)]]}
    metrics, details = mot({'video': [[box()]]}, predictions)

    assert metrics['TP'] == 1
    assert metrics['FP'] == 1
    assert metrics['FN'] == 0
    assert metrics['MOTA'] == 0
    assert metrics['IDF1'] == pytest.approx(2 / 3)
    assert metrics['HOTA'] == pytest.approx(np.sqrt(0.5))
    assert details['zero_area_predictions'] == 1
    assert details['zero_area_predictions_per_video'] == {'video': 1}
    assert predictions['video'][0][1]['box_xyxy'] == list(coordinates)


def test_collapsed_ground_truth_is_still_rejected():
    with pytest.raises(ValueError, match='invalid xyxy'):
        score([[box(xyxy=(0, 0, 0, 10))]], [[box(10)]])


@pytest.mark.parametrize('coordinates', [(0, 0, float('nan'), 10), (0, 0, float('inf'), 10)])
def test_nonfinite_predictions_are_still_rejected(coordinates):
    with pytest.raises(ValueError, match='invalid xyxy'):
        score([[box()]], [[box(10, xyxy=coordinates)]])
