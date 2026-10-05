"""Check experiment behavior with synthetic inputs."""

from types import SimpleNamespace

import pytest

from adaptive_video.evaluation.coco import BDD_TO_COCO, DEFAULT_EVAL_CLASSES, IDS
from adaptive_video.experiments import evaluate


def test_accuracy_counts_only_first_repeat_but_validates_all(monkeypatch):
    source = SimpleNamespace(
        videos=['clip'], frames=lambda video=None: [{'image_id': 1}]
    )
    records = [dict(video='clip', repeat=i, frames=1, outputs=[[]]) for i in range(2)]

    assert evaluate.collect_predictions(records, live=True, source=source) == ([], [1])
    records[1]['outputs'] = []

    with pytest.raises(ValueError, match='Complete clips'):
        evaluate.collect_predictions(records, live=True, source=source)


def test_duplicate_accuracy_frame_rejected(monkeypatch):
    source = SimpleNamespace(
        videos=['clip'], frames=lambda video=None: [{'image_id': 1}]
    )
    record = dict(video='clip', outputs=[[]])

    with pytest.raises(ValueError, match='Duplicate frame'):
        evaluate.collect_predictions([record, record], live=False, source=source)


def test_resource_denominators_include_skipped_frames(monkeypatch, tmp_path):
    samples = [{'t': 0, 'power1_W': 10}, {'t': 0.2, 'power1_W': 10}]
    monkeypatch.setattr(
        evaluate, 'read', lambda path: dict(samples=samples, start=0, end=0.2)
    )
    record = dict(
        repeat=0,
        variant='test',
        video='clip',
        frames=2,
        calls=1,
        wall_s=0.2,
        energy_joules=2,
        trace=[
            dict(frame_idx=0, ran_inference=True),
            dict(frame_idx=1, ran_inference=False),
        ],
    )
    result = evaluate.summarize_resources(tmp_path, [record])

    assert result['latency_ms_per_frame'] == 100
    assert result['module_joules_per_frame'] == 1
    assert result['skipped_frames_percent'] == 50
    record['calls'] = 2

    with pytest.raises(ValueError, match='Invocation count'):
        evaluate.summarize_resources(tmp_path, [record])


def test_missing_scheduled_repeat_rejected():
    trials = [dict(repeat=i, variant='test', video='clip') for i in range(2)]

    with pytest.raises(ValueError, match='scheduled trials'):
        evaluate.validate_schedule(
            {'test': trials[:1]}, {'arms': ['test'], 'schedule': trials}
        )


def test_evaluation_class_mapping():
    assert {BDD_TO_COCO[c] for c in DEFAULT_EVAL_CLASSES} == set(IDS)


def test_latency_summary_without_power_trace(tmp_path):
    record = {
        'frames': 2,
        'trace': [
            {'frame_idx': 0, 'ran_inference': True},
            {'frame_idx': 1, 'ran_inference': False},
        ],
        'calls': 1,
        'wall_s': 0.2,
        'energy_joules': None,
        'power_backend': 'none',
    }

    summary = evaluate.summarize_resources(tmp_path, [record])

    assert summary['latency_ms_per_frame'] == 100.0
    assert summary['skipped_frames_percent'] == 50.0
    assert summary['module_joules_per_frame'] is None


def test_missing_energy_is_not_accepted_for_power_measured_trial(tmp_path):
    record = {
        'frames': 1,
        'trace': [{'frame_idx': 0, 'ran_inference': True}],
        'calls': 1,
        'wall_s': 0.1,
        'energy_joules': None,
        'power_backend': 'vdd_in',
    }

    with pytest.raises(ValueError, match='Missing energy'):
        evaluate.summarize_resources(tmp_path, [record])
