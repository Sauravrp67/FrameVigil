"""Fresh flow analysis works"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import csvwrite, read, save
from ultralytics.engine.results import Results
from src.flow import reliability, reproduce, shadow
from src.tracking import scheduler
from src.tracking.kalman import BoxTrack
from src.tracking.output import (
    DetectorOutputTracker,
    digest,
    emit,
    make,
    state,
)


def make_capture(tmp_path, monkeypatch):
    frames = [
        {
            'sequence': video,
            'frame_index': index,
            'image_id': video_index * 2 + index,
            'width': 1280,
            'height': 720,
        }
        for video_index, video in enumerate(['first', 'second'])
        for index in range(2)
    ]

    def labels(video):
        return [
            {
                'frameIndex': index,
                'labels': [
                    {
                        'id': 'car',
                        'category': 'car',
                        'box2d': {'x1': 100, 'y1': 100, 'x2': 180, 'y2': 180},
                    },
                ],
            }
            for index in range(2)
        ]

    source = SimpleNamespace(
        videos=['first', 'second'],
        frames=lambda video=None: [
            frame for frame in frames if video is None or frame['sequence'] == video
        ],
        labels=labels,
    )
    monkeypatch.setattr(shadow, 'EvaluationDataset', lambda path: source)
    capture = tmp_path / 'capture'
    save(capture / 'capture.json', {'schema_version': 1, 'videos': source.videos})
    save(capture / 'dataset.json', {})
    detection = {
        'track_id': 7,
        'class_id': 2,
        'conf': 0.9,
        'box_xyxy': [100, 100, 180, 180],
    }

    for video in source.videos:
        rows = []

        for index, frame in enumerate(source.frames(video)):
            alternative = None

            if index:
                features = {
                    name: 0.1
                    for name in reliability.PRE_FEATURES + reliability.FLOW_FEATURES
                }
                alternative = {
                    'predicted': [detection],
                    'weighted': [detection],
                    'direct': [detection],
                    'features': features,
                    'objects': [{'track_id': 7, 'valid': True}],
                    'flow_and_correction_ms': 2.0,
                }

            rows.append(
                {
                    'video': video,
                    'index': index,
                    'image_id': frame['image_id'],
                    'ran_detector': True,
                    'reasons': {'eligible': bool(index)},
                    'baseline': [detection],
                    'alternative': alternative,
                    'failure': None,
                }
            )

        save(capture / 'videos' / f'{video}.json.gz', rows)

    return capture


def test_fresh_capture_through_pooled_ap_and_grouped_reliability(tmp_path, monkeypatch):
    capture = make_capture(tmp_path, monkeypatch)
    output = tmp_path / 'analysis'
    reproduce.run(capture, output)
    summary = read(output / 'summary/summary.json')

    assert summary['events'] == 2
    assert summary['safe'] == 2
    assert summary['method_energy'] is None
    assert summary['metrics']['baseline'] == summary['metrics']['weighted']
    assert not (tmp_path / 'records').exists()

    for fold in read(output / 'reliability/folds.json'):
        assert fold['held_out'] not in fold['train_videos']

    assert (
        pd.read_csv(output / 'reliability/reliability_summary.csv').AUROC.isna().all()
    )


def test_shadow_rejects_incomplete_video(tmp_path, monkeypatch):
    capture = make_capture(tmp_path, monkeypatch)
    path = capture / 'videos/first.json.gz'
    save(path, read(path)[:1])

    with pytest.raises(ValueError, match='Incomplete capture'):
        shadow.run(capture, tmp_path / 'analysis')


def test_shadow_rejects_wrong_image_identity(tmp_path, monkeypatch):
    capture = make_capture(tmp_path, monkeypatch)
    path = capture / 'videos/first.json.gz'
    rows = read(path)
    rows[0]['image_id'] = 99
    save(path, rows)

    with pytest.raises(ValueError, match='identity mismatch'):
        shadow.run(capture, tmp_path / 'analysis')


def test_no_candidates_is_explicit(tmp_path, monkeypatch):
    capture = make_capture(tmp_path, monkeypatch)

    for video in ['first', 'second']:
        path = capture / 'videos' / f'{video}.json.gz'
        rows = read(path)

        for row in rows:
            row['alternative'] = None
            row['reasons']['eligible'] = False

        save(path, rows)

    reproduce.run(capture, tmp_path / 'analysis')
    result = read(tmp_path / 'analysis/reliability/status.json')

    assert result['status'] == 'INSUFFICIENT_DATA'


@pytest.mark.parametrize('flow_fails', [False, True])
def test_collector_keeps_actual_baseline_trajectory_and_never_reads_labels(
    monkeypatch, flow_fails
):
    monkeypatch.setattr(scheduler, 'load_scene_classifier', lambda device: None)
    monkeypatch.setattr(scheduler, 'classify_scene', lambda *args: 1)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    frames = [{'image_id': index} for index in range(50)]
    source = SimpleNamespace(frames=lambda video: frames, image=lambda frame: image)

    class Model:
        def predict(self, frame, **kwargs):
            return [
                Results(
                    frame,
                    path='',
                    names={2: 'car'},
                    boxes=torch.tensor([[100, 100, 180, 180, 0.9, 2]]),
                )
            ]

    class Backend:
        def acquire(self, previous, current, index):
            if flow_fails:
                raise RuntimeError('Synthetic flow failure')

            return np.zeros((180, 320, 2), dtype=np.float32)

    BoxTrack.reset_id_counter()
    tracker = make(DetectorOutputTracker)
    captured = shadow.collect_video(
        source, 'video', tracker, Model(), Backend(), device='cpu'
    )

    if flow_fails:
        assert any(row['failure'] is not None for row in captured)
        assert all(row['alternative'] is None for row in captured)
    else:
        assert any(row['alternative'] is not None for row in captured)

    BoxTrack.reset_id_counter()
    baseline = make(DetectorOutputTracker)

    for row in captured:
        result = baseline.step(image, Model(), 'cpu')

        assert row['ran_detector'] == result['ran_inference']
        assert row['baseline'] == emit(baseline, 'constant_acceleration')
        assert row['baseline_state_sha256'] == digest(state(baseline))


def test_reliability_fits_only_other_videos_with_both_classes(tmp_path):
    analysis = tmp_path / 'input'
    save(analysis / 'analysis.json', {'events': 12})
    feature_rows = []
    label_rows = []

    for video_index in range(3):
        for index in range(4):
            identity = {
                'video': f'video_{video_index}',
                'index': index,
                'image_id': video_index * 4 + index,
            }
            feature_rows.append(
                {
                    **identity,
                    **{
                        name: float(index)
                        for name in reliability.PRE_FEATURES + reliability.FLOW_FEATURES
                    },
                }
            )
            label_rows.append({**identity, 'safe': index >= 2})

    csvwrite(analysis / 'features.csv', feature_rows)
    csvwrite(analysis / 'labels.csv', label_rows)
    reliability.run(analysis, tmp_path / 'output')
    probabilities = pd.read_csv(tmp_path / 'output/probabilities.csv')

    assert probabilities.probability.between(0, 1).all()
    assert len(probabilities) == 36

    for fold in read(tmp_path / 'output/folds.json'):
        assert fold['held_out'] not in fold['train_videos']
        assert len(fold['train_videos']) == 2
        assert len(fold['coefficients_standardized']) == len(fold['features'])
