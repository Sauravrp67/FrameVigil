"""Select complete videos and pipelines without changing image identities."""

from collections import Counter

import pytest
import yaml

from adaptive_video.evaluation.dataset import EvaluationDataset
from adaptive_video.experiments.evaluate import collect_predictions, validate_schedule
from adaptive_video.measurement import stages
from adaptive_video.project import read, save


@pytest.fixture
def inventory(tmp_path):
    videos = [f'video_{index:03d}' for index in reversed(range(40))]
    frames = [
        {
            'sequence': video,
            'image_id': video_index * 2 + frame_index + 1,
            'frame_index': frame_index,
            'path': f'{video}/{frame_index}.jpg',
            'width': 1280,
            'height': 720,
        }
        for video_index, video in enumerate(videos)
        for frame_index in range(2)
    ]
    path = tmp_path / 'dataset.json'
    save(path, {
        'schema_version': 1,
        'videos': videos,
        'frames': frames,
        'labels': {video: f'{video}.json' for video in videos},
        'label_sha256': {video: 'label_hash' for video in videos},
        'image_sha256': {frame['path']: 'image_hash' for frame in frames},
    })

    return path


def test_selected_variants_videos_and_evaluation_population(tmp_path, inventory):
    config = tmp_path / 'study.yaml'
    variants = ['constant_acceleration', 'visibility_ca10', 'cv8']
    config.write_text(yaml.safe_dump({
        'dataset': {'sequence': {'start': 0, 'end': 30}},
        'experiment': {'variants': variants, 'repeats': 2},
    }))
    output = tmp_path / 'experiment'

    stages.prepare(output, inventory, config=config)

    protocol = read(output / 'manifests/freeze.json')
    source = EvaluationDataset(output / protocol['dataset_manifest'])
    expected_videos = [f'video_{index:03d}' for index in range(30)]

    assert protocol['arms'] == variants
    assert source.videos == expected_videos
    assert len(protocol['schedule']) == 180
    assert len(source.frames()) == 60
    assert len(source.manifest['image_sha256']) == 60
    assert set(source.manifest['labels']) == set(expected_videos)

    original_frames = {
        frame['image_id']: frame for frame in read(inventory)['frames']
    }
    assert all(frame == original_frames[frame['image_id']] for frame in source.frames())

    counts = Counter(
        (trial['repeat'], trial['video'], trial['variant'])
        for trial in protocol['schedule']
    )
    assert set(counts.values()) == {1}
    assert set(trial['variant'] for trial in protocol['schedule']) == set(variants)

    groups = {
        variant: [
            {**trial, 'frames': 2, 'outputs': [[], []]}
            for trial in protocol['schedule'] if trial['variant'] == variant
        ]
        for variant in variants
    }
    validate_schedule(groups, protocol)
    predictions, image_ids = collect_predictions(groups['cv8'], live=True, source=source)

    assert predictions == []
    assert set(image_ids) == {frame['image_id'] for frame in source.frames()}
    assert len(image_ids) == 60


@pytest.mark.parametrize('selection, expected', [
    ({'start': 0, 'end': 3}, ['a', 'b', 'c']),
    ({'start': 0, 'end': 2}, ['a', 'b']),
    ({'start': 1, 'end': 3}, ['b', 'c']),
    ({'end': 1}, ['a']),
    ({'start': -1}, ['c']),
    ({'start': 0, 'end': 200}, ['a', 'b', 'c']),
    ({}, ['a', 'b', 'c']),
])
def test_video_range_semantics(selection, expected):
    assert stages.select_videos(['c', 'a', 'b'], selection) == expected


@pytest.mark.parametrize('selection', [
    '0:30',
    0,
    {'start': '0', 'end': 2},
    {'start': 0, 'end': 1.5},
    {'start': False, 'end': 2},
    {'start': 2, 'end': 2},
    {'start': 9, 'end': 10},
])
def test_invalid_or_empty_video_selection(selection):
    with pytest.raises(ValueError):
        stages.select_videos(['a', 'b', 'c'], selection)


@pytest.mark.parametrize('variants', [[], ['unknown'], ['cv8', 'cv8'], 'cv8'])
def test_invalid_variant_selection(tmp_path, inventory, variants):
    config = tmp_path / 'study.yaml'
    config.write_text(yaml.safe_dump({'experiment': {'variants': variants}}))

    with pytest.raises(ValueError):
        stages.prepare(tmp_path / 'experiment', inventory, config=config)


def test_new_registered_pipeline_and_repeat_override(tmp_path, inventory, monkeypatch):
    monkeypatch.setitem(stages.VARIANTS, 'new_pipeline', stages.VARIANTS['cv8'])
    config = tmp_path / 'study.yaml'
    config.write_text(yaml.safe_dump({
        'dataset': {'sequence': {'start': 30, 'end': 40}},
        'experiment': {'variants': ['new_pipeline'], 'repeats': 3},
    }))
    output = tmp_path / 'experiment'

    stages.prepare(output, inventory, repeats=1, config=config)
    protocol = read(output / 'manifests/freeze.json')

    assert protocol['arms'] == ['new_pipeline']
    assert protocol['repeats'] == 1
    assert len(protocol['videos']) == 10
    assert len(protocol['schedule']) == 10


@pytest.mark.parametrize('backend', ['torch', 'tensorrt'])
@pytest.mark.parametrize('precision', ['fp32', 'tf32'])
def test_preparation_records_runtime_choices(tmp_path, inventory, backend, precision):
    config = tmp_path / 'study.yaml'
    config.write_text(yaml.safe_dump({
        'inference': {'backend': backend, 'precision': precision, 'nms': 'native'},
    }))
    output = tmp_path / 'experiment'

    stages.prepare(output, inventory, config=config)
    settings = read(output / 'manifests/freeze.json')['inference']

    assert settings['backend'] == backend
    assert settings['precision'] == precision
    assert settings['nms'] == 'native'
    assert settings['engine'].endswith(f'_{precision}.engine')
