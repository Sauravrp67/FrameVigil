"""Experiment preparation and resumption work without fingerprint metadata."""

import pytest

from adaptive_video import project
from adaptive_video.dataset import save
from adaptive_video.measurement import stages
from adaptive_video.measurement.runner import trial
from adaptive_video.measurement.support import exact_repeat


def prepare_inputs(root):
    (root / 'models').mkdir()
    (root / 'src').mkdir()
    (root / 'docs').mkdir()
    (root / 'models/detector.pt').write_bytes(b'weights')
    (root / 'src/kalman.py').write_text('value = 1\n')
    (root / 'docs/FULL_VALIDATION_PROTOCOL.md').write_text('protocol\n')
    project.save(root / 'dataset.json', {})


def test_experiment_configuration_is_not_hash_enforced(tmp_path, monkeypatch):
    prepare_inputs(tmp_path)
    monkeypatch.setattr(stages, 'ROOT', tmp_path)
    videos = [f'video_{index:03d}' for index in range(200)]
    project.save(tmp_path / 'dataset.json', {
        'videos': videos,
        'frames': [
            {'sequence': video, 'image_id': index + 1, 'frame_index': 0, 'path': f'{video}.jpg'}
            for index, video in enumerate(videos)
        ],
        'labels': {video: f'{video}.json' for video in videos},
        'label_sha256': {video: 'unused' for video in videos},
        'image_sha256': {f'{video}.jpg': 'unused' for video in videos},
    })
    output = tmp_path / 'experiment'
    stages.prepare(output, tmp_path / 'dataset.json', repeats=1)

    assert not (tmp_path / 'records').exists()
    assert not (tmp_path / 'manifests').exists()
    protocol = project.read(output / 'manifests/freeze.json')

    assert len(protocol['schedule']) == 2200
    assert len(protocol['videos']) == 200
    assert not (output / 'runtime_sha256.json').exists()
    assert not (output / 'models_sha256.json').exists()
    assert not (output / 'integrity.json').exists()
    assert 'dataset_sha256' not in protocol

    # Editing the schedule and inventory requires no replacement fingerprints.
    protocol['schedule'].reverse()
    project.save(output / 'manifests/freeze.json', protocol)
    project.save(output / 'manifests/dataset.json', {'edited': True})
    (tmp_path / 'src/kalman.py').write_text('value = 100\n')
    (tmp_path / 'models/detector.pt').write_bytes(b'changed weights')
    monkeypatch.setattr(stages, 'bootstrap', lambda: None)

    assert stages.checked_experiment(output) == output


def test_completed_trial_resume_needs_no_configuration_hash(tmp_path):
    destination = tmp_path / 'results/live/r0_original_video.json.gz'

    save(destination, {'status': 'PASS', 'power_backend': 'none'})
    trial(None, 'original', 'video', 0, {}, tmp_path, None)

    assert not (tmp_path / 'integrity.json').exists()


def test_repeats_reject_different_predictions_calls_or_state():
    d = dict(
        variant='cv8',
        video='synthetic',
        outputs=[[]],
        trace=[{}],
        detector_raw_hashes=['x'],
        calls=1,
        extended_final_state_sha256='a',
    )
    exact_repeat(d, d)

    for key, value in [
        ('outputs', [[{'conf': 0.5}]]),
        ('calls', 0),
        ('extended_final_state_sha256', 'b'),
    ]:
        with pytest.raises(AssertionError):
            exact_repeat(d, dict(d, **{key: value}))
