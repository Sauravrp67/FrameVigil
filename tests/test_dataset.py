"""Dataset preparation must fail safely without evaluating or selecting videos."""

from pathlib import Path
import json
import os
import subprocess

import pytest

from adaptive_video import project
from adaptive_video.evaluation.dataset import (
    EvaluationDataset,
    prepare_dataset_manifest,
)
from adaptive_video.experiments import evaluate


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_dataset.sh'


def test_help_needs_no_dataset_and_creates_nothing(tmp_path):
    result = subprocess.run(
        ['bash', str(SCRIPT), '--help'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert '200' in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_corrupt_cached_archive_is_preserved_and_never_extracted(tmp_path):
    cache = tmp_path / 'archive cache'
    cache.mkdir()
    archive = cache / 'images20-track-val-1.zip'
    archive.write_bytes(b'corrupt archive')
    result = subprocess.run(
        [
            'bash',
            str(SCRIPT),
            '--data-dir',
            str(tmp_path / 'dataset'),
            '--archive-dir',
            str(cache),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert 'Checksum mismatch' in result.stderr
    assert archive.read_bytes() == b'corrupt archive'
    assert not (tmp_path / 'dataset/bdd100k').exists()


def test_failed_transfer_keeps_partial_and_does_not_extract(tmp_path):
    tools = tmp_path / 'bin'
    tools.mkdir()
    curl = tools / 'curl'
    curl.write_text(
        '''#!/usr/bin/env bash
while (($#)); do
    if [[ "$1" == --output ]]; then printf 'partial bytes' > "$2"; exit 22; fi
    shift
done
exit 99
'''
    )
    curl.chmod(0o755)
    dest = tmp_path / 'dataset'
    result = subprocess.run(
        ['bash', str(SCRIPT), '--data-dir', str(dest)],
        env={**os.environ, 'PATH': str(tools) + os.pathsep + os.environ['PATH']},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert 'Download failed' in result.stderr
    assert (
        dest / 'archives/images20-track-val-1.zip.part'
    ).read_bytes() == b'partial bytes'
    assert not (dest / 'archives/images20-track-val-1.zip').exists()
    assert not (dest / 'bdd100k').exists()


def make_dataset(root, count=200):
    for index in range(count):
        video = f'video_{index:03d}'
        images = root / 'data/bdd100k/images/track/val' / video
        labels = root / 'data/bdd100k/labels/box_track_20/val'
        images.mkdir(parents=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / 'frame.jpg').write_bytes(
            b'fixture image bytes; decoding is not exercised'
        )
        (labels / f'{video}.json').write_text(
            json.dumps([dict(name='frame.jpg', frameIndex=0, labels=[])])
        )

    manifest = root / 'population.json'
    prepare_dataset_manifest(
        manifest, root, expected_videos=count, expected_frames=count
    )

    return manifest


def test_all_200_videos_have_distinct_images_and_complete_outputs(tmp_path):
    manifest = make_dataset(tmp_path)
    source = EvaluationDataset(manifest, tmp_path)

    assert len(source.videos) == 200
    assert len({f['image_id'] for f in source.frames()}) == 200
    records = [dict(video=video, outputs=[[]]) for video in source.videos]
    predictions, ids = evaluate.collect_predictions(records, live=False, source=source)

    assert predictions == []
    assert len(ids) == 200
    records[-1]['outputs'] = []

    with pytest.raises(ValueError, match='Complete clips'):
        evaluate.collect_predictions(records, live=False, source=source)


def test_annotation_mutation_rejected(tmp_path):
    source = EvaluationDataset(make_dataset(tmp_path, 1), tmp_path)

    assert len(source.labels(source.videos[0])) == 1
    path = tmp_path / source.manifest['labels'][source.videos[0]]
    path.write_text('[]')

    with pytest.raises(ValueError, match='Annotation changed'):
        source.labels(source.videos[0])


def test_inventory_requires_exact_image_annotation_correspondence(tmp_path):
    manifest = make_dataset(tmp_path, 1)
    manifest.unlink()
    (tmp_path / 'data/bdd100k/images/track/val/video_000/extra.jpg').touch()

    with pytest.raises(ValueError, match='frames differ'):
        prepare_dataset_manifest(
            manifest, tmp_path, expected_videos=1, expected_frames=1
        )


def test_full_evaluator_refuses_a_missing_video(monkeypatch, tmp_path):
    manifest = make_dataset(tmp_path, 2)
    source = EvaluationDataset(manifest, tmp_path)
    monkeypatch.setattr(evaluate, 'EvaluationDataset', lambda path: source)
    monkeypatch.setattr(
        evaluate,
        'load_results',
        lambda *args: {'original': [dict(video=source.videos[0], outputs=[[]])]},
    )

    with pytest.raises(ValueError, match='every selected dataset video'):
        evaluate.run(tmp_path, tmp_path / 'metrics', dataset_manifest=manifest)


def test_manifest_does_not_overwrite(tmp_path):
    manifest = make_dataset(tmp_path, 1)

    with pytest.raises(FileExistsError):
        prepare_dataset_manifest(
            manifest, tmp_path, expected_videos=1, expected_frames=1
        )


def test_dataset_verification_needs_no_model_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(project, 'ROOT', tmp_path)
    image = tmp_path / 'image.jpg'
    image.write_bytes(b'image')
    manifest = tmp_path / 'dataset.json'
    project.save(
        manifest,
        {
            'image_sha256': {'image.jpg': project.sha(image)},
            'labels': {},
            'label_sha256': {},
            'videos': [],
            'frames': [],
        },
    )

    assert project.verify(manifest)['status'] == 'PASS'
    assert not (tmp_path / 'models').exists()
    assert not (tmp_path / 'manifests').exists()

    image.write_bytes(b'changed image')

    with pytest.raises(ValueError, match='Image changed'):
        project.verify(manifest)


def test_yaml_dataset_paths_are_used(tmp_path):
    manifest = make_dataset(tmp_path, count=2)
    manifest.unlink()
    (tmp_path / 'data').rename(tmp_path / 'custom_data')

    config = tmp_path / 'study.yaml'
    config.write_text(
        'dataset:\n'
        '  images: custom_data/bdd100k/images/track/val\n'
        '  labels: custom_data/bdd100k/labels/box_track_20/val\n'
    )

    result = prepare_dataset_manifest(
        manifest,
        root=tmp_path,
        expected_videos=2,
        expected_frames=2,
        config=config,
    )

    assert len(result['videos']) == 2
    assert all(frame['path'].startswith('custom_data/') for frame in result['frames'])
    source = EvaluationDataset(manifest, tmp_path)
    assert len(source.labels(result['videos'][0])) == 1
