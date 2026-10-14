"""Check explicit visualization inputs, truthful labels and actual video encoding."""

import copy
import shutil

import cv2
import numpy as np
import pytest

from src.dataset import save, sha
from src.evaluation.dataset import EvaluationDataset
from src.experiments import figures, render


def write_table(path, energy=False):
    header = 'variant,AP50_95_percent,unique_frames,latency_ms_per_frame'
    values = ['constant_acceleration,32,39973,10', 'visibility_ca10,34,39973,11']

    if energy:
        header += ',module_joules_per_frame'
        values = [value + ',0.1' for value in values]

    path.write_text('\n'.join([header, *values]) + '\n')


@pytest.mark.parametrize('energy', [False, True])
def test_plots_only_available_diagnostics(tmp_path, energy):
    source = tmp_path / 'summary.csv'
    write_table(source, energy)
    (tmp_path / 'localization.csv').write_text(
        'variant,iou,AP_percent\nconstant_acceleration,0.5,40\nvisibility_ca10,0.5,42\n'
    )
    output = tmp_path / 'figures'
    paths = figures.run(source, output)

    assert {path.name for path in paths} == {'localization.png', 'localization.svg'}
    assert all(path.stat().st_size > 0 for path in paths)
    assert 'Pooled AP (%)' in (output / 'localization.svg').read_text()
    assert not list(output.glob('*.md'))

    with pytest.raises(FileExistsError):
        figures.run(source, output)


@pytest.mark.parametrize(
    'contents, message',
    [
        ('variant,AP50_95_percent\na,30\n', 'No measured'),
        (
            'variant,AP50_95_percent,latency_ms_per_frame\na,30,10\nb,31,\n',
            'Incomplete cost',
        ),
        ('variant,AP50_95_percent,latency_ms_per_frame\na,nan,10\n', 'Invalid'),
        (
            'variant,AP50_95_percent,latency_ms_per_frame,unique_frames\na,30,10,2\nb,31,11,3\n',
            'same positive',
        ),
    ],
)
def test_invalid_plot_tables_rejected(tmp_path, contents, message):
    source = tmp_path / 'bad.csv'
    source.write_text(contents)

    with pytest.raises(ValueError, match=message):
        figures.load_table(source)


def prediction_fixture(tmp_path, monkeypatch):
    frames = []
    hashes = {}

    for index in range(2):
        path = tmp_path / f'frame{index}.jpg'

        assert cv2.imwrite(str(path), np.full((96, 128, 3), 80, dtype=np.uint8))
        frames.append(
            dict(
                image_id=index + 1,
                sequence='clip',
                frame_index=index,
                path=path.name,
                width=128,
                height=96,
            )
        )
        hashes[path.name] = sha(path)

    manifest_path = tmp_path / 'dataset.json'
    save(manifest_path, dict(videos=['clip'], frames=frames, image_sha256=hashes))
    monkeypatch.setattr(
        render, 'EvaluationDataset', lambda path: EvaluationDataset(path, root=tmp_path)
    )
    paths = []

    for variant in ['constant_acceleration', 'visibility_ca10']:
        path = tmp_path / f'{variant}.json.gz'
        detection = dict(
            box_xyxy=[10, 10, 50, 50], class_id=2, conf=0.2, track_id='object-7'
        )
        save(
            path,
            dict(
                video='clip',
                variant=variant,
                outputs=[[detection], []],
                trace=[
                    dict(frame_idx=0, ran_inference=True),
                    dict(frame_idx=1, ran_inference=False),
                ],
            ),
        )
        paths.append(path)

    return manifest_path, paths


def test_panel_preserves_image_and_prediction_values():
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    boxes = [
        dict(
            box_xyxy=np.array([-5.0, 3.0, 140.0, 100.0]),
            conf=0.01,
            class_id=2,
            track_id='a',
        )
    ]
    original = copy.deepcopy(boxes)
    panel = render.render_panel(image, boxes, 'test', 0, None)

    np.testing.assert_array_equal(boxes[0]['box_xyxy'], original[0]['box_xyxy'])

    assert not image.any()
    assert panel[render.HEADER_HEIGHT :].any()
    assert render.track_color('a') == render.track_color('a')


@pytest.mark.skipif(
    not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
    reason='Video integration test needs FFmpeg and ffprobe',
)
def test_real_video_encoding_and_input_provenance(tmp_path, monkeypatch):
    manifest_path, paths = prediction_fixture(tmp_path, monkeypatch)
    result = render.run(manifest_path, paths, 'clip', tmp_path / 'demo', fps=5)

    assert result['output']['frames'] == 2
    assert result['output']['width'] == 256
    assert result['output']['height'] == 186
    assert result['output']['frame_rate'] == '5/1'
    assert result['output']['duration_s'] == pytest.approx(0.4)
    assert result['prediction_sha256'] == {str(path): sha(path) for path in paths}
    assert (tmp_path / 'demo/preview.png').is_file()
    assert (tmp_path / 'demo/manifest.json').is_file()
    assert not (tmp_path / 'demo/comparison.partial.mp4').exists()

    with pytest.raises(FileExistsError):
        render.run(manifest_path, paths, 'clip', tmp_path / 'demo')


def test_wrong_sequence_partial_and_duplicate_trials_rejected(tmp_path, monkeypatch):
    _, paths = prediction_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match='selected video'):
        render.load_predictions(paths, 'other-video', 2)

    with pytest.raises(ValueError, match='complete selected video'):
        render.load_predictions(paths, 'clip', 3)

    with pytest.raises(ValueError, match='one trial per variant'):
        render.load_predictions([paths[0], paths[0]], 'clip', 2)


def test_missing_ids_and_trace_remain_missing(tmp_path):
    path = tmp_path / 'always.json'
    save(
        path,
        dict(
            video='clip',
            variant='baseline',
            outputs=[[dict(box_xyxy=[1, 1, 5, 5], class_id=2, conf=0.7)]],
        ),
    )
    record = render.load_predictions([path], 'clip', 1)[0]

    assert 'trace' not in record
    assert 'track_id' not in record['outputs'][0][0]
