"""Model download definitions remain separate and reject unused sizes."""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('model', ['yolo11x', 'yolo26x', 'rtdetrv2'])
def test_model_definition_only_declares_downloads(model):
    script = ROOT / 'scripts/model_downloads' / f'{model}.sh'
    result = subprocess.run(
        ['bash', '-c', 'download() { printf "%s\\n" "$1"; }; source "$1"', 'bash', str(script)],
        capture_output=True, text=True, check=True,
    )
    files = result.stdout.splitlines()

    if model == 'rtdetrv2':
        assert files == ['rtdetrv2/config.json', 'rtdetrv2/preprocessor_config.json', 'rtdetrv2/model.safetensors']
    else:
        assert files == [f'{model}.pt']


@pytest.mark.parametrize('model', ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l'])
def test_downloader_rejects_unused_sizes_without_network(tmp_path, model):
    result = subprocess.run([
        'bash', str(ROOT / 'scripts/download_models.sh'),
        '--detector', model, '--models-dir', str(tmp_path),
    ], capture_output=True, text=True)

    assert result.returncode != 0
    assert 'Unsupported detector' in result.stderr
    assert not list(tmp_path.iterdir())
