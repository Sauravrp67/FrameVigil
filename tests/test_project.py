"""Fresh-checkout helpers must not require stored experiment results."""

import sys

from adaptive_video import dataset, project


def test_json_and_compressed_json_roundtrip(tmp_path):
    value = {'frames': [1, 2], 'status': 'test'}

    for filename in ['result.json', 'result.json.gz']:
        path = tmp_path / filename
        dataset.save(path, value)

        assert project.read(path) == value
        assert len(project.sha(path)) == 64


def test_bootstrap_does_not_require_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(project, 'ROOT', tmp_path)
    project.bootstrap()

    assert not list(tmp_path.iterdir())
    assert not any('trt-quantize-partition' in str(path) for path in sys.path)
