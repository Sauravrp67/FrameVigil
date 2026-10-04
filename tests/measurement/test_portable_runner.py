"""Exercise the live runner's no-energy path with a CPU stub detector."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from adaptive_video.measurement import runner
from adaptive_video.project import read


def test_trial_without_power_never_starts_sampler(monkeypatch, tmp_path):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    frame = {'path': 'unused.jpg', 'height': 2, 'width': 2}
    source = SimpleNamespace(frames=lambda video: [frame], image=lambda frame: image)
    prediction = SimpleNamespace(boxes=SimpleNamespace(data=torch.empty((0, 6))), names={})
    model = SimpleNamespace(predict=lambda *args, **kwargs: [prediction])
    sampler = Mock(side_effect=AssertionError('Power sampler must not run'))

    monkeypatch.setattr(runner, 'IsolatedPower', sampler)
    monkeypatch.setattr(runner, 'configure', lambda variant: (None, None, None))
    monkeypatch.setattr(runner.guard, 'check_competing_benchmark', lambda: None)
    monkeypatch.setattr(runner.guard, 'device_snapshot', lambda: {})
    monkeypatch.setattr(runner.cv2, 'imread', lambda path: image)
    monkeypatch.setattr(runner.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(runner.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(runner.torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(runner.torch.cuda, 'max_memory_reserved', lambda: 0)

    runner.trial(
        model,
        variant='always_on',
        video='clip',
        repeat=0,
        freeze={},
        experiment=tmp_path,
        source=source,
        power_backend='none',
    )

    record = read(tmp_path / 'results/live/r0_always_on_clip.json.gz')

    assert record['status'] == 'PASS'
    assert record['calls'] == 1
    assert record['energy_joules'] is None
    assert record['power_backend'] == 'none'
    assert not (tmp_path / 'results/power').exists()
    sampler.assert_not_called()
