"""Read the detector runtime choices stored in the study configuration."""

from pathlib import Path

import torch


def inference_settings(configuration=None):
    settings = {
        'backend': 'torch',
        'precision': 'tf32',
        'nms': 'native',
        'checkpoint': 'models/yolo11x.pt',
    }
    supplied = {} if configuration is None else configuration

    if not isinstance(supplied, dict):
        raise ValueError('inference must be a YAML mapping')

    unknown = set(supplied) - set(settings) - {'engine'}

    if unknown:
        raise ValueError(f'Unknown inference settings: {sorted(unknown)}')

    settings.update(supplied)

    for key, choices in {
        'backend': ('torch', 'tensorrt'),
        'precision': ('fp32', 'tf32'),
        'nms': ('native', 'custom'),
    }.items():
        if settings[key] not in choices:
            raise ValueError(f'inference.{key} must be one of {choices}')

    checkpoint = settings['checkpoint']

    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError('inference.checkpoint must be a nonempty path')

    if not settings.get('engine'):
        path = Path(checkpoint)
        settings['engine'] = str(path.with_name(f"{path.stem}_{settings['precision']}.engine"))

    if not isinstance(settings['engine'], str):
        raise ValueError('inference.engine must be a path')

    return settings


def apply_precision(precision):
    """Keep FP32 tensors; optionally allow TF32 matmul and convolution arithmetic."""
    if precision not in ('fp32', 'tf32'):
        raise ValueError(f'Unknown precision: {precision}')

    allow_tf32 = precision == 'tf32'

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
