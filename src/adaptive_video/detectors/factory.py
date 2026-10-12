"""Select a detector family for inference or engine export."""

from . import rtdetrv2, yolo
from .configuration import apply_precision, inference_settings


def detector_module(settings):
    if settings.get('model') == 'rtdetrv2':
        return rtdetrv2

    return yolo


def load_detector(device='cuda', settings=None):
    """Create the selected detector with its requested arithmetic policy."""
    settings = inference_settings(settings)
    apply_precision(settings['precision'])

    return detector_module(settings).load_detector(settings, device=device)


def prepare_export(settings, device):
    """Return the model graph, example input, output names and output format."""
    return detector_module(settings).prepare_export(settings, device)
