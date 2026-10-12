"""Ultralytics YOLO with explicit precision and NMS choices."""

import torch
import torchvision.ops
import torchvision.ops.boxes
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..project import ROOT
from .configuration import apply_precision, inference_settings


NATIVE_NMS = torchvision.ops.nms


def _overlap_with_selected(boxes, area, selected, candidates):
    edge = boxes[selected]
    x_low = torch.clamp(boxes[candidates, 0], min=edge[0].item())
    y_low = torch.clamp(boxes[candidates, 1], min=edge[1].item())
    x_high = torch.clamp(boxes[candidates, 2], max=edge[2].item())
    y_high = torch.clamp(boxes[candidates, 3], max=edge[3].item())
    intersection = (x_high - x_low).clamp(min=0) * (y_high - y_low).clamp(min=0)

    return intersection / (area[selected] + area[candidates] - intersection + 1e-7)


def _pure_torch_nms(boxes, scores, iou_threshold):
    extent = boxes[:, 2:] - boxes[:, :2]
    area = extent[:, 0] * extent[:, 1]
    candidates = scores.argsort(descending=True)
    selected = []

    while candidates.numel():
        winner = candidates[0].item()
        selected.append(winner)
        candidates = candidates[1:]

        if candidates.numel():
            overlap = _overlap_with_selected(boxes, area, winner, candidates)
            candidates = candidates[overlap <= iou_threshold]

    return torch.tensor(selected, dtype=torch.int64, device=boxes.device)


def select_nms(implementation):
    """Select suppression for the one detector runtime used by this process."""
    if implementation not in ('native', 'custom'):
        raise ValueError(f'Unknown NMS implementation: {implementation}')

    function = NATIVE_NMS if implementation == 'native' else _pure_torch_nms
    torchvision.ops.nms = function
    torchvision.ops.boxes.nms = function


def load_detector(device='cuda', settings=None):
    settings = inference_settings(settings)
    apply_precision(settings['precision'])
    select_nms(settings['nms'])
    model = YOLO(str(ROOT / settings['checkpoint'])).to(device)

    if settings['backend'] == 'tensorrt':
        # TensorRT remains optional for reviewers using the eager route.
        from .tensorrt_runtime import TensorRTDetector

        return TensorRTDetector(model, ROOT / settings['engine'], settings['precision'])

    return model
