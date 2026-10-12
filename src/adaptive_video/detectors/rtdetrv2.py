"""RT-DETRv2 R50-vd"""

from types import SimpleNamespace

import cv2
import torch
from transformers import RTDetrImageProcessor, RTDetrV2Config, RTDetrV2ForObjectDetection
from ultralytics.engine.results import Results

from ..project import ROOT
from .tensorrt_runtime import TensorRTForward, read_engine_metadata


def load_network(checkpoint, device):
    network = RTDetrV2ForObjectDetection.from_pretrained(
        checkpoint, local_files_only=True,
    )
    network.config.disable_custom_kernels = True

    return network.to(device).float().eval()


class ExportModel(torch.nn.Module):
    """Keep class logits and normalized cxcywh boxes in the exported graph."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs):
        output = self.network(pixel_values=inputs, return_dict=True)

        return output.logits, output.pred_boxes


class RTDETRv2Detector:
    def __init__(self, settings, device='cuda'):
        checkpoint = ROOT / settings['checkpoint']
        config = RTDetrV2Config.from_pretrained(checkpoint, local_files_only=True)
        self.processor = RTDetrImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.device = torch.device(device)
        self.names = {int(index): name for index, name in config.id2label.items()}
        self.names[3] = 'motorcycle'
        self.engine_path = ROOT / settings['engine']
        self.runtime = None
        self.network = None
        self.metadata = {'detector': 'rtdetrv2', 'input_shape': [1, 3, 640, 640]}

        if settings['backend'] == 'torch':
            self.network = load_network(checkpoint, device)
        else:
            self.metadata = read_engine_metadata(self.engine_path, settings['precision'])

            if self.metadata.get('prediction_format') != 'logits_boxes':
                raise ValueError('RT-DETRv2 requires an engine with logits and normalized boxes')

    @torch.inference_mode()
    def predict(self, frame, imgsz=None, conf=0.5, verbose=False):
        # RT-DETRv2 uses its checkpoint's 640x640 resize.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors='pt')['pixel_values'].to(self.device)

        if self.network is not None:
            output = self.network(pixel_values=inputs)
        else:
            if self.runtime is None:
                self.runtime = TensorRTForward(self.engine_path, inputs)

            tensors = self.runtime(inputs)
            output = SimpleNamespace(logits=tensors['logits'], pred_boxes=tensors['boxes'])

        predictions = self.processor.post_process_object_detection(
            output, threshold=conf, target_sizes=[frame.shape[:2]],
        )[0]
        rows = torch.cat([
            predictions['boxes'],
            predictions['scores'][:, None],
            predictions['labels'][:, None].to(torch.float32),
        ], dim=1)

        return [Results(frame, path='', names=self.names, boxes=rows)]


def load_detector(settings, device='cuda'):
    return RTDETRv2Detector(settings, device=device)


def prepare_export(settings, device):
    """Prepare the RT-DETRv2 graph for the shared engine builder."""
    network = load_network(ROOT / settings['checkpoint'], device)
    inputs = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float32)

    exported = ExportModel(network).eval()

    return exported, inputs, ['logits', 'boxes'], 'logits_boxes'
