"""YOLO loading, export and Ultralytics predictor integration."""

import torch
from ultralytics import YOLO

from ..project import ROOT
from .nms import select_nms
from .tensorrt_runtime import TensorRTForward, read_engine_metadata, require_tensorrt


def configure_detector(model, nms):
    """Choose the trained output head before Ultralytics fuses the model."""
    if nms == 'none':
        head = model.model.model[-1]

        if getattr(head, 'one2one_cv2', None) is None:
            raise ValueError('nms: none requires a detector with a one-to-one head, such as YOLO26')

    if hasattr(model.model, 'end2end'):
        model.model.end2end = nms == 'none'

    # Ultralytics applies these options when it constructs the predictor.
    model.overrides['nms'] = nms != 'none'
    model.overrides['max_det'] = 300


class ExportModel(torch.nn.Module):
    """Export decoded detections without depending on auxiliary-head structures."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs):
        predictions = self.network(inputs)

        return predictions[0] if isinstance(predictions, (tuple, list)) else predictions


class YoloDetector:
    """Run YOLO through its normal predictor with either Torch or TensorRT.

    Ultralytics owns preprocessing and postprocessing for both backends. Its
    eager model remains resident for predictor setup when TensorRT is selected.
    """

    def __init__(self, settings, device='cuda'):
        select_nms(settings['nms'])
        self.network = YOLO(str(ROOT / settings['checkpoint'])).to(device)
        configure_detector(self.network, settings['nms'])
        self.backend = settings['backend']
        self.engine_path = ROOT / settings['engine']
        self.runtime = None

        if self.backend == 'tensorrt':
            require_tensorrt()
            self.metadata = read_engine_metadata(self.engine_path, settings['precision'])
            expected = 'xyxy_score_class' if settings['nms'] == 'none' else 'xywh_scores'

            if self.metadata.get('prediction_format', 'xywh_scores') != expected:
                raise ValueError('Engine output head differs from inference.nms; build the matching engine')

    def engine_forward(self, inputs, *args, **kwargs):
        """Return engine tensors in the format expected by Ultralytics."""
        outputs = self.runtime(inputs)

        if len(outputs) == 1:
            return outputs['decoded']

        # Existing four-output engines also return three auxiliary head tensors.
        return outputs['decoded'], [outputs[f'head{index}'] for index in range(3)]

    @torch.inference_mode()
    def predict(self, frame, **kwargs):
        if self.backend == 'tensorrt' and self.runtime is None:
            # Initialize during untimed warmup, keeping Ultralytics' image pipeline.
            self.network.predict(frame, **kwargs)
            inputs = self.network.predictor.preprocess([frame])
            runtime = TensorRTForward(self.engine_path, inputs)
            names = set(runtime.outputs)

            if names not in ({'decoded'}, {'decoded', 'head0', 'head1', 'head2'}):
                raise ValueError('YOLO engine must return decoded predictions')

            self.runtime = runtime
            self.network.predictor.inference = self.engine_forward

        return self.network.predict(frame, **kwargs)


def load_detector(settings, device='cuda'):
    return YoloDetector(settings, device=device)


def prepare_export(settings, device):
    """Prepare the selected YOLO head for the shared engine builder."""
    detector = YOLO(str(ROOT / settings['checkpoint']))
    configure_detector(detector, settings['nms'])

    network = detector.model.to(device).float().eval()
    network.fuse(verbose=False)

    inputs = torch.zeros((1, 3, 736, 1280), device=device, dtype=torch.float32)
    exported = ExportModel(network).eval()
    prediction_format = 'xyxy_score_class' if settings['nms'] == 'none' else 'xywh_scores'

    return exported, inputs, ['decoded'], prediction_format
