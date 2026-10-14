"""Runtime choices must reach the detector without changing its image pipeline."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torchvision.ops

from src.detectors import factory, nms, yolo
from src.detectors.configuration import apply_precision, inference_settings
from src.detectors import tensorrt_runtime as runtime


@pytest.fixture(autouse=True)
def restore_runtime_settings():
    matmul = torch.backends.cuda.matmul.allow_tf32
    convolution = torch.backends.cudnn.allow_tf32
    nms = torchvision.ops.nms
    boxes_nms = torchvision.ops.boxes.nms

    yield

    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = convolution
    torchvision.ops.nms = nms
    torchvision.ops.boxes.nms = boxes_nms


@pytest.mark.parametrize('settings', [[], '', {'precision': 'fp16'}, {'backend': 'other'}, {'nms': 'other'}])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        inference_settings(settings)


@pytest.mark.parametrize('backend', ['torch', 'tensorrt'])
@pytest.mark.parametrize('precision', ['fp32', 'tf32'])
def test_runtime_selection(monkeypatch, backend, precision):
    model = Mock()
    model.to.return_value = model
    model.overrides = {}
    constructor = Mock(return_value=model)
    monkeypatch.setattr(yolo, 'YOLO', constructor)
    monkeypatch.setattr(yolo, 'require_tensorrt', Mock())
    metadata = Mock(return_value={'prediction_format': 'xywh_scores'})
    monkeypatch.setattr(yolo, 'read_engine_metadata', metadata)
    settings = inference_settings({'backend': backend, 'precision': precision})

    result = factory.load_detector(device='cpu', settings=settings)

    assert settings['engine'] == f'models/yolo11x_{precision}.engine'
    assert torch.backends.cuda.matmul.allow_tf32 is (precision == 'tf32')
    assert torch.backends.cudnn.allow_tf32 is (precision == 'tf32')
    assert torchvision.ops.nms is nms.NATIVE_NMS

    assert isinstance(result, yolo.YoloDetector)
    assert result.network is model
    assert result.backend == backend

    if backend == 'torch':
        metadata.assert_not_called()
    else:
        metadata.assert_called_once_with(yolo.ROOT / settings['engine'], precision)


def test_native_and_custom_suppression_agree():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [20., 20., 30., 30.]])
    scores = torch.tensor([0.9, 0.8, 0.7])

    nms.select_nms('custom')
    custom = torchvision.ops.nms(boxes, scores, 0.5)
    nms.select_nms('native')
    native = torchvision.ops.nms(boxes, scores, 0.5)

    assert native.tolist() == custom.tolist() == [0, 2]


def test_precision_mismatch_is_rejected(tmp_path):
    engine = tmp_path / 'detector.engine'
    engine.write_bytes(b'fixture')
    engine.with_suffix('.json').write_text(json.dumps({'precision': 'tf32'}))

    with pytest.raises(ValueError, match='precision'):
        runtime.read_engine_metadata(engine, 'fp32')


def test_adapter_preserves_predict_arguments_and_initializes_once(tmp_path, monkeypatch):
    engine = tmp_path / 'detector.engine'
    engine.write_bytes(b'fixture')
    engine.with_suffix('.json').write_text(json.dumps({'precision': 'tf32'}))
    model = Mock()
    inputs = torch.zeros(1, 3, 32, 32)
    model.predictor.preprocess.return_value = inputs
    engine_runtime = Mock()
    engine_runtime.outputs = {'decoded': object()}
    constructor = Mock(return_value=engine_runtime)
    monkeypatch.setattr(yolo, 'require_tensorrt', lambda: None)
    monkeypatch.setattr(yolo, 'TensorRTForward', constructor)
    model.to.return_value = model
    model.overrides = {}
    monkeypatch.setattr(yolo, 'YOLO', Mock(return_value=model))
    settings = inference_settings({'backend': 'tensorrt', 'engine': str(engine)})
    detector = yolo.YoloDetector(settings, device='cpu')
    options = {'imgsz': (736, 1280), 'conf': 0.5, 'verbose': False}

    detector.predict('first frame', **options)
    detector.predict('second frame', **options)

    constructor.assert_called_once_with(engine, inputs)
    assert model.predictor.inference == detector.engine_forward
    assert model.predict.call_count == 3  # One eager setup, then two normal calls.
    model.predict.assert_called_with('second frame', **options)


def test_forward_rebinds_each_new_frame(monkeypatch):
    forward = runtime.TensorRTForward.__new__(runtime.TensorRTForward)
    forward.input_shape = (1, 3, 2, 2)
    forward.device = torch.device('cpu')
    forward.context = Mock()
    forward.outputs = {name: object() for name in ('decoded', 'head0', 'head1', 'head2')}
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: SimpleNamespace(cuda_stream=123))
    first = torch.zeros(forward.input_shape)
    second = torch.ones(forward.input_shape)

    forward(first)
    outputs = forward(second)

    assert forward.context.set_tensor_address.call_args_list[0].args == ('images', first.data_ptr())
    assert forward.context.set_tensor_address.call_args_list[1].args == ('images', second.data_ptr())
    assert forward.context.execute_async_v3.call_count == 2
    assert outputs is forward.outputs


def test_yolo26_engine_paths_distinguish_output_heads():
    normal = inference_settings({'checkpoint': 'models/yolo26x.pt'})
    end_to_end = inference_settings({'checkpoint': 'models/yolo26x.pt', 'nms': 'none'})

    assert normal['engine'] == 'models/yolo26x_tf32.engine'
    assert end_to_end['engine'] == 'models/yolo26x_tf32_nmsfree.engine'


@pytest.mark.parametrize('nms, end_to_end', [('native', False), ('custom', False), ('none', True)])
def test_detector_head_selected_before_predictor_setup(nms, end_to_end):
    head = SimpleNamespace(one2one_cv2=object())
    network = SimpleNamespace(model=[head], end2end=False)
    model = SimpleNamespace(model=network, overrides={})

    yolo.configure_detector(model, nms)

    assert model.model.end2end is end_to_end
    assert model.overrides['nms'] is (not end_to_end)
    assert model.overrides['max_det'] == 300


def test_nms_free_request_rejected_for_ordinary_yolo_head():
    model = SimpleNamespace(model=SimpleNamespace(model=[SimpleNamespace()]), overrides={})

    with pytest.raises(ValueError, match='one-to-one'):
        yolo.configure_detector(model, 'none')


def test_engine_head_mismatch_rejected_before_execution(tmp_path, monkeypatch):
    engine = tmp_path / 'detector.engine'
    engine.write_bytes(b'fixture')
    engine.with_suffix('.json').write_text(json.dumps({
        'precision': 'tf32', 'prediction_format': 'xyxy_score_class',
    }))
    monkeypatch.setattr(yolo, 'require_tensorrt', lambda: None)

    model = Mock()
    model.to.return_value = model
    model.overrides = {}
    monkeypatch.setattr(yolo, 'YOLO', Mock(return_value=model))
    settings = inference_settings({'backend': 'tensorrt', 'engine': str(engine)})

    with pytest.raises(ValueError, match='output head'):
        yolo.YoloDetector(settings, device='cpu')


@pytest.mark.parametrize('with_auxiliary', [False, True])
def test_export_uses_decoded_tensor_with_any_auxiliary_structure(with_auxiliary):
    from src.detectors.yolo import ExportModel

    class Detector(torch.nn.Module):
        def forward(self, inputs):
            if with_auxiliary:
                return inputs, {'one2many': {'boxes': inputs}, 'one2one': {'scores': inputs}}

            return inputs

    inputs = torch.zeros(1, 300, 6)
    assert ExportModel(Detector())(inputs) is inputs


@pytest.mark.parametrize('model, checkpoint', [
    ('yolo11x', 'models/yolo11x.pt'),
    ('yolo26x', 'models/yolo26x.pt'),
    ('rtdetrv2', 'models/rtdetrv2'),
])
def test_named_detector_defaults(model, checkpoint):
    settings = inference_settings({'model': model})

    assert settings['checkpoint'] == checkpoint
    assert settings['engine'] == f'models/{model}_tf32.engine'
    assert settings['nms'] == ('none' if model == 'rtdetrv2' else 'native')


@pytest.mark.parametrize('model', ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'unknown'])
def test_removed_model_choices_rejected(model):
    with pytest.raises(ValueError, match='inference.model'):
        inference_settings({'model': model})


def test_rtdetrv2_rejects_yolo_nms_setting():
    with pytest.raises(ValueError, match='NMS-free'):
        inference_settings({'model': 'rtdetrv2', 'nms': 'native'})


def test_rtdetrv2_original_coordinates_and_class_mapping():
    import numpy as np
    from transformers import RTDetrImageProcessor
    from src.detectors.rtdetrv2 import RTDETRv2Detector

    detector = RTDETRv2Detector.__new__(RTDETRv2Detector)
    detector.processor = RTDetrImageProcessor(size={'height': 640, 'width': 640})
    detector.device = torch.device('cpu')
    detector.names = {3: 'motorcycle'}
    logits = torch.full((1, 300, 80), -100.)
    logits[0, 0, 3] = 10.
    boxes = torch.tensor([0.5, 0.5, 0.2, 0.4]).repeat(1, 300, 1)
    detector.network = Mock(return_value=SimpleNamespace(logits=logits, pred_boxes=boxes))
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    result = detector.predict(image, imgsz=(736, 1280), conf=0.5)[0]
    inputs = detector.network.call_args.kwargs['pixel_values']

    assert tuple(inputs.shape) == (1, 3, 640, 640)
    assert inputs[0, 0].mean() == 1  # BGR input is converted to RGB, scaled to 0..1.
    assert inputs[0, 2].mean() == 0
    assert len(result.boxes) == 1
    assert result.boxes.data[0, 5] == 3
    torch.testing.assert_close(result.boxes.data[0, :4], torch.tensor([512., 216., 768., 504.]))


@pytest.mark.parametrize('model_name', ['yolo11x', 'yolo26x', 'rtdetrv2'])
def test_factory_routes_loading_and_export(monkeypatch, model_name):
    settings = inference_settings({'model': model_name})
    expected_module = factory.rtdetrv2 if model_name == 'rtdetrv2' else yolo
    loader = Mock()
    exporter = Mock()
    monkeypatch.setattr(expected_module, 'load_detector', loader)
    monkeypatch.setattr(expected_module, 'prepare_export', exporter)

    detector = factory.load_detector(device='cpu', settings=settings)
    exported = factory.prepare_export(settings, 'cpu')

    assert detector is loader.return_value
    assert exported is exporter.return_value
    loader.assert_called_once_with(settings, device='cpu')
    exporter.assert_called_once_with(settings, 'cpu')


@pytest.mark.parametrize('legacy_heads', [False, True])
def test_yolo_adapts_named_engine_outputs(monkeypatch, legacy_heads):
    outputs = {'decoded': torch.zeros(1, 84, 10)}

    if legacy_heads:
        outputs.update({f'head{index}': object() for index in range(3)})

    engine_runtime = Mock(return_value=outputs)
    engine_runtime.outputs = outputs
    detector = yolo.YoloDetector.__new__(yolo.YoloDetector)
    detector.runtime = engine_runtime
    inputs = torch.zeros(1, 3, 32, 32)

    predictions = detector.engine_forward(inputs)

    if legacy_heads:
        assert predictions[0] is outputs['decoded']
        assert predictions[1] == [outputs[f'head{index}'] for index in range(3)]
    else:
        assert predictions is outputs['decoded']

    engine_runtime.assert_called_once_with(inputs)


def test_yolo_eager_predict_preserves_arguments(monkeypatch):
    network = Mock()
    network.to.return_value = network
    network.overrides = {}
    monkeypatch.setattr(yolo, 'YOLO', Mock(return_value=network))
    engine_runtime = Mock()
    monkeypatch.setattr(yolo, 'TensorRTForward', engine_runtime)
    settings = inference_settings({'backend': 'torch'})
    detector = yolo.YoloDetector(settings, device='cpu')
    options = {'imgsz': (736, 1280), 'conf': 0.5, 'verbose': False}

    result = detector.predict('frame', **options)

    assert result is network.predict.return_value
    network.predict.assert_called_once_with('frame', **options)
    engine_runtime.assert_not_called()


def test_rtdetrv2_export_returns_logits_and_boxes():
    network = Mock(return_value=SimpleNamespace(
        logits=torch.zeros(1, 300, 80),
        pred_boxes=torch.zeros(1, 300, 4),
    ))
    inputs = torch.zeros(1, 3, 16, 16)
    exported = factory.rtdetrv2.ExportModel(network)

    logits, boxes = exported(inputs)

    assert logits is network.return_value.logits
    assert boxes is network.return_value.pred_boxes
    network.assert_called_once_with(pixel_values=inputs, return_dict=True)
