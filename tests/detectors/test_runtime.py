"""Runtime choices must reach the detector without changing its image pipeline."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torchvision.ops

from adaptive_video.detectors import yolo
from adaptive_video.detectors.configuration import apply_precision, inference_settings
from adaptive_video.detectors import tensorrt_runtime as runtime


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
    constructor = Mock(return_value=model)
    adapter = Mock()
    monkeypatch.setattr(yolo, 'YOLO', constructor)
    monkeypatch.setattr(runtime, 'TensorRTDetector', adapter)
    settings = inference_settings({'backend': backend, 'precision': precision})

    result = yolo.load_detector(device='cpu', settings=settings)

    assert settings['engine'] == f'models/yolo11x_{precision}.engine'
    assert torch.backends.cuda.matmul.allow_tf32 is (precision == 'tf32')
    assert torch.backends.cudnn.allow_tf32 is (precision == 'tf32')
    assert torchvision.ops.nms is yolo.NATIVE_NMS

    if backend == 'torch':
        assert result is model
        adapter.assert_not_called()
    else:
        assert result is adapter.return_value
        adapter.assert_called_once_with(model, yolo.ROOT / settings['engine'], precision)


def test_native_and_custom_suppression_agree():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [20., 20., 30., 30.]])
    scores = torch.tensor([0.9, 0.8, 0.7])

    yolo.select_nms('custom')
    custom = torchvision.ops.nms(boxes, scores, 0.5)
    yolo.select_nms('native')
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
    forward = Mock()
    constructor = Mock(return_value=forward)
    monkeypatch.setattr(runtime, 'require_tensorrt', lambda: None)
    monkeypatch.setattr(runtime, 'TensorRTForward', constructor)
    detector = runtime.TensorRTDetector(model, engine, 'tf32')
    options = {'imgsz': (736, 1280), 'conf': 0.5, 'verbose': False}

    detector.predict('first frame', **options)
    detector.predict('second frame', **options)

    constructor.assert_called_once_with(engine, inputs)
    assert model.predictor.inference is forward
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
    decoded, heads = forward(second)

    assert forward.context.set_tensor_address.call_args_list[0].args == ('images', first.data_ptr())
    assert forward.context.set_tensor_address.call_args_list[1].args == ('images', second.data_ptr())
    assert forward.context.execute_async_v3.call_count == 2
    assert decoded is forward.outputs['decoded']
    assert heads == [forward.outputs[f'head{index}'] for index in range(3)]
