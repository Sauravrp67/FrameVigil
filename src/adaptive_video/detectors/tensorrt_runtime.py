"""Model-independent TensorRT loading, binding and execution."""

import json
from pathlib import Path

import torch

try:
    import tensorrt as trt
except ImportError:
    trt = None


def require_tensorrt():
    if trt is None:
        raise RuntimeError('TensorRT Python bindings are required for the TensorRT route')

    if int(trt.__version__.split('.')[0]) < 10:
        raise RuntimeError('This runtime requires TensorRT 10 or newer')

    return trt


def read_engine_metadata(path, precision):
    path = Path(path)
    metadata_path = path.with_suffix('.json')

    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f'Missing engine or metadata: {path}. '
            'Run scripts/build_detector_engine.py --config configs/study.yaml on this machine.'
        )

    metadata = json.loads(metadata_path.read_text())

    if metadata['precision'] != precision:
        raise ValueError(f"Engine precision is {metadata['precision']}, but the experiment requests {precision}")

    return metadata


class TensorRTForward:
    """Bind each new input and execute on PyTorch's current CUDA stream."""

    def __init__(self, path, inputs):
        runtime_api = require_tensorrt()
        self.logger = runtime_api.Logger(runtime_api.Logger.WARNING)
        self.runtime = runtime_api.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())

        if self.engine is None:
            raise RuntimeError('Cannot load TensorRT engine; rebuild it on this GPU and TensorRT version')

        self.context = self.engine.create_execution_context()
        self.outputs = {}
        self.input_shape = tuple(inputs.shape)
        self.device = inputs.device

        if self.context is None:
            raise RuntimeError('Cannot create TensorRT execution context')

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))

            if self.engine.get_tensor_dtype(name) != runtime_api.float32:
                raise ValueError(f'Expected FP32 tensor: {name}')

            if self.engine.get_tensor_mode(name) == runtime_api.TensorIOMode.INPUT:
                if name != 'images' or shape != self.input_shape:
                    raise ValueError(f'Engine input {name} has incompatible shape {shape}')
            else:
                output = torch.empty(shape, device=self.device, dtype=torch.float32)
                self.outputs[name] = output
                if not self.context.set_tensor_address(name, output.data_ptr()):
                    raise RuntimeError(f'Cannot bind TensorRT output: {name}')

        if not self.outputs:
            raise ValueError('Engine has no output tensors')

    def __call__(self, inputs, *args, **kwargs):
        if tuple(inputs.shape) != self.input_shape or inputs.dtype != torch.float32:
            raise ValueError(f'Engine expects FP32 input with shape {self.input_shape}')

        if inputs.device != self.device or not inputs.is_contiguous():
            raise ValueError('Engine input must be contiguous and on the engine CUDA device')

        # Keep this frame alive until the next invocation; output buffers are reused.
        self.inputs = inputs

        if not self.context.set_tensor_address('images', inputs.data_ptr()):
            raise RuntimeError('Cannot bind TensorRT input')

        stream = torch.cuda.current_stream(self.device)

        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT inference failed')

        return self.outputs
