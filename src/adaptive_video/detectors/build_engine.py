"""Export a detector forward graph and build a device-local FP32 TensorRT engine."""

import json
from importlib.metadata import version
import time

import torch
from ultralytics import __version__ as ultralytics_version

from ..project import ROOT
from .configuration import apply_precision, inference_settings
from .tensorrt_runtime import require_tensorrt
from .factory import prepare_export


@torch.inference_mode()
def build_engine(configuration, device='cuda'):
    # ONNX is only needed when exporting, not for an eager or engine-only run.
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError('Engine export requires ONNX: run bash scripts/setup.sh') from error

    trt = require_tensorrt()
    settings = inference_settings(configuration)
    destination = ROOT / settings['engine']
    onnx_path = destination.with_suffix('.onnx')
    metadata_path = destination.with_suffix('.json')

    if any(path.exists() for path in (destination, onnx_path, metadata_path)):
        raise FileExistsError(f'Build outputs already exist at {destination}; choose another inference.engine path')

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    apply_precision('fp32')
    exported, inputs, output_names, prediction_format = prepare_export(settings, device)

    exported(inputs)

    print('Exporting FP32 detector forward graph...', flush=True)
    torch.onnx.export(
        exported,
        inputs,
        str(onnx_path),
        opset_version=17,
        dynamo=False,
        input_names=['images'],
        output_names=output_names,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(onnx_path))

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 0

    if int(trt.__version__.split('.')[0]) < 11:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(index)) for index in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if settings['precision'] == 'tf32':
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)

    print(f"Building TensorRT engine with {settings['precision']} arithmetic policy...", flush=True)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')

    destination.write_bytes(bytes(serialized))
    metadata = {
        'precision': settings['precision'],
        'tf32_allowed': config.get_flag(trt.BuilderFlag.TF32),
        'input_shape': list(inputs.shape),
        'input_dtype': 'float32',
        'checkpoint': settings['checkpoint'],
        'tensorrt': trt.__version__,
        'gpu': torch.cuda.get_device_name(inputs.device),
        'torch': torch.__version__,
        'build_seconds': time.perf_counter() - started,
        'outputs': output_names,
        'ultralytics': ultralytics_version,
        'prediction_format': prediction_format,
        'max_det': 300,
        'detector': settings.get('model', 'yolo'),
        'transformers': version('transformers') if settings.get('model') == 'rtdetrv2' else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print('Saved engine:', destination)
    print('Saved metadata:', metadata_path)

    return destination
