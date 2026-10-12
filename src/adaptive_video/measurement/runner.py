"""Run each variant on a CUDA GPU and record predictions, time and optional energy."""

import os

# CUDA reads this setting when its libraries initialize; set it before Torch.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import platform
from importlib.metadata import version
import resource
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import __version__ as ultralytics_version
from packaging.version import Version

from .. import dataset as common
from ..detectors.factory import load_detector
from ..detectors.configuration import inference_settings
from pathlib import Path

from ..project import ROOT
from ..evaluation.dataset import EvaluationDataset
from ..tracking.output import digest, emit, state
from . import guard
from .isolated import IsolatedPower
from .power import integrate
from .protocol import ACTION_KEYS, always_rows
from .support import configure, exact_repeat, extended_state
from ..variants.registry import VARIANTS


class TimedDetector:
    """Wrap detector calls and record their elapsed time and raw outputs."""

    def __init__(self, model):
        self.model = model
        self.times = []
        self.raw = []

    def predict(self, *args, **kwargs):
        call_start = time.monotonic()
        result = self.model.predict(*args, **kwargs)
        torch.cuda.synchronize()
        self.times.append(1000 * (time.monotonic() - call_start))
        self.raw.append(result[0].boxes.data.detach())

        return result


def trial(model, variant, video, repeat, freeze, experiment, source, power_backend='none'):
    """Measure one complete video, excluding setup and final file serialization."""
    runtime_settings = inference_settings(freeze.get('inference', {'nms': 'custom'}))
    destination = experiment / 'results/live' / f'r{repeat}_{variant}_{video}.json.gz'
    power_path = experiment / 'results/power' / f'r{repeat}_{variant}_{video}.json'

    if destination.exists():
        completed = common.read(destination)
        assert completed['status'] == 'PASS'
        recorded_backend = completed.get('power_backend', 'vdd_in')

        if recorded_backend != power_backend:
            raise ValueError('Choose a new experiment directory when changing power backend')

        previous_settings = completed.get('inference', inference_settings({'nms': 'custom'}))

        if previous_settings != runtime_settings:
            raise ValueError('Choose a new experiment directory when changing detector runtime')

        return

    assert not power_path.exists(), 'Preserve and diagnose existing partial power trace before retrying'
    guard.check_competing_benchmark()
    guard.device_snapshot()
    frames = source.frames(video)
    detector, track_type, emission = configure(variant)
    warmup_image = source.image(frames[0])

    for _ in range(3):
        model.predict(warmup_image, imgsz=(736, 1280), conf=0.5, verbose=False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    meter = TimedDetector(model)
    outputs = []
    actions = []
    frame_ms = []
    decode_ms = []
    emission_ms = []
    controller_ms = []
    raw_indices = []
    monitor_ms = []
    device_checks = []
    feature_records = []
    power = None

    if power_backend == 'vdd_in':
        power = IsolatedPower().start()
        power.phase = f'r{repeat}_{variant}_{video}'
        time.sleep(0.25)

    frame_progress = tqdm(
        frames,
        desc=f'{variant} | {video} | repeat {repeat + 1}',
        unit='frame',
        position=1,
        leave=False,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    start = time.monotonic()

    try:
        for frame_index, frame in enumerate(frame_progress):
            if frame_index % 20 == 0:
                monitor_start = time.monotonic()
                guard.check_competing_benchmark()
                device_checks.append(guard.device_snapshot())
                monitor_ms.append(1000 * (time.monotonic() - monitor_start))

            frame_start = time.monotonic()
            image = cv2.imread(str(ROOT / frame['path']))
            assert image is not None and image.shape[:2] == (frame['height'], frame['width'])
            decode_end = time.monotonic()
            calls_before = len(meter.times)

            if detector is None:
                prediction = meter.predict(image, imgsz=(736, 1280), conf=0.5, verbose=False)
                result = {
                    'frame_idx': frame_index,
                    'ran_inference': True,
                    'ran_erd': False,
                    'erd_result': None,
                    'num_rescued': 0,
                    'num_tracks': 0,
                    'num_confirmed': 0,
                }
            else:
                result = detector.step(image, meter, 'cuda')

            inference_end = time.monotonic()
            output = always_rows(prediction) if detector is None else emit(detector, emission)
            torch.cuda.synchronize()
            frame_end = time.monotonic()
            outputs.append(output)
            actions.append({key: result[key] for key in ACTION_KEYS})
            raw_indices.append(calls_before if len(meter.times) > calls_before else None)
            frame_ms.append(1000 * (frame_end - frame_start))
            decode_ms.append(1000 * (decode_end - frame_start))
            emission_ms.append(1000 * (frame_end - inference_end))
            controller_ms.append(1000 * (frame_end - decode_end) - sum(meter.times[calls_before:]))
            feature_records.append({
                key: getattr(detector, key)
                for key in ['camera_info', 'appearance_info', 'visibility_info']
                if hasattr(detector, key)
            })

        end = time.monotonic()

        if power is not None:
            time.sleep(0.25)

    finally:
        frame_progress.close()

        if power is not None:
            power.stop()
            power.save(power_path, {
                'start': start,
                'end': locals().get('end'),
                'variant': variant,
                'video': video,
                'repeat': repeat,
            })

    joules = None

    if power is not None:
        assert not any('error' in sample for sample in power.rows)
        joules = integrate(power.rows, start, end)

    guard.check_competing_benchmark()
    device_checks.append(guard.device_snapshot())
    raw_hashes = [digest(tensor.cpu().numpy()) for tensor in meter.raw]
    final_state = None if detector is None else digest(state(detector))
    extended_final_state = digest(extended_state(detector, track_type))
    data = {
        'status': 'PASS',
        'variant': variant,
        'video': video,
        'repeat': repeat,
        'frames': len(frames),
        'calls': len(meter.times),
        'outputs': outputs,
        'trace': actions,
        'detector_raw_hashes': [None if index is None else raw_hashes[index] for index in raw_indices],
        'final_state_sha256': final_state,
        'extended_final_state_sha256': extended_final_state,
        'frame_ms': frame_ms,
        'decode_ms': decode_ms,
        'emission_ms': emission_ms,
        'controller_ms': controller_ms,
        'detector_ms': meter.times,
        'monitor_ms': monitor_ms,
        'device_checks': device_checks,
        'feature_records': feature_records,
        'wall_s': end - start,
        'energy_joules': joules,
        'power_backend': power_backend,
        'inference': runtime_settings,
        'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'rss_process_lifetime_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'failure_counters': {
            key: getattr(track_type, key)
            for key in ['domain_failures_total', 'correction_restarts_total']
            if track_type is not None and hasattr(track_type, key)
        },
        'scope': 'Complete unpaced live pipeline; decode, neural/ERD, features, tracking, emission, recording and monitoring; hashing/setup/serialization outside clip',
    }

    if repeat:
        first_repeat = common.read(experiment / 'results/live' / f'r0_{variant}_{video}.json.gz')
        exact_repeat(data, first_repeat)

    # Check raw detector parity here when the corresponding always-on trial is
    # already present. This check alone does not cover a later always-on trial.
    always_path = experiment / 'results/live' / f'r{repeat}_always_on_{video}.json.gz'

    if variant != 'always_on' and always_path.exists():
        always_on = common.read(always_path)

        for frame_index, checksum in enumerate(data['detector_raw_hashes']):
            if checksum is not None:
                assert checksum == always_on['detector_raw_hashes'][frame_index], (
                    variant, video, frame_index, 'raw detector drift',
                )

    common.save(destination, data)
    tqdm.write(
        f"repeat {repeat + 1} | {variant} | {video} | "
        f"calls {data['calls']}/{len(frames)} | "
        f"{data['wall_s']:.2f}s | energy {joules} J | PASS"
    )


def main(experiment, power_backend='none'):
    """Validate the prepared campaign, initialize YOLO and execute its schedule."""
    if not torch.cuda.is_available():
        raise RuntimeError('Live measurement requires a CUDA-capable GPU and compatible PyTorch')

    if power_backend not in {'none', 'vdd_in'}:
        raise ValueError(f'Unknown power backend: {power_backend}')

    if power_backend == 'vdd_in':
        guard.require_module_power()
    experiment = Path(experiment).resolve()
    freeze = common.read(experiment / 'manifests/freeze.json')
    source = EvaluationDataset(experiment / freeze['dataset_manifest'])

    if Version(np.__version__) < Version('1.26') or Version(cv2.__version__) < Version('4.8'):
        raise RuntimeError('Measurement requires NumPy >= 1.26 and OpenCV >= 4.8')
    if power_backend == 'vdd_in':
        assert common.read(experiment / 'results/power_check.json')['status'] == 'PASS'
    available_variants = set(VARIANTS) | {'always_on'}

    if not freeze['arms'] or not set(freeze['arms']) <= available_variants:
        raise ValueError('Experiment contains an unknown variant or no variants')

    inventory = common.read(experiment / freeze['dataset_manifest'])

    for path, checksum in tqdm(
        inventory['image_sha256'].items(),
        desc='Checking input images',
        unit='image',
        mininterval=1.0,
        dynamic_ncols=True,
    ):
        assert common.sha(ROOT / path) == checksum, path

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
    guard.check_competing_benchmark()
    guard.device_snapshot()
    
    tqdm.write('Loading detector and warming up...')
    runtime_settings = inference_settings(freeze.get('inference', {'nms': 'custom'}))
    model = load_detector(settings=runtime_settings)

    for _ in range(8):
        first_frame = source.image(source.frames(freeze['videos'][0])[0])
        model.predict(first_frame, imgsz=(736, 1280), conf=0.5, verbose=False)

    environment_path = experiment / 'results/environment.json'

    if not environment_path.exists():
        common.save(environment_path, {
            'time': common.now(),
            'python': sys.version,
            'torch': torch.__version__,
            'ultralytics': ultralytics_version,
            'transformers': version('transformers') if runtime_settings.get('model') == 'rtdetrv2' else None,
            'numpy': np.__version__,
            'opencv': cv2.__version__,
            'promotion': getattr(np, '_get_promotion_state', lambda: 'not exposed')(),
            'device': torch.cuda.get_device_name(0),
            'architecture': platform.machine(),
            'power_backend': power_backend,
            'inference': runtime_settings,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'engine_metadata': getattr(model, 'metadata', None),
            'memory_note': 'Torch allocator statistics exclude TensorRT internal allocations',
        })

    with tqdm(
        freeze['schedule'],
        desc='Experiment trials',
        unit='trial',
        position=0,
        mininterval=1.0,
        dynamic_ncols=True,
    ) as trial_progress:
        for scheduled_trial in trial_progress:
            try:
                trial(
                    model, **scheduled_trial, freeze=freeze, experiment=experiment,
                    source=source, power_backend=power_backend,
                )
            except Exception as error:
                filename = f"r{scheduled_trial['repeat']}_{scheduled_trial['variant']}_{scheduled_trial['video']}.json"
                common.save(experiment / 'results/failures' / filename, {
                    'time': common.now(),
                    'trial': scheduled_trial,
                    'error': repr(error),
                    'traceback': traceback.format_exc(),
                })
                raise

    print(f"COMPLETE all {len(freeze['schedule'])} live trials", flush=True)

