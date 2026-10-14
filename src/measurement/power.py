"""Sample INA3221 module-input power and integrate energy over workload intervals.

The boundary is Jetson VDD_IN, not wall AC or the complete carrier board.
Sysfs and tegrastats read the same sensor; agreement is not independent calibration.
"""

import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from .guard import require_module_power


class Power:
    """Collect timestamped sysfs samples and the parallel tegrastats output."""

    def __init__(self):
        self.sensor = next(
            path.parent
            for path in Path('/sys/class/hwmon').glob('hwmon*/in1_label')
            if path.read_text().strip() == 'VDD_IN'
        )
        self.rows = []
        self.tgs = []
        self.stop_event = threading.Event()
        self.phase = 'setup'

    def start(self):
        self.proc = subprocess.Popen(
            ['tegrastats', '--interval', '100'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.tt = threading.Thread(target=self.read_tgs, daemon=True)
        self.tt.start()

        return self

    def read_tgs(self):
        for line in self.proc.stdout:
            self.tgs.append({
                't': time.monotonic(),
                'phase': self.phase,
                'line': line.strip(),
            })

    def loop(self):
        while not self.stop_event.is_set():
            sample_start = time.monotonic()
            sample = {'t': sample_start, 'phase': self.phase}

            try:
                for channel in [1, 2, 3]:
                    voltage_mv = float((self.sensor / f'in{channel}_input').read_text())
                    current_ma = float((self.sensor / f'curr{channel}_input').read_text())
                    sample[f'voltage{channel}_mV'] = voltage_mv
                    sample[f'current{channel}_mA'] = current_ma
                    sample[f'power{channel}_W'] = voltage_mv * current_ma / 1e6

                sample['read_s'] = time.monotonic() - sample_start

            except Exception as error:
                sample['error'] = repr(error)

            self.rows.append(sample)
            self.stop_event.wait(0.05)

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.proc.terminate()

        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

        self.tt.join(timeout=1)

    def save(self, path, extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            'sensor': str(self.sensor),
            'boundary': 'Jetson module VDD_IN',
            'samples': self.rows,
            'tegrastats': self.tgs,
            **extra,
        }
        path.write_text(json.dumps(result, indent=2) + '\n')


def integrate(rows, start, end):
    """Integrate watts over seconds using interpolated endpoints and trapezoids."""
    valid_samples = [sample for sample in rows if 'power1_W' in sample]
    timestamps = np.array([sample['t'] for sample in valid_samples])
    power_watts = np.array([sample['power1_W'] for sample in valid_samples])

    if len(timestamps) < 2 or start < timestamps[0] or end > timestamps[-1]:
        raise ValueError('Power samples do not bracket workload')

    inside_interval = (timestamps > start) & (timestamps < end)
    interval_times = np.r_[start, timestamps[inside_interval], end]
    interval_power = np.r_[
        np.interp(start, timestamps, power_watts),
        power_watts[inside_interval],
        np.interp(end, timestamps, power_watts),
    ]

    if np.diff(interval_times).max() > 0.25:
        raise ValueError('Sampling gap >250ms')

    energy_joules = np.sum(
        np.diff(interval_times) * (interval_power[:-1] + interval_power[1:]) * 0.5
    )

    return float(energy_joules)


def validate(out, sampler_factory=None):
    """Capture two idle/load/recovery cycles on Jetson for telemetry checks."""
    require_module_power()

    # Keep Torch out of the sampler subprocess and CPU-only integration imports.
    import torch

    torch.set_num_threads(1)
    inputs = torch.ones((2048, 2048), device='cuda')
    weights = torch.ones_like(inputs) * 0.01

    for _ in range(5):
        product = inputs @ weights

    torch.cuda.synchronize()
    sampler_type = Power if sampler_factory is None else sampler_factory
    sampler = sampler_type().start()
    intervals = []

    try:
        for repeat in range(2):
            for phase, duration in [('idle', 10), ('gpu_load', 20), ('recovery', 10)]:
                sampler.phase = f'{repeat}_{phase}'
                start = time.monotonic()
                iterations = 0

                while time.monotonic() - start < duration:
                    if phase == 'gpu_load':
                        product = inputs @ weights
                        torch.cuda.synchronize()
                        iterations += 1
                    else:
                        time.sleep(0.05)

                end = time.monotonic()
                intervals.append({
                    'phase': sampler.phase,
                    'start': start,
                    'end': end,
                    'iterations': iterations,
                })

        sampler.phase = 'tail'
        time.sleep(0.3)

    finally:
        sampler.stop()

    sampler.save(out, {'intervals': intervals, 'status': 'raw_capture_complete'})

    for interval in intervals:
        interval['joules'] = integrate(sampler.rows, interval['start'], interval['end'])
        interval['mean_W'] = interval['joules'] / (interval['end'] - interval['start'])

    sampler.save(out, {
        'intervals': intervals,
        'torch_version': torch.__version__,
        'cuda_device': torch.cuda.get_device_name(0),
        'validation_scope': 'plausibility and dynamic response, not external absolute calibration',
    })
    print(json.dumps(intervals, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    validate(parser.parse_args().output)
