"""Check sensor plausibility, sampling cadence and repeatable load response."""

import json
import re

import numpy as np

from .. import dataset as common
from pathlib import Path
from .power import integrate


def main(experiment):
    directory = Path(experiment).resolve()
    capture = common.read(directory / 'results/power_validation.json')
    samples = capture['samples']
    sample_intervals = np.diff([sample['t'] for sample in samples])
    comparisons = []

    for interval in capture['intervals']:
        interval['joules'] = integrate(samples, interval['start'], interval['end'])
        interval['mean_W'] = interval['joules'] / (interval['end'] - interval['start'])
        sysfs_power = [
            sample['power1_W'] for sample in samples
            if sample['phase'] == interval['phase']
        ]
        tegrastats_power = [
            float(re.search(r'VDD_IN (\d+)mW', sample['line']).group(1)) / 1000
            for sample in capture['tegrastats']
            if sample['phase'] == interval['phase'] and 'VDD_IN ' in sample['line']
        ]
        comparisons.append({
            'phase': interval['phase'],
            'sysfs_mean_W': float(np.mean(sysfs_power)),
            'tegrastats_mean_W': float(np.mean(tegrastats_power)),
            'difference_W': float(np.mean(sysfs_power) - np.mean(tegrastats_power)),
        })

    checks = {
        'no_sensor_errors': all('error' not in sample for sample in samples),
        'positive_plausible_readings': all(0 < sample['power1_W'] < 60 for sample in samples),
        'max_interval_below_250ms': bool(sample_intervals.max() < 0.25),
        'software_readers_agree_within_005W': all(
            abs(comparison['difference_W']) < 0.05 for comparison in comparisons
        ),
        'both_gpu_loads_exceed_idle_by_5W': all(
            capture['intervals'][index + 1]['mean_W'] - capture['intervals'][index]['mean_W'] > 5
            for index in [0, 3]
        ),
        'repeated_load_means_within_2percent': abs(
            capture['intervals'][1]['mean_W'] / capture['intervals'][4]['mean_W'] - 1
        ) < 0.02,
    }
    result = {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'samples': len(samples),
        'dt_ms_quantiles': np.quantile(sample_intervals * 1000, [0.5, 0.95, 1]).tolist(),
        'comparison': comparisons,
        'intervals': capture['intervals'],
        'independent_calibration': False,
    }
    common.save(directory / 'results/power_check.json', result)
    print(json.dumps(result, indent=2))

