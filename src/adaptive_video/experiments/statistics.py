"""Summarize recorded timing, decisions and power."""

import numpy as np

from ..dataset import csvwrite
from ..measurement.power import integrate
from ..project import read


STATES = ('empty_skip', 'prediction_only', 'erd_detection', 'detection', 'rescue_detection')


def frame_state(action):
    if not action['ran_inference']:
        return 'empty_skip' if action['ran_erd'] and action['erd_result'] == 'empty' else 'prediction_only'

    if action['num_rescued'] > 0:
        return 'rescue_detection'

    return 'erd_detection' if action['ran_erd'] else 'detection'


def timing_summary(records):
    """Use non-overlapping components; controller_ms already includes emission."""
    records = list(records)
    frame_count = sum(record['frames'] for record in records)
    totals = {
        'decode_ms': sum(sum(record['decode_ms']) for record in records),
        'detector_ms': sum(sum(record['detector_ms']) for record in records),
        'controller_output_ms': sum(sum(record['controller_ms']) for record in records),
    }
    wall_ms = sum(record['wall_s'] * 1000 for record in records)
    totals['other_ms'] = wall_ms - sum(totals.values())

    if totals['other_ms'] < -0.01:
        raise ValueError('Timing components exceed the measured workload interval')

    result = {name: value / frame_count for name, value in totals.items()}
    frame_times = np.concatenate([record['frame_ms'] for record in records])
    result['frame_latency_p50_ms'] = float(np.quantile(frame_times, 0.5))
    result['frame_latency_p95_ms'] = float(np.quantile(frame_times, 0.95))

    for name in STATES:
        count = sum(frame_state(action) == name for record in records for action in record['trace'])
        result[f'{name}_percent'] = 100 * count / frame_count

    return result


def frame_rows(record, frames, errors):
    """Join saved decisions and frame times to offline COCO matching counts."""
    rows = []

    for index, (frame, action) in enumerate(zip(frames, record['trace'])):
        rows.append({
            'variant': record['variant'],
            'video': record['video'],
            'frame_index': index,
            'image_id': frame['image_id'],
            'state': frame_state(action),
            'frame_ms': record['frame_ms'][index],
            'num_tracks': action['num_tracks'],
            'num_rescued': action['num_rescued'],
            **errors.get(frame['image_id'], {'TP75': 0, 'FP75': 0, 'FN75': 0}),
        })

    return rows


def power_rows(folder, record):
    """Clip module-power traces to the exact measured interval and integrate them.

    Frame timestamps were not saved. These traces use elapsed wall time only.
    Auxiliary channels are not assigned rail names without recorded labels.
    """
    if record['energy_joules'] is None:
        return []

    filename = f"r{record['repeat']}_{record['variant']}_{record['video']}.json"
    trace = read(folder / 'results/power' / filename)
    samples = trace['samples']
    expected = integrate(samples, trace['start'], trace['end'])
    timestamps = np.array([sample['t'] for sample in samples])
    watts = np.array([sample['power1_W'] for sample in samples])
    inside = (timestamps > trace['start']) & (timestamps < trace['end'])
    times = np.r_[trace['start'], timestamps[inside], trace['end']]
    powers = np.interp(times, timestamps, watts)
    cumulative = np.r_[0, np.cumsum(np.diff(times) * (powers[:-1] + powers[1:]) / 2)]

    if not np.isclose(cumulative[-1], expected, rtol=1e-10):
        raise ValueError('Cumulative energy differs from integrated module energy')

    return [
        {
            'variant': record['variant'], 'video': record['video'],
            'repeat': record['repeat'], 'elapsed_s': float(time - times[0]),
            'module_W': float(power), 'cumulative_J': float(energy),
        }
        for time, power, energy in zip(times, powers, cumulative)
    ]


def write_report_tables(summary, timing, output):
    """Save numerical comparison tables; positive reductions mean savings."""
    timing_by_variant = {row['variant']: row for row in timing}
    baseline = next((row for row in summary if row['variant'] == 'baseline'), {})
    accuracy = []
    tracking = []
    power = []

    def value(row, field):
        number = row.get(field)
        return float(number) if number not in (None, '') else None

    def reduction(row, field):
        reference = value(baseline, field)
        measured = value(row, field)

        if reference is None or reference <= 0 or measured is None:
            return None

        return 100 * (1 - measured / reference)

    for row in summary:
        variant = row['variant']
        frame_timing = timing_by_variant.get(variant, {})
        accuracy.append({
            'variant': variant,
            'AP50_95_percent': value(row, 'AP50_95_percent'),
            'frame_latency_p95_ms': value(frame_timing, 'frame_latency_p95_ms'),
            'module_joules_per_frame': value(row, 'module_joules_per_frame'),
        })
        tracking.append({
            'variant': variant,
            'HOTA_percent': value(row, 'HOTA_percent'),
            'IDF1_percent': value(row, 'IDF1_percent'),
            'MOTA_percent': value(row, 'MOTA_percent'),
        })
        power.append({
            'variant': variant,
            'module_average_W': value(row, 'module_average_W'),
            'power_reduction_percent': reduction(row, 'module_average_W'),
            'module_joules_per_frame': value(row, 'module_joules_per_frame'),
            'energy_reduction_percent': reduction(row, 'module_joules_per_frame'),
        })

    csvwrite(output / 'accuracy_cost.csv', accuracy)
    csvwrite(output / 'tracking.csv', tracking)
    csvwrite(output / 'power_comparison.csv', power)
