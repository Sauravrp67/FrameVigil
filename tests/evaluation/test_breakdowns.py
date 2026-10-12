"""Keep diagnostic curves and cost decompositions consistent with headline metrics."""

import numpy as np
import pytest
from pycocotools.coco import COCO

from adaptive_video.evaluation.coco import coco
from adaptive_video.evaluation.coco import frame_errors, localization_curve
from adaptive_video.experiments.statistics import frame_state, timing_summary, power_rows
from adaptive_video.experiments.prediction_plots import select_cases
from adaptive_video.project import save


def test_localization_and_frame_errors_use_original_coco_matching():
    truth = COCO()
    truth.dataset = {
        'info': {}, 'images': [{'id': 1}, {'id': 2}],
        'categories': [{'id': 2, 'name': 'car'}],
        'annotations': [
            {'id': index, 'image_id': index, 'category_id': 2,
             'bbox': [0, 0, 20, 20], 'area': 400, 'iscrowd': 0}
            for index in (1, 2)
        ],
    }
    truth.createIndex()
    predictions = [
        {'image_id': 1, 'category_id': 2, 'bbox': [0, 0, 20, 20], 'score': 0.9},
        {'image_id': 2, 'category_id': 2, 'bbox': [100, 100, 20, 20], 'score': 0.8},
    ]
    metrics, evaluator = coco(truth, predictions, [1, 2])
    curve = localization_curve(evaluator)
    errors = frame_errors(evaluator)

    assert len(curve) == 10
    assert np.mean([point['AP_percent'] for point in curve]) == pytest.approx(metrics['AP50_95'] * 100)
    assert curve[5]['AP_percent'] == pytest.approx(metrics['AP75'] * 100)
    assert errors[1] == {'TP75': 1, 'FP75': 0, 'FN75': 0}
    assert errors[2] == {'TP75': 0, 'FP75': 1, 'FN75': 1}


def test_nonoverlapping_timing_includes_emission_once():
    actions = [
        dict(ran_inference=True, ran_erd=False, erd_result=None, num_rescued=1),
        dict(ran_inference=False, ran_erd=False, erd_result=None, num_rescued=0),
    ]
    record = {
        'frames': 2, 'wall_s': 0.2, 'decode_ms': [5, 5], 'detector_ms': [100],
        'controller_ms': [20, 30], 'emission_ms': [10, 10], 'frame_ms': [125, 35],
        'trace': actions,
    }
    result = timing_summary([record])

    assert sum(result[key] for key in ('decode_ms', 'detector_ms', 'controller_output_ms', 'other_ms')) == 100
    assert result['controller_output_ms'] == 25
    assert result['other_ms'] == 20
    assert sum(value for key, value in result.items() if key.endswith('_percent')) == 100
    assert frame_state(actions[0]) == 'rescue_detection'


def test_power_curve_clips_boundaries_and_matches_energy(tmp_path):
    path = tmp_path / 'results/power/r0_constant_acceleration_clip.json'
    save(path, {'start': 0.05, 'end': 0.15, 'samples': [
        {'t': 0, 'power1_W': 10}, {'t': 0.1, 'power1_W': 20}, {'t': 0.2, 'power1_W': 30},
    ]})
    rows = power_rows(tmp_path, {'repeat': 0, 'variant': 'constant_acceleration', 'video': 'clip', 'energy_joules': 2})

    assert rows[0]['elapsed_s'] == 0
    assert rows[0]['module_W'] == 15
    assert rows[-1]['elapsed_s'] == pytest.approx(0.1)
    assert rows[-1]['module_W'] == 25
    assert rows[-1]['cumulative_J'] == pytest.approx(2)


def test_cases_include_harm_and_benefit_without_manual_selection():
    rows = []

    for index, errors in enumerate([0, 2, 4]):
        for variant, count in [('baseline', 2), ('raw_both', errors)]:
            rows.append({'variant': variant, 'image_id': str(index), 'frame_index': str(index),
                         'video': 'clip', 'FP75': str(count), 'FN75': '0'})

    cases = select_cases(rows, 'raw_both')

    assert [case['error_reduction'] for case in cases] == [-2, 0, 2]
    assert [case['frame_index'] for case in cases] == [2, 1, 0]


def test_heatmap_and_skip_plot_allow_no_energy(tmp_path):
    from adaptive_video.experiments.figures import per_video_plots, color_map

    rows = [
        {'variant': variant, 'video': 'clip', 'AP50_95_percent': '30',
         'latency_ms_per_frame': latency, 'module_joules_per_frame': '',
         'skipped_frames_percent': skipped}
        for variant, latency, skipped in [('baseline', '100', '0'), ('constant_acceleration', '70', '30')]
    ]
    paths = per_video_plots(rows, tmp_path, color_map(['baseline', 'constant_acceleration']))

    assert len(paths) == 4
    assert all(path.stat().st_size > 0 for path in paths)
    assert 'Energy reduction' not in (tmp_path / 'per_video_effects.svg').read_text()


def test_csv_comparisons_use_p95_and_preserve_missing_metrics(tmp_path):
    import csv

    from adaptive_video.experiments.statistics import write_report_tables

    summary = [
        dict(variant='baseline', AP50_95_percent=30, latency_ms_per_frame=100,
             module_average_W=20, module_joules_per_frame=2),
        dict(variant='constant_acceleration', AP50_95_percent=28, latency_ms_per_frame=80,
             module_average_W=15, module_joules_per_frame=1.2, HOTA_percent=40),
    ]
    timing = [dict(variant='baseline', frame_latency_p95_ms=120),
              dict(variant='constant_acceleration', frame_latency_p95_ms=130)]
    write_report_tables(summary, timing, tmp_path)

    with (tmp_path / 'accuracy_cost.csv').open() as stream:
        accuracy = list(csv.DictReader(stream))

    with (tmp_path / 'power_comparison.csv').open() as stream:
        power = list(csv.DictReader(stream))

    with (tmp_path / 'tracking.csv').open() as stream:
        tracking = list(csv.DictReader(stream))

    assert len(accuracy[0]) == 4
    assert float(accuracy[1]['frame_latency_p95_ms']) == 130
    assert float(power[0]['energy_reduction_percent']) == 0
    assert float(power[1]['power_reduction_percent']) == 25
    assert float(power[1]['energy_reduction_percent']) == pytest.approx(40)
    assert tracking[0]['HOTA_percent'] == ''


def test_aggregate_power_grids_weight_clips_equally(tmp_path):
    from adaptive_video.experiments.figures import power_profiles, power_plots

    traces = []

    for variant in ('baseline', 'constant_acceleration', 'raw_both'):
        for video, duration, watts in [('short', 2, 10), ('long', 20, 30)]:
            for elapsed in (0, duration):
                traces.append(dict(video=video, variant=variant, repeat=0, elapsed_s=elapsed,
                                   module_W=watts, cumulative_J=elapsed * watts))

    traces.append(dict(video='unpaired', variant='baseline', repeat=0,
                       elapsed_s=0, module_W=999, cumulative_J=0))
    traces.append(dict(video='short', variant='baseline', repeat=1,
                       elapsed_s=0, module_W=999, cumulative_J=0))
    progress, videos, profiles = power_profiles(traces)

    assert videos == ['long', 'short']
    assert progress[50] == 50
    assert np.median(profiles['baseline'], axis=0)[50] == 20
    paths = power_plots(traces, tmp_path)

    assert len(paths) == 2
    assert not (tmp_path / 'energy_aggregate.svg').exists()
    for filename in ('power_aggregate.svg',):
        svg = (tmp_path / filename).read_text()
        assert 'Constant acceleration' in svg
        assert 'Raw boxes + scores' in svg
        assert 'Baseline' in svg
        assert '2 paired clips' in svg
        assert 'median' in svg
