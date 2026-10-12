"""Validate an evaluation summary and generate diagnostic plots."""

import csv
import math

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pathlib import Path

from ..project import ROOT


COST_AXES = {
    'latency_ms_per_frame': 'Pipeline latency (ms/frame)',
    'module_joules_per_frame': 'Module-input energy (J/frame)',
}


def load_table(source):
    """Require comparable populations and complete values for each plotted cost."""
    with Path(source).open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        rows = list(reader)

    if not rows or not {'variant', 'AP50_95_percent'} <= columns:
        raise ValueError('CSV must contain variants and AP50_95_percent values')

    variants = [row['variant'] for row in rows]

    if any(not variant for variant in variants) or len(set(variants)) != len(variants):
        raise ValueError('Each row must identify a unique, nonempty variant')

    costs = []

    for key in COST_AXES:
        present = [bool(row.get(key, '').strip()) for row in rows]

        if any(present) and not all(present):
            raise ValueError(f'Incomplete cost column: {key}')

        if all(present):
            costs.append(key)

    if not costs:
        raise ValueError('No measured latency or energy column is available')

    for row in rows:
        for key in ['AP50_95_percent', *costs]:
            value = float(row[key])

            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {key} for {row["variant"]}')

            if key == 'AP50_95_percent' and value > 100:
                raise ValueError('AP50_95_percent must be between 0 and 100')

            row[key] = value

    frame_count = None

    if 'unique_frames' in columns:
        counts = {int(row['unique_frames']) for row in rows}

        if len(counts) != 1 or min(counts) <= 0:
            raise ValueError('Variants must have the same positive unique_frames count')

        frame_count = counts.pop()

    return rows, costs, frame_count


def run(source, output):
    """Validate the summary and write diagnostic plots to a new directory."""
    load_table(source)
    output = Path(output).resolve()

    if output == ROOT / 'data' or ROOT / 'data' in output.parents:
        raise ValueError('Dataset inputs are read-only')

    output.mkdir(parents=True, exist_ok=False)
    return plot_diagnostics(source, output)


LABELS = {
    'baseline': 'Baseline', 'constant_acceleration': 'Constant acceleration', 'raw_both': 'Raw boxes + scores',
    'cv8': 'Constant velocity', 'damped_ca10': 'Damped acceleration',
    'log_cv8': 'Log size', 'aspect_cv8': 'Aspect ratio',
    'projective_restart_cv8': 'Projective restart', 'camera_ca10': 'Camera motion',
    'appearance_ca10': 'Appearance', 'visibility_ca10': 'Visibility',
}


def read_table(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))

    return rows


def number(row, key):
    value = row.get(key)
    return float(value) if value not in ('', None) else np.nan


def create_figure(rows=1, columns=1, width=12, height=5):
    figure = Figure(figsize=(width, height), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    return figure, axes


def save_figure(figure, output, name):
    paths = []

    for extension in ('png', 'svg'):
        path = output / f'{name}.{extension}'

        if path.exists():
            raise FileExistsError(path)

        figure.savefig(path, dpi=180)
        paths.append(path)

    figure.clear()
    return paths


def color_map(variants):
    palette = matplotlib.colormaps['tab20']
    return {variant: palette(index) for index, variant in enumerate(variants)}


def localization_plot(table, output, colors):
    figure, axes = create_figure(width=10, height=6)
    axis = axes[0, 0]

    for variant in colors:
        rows = [row for row in table if row['variant'] == variant]
        axis.plot([number(row, 'iou') for row in rows],
                  [number(row, 'AP_percent') for row in rows],
                  marker='.', color=colors[variant], label=LABELS.get(variant, variant))

    axis.set_xlabel('IoU threshold')
    axis.set_ylabel('Pooled AP (%)')
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    axis.set_title('Accuracy at increasingly strict localization thresholds')
    return save_figure(figure, output, 'localization')


def per_video_plots(table, output, colors):
    baseline = {row['video']: row for row in table if row['variant'] == 'baseline'}

    if not baseline:
        return []

    videos = sorted(baseline)
    variants = [variant for variant in colors if variant != 'baseline']
    if not variants:
        return []

    lookup = {(row['variant'], row['video']): row for row in table}
    fields = [('AP50_95_percent', 'AP50:95 change (percentage points)'),
              ('latency_ms_per_frame', 'Latency reduction (%)'),
              ('module_joules_per_frame', 'Energy reduction (%)')]
    fields = [(field, title) for field, title in fields
              if any(np.isfinite(number(row, field)) for row in table)]
    figure, axes = create_figure(columns=len(fields), width=6 * len(fields), height=6)

    for axis, (field, title) in zip(axes[0], fields):
        values = np.full((len(variants), len(videos)), np.nan)

        for i, variant in enumerate(variants):
            for j, video in enumerate(videos):
                reference = number(baseline[video], field)
                measured = number(lookup[variant, video], field)
                values[i, j] = measured - reference if field.startswith('AP') else 100 * (1 - measured / reference)

        finite = np.abs(values[np.isfinite(values)])
        limit = max(float(finite.max()), 0.01) if finite.size else 1
        heatmap = axis.imshow(values, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(videos)), [f'V{i + 1}' for i in range(len(videos))], rotation=90)
        axis.set_yticks(range(len(variants)), [LABELS[v] for v in variants])
        axis.set_title(title)
        figure.colorbar(heatmap, ax=axis, shrink=0.8)

    paths = save_figure(figure, output, 'per_video_effects')
    (output / 'video_keys.txt').write_text('\n'.join(f'V{i + 1}: {v}' for i, v in enumerate(videos)) + '\n')
    figure, axes = create_figure(columns=len(fields) - 1, width=6.5 * (len(fields) - 1), height=5)

    for axis, (field, title) in zip(axes[0], fields[1:]):
        xs, ys, gains = [], [], []

        for row in table:
            if row['variant'] == 'baseline':
                continue

            reference = baseline[row['video']]
            xs.append(number(row, 'skipped_frames_percent'))
            ys.append(100 * (1 - number(row, field) / number(reference, field)))
            gains.append(number(row, 'AP50_95_percent') - number(reference, 'AP50_95_percent'))

        finite_gains = [abs(gain) for gain in gains if np.isfinite(gain)]
        limit = max(max(finite_gains, default=1), 0.01)
        points = axis.scatter(xs, ys, c=gains, cmap='RdBu', vmin=-limit, vmax=limit,
                              s=32, edgecolors='black', linewidths=0.3)
        axis.axhline(0, color='black', linewidth=0.7)
        axis.set_xlabel('Frames skipped (%)')
        axis.set_ylabel(title)
        axis.grid(alpha=0.2)
        figure.colorbar(points, ax=axis, label='AP50:95 change (pp)')

    figure.suptitle('Each point is one variant on one video; paired with baseline detector')
    return paths + save_figure(figure, output, 'skipping_value')


def timing_plots(table, output):
    figure, axes = create_figure(columns=2, width=16, height=6)
    labels = [LABELS[row['variant']] for row in table]
    positions = np.arange(len(table))
    bottom = np.zeros(len(table))
    components = [('decode_ms', 'Decode'), ('detector_ms', 'Detector'),
                  ('controller_output_ms', 'Controller + output'), ('other_ms', 'Loop/monitoring overhead')]

    for field, label in components:
        values = np.array([number(row, field) for row in table])
        axes[0, 0].barh(positions, values, left=bottom, label=label)
        bottom += values

    axes[0, 0].set_xlabel('Mean wall time per input frame (ms)')
    axes[0, 0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    bottom = np.zeros(len(table))

    for field, label in [('empty_skip', 'Empty-scene skip'), ('prediction_only', 'Prediction only'),
                         ('erd_detection', 'ERD then detection'), ('detection', 'Detection'),
                         ('rescue_detection', 'Detection with rescue')]:
        values = np.array([number(row, f'{field}_percent') for row in table])
        axes[0, 1].barh(positions, values, left=bottom, label=label)
        bottom += values

    axes[0, 1].set_xlabel('Frames (%)')
    axes[0, 1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

    for axis in axes[0]:
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()

    return save_figure(figure, output, 'timing_and_decisions')


def timeline_plots(frames, videos, output):
    paths = []
    variants = ('baseline', 'constant_acceleration', 'raw_both', 'cv8')
    states = ['empty_skip', 'prediction_only', 'erd_detection', 'detection', 'rescue_detection']

    for video in videos:
        figure, axes = create_figure(rows=5, width=13, height=12)

        for variant in variants:
            rows = [row for row in frames if row['video'] == video and row['variant'] == variant]

            if not rows:
                continue

            x = [int(row['frame_index']) for row in rows]
            axes[0, 0].scatter(x, [variants.index(variant)] * len(x),
                               c=[states.index(row['state']) for row in rows],
                               cmap='viridis', vmin=0, vmax=4, marker='s', s=12)
            for axis, field in [(axes[1, 0], 'frame_ms'), (axes[2, 0], 'num_tracks'), (axes[3, 0], 'num_rescued')]:
                axis.plot(x, [number(row, field) for row in rows], label=LABELS[variant], alpha=0.8)

            axes[4, 0].plot(x, [number(row, 'FP75') + number(row, 'FN75') for row in rows],
                           label=LABELS[variant], alpha=0.8)

        axes[0, 0].set_yticks(range(len(variants)), [LABELS[v] for v in variants])
        axes[0, 0].set_title(' / '.join(f'{i}: {state}' for i, state in enumerate(states)), fontsize=9)
        axes[0, 0].figure.colorbar(matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(0, 4), cmap='viridis'), ax=axes[0, 0], ticks=range(5))
        axes[1, 0].set_ylabel('Frame time (ms)')
        axes[1, 0].legend(ncol=4, fontsize=8)
        axes[2, 0].set_ylabel('Active tracks')
        axes[3, 0].set_ylabel('Rescued objects')
        axes[4, 0].set_ylabel('FP + FN at IoU 0.75')
        axes[4, 0].set_xlabel('Frame index (video order, not elapsed wall time)')
        figure.suptitle(video)
        paths.extend(save_figure(figure, output, f'timeline_{video}'))

    return paths


def power_profiles(table):
    """Normalize repeat 1 by elapsed time; every paired clip has equal weight."""
    traces = {}

    for row in table:
        if int(row['repeat']) == 0:
            traces.setdefault((row['variant'], row['video']), []).append(row)

    variants = sorted({variant for variant, video in traces})

    if 'baseline' not in variants or len(variants) < 2:
        return [], [], {}

    videos = sorted(set.intersection(*[
        {video for name, video in traces if name == variant} for variant in variants
    ]))
    progress = np.linspace(0, 100, 101)
    profiles = {}

    for variant in variants:
        profiles[variant] = []

        for video in videos:
            rows = sorted(traces[variant, video], key=lambda row: number(row, 'elapsed_s'))
            elapsed = np.array([number(row, 'elapsed_s') for row in rows])
            duration = elapsed[-1] - elapsed[0]

            if duration <= 0:
                raise ValueError(f'Nonpositive trace duration: {variant}, {video}')

            position = 100 * (elapsed - elapsed[0]) / duration

            values = [number(row, 'module_W') for row in rows]
            profiles[variant].append(np.interp(progress, position, values))

    return progress, videos, profiles


def power_plots(table, output):
    """Compare all variants with baseline using across-clip medians and quartiles."""
    progress, videos, profiles = power_profiles(table)

    if not videos:
        return []

    variants = [variant for variant in profiles if variant != 'baseline']
    columns = min(4, len(variants))
    rows = math.ceil((len(variants) + 1) / columns)
    figure = Figure(figsize=(4 * columns, 2.7 * rows), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False).ravel()

    for axis, variant in zip(axes, variants):
        for name, color, legend in (
            ('baseline', '#2166ac', 'Baseline'),
            (variant, '#d6604d', 'Variant named above panel'),
        ):
            lower, median, upper = np.percentile(profiles[name], [25, 50, 75], axis=0)
            axis.plot(progress, median, color=color, label=legend, linewidth=1.3)
            axis.fill_between(progress, lower, upper, color=color, alpha=0.18)

        axis.set_title(LABELS.get(variant, variant), fontsize=11)
        axis.set_xlim(0, 100)
        axis.grid(alpha=0.2)
        axis.tick_params(labelbottom=True, labelleft=True, labelsize=8)

    for axis in axes[len(variants):]:
        axis.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    axes[len(variants)].legend(handles, labels, loc='center', frameon=False, fontsize=10)
    figure.supxlabel('Elapsed execution time (% of each run)\n'
                       'Equal clip weights; normalized time does not align video frames.', fontsize=11)
    figure.supylabel('Module-input power (W)')
    figure.suptitle(f'{len(videos)} paired clips | repeat 1 | median and 25–75% across clips', fontsize=14)
    return save_figure(figure, output, 'power_aggregate')


def plot_diagnostics(summary_path, output):
    """Draw diagnostics for the available summary CSV files."""
    directory = Path(summary_path).parent
    summary = read_table(summary_path)
    colors = color_map([row['variant'] for row in summary])
    paths = []

    if (directory / 'localization.csv').exists():
        table = read_table(directory / 'localization.csv')
        paths.extend(localization_plot(table, output, colors))

    if (directory / 'timing.csv').exists():
        table = read_table(directory / 'timing.csv')
        paths.extend(timing_plots(table, output))

    if not (directory / 'per_video.csv').exists():
        return paths

    videos = read_table(directory / 'per_video.csv')
    paths.extend(per_video_plots(videos, output, colors))

    # Use low, middle and high skip rates for the example traces.
    reference = [row for row in videos if row['variant'] == 'constant_acceleration']
    reference.sort(key=lambda row: (number(row, 'skipped_frames_percent'), row['video']))
    selected = []

    if reference:
        for index in (0, len(reference) // 2, len(reference) - 1):
            video = reference[index]['video']

            if video not in selected:
                selected.append(video)

    if (directory / 'frames.csv').exists():
        frames = read_table(directory / 'frames.csv')
        paths.extend(timeline_plots(frames, selected, output))

    if (directory / 'power.csv').exists():
        power = read_table(directory / 'power.csv')
        paths.extend(power_plots(power, output))

    return paths
