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
    'module_edp_j_ms': 'Energy–delay product (J·ms)',
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
    'baseline': 'Baseline', 'constant_acceleration': 'Constant acceleration',
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


def edp_plot(table, output, colors, metric="AP50_95_percent"):
    """Plot one operating point per variant with an available metric."""
    metrics = {
        'AP50_95_percent': ('Pooled AP50:95', 'accuracy_edp'),
        'AP50_percent': ('Pooled AP50', 'accuracy_edp_ap50'),
        'HOTA_percent': ('HOTA', 'hota_edp'),
        'IDF1_percent': ('IDF1', 'idf1_edp'),
        'MOTA_percent': ('MOTA', 'mota_edp'),
    }
    label, filename = metrics[metric]
    tracking = metric in ('HOTA_percent', 'IDF1_percent', 'MOTA_percent')

    if tracking:
        table = [row for row in table if row['variant'] != 'baseline'
                 and np.isfinite(number(row, metric))
                 and np.isfinite(number(row, 'module_edp_j_ms'))]

    if not any(np.isfinite(number(row, 'module_edp_j_ms')) for row in table):
        return []

    if not all(np.isfinite(number(row, metric)) for row in table):
        return []

    figure, axes = create_figure(width=10, height=6)
    axis = axes[0, 0]

    for row in table:
        variant = row['variant']
        axis.scatter(number(row, 'module_edp_j_ms'), number(row, metric),
                     color=colors[variant], marker='*' if variant == 'baseline' else 'o',
                     s=180 if variant == 'baseline' else 65, edgecolors='black',
                     linewidths=0.5, label=LABELS.get(variant, variant))

    axis.set_xlabel('Energy–delay product (J·ms)')
    axis.set_ylabel(f'{label} (%)')
    axis.grid(alpha=0.2)
    axis.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    return save_figure(figure, output, filename)


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


def skipping_plot(table, output):
    baseline = {row['video']: row for row in table if row['variant'] == 'baseline'}

    if not baseline or not any(row['variant'] != 'baseline' for row in table):
        return []

    fields = [('latency_ms_per_frame', 'Latency reduction (%)'),
              ('module_joules_per_frame', 'Energy reduction (%)')]
    fields = [(field, title) for field, title in fields
              if any(np.isfinite(number(row, field)) for row in table)]

    if not fields:
        return []

    figure, axes = create_figure(columns=len(fields), width=6.5 * len(fields), height=5)

    for axis, (field, title) in zip(axes[0], fields):
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

    return save_figure(figure, output, 'skipping_value')


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
    variants = ('baseline', 'constant_acceleration', 'visibility_ca10', 'cv8')
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
    """Normalize repeat 1 by elapsed time."""
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
    figure.supxlabel('Elapsed execution time (% of each run)', fontsize=11)
    figure.supylabel('Module-input power (W)')
    return save_figure(figure, output, 'power_aggregate')


def plot_diagnostics(summary_path, output):
    """Draw diagnostics for the available summary CSV files."""
    directory = Path(summary_path).parent
    summary = read_table(summary_path)
    colors = color_map([row['variant'] for row in summary])
    paths = edp_plot(summary, output, colors)
    for metric in ("AP50_percent", "HOTA_percent", "IDF1_percent", "MOTA_percent"):
        paths.extend(edp_plot(summary, output, colors, metric=metric))

    if (directory / 'localization.csv').exists():
        table = read_table(directory / 'localization.csv')
        paths.extend(localization_plot(table, output, colors))

    if (directory / 'timing.csv').exists():
        table = read_table(directory / 'timing.csv')
        paths.extend(timing_plots(table, output))

    if not (directory / 'per_video.csv').exists():
        return paths

    videos = read_table(directory / 'per_video.csv')
    paths.extend(skipping_plot(videos, output))

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
