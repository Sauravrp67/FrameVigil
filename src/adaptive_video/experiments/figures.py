"""Plot accuracy against measured costs from an evaluation summary CSV."""

import csv
import math
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

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
    """Write PNG and SVG plots without assuming a dataset size or hardware type."""
    rows, costs, frame_count = load_table(source)
    output = Path(output).resolve()

    for protected in (ROOT / 'data',):
        if output == protected or protected in output.parents:
            raise ValueError('Dataset inputs are read-only')

    paths = [output / f'accuracy_cost.{extension}' for extension in ('png', 'svg')]

    if any(path.exists() for path in paths):
        raise FileExistsError('Choose a new plot directory to preserve existing figures')

    output.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(7.5 * len(costs), 6), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(costs), squeeze=False)[0]
    colors = matplotlib.colormaps['tab20']
    handles = []

    for axis_index, (axis, cost) in enumerate(zip(axes, costs)):
        for row_index, row in enumerate(rows):
            point = axis.scatter(
                row[cost], row['AP50_95_percent'],
                s=55, color=colors(row_index % 20),
            )

            if axis_index == 0:
                handles.append(point)

        axis.set_xlabel(COST_AXES[cost])
        axis.set_ylabel('Pooled AP50:95 (%)')
        axis.grid(alpha=0.2)

    title = f'{len(rows)} variants'

    if frame_count is not None:
        title += f' | {frame_count:,} unique frames'

    figure.suptitle(title)
    figure.legend(
        handles, [row['variant'] for row in rows],
        loc='outside lower center', ncol=min(4, len(rows)), fontsize=9,
    )

    for path in paths:
        figure.savefig(path, dpi=180)

    return paths
