"""Summarize explicit shadow/reliability outputs"""

from pathlib import Path

import pandas as pd

from ..dataset import csvwrite, read, save


def run(analysis, reliability, output, detector_cost=None):
    """Optionally model headroom using a supplied CSV with detector_ms samples.

    Component estimates are not measured end-to-end latency or energy savings.
    No automatic GO decision is made without a prospective policy experiment.
    """
    analysis = Path(analysis)
    reliability = Path(reliability)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = read(analysis / 'analysis.json')
    status = read(reliability / 'status.json')
    summary = {
        **metadata,
        'reliability_status': status['status'],
        'metrics': read(analysis / 'shadow_AP.json')['metrics'],
        'method_latency': None,
        'method_energy': None,
        'decision': 'INCONCLUSIVE: one-step diagnostic requires prospective policy validation',
    }

    if metadata['events']:
        labels = pd.read_csv(analysis / 'labels.csv')
        features = pd.read_csv(analysis / 'features.csv')
        actions = pd.read_csv(analysis / 'actions.csv')
        calls = int(actions.ran_detector.sum())
        summary.update({
            'detector_calls': calls,
            'safe': int(labels.safe.sum()),
            'raw_safe': int(labels.raw_safe.sum()),
            'kalman_safe': int(labels.kalman_safe.sum()),
            'unsupported_events': int((labels.unsupported_gt > 0).sum()),
            'catastrophic_objects': int(labels.catastrophic.sum()),
            'safe_share': float(labels.safe.mean()),
            'eligible_share_detector': float(actions.eligible.sum() / calls) if calls else None,
            'successful_flow_share_detector': len(labels) / calls if calls else None,
            'component_flow_mean_ms': float(features.flow_and_correction_ms.mean()),
        })
        per_video = labels.groupby('video').agg(events=('safe', 'size'), safe=('safe', 'sum'))
        csvwrite(destination / 'per_video.csv', per_video.reset_index().to_dict('records'))

        if detector_cost is not None and status['status'] == 'COMPLETE':
            costs = pd.read_csv(detector_cost).detector_ms

            if costs.empty or costs.isna().any() or not costs.between(0, float('inf'), inclusive='neither').all():
                raise ValueError('Detector cost samples must be finite and positive')

            detector_ms = float(costs.mean())
            thresholds = pd.read_csv(reliability / 'thresholds.csv')
            thresholds['headroom_model_ms'] = (
                thresholds.safe_accepted * detector_ms
                - thresholds.probed * summary['component_flow_mean_ms']
            )
            thresholds['scope'] = 'Component model only; not measured policy savings'
            csvwrite(destination / 'thresholds_with_cost.csv', thresholds.to_dict('records'))
            summary['component_detector_mean_ms'] = detector_ms

    save(destination / 'summary.json', summary)
    (destination / 'REPORT.md').write_text(
        '# Optical-flow diagnostic\n\n'
        f"Analyzed {metadata['frames']} frames and {metadata['events']} successful flow events.\n\n"
        f"Flow failures: {metadata['failed_flow_events']}. Reliability: {status['status']}.\n\n"
        'AP describes hypothetical one-frame substitutions on baseline history. '
        'Corrections do not propagate into later frames. Labels use annotations only offline. '
        'The event safety label checks matches at IoU 0.5 and 0.75; it is not a guarantee '
        'of preserved pooled AP or future object discovery.\n\n'
        'No live policy latency or energy savings have been established by this analysis.\n'
    )
