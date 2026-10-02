"""Evaluate flow-quality features using leave-one-video-out validation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..dataset import csvwrite, read, save


PRE_FEATURES = ['n_tracks', 'max_age', 'max_cov', 'min_diag', 'max_speed']
FLOW_FEATURES = ['invalid_fraction', 'max_dispersion', 'max_disagreement',
                 'min_inlier_fraction', 'max_motion_spread']


def run(analysis, output):
    """Fit only on other videos; never select a winning seed or refit held-out data."""
    analysis = Path(analysis)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)

    if read(analysis / 'analysis.json')['events'] == 0:
        save(destination / 'status.json', {'status': 'INSUFFICIENT_DATA', 'reason': 'No eligible successful flow events'})
        return

    features = pd.read_csv(analysis / 'features.csv')
    labels = pd.read_csv(analysis / 'labels.csv')
    keys = ['video', 'index', 'image_id']
    data = features.merge(labels, on=keys, validate='one_to_one')

    if len(data) != len(features) or len(data) != len(labels):
        raise ValueError('Feature and label identities do not match')

    if data.video.nunique() < 2:
        save(destination / 'status.json', {'status': 'INSUFFICIENT_DATA', 'reason': 'At least two videos with events are required'})
        return

    targets = data.safe.astype(int).to_numpy()
    families = {'pre': PRE_FEATURES, 'flow': FLOW_FEATURES,
                'combined': PRE_FEATURES + FLOW_FEATURES}
    summary = []
    probabilities = []
    folds = []
    thresholds = []
    calibration = []

    for family, names in families.items():
        values = data[names].to_numpy(float)

        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite features: {family}')

        predicted = np.zeros(len(data))

        for video in sorted(data.video.unique()):
            testing = data.video.eq(video).to_numpy()
            training = ~testing
            coefficients = []
            intercept = None

            if len(np.unique(targets[training])) < 2:
                predicted[testing] = targets[training].mean()
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1, max_iter=1000, random_state=0,
                ))
                model.fit(values[training], targets[training])
                predicted[testing] = model.predict_proba(values[testing])[:, 1]
                coefficients = model[-1].coef_[0].tolist()
                intercept = float(model[-1].intercept_[0])

            folds.append({
                'family': family, 'held_out': video,
                'train_videos': sorted(data.video[training].unique()),
                'features': names, 'coefficients_standardized': coefficients,
                'intercept': intercept,
            })

        both_classes = len(np.unique(targets)) == 2
        summary.append({
            'family': family,
            'AUROC': float(roc_auc_score(targets, predicted)) if both_classes else None,
            'AUPRC': float(average_precision_score(targets, predicted)) if targets.any() else None,
            'Brier': float(brier_score_loss(targets, predicted)),
            'prevalence': float(targets.mean()),
        })

        for identity, target, probability in zip(data[keys].to_dict('records'), targets, predicted):
            probabilities.append({**identity, 'family': family,
                                  'safe': bool(target), 'probability': float(probability)})

        for bin_index in range(10):
            lower = bin_index / 10
            upper = (bin_index + 1) / 10
            selected = (predicted >= lower) & (predicted < upper)

            if bin_index == 9:
                selected = (predicted >= lower) & (predicted <= upper)

            calibration.append({
                'family': family, 'lower': lower, 'upper': upper,
                'count': int(selected.sum()),
                'predicted': float(predicted[selected].mean()) if selected.any() else None,
                'observed': float(targets[selected].mean()) if selected.any() else None,
            })

        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            accepted = predicted >= threshold
            count = int(accepted.sum())
            good = int(targets[accepted].sum())
            thresholds.append({
                'family': family, 'threshold': threshold,
                'accepted': count, 'safe_accepted': good, 'false_safe': count - good,
                'coverage': count / len(data),
                'false_safe_rate': (count - good) / count if count else None,
                'videos_with_accepts': int(data.video[accepted].nunique()),
                'probed': count if family == 'pre' else len(data),
            })

    csvwrite(destination / 'reliability_summary.csv', summary)
    csvwrite(destination / 'probabilities.csv', probabilities)
    csvwrite(destination / 'thresholds.csv', thresholds)
    csvwrite(destination / 'calibration.csv', calibration)
    save(destination / 'folds.json', folds)
    save(destination / 'status.json', {'status': 'COMPLETE', 'events': len(data),
                                     'scope': 'Grouped offline diagnostic; not a deployed policy'})
