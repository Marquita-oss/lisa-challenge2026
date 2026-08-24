from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)

ARTIFACT_COLS = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']


def compute_challenge_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    col_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """accuracy, F1, F2, precision, recall por clase. y_true/y_pred: (N, C) binario."""
    if col_names is None:
        col_names = ARTIFACT_COLS
    results = {}
    for i, col in enumerate(col_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        results[col] = {
            'accuracy':   float(accuracy_score(yt, yp)),
            'f1':         float(fbeta_score(yt, yp, beta=1, zero_division=0)),
            'f2':         float(fbeta_score(yt, yp, beta=2, zero_division=0)),
            'precision':  float(precision_score(yt, yp, zero_division=0)),
            'recall':     float(recall_score(yt, yp, zero_division=0)),
            'n_positive': int(yt.sum()),
            'n_total':    int(len(yt)),
        }
    return results


def compute_auc_per_class(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    col_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    if col_names is None:
        col_names = ARTIFACT_COLS
    aucs = {}
    for i, col in enumerate(col_names):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            aucs[col] = float('nan')
        else:
            aucs[col] = float(roc_auc_score(yt, y_probs[:, i]))
    return aucs


def mean_metric(per_class: Dict, metric: str) -> float:
    vals = [v[metric] for v in per_class.values()
            if not np.isnan(v[metric])]
    return float(np.mean(vals)) if vals else 0.0


def print_metrics_table(per_class: Dict, aucs: Optional[Dict] = None) -> None:
    header = f"{'Artifact':15s} {'AUC':>6s} {'Acc':>6s} {'F1':>6s} {'F2':>6s} {'Prec':>6s} {'Rec':>6s} {'N+':>5s}"
    print(header)
    print("-" * len(header))
    for col, m in per_class.items():
        auc_str = f"{aucs[col]:.3f}" if (aucs and not np.isnan(aucs.get(col, float('nan')))) else "  n/a"
        print(
            f"{col:15s} {auc_str:>6s} "
            f"{m['accuracy']:>6.3f} {m['f1']:>6.3f} {m['f2']:>6.3f} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['n_positive']:>5d}"
        )
