"""Binary screening metrics with explicit positive class and threshold."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true, probability, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError('Expected nonempty aligned 1D labels and probabilities')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Labels must be binary and probabilities finite in [0,1]')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    predicted = (p >= threshold).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        'n': len(y), 'class_counts': {str(c): int((y == c).sum()) for c in [0, 1]},
        'threshold': threshold, 'positive_class': 1,
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'macro_f1': float(f1_score(y, predicted, labels=[0, 1], average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y, predicted)),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'specificity': float(tn / (tn + fp)) if tn + fp else None,
        'confusion_matrix': matrix.tolist(), 'confusion_matrix_order': [0, 1],
    }
