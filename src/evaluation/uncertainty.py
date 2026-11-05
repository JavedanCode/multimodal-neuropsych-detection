"""Participant-level resampling for exploratory uncertainty estimates."""

import numpy as np


def bootstrap_weights(y, domains, draws=2000, seed=20261104):
    y = np.asarray(y)
    domains = np.asarray(domains)
    rng = np.random.default_rng(seed)
    strata = np.array([f"{d}_{t}" for d, t in zip(domains, y)])
    weights = np.zeros((draws, len(y)), int)
    for name in np.unique(strata):
        indices = np.flatnonzero(strata == name)
        samples = rng.choice(indices, size=(draws, len(indices)), replace=True)
        for i, row in enumerate(samples):
            weights[i] += np.bincount(row, minlength=len(y))
    return weights


def sampled_metrics(y, prediction, weights):
    y = np.asarray(y)[:, None]
    p = np.asarray(prediction)
    tp = weights @ ((y == 1) & (p == 1))
    fp = weights @ ((y == 0) & (p == 1))
    fn = weights @ ((y == 1) & (p == 0))
    return {
        "precision": np.mean(tp / np.maximum(tp + fp, 1), axis=1),
        "f1": np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1), axis=1),
    }
