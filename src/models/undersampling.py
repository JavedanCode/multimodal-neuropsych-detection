"""Training-only, domain-proportional negative undersampling."""
import numpy as np


def sample_positions(y, ratio, seed):
    if ratio not in (1, 2):
        raise ValueError('Expected ratio 1 or 2')
    positive = np.flatnonzero(y.to_numpy() == 1)
    negative = np.flatnonzero(y.to_numpy() == 0)
    if not len(positive) or len(positive) + len(negative) != len(y):
        raise ValueError('Binary targets with positives required')
    count = min(len(negative), ratio * len(positive))
    groups = [negative[np.asarray(y.index)[negative] < 600],
              negative[np.asarray(y.index)[negative] >= 600]]
    ideal = np.array([len(g) for g in groups]) * count / max(1, len(negative))
    quota = np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal-quota), kind='stable')[:count-int(quota.sum())]:
        quota[i] += 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([positive] + [rng.choice(g, n, replace=False)
                   for g, n in zip(groups, quota)]))


def decision(bundle, x):
    z = bundle['preprocess'].transform(x)
    return np.mean([model.decision_function(z) for model in bundle['models']], axis=0)
