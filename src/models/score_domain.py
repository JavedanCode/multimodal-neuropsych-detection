"""Fold-fitted PHQ-score ridge and domain-conditional standardization.

Protocol: docs/methods/score_functionals_domain.md. Every statistic is fitted on
the supplied fitting participants only; domain comes from participant IDs.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from src.dataset.contracts import domain
from src.models.fusion import BranchPreprocessor
from src.models.sparse_fusion import branch_scores, fit_branch

SHRINKAGE = 10.0
THRESHOLD = 9.5
ALPHAS = [1.0, 10.0, 100.0, 1000.0]


class DomainScaler:
    """Re-standardize within interview domain, shrunk toward pooled statistics.

    Input is already pooled-standardized (mean 0, variance 1 on fitting rows),
    so the pooled target is mean 0 / variance 1.
    """

    def fit(self, values, domains):
        values, domains = np.asarray(values, float), np.asarray(domains)
        self.stats_ = {}
        for name in ("WoZ", "AI"):
            rows = values[domains == name]
            n = len(rows)
            mean = rows.mean(0) if n else np.zeros(values.shape[1])
            var = rows.var(0) if n else np.ones(values.shape[1])
            shrunk_mean = n * mean / (n + SHRINKAGE)
            shrunk_var = (n * var + SHRINKAGE) / (n + SHRINKAGE)
            self.stats_[name] = (shrunk_mean, np.sqrt(shrunk_var), n)
        return self

    def transform(self, values, domains):
        values, domains = np.asarray(values, float), np.asarray(domains)
        result = np.empty_like(values)
        for name, (mean, scale, _) in self.stats_.items():
            rows = domains == name
            result[rows] = (values[rows] - mean) / scale
        if not set(np.unique(domains)) <= set(self.stats_):
            raise ValueError("Unknown interview domain")
        return result


def balanced_weights(labels):
    labels = np.asarray(labels, int)
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("Both classes required for balanced weights")
    return len(labels) / (2.0 * counts[labels])


def fit_model(features, labels, phq, spec, seed):
    """spec: {'target': 'binary'|'score', 'domain': bool, 'alpha': float|None}."""
    if spec["target"] == "binary" and not spec["domain"]:
        branch = fit_branch(features, labels, {"selection": "all", "C": 0.1}, seed)
        return {"spec": spec, "branch": branch}
    preprocess = BranchPreprocessor().fit(features)
    values = preprocess.transform(features)
    scaler = None
    if spec["domain"]:
        scaler = DomainScaler().fit(values, domain(features.index))
        values = scaler.transform(values, domain(features.index))
    if spec["target"] == "binary":
        estimator = LogisticRegression(
            C=0.1, class_weight="balanced", solver="liblinear",
            max_iter=5000, random_state=seed,
        ).fit(values, labels)
    elif spec["target"] == "score":
        estimator = Ridge(alpha=spec["alpha"]).fit(
            values, np.asarray(phq, float) - THRESHOLD,
            sample_weight=balanced_weights(labels),
        )
    else:
        raise ValueError("Unknown target")
    return {
        "spec": spec,
        "preprocess": preprocess,
        "domain_scaler": scaler,
        "estimator": estimator,
        "prior": float(np.mean(labels)),
    }


def model_scores(model, features):
    """Margin thresholded at zero (binary margin, or predicted score - 9.5)."""
    if "branch" in model:
        return branch_scores(model["branch"], features)
    values = model["preprocess"].transform(features)
    if model["domain_scaler"] is not None:
        values = model["domain_scaler"].transform(values, domain(features.index))
    if model["spec"]["target"] == "binary":
        return model["estimator"].decision_function(values)
    return model["estimator"].predict(values)
