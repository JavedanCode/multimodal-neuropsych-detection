"""Shared classical estimators, bounded candidate grids and inner-fold selection."""

from __future__ import annotations
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.dataset.preprocessing import ParticipantPreprocessor
from src.evaluation.metrics import binary_metrics

SEEDS = [42, 137, 2026]


THRESHOLDS = [0.0, -0.5, 0.5, -1.0, 1.0]


def candidates(family):
    return [
        {
            "C": c,
            "class_weight": weight,
            **({"gamma": gamma} if family == "rbf" else {}),
        }
        for c in [0.01, 0.1, 1.0]
        for weight in [None, "balanced"]
        for gamma in (["scale", 0.01] if family == "rbf" else [None])
    ]


def estimator(family, params, seed):
    if family == "logistic":
        return LogisticRegression(
            solver="liblinear", max_iter=5000, random_state=seed, **params
        )
    if family not in ["linear", "rbf"]:
        raise ValueError("Unknown model family")
    # Default is uncalibrated margins. Explicit probability=False is deprecated
    # by sklearn 1.9; omitting it also preserves older versions' false default.
    return SVC(kernel=family, cache_size=64, random_state=seed, **params)


def inner_data(x, y, splits):
    """Fit preprocessing only on each inner fitting fold; reuse across candidates."""
    result = []
    for a, b in splits:
        prep = ParticipantPreprocessor().fit(x.iloc[a])
        result.append((a, b, prep.transform(x.iloc[a]), prep.transform(x.iloc[b])))
    return result


def select_inner(x, y, family, prepared, seed, thresholds=THRESHOLDS, mean_fold=False):
    """Only inner data/labels are accepted; outer outcomes cannot select settings."""
    records = []
    for params in candidates(family):
        oof = np.full(len(y), np.nan)
        for a, b, fit, valid in prepared:
            model = estimator(family, params, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(fit, y.iloc[a])
            oof[b] = model.decision_function(valid)
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete/nonfinite inner OOF scores")
        for threshold in thresholds:
            predicted = oof >= threshold
            fold_scores = [
                float(f1_score(y.iloc[b], predicted[b], zero_division=0))
                for _, b, _, _ in prepared
            ]
            pooled = float(f1_score(y, predicted, zero_division=0))
            records.append(
                {
                    "parameters": params,
                    "threshold": threshold,
                    "pooled_f1": pooled,
                    "fold_f1": fold_scores,
                    "selection_f1": (
                        float(np.mean(fold_scores)) if mean_fold else pooled
                    ),
                }
            )
    # Python max returns the first tied item, preserving the predeclared order.
    default = max(
        [r for r in records if r["threshold"] == 0], key=lambda r: r["selection_f1"]
    )
    tuned = max(records, key=lambda r: r["selection_f1"])
    return default, tuned, records


def fit_outer(x, y, valid, family, selected, seed):
    model = Pipeline(
        [
            ("preprocess", ParticipantPreprocessor()),
            ("classifier", estimator(family, selected["parameters"], seed)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    scores = model.decision_function(valid)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite outer scores")
    return model, scores


def metric_row(y, predicted):
    metrics = binary_metrics(y, np.asarray(predicted, dtype=float))
    result = {
        k: metrics[k]
        for k in ["precision", "recall", "f1", "macro_f1", "accuracy", "specificity"]
    }
    result["balanced_accuracy"] = (result["recall"] + result["specificity"]) / 2
    result.update(
        zip(
            ["tn", "fp", "fn", "tp"],
            np.array(metrics["confusion_matrix"]).ravel().tolist(),
        )
    )
    return result
