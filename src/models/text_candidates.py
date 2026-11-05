"""Precision thresholds and conservative quality-adjusted text predictions."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models.text_preprocessing import TextPreprocessor
from src.training.classical import estimator

SCENARIOS = ("clean", "missing_audio", "missing_face", "missing_both")


def measures(y, scores):
    p = np.asarray(scores) >= 0
    return {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p)),
        "f1": float(f1_score(y, p)),
    }


def choose(records, reference):
    eligible = [
        r
        for r in records
        if r["metrics"]["recall"] >= 0.65
        and r["metrics"]["f1"] >= reference["metrics"]["f1"] - 0.02
    ]
    return (
        max(
            eligible,
            key=lambda r: (
                r["metrics"]["precision"],
                r["metrics"]["f1"],
                r["metrics"]["recall"],
            ),
        )
        if eligible
        else reference
    )


def fit_text(x, y, params, seed):
    return Pipeline(
        [("prep", TextPreprocessor()), ("model", estimator("linear", params, seed))]
    ).fit(x, y)


def fit_aux(data, y, seed):
    models = {}
    scales = []
    for name in ["audio", "face"]:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ).fit(data[name], y)
        models[name] = model
        scales.append(max(float(model.decision_function(data[name]).std()), 1e-6))
    return models, np.asarray(scales)


def eligible_quality(data, scenario):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    q = data["quality"]
    a = data["available"].astype(bool)
    audio = a[:, 1] & (q[:, 2] <= 0.001) & (q[:, 5] <= 0.01) & (q[:, 1] >= 0.95)
    face = a[:, 2] & (q[:, 6] >= 0.8) & (q[:, 7] >= 0.8)
    if scenario in ["missing_audio", "missing_both"]:
        audio[:] = False
    if scenario in ["missing_face", "missing_both"]:
        face[:] = False
    return np.column_stack([audio, face])


def residual(models, scales, data, scenario="clean"):
    mask = eligible_quality(data, scenario)
    score = (
        np.column_stack(
            [models[n].decision_function(data[n]) for n in ["audio", "face"]]
        )
        / scales
    )
    return (np.where(mask, score, 0)).sum(1) / np.maximum(mask.sum(1), 1)


def predict(bundle, data, scenario="clean"):
    score = bundle["text"].decision_function(data[bundle["feature"]])
    if bundle["alpha"]:
        score = score + bundle["alpha"] * residual(
            bundle["aux"], bundle["scales"], data, scenario
        )
    return score - bundle["threshold"]
