"""Cross-fitted base predictions for leakage-controlled multimodal stacking."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.models.text_preprocessing import TextPreprocessor
from src.training.splits import splits
from src.dataset.contracts import domain

SCENARIOS = ("clean", "missing_audio", "missing_face", "missing_both")


def take(data, indices):
    return {
        k: v.iloc[indices] if hasattr(v, "iloc") else v[indices]
        for k, v in data.items()
    }


def base_fit(data, y, seed):
    models = {}
    for name in ["text", "audio", "face"]:
        prep = TextPreprocessor() if name == "text" else StandardScaler()
        model = (
            SVC(kernel="linear", C=0.1, class_weight="balanced", random_state=seed)
            if name == "text"
            else LogisticRegression(
                C=0.1,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
                max_iter=5000,
            )
        )
        models[name] = Pipeline([("prep", prep), ("model", model)]).fit(data[name], y)
    return models


def base_scores(models, data):
    score = np.column_stack(
        [models[n].decision_function(data[n]) for n in ["text", "audio", "face"]]
    )
    score[:, 1:] *= data["available"][:, 1:]
    return score


def crossfit(data, y, seed):
    strata = [f"{d}_{int(t)}" for d, t in zip(domain(y.index), y)]
    minimum = min(strata.count(s) for s in set(strata))
    n = min(3, minimum)
    if n < 2:
        raise ValueError("Insufficient subgroup size for honest cross-fitting")
    scores = np.full((len(y), 3), np.nan)
    records = []
    for a, b in splits(y, n, seed):
        models = base_fit(take(data, a), y.iloc[a], seed)
        scores[b] = base_scores(models, take(data, b))
        records.append(
            {
                "fit_ids": list(map(int, y.index[a])),
                "heldout_ids": list(map(int, y.index[b])),
            }
        )
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete base cross-fitting")
    return scores, records


def design(scores, quality, available, kind, scenario="clean"):
    if scenario not in SCENARIOS or kind not in ["text", "ordinary", "quality"]:
        raise ValueError("Unknown design")
    s = np.array(scores, copy=True)
    q = np.array(quality, copy=True)
    a = np.array(available, copy=True)
    for modality, columns in [(1, slice(0, 6)), (2, slice(6, 9))]:
        if scenario == "missing_both" or scenario == (
            "missing_audio" if modality == 1 else "missing_face"
        ):
            s[:, modality] = 0
            q[:, columns] = 0
            a[:, modality] = 0
    if kind == "text":
        return s[:, :1]
    parts = [s, a[:, 1:]]
    if kind == "quality":
        parts += [s[:, :1] * q[:, 9:11], s[:, 1:2] * q[:, :6], s[:, 2:3] * q[:, 6:9]]
    return np.concatenate(parts, 1)
