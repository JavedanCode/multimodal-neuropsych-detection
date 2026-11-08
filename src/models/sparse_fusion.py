"""Fold-fitted sparse logistic branches and matched quality-adjusted late fusion."""

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from src.models.fusion import BranchPreprocessor


def specifications():
    return [
        {"selection": selection, "C": c}
        for selection in ("all", "top5", "top10")
        for c in (0.001, 0.01, 0.1)
    ]


def fit_branch(features, labels, spec, seed):
    """Select and fit using only the supplied fitting participants."""
    preprocess = BranchPreprocessor().fit(features)
    values = preprocess.transform(features)
    indices = np.arange(values.shape[1])
    selector_coefficients = None
    if spec["selection"] != "all":
        legacy = LogisticRegression().get_params().get("penalty") == "l2"
        penalty = {"penalty": "l1"} if legacy else {"l1_ratio": 1.0}
        selector = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
            **penalty,
        ).fit(values, labels)
        selector_coefficients = np.abs(selector.coef_[0])
        ranked = np.argsort(-selector_coefficients, kind="stable")
        nonzero = ranked[selector_coefficients[ranked] > 1e-8]
        indices = nonzero[: int(spec["selection"][3:])]
    classifier = None
    if len(indices):
        classifier = LogisticRegression(
            C=spec["C"],
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        ).fit(values[:, indices], labels)
    names = np.asarray(features.columns)[preprocess.keep_][indices].tolist()
    return {
        "preprocess": preprocess,
        "indices": indices,
        "classifier": classifier,
        "prior": float(np.mean(labels)),
        "selected_features": names,
        "selector_coefficients": selector_coefficients,
        "spec": spec,
    }


def branch_scores(model, features):
    values = model["preprocess"].transform(features)[:, model["indices"]]
    if model["classifier"] is None:
        return np.full(len(features), logit(model["prior"]))
    return model["classifier"].decision_function(values)


def fusion_weights(quality, available, audio_weight, reliability=False):
    """Normalize fixed base weights, optionally discounting measured poor inputs."""
    quality = np.asarray(quality, dtype=float)
    available = np.asarray(available, dtype=bool)
    if quality.shape != (len(available), 9) or available.shape[1:] != (2,):
        raise ValueError("Expected nine quality fields and two availability masks")
    if not np.isfinite(quality).all() or not 0 < audio_weight < 1:
        raise ValueError(
            "Finite quality and a strictly interior fusion weight required"
        )
    weights = np.tile([audio_weight, 1 - audio_weight], (len(available), 1))
    if reliability:
        audio_poor = (
            (quality[:, 2] > 0.001) | (quality[:, 5] > 0.01) | (quality[:, 1] < 0.95)
        )
        face_poor = (quality[:, 6] < 0.8) | (quality[:, 7] < 0.8)
        weights *= np.where(np.column_stack([audio_poor, face_poor]), 0.5, 1.0)
    weights *= available
    total = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)


def margin_scale(scores):
    """Root-mean-square margin around the branch's own zero decision boundary."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Finite out-of-fold margins required")
    scale = float(np.sqrt(np.mean(scores**2)))
    return scale if scale > 0 else 1.0


def fuse_scores(
    audio, face, quality, available, weight, prior, reliability=False, scales=None
):
    """Fuse two branch margins; return a score thresholded at zero.

    Without ``scales`` this is the historical rule (weighted mean of
    ``expit(margin)`` minus .5). With ``scales`` each margin is divided by its
    training-only scale first, so the applied weight equals the nominal weight.
    """
    weights = fusion_weights(quality, available, weight, reliability)
    if scales is not None:
        scales = np.asarray(scales, dtype=float)
        if scales.shape != (2,) or not (scales > 0).all():
            raise ValueError("Two positive branch scales required")
        margins = np.column_stack([audio, face]) / scales
        margins = np.where(np.asarray(available, dtype=bool), margins, 0.0)
        prediction = (margins * weights).sum(axis=1)
        no_input = weights.sum(axis=1) == 0
        prediction[no_input] = logit(np.broadcast_to(prior, prediction.shape))[
            no_input
        ]
        return prediction
    probabilities = expit(np.column_stack([audio, face]))
    probabilities = np.where(np.asarray(available, dtype=bool), probabilities, 0.0)
    prediction = (probabilities * weights).sum(axis=1)
    no_input = weights.sum(axis=1) == 0
    prediction[no_input] = np.broadcast_to(prior, prediction.shape)[no_input]
    return prediction - 0.5


def predict_policy(bundle, features, quality, available):
    if bundle["kind"] == "single":
        return branch_scores(bundle["model"], features[bundle["input"]])
    audio = branch_scores(bundle["audio"], features["audio"])
    face = branch_scores(bundle["face"], features["face"])
    return fuse_scores(
        audio,
        face,
        quality,
        available,
        bundle["weight"],
        bundle["prior"],
        bundle["reliability"],
        bundle.get("scales"),
    )
