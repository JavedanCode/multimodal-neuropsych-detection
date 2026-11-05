"""CV fitness and constrained L1 selection; APIs accept training data only."""

from __future__ import annotations
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.dataset.preprocessing import ParticipantPreprocessor
from src.training.validation import validate_partition
from src.evaluation.metrics import binary_metrics


class TrainingFitness:
    def __init__(self, X, y, seed=42, folds=5, alpha=0.95, beta=0.05, C=0.1):
        validate_partition(X, y)
        if folds < 2 or y.nunique() != 2 or y.value_counts().min() < folds:
            raise ValueError("Insufficient class counts for CV")
        if (
            not np.isfinite([alpha, beta, C]).all()
            or min(alpha, beta) < 0
            or not np.isclose(alpha + beta, 1)
            or C <= 0
        ):
            raise ValueError("Require alpha+beta=1, nonnegative weights and positive C")
        self.X, self.y, self.seed, self.alpha, self.beta, self.C = (
            X.copy(),
            y.copy(),
            seed,
            alpha,
            beta,
            C,
        )
        self.splits = list(
            StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)
        )
        self.cache = {}
        self.records = []

    def __call__(self, mask):
        mask = np.asarray(mask)
        if mask.shape != (len(self.X.columns),) or not np.isin(mask, [0, 1]).all():
            raise ValueError("Mask must be binary and match descriptor count")
        mask = mask.astype(bool)
        key = np.packbits(mask).tobytes()
        if key in self.cache:
            return self.cache[key]["fitness"]
        columns = self.X.columns[mask].tolist()
        started = time.perf_counter()
        scores = []
        invalid = None
        if not all(
            any(c.startswith(m + "__") for c in columns) for m in ["audio", "facial"]
        ):
            invalid = "Both modalities must be selected"
        else:
            for a, b in self.splits:
                prep = ParticipantPreprocessor()
                try:
                    fit = prep.fit_transform(self.X.iloc[a][columns])
                except ValueError as error:
                    if "No nonconstant training features" not in str(error):
                        raise
                    invalid = str(error)
                    break
                if not all(
                    any(c.startswith(m + "__") for c in fit)
                    for m in ["audio", "facial"]
                ):
                    invalid = "A modality has no surviving fold features"
                    break
                val = prep.transform(self.X.iloc[b][columns])
                model = LogisticRegression(
                    C=self.C,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=self.seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(fit, self.y.iloc[a])
                scores.append(
                    binary_metrics(self.y.iloc[b], model.predict_proba(val)[:, 1])["f1"]
                )
        score = float(np.mean(scores)) if invalid is None else None
        objective = (
            self.alpha * (1 - score) + self.beta * len(columns) / len(mask)
            if score is not None
            else 2.0
        )
        record = {
            "mask": mask.tolist(),
            "selected_count": len(columns),
            "mean_cv_f1": score,
            "fold_f1": scores,
            "fitness": float(objective),
            "invalid_reason": invalid,
            "seconds": time.perf_counter() - started,
        }
        self.cache[key] = record
        self.records.append(record)
        return float(objective)


def select_l1(X, y, seed=42, C=0.1):
    validate_partition(X, y)
    if not np.isfinite(C) or C <= 0:
        raise ValueError("Positive finite L1 C required")
    started = time.perf_counter()
    prep = ParticipantPreprocessor()
    frame = prep.fit_transform(X)
    # sklearn >=1.8 expresses L1 via l1_ratio; retain support for >=1.4.
    default_penalty = LogisticRegression().get_params().get("penalty")
    regularization = {"penalty": "l1"} if default_penalty == "l2" else {"l1_ratio": 1.0}
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
        **regularization,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(frame, y)
    coefficients = np.abs(model.coef_[0])
    chosen = coefficients > 1e-8
    forced = []
    for modality in ["audio", "facial"]:
        indices = np.array(
            [i for i, c in enumerate(frame) if c.startswith(modality + "__")]
        )
        if not len(indices):
            raise ValueError("No usable modality descriptors for L1")
        if not chosen[indices].any():
            index = indices[np.argmax(coefficients[indices])]
            chosen[index] = True
            forced.append(frame.columns[index])
    names = frame.columns[chosen].tolist()
    return {
        "mask": [c in names for c in X],
        "selected_features": names,
        "forced_modality_features": forced,
        "coefficient_abs": dict(zip(frame.columns, map(float, coefficients))),
        "seconds": time.perf_counter() - started,
        "C": C,
    }
