"""Fold-fitted preprocessing and classification for representation comparisons."""

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from src.models.representation_preprocessing import RepresentationPreprocessor
from src.training.classical import estimator


def prepare_inner(x, y, splits):
    prepared = []
    for a, b in splits:
        prep = RepresentationPreprocessor().fit(x.iloc[a])
        prepared.append((a, b, prep.transform(x.iloc[a]), prep.transform(x.iloc[b])))
    return prepared


def fit_outer(x, y, valid, family, selected, seed):
    model = Pipeline(
        [
            ("preprocess", RepresentationPreprocessor()),
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
