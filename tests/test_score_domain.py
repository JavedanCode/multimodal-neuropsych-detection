"""Guard the PHQ-score ridge target and domain-conditional standardization."""

import numpy as np
import pandas as pd
import pytest

from src.models.score_domain import (
    SHRINKAGE,
    DomainScaler,
    balanced_weights,
    fit_model,
    model_scores,
)
from src.models.sparse_fusion import branch_scores, fit_branch


def cohort(n_woz=40, n_ai=12, seed=0):
    random = np.random.default_rng(seed)
    ids = list(range(300, 300 + n_woz)) + list(range(600, 600 + n_ai))
    frame = pd.DataFrame(
        random.normal(size=(len(ids), 6)),
        index=pd.Index(ids, name="Participant_ID"),
        columns=[f"audio__{i}" for i in range(6)],
    )
    frame.loc[frame.index >= 600] += 3.0  # acquisition shift in the AI domain
    phq = pd.Series(np.clip(6 + 4 * frame["audio__0"] + random.normal(size=len(ids)), 0, 24), index=frame.index)
    labels = (phq >= 10).astype(int)
    return frame, labels, phq


def test_domain_scaler_shrinks_small_domains_toward_pooled():
    values = np.vstack([np.zeros((30, 1)), np.full((5, 1), 4.0)])
    domains = np.array(["WoZ"] * 30 + ["AI"] * 5)
    scaler = DomainScaler().fit(values, domains)
    mean, scale, n = scaler.stats_["AI"]
    assert n == 5
    np.testing.assert_allclose(mean, 5 * 4.0 / (5 + SHRINKAGE))
    np.testing.assert_allclose(scale, np.sqrt(SHRINKAGE / (5 + SHRINKAGE)))


def test_domain_scaling_ignores_heldout_rows():
    frame, labels, phq = cohort()
    spec = {"target": "binary", "domain": True, "alpha": None}
    model = fit_model(frame.iloc[:45], labels.iloc[:45], phq.iloc[:45], spec, 42)
    before = {k: tuple(np.copy(v) for v in s[:2]) for k, s in model["domain_scaler"].stats_.items()}
    model_scores(model, frame.iloc[45:] * 1e6)
    for name, (mean, scale) in before.items():
        np.testing.assert_array_equal(mean, model["domain_scaler"].stats_[name][0])
        np.testing.assert_array_equal(scale, model["domain_scaler"].stats_[name][1])


def test_balanced_weights_equalize_classes():
    weights = balanced_weights([0, 0, 0, 1])
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
    with pytest.raises(ValueError):
        balanced_weights([0, 0])


def test_score_target_margin_is_predicted_score_minus_threshold():
    frame, labels, phq = cohort()
    model = fit_model(frame, labels, phq, {"target": "score", "domain": False, "alpha": 1.0}, 42)
    scores = model_scores(model, frame)
    assert np.corrcoef(scores, phq)[0, 1] > 0.8
    np.testing.assert_allclose(scores, model["estimator"].predict(model["preprocess"].transform(frame)))


def test_binary_without_domain_matches_historical_branch():
    frame, labels, phq = cohort()
    model = fit_model(frame, labels, phq, {"target": "binary", "domain": False, "alpha": None}, 42)
    historical = fit_branch(frame, labels, {"selection": "all", "C": 0.1}, 42)
    np.testing.assert_array_equal(model_scores(model, frame), branch_scores(historical, frame))


def test_cli_score_study_runs_alone():
    from src.__main__ import main

    with pytest.raises(SystemExit):
        main(["run", "audio-visual", "--score-study", "--corrected"])
