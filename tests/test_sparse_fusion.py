"""Guard training-only sparse selection and observable quality-weight behavior."""

import numpy as np
import pandas as pd
import pytest

from src.models.sparse_fusion import (
    fit_branch,
    branch_scores,
    fusion_weights,
    fuse_scores,
)


def test_feature_selection_ignores_heldout_extremes():
    random = np.random.default_rng(42)
    frame = pd.DataFrame(
        random.normal(size=(60, 12)), columns=[f"audio__{i}" for i in range(12)]
    )
    labels = pd.Series((frame["audio__0"] > 0).astype(int))
    model = fit_branch(
        frame.iloc[:50], labels.iloc[:50], {"selection": "top5", "C": 0.01}, 42
    )
    assert 0 < len(model["selected_features"]) <= 5
    names, means = (
        list(model["selected_features"]),
        model["preprocess"].scale_.mean_.copy(),
    )
    branch_scores(model, frame.iloc[50:] * 1e8)
    assert names == model["selected_features"]
    np.testing.assert_array_equal(means, model["preprocess"].scale_.mean_)
    np.testing.assert_allclose(means, frame.iloc[:50].mean(), atol=1e-12)


def test_quality_discount_and_missing_inputs():
    quality = np.zeros((4, 9))
    quality[:, [1, 6, 7]] = 1
    quality[0, 2] = 0.01
    available = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=bool)
    weights = fusion_weights(quality, available, 0.5, True)
    np.testing.assert_allclose(weights, [[1 / 3, 2 / 3], [1, 0], [0, 1], [0, 0]])
    scores = fuse_scores(
        [1, 1, 100, 100], [-1, -100, -1, 100], quality, available, 0.5, 0.2, True
    )
    changed = fuse_scores(
        [1, 1, -100, -100], [-1, 100, -1, -100], quality, available, 0.5, 0.2, True
    )
    np.testing.assert_array_equal(scores, changed)
    assert scores[-1] == pytest.approx(-0.3)


def test_missing_oof_inputs_use_each_fitting_prior():
    quality = np.zeros((2, 9))
    score = fuse_scores([0, 0], [0, 0], quality, np.zeros((2, 2)), 0.5, [0.2, 0.3])
    np.testing.assert_allclose(score, [-0.3, -0.2])


def test_cli_disallows_two_followups_at_once():
    from src.__main__ import main

    with pytest.raises(SystemExit):
        main(["run", "audio-visual", "--preprocess-audio", "--sparse-fusion"])
