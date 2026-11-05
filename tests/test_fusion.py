import numpy as np
import pandas as pd
import torch
from src.models.fusion import BranchPreprocessor, BranchGate


def test_preprocessing_uses_only_training_statistics():
    x = pd.DataFrame(
        {
            "audio__a": [0.0, 2.0, np.nan],
            "audio__dup": [0.0, 2.0, np.nan],
            "audio__constant": [1.0, 1.0, 1.0],
            "audio__empty": [np.nan] * 3,
        }
    )
    prep = BranchPreprocessor().fit(x)
    assert len(prep.keep_) == 1
    np.testing.assert_allclose(prep.scale_.mean_, [1.0])
    before = prep.scale_.mean_.copy()
    out = prep.transform(x.fillna(10000.0))
    np.testing.assert_array_equal(before, prep.scale_.mean_)
    assert out.shape == (3, 1)


def test_neural_ablation_matches_initialization_and_masks_missing_faces():
    models = []
    for mode in ["equal", "adaptive", "quality"]:
        torch.manual_seed(42)
        models.append(BranchGate(3, 4, mode).eval())
    for model in models[1:]:
        for part in ["audio_encoder", "facial_encoder", "head"]:
            for a, b in zip(
                getattr(models[0], part).parameters(), getattr(model, part).parameters()
            ):
                torch.testing.assert_close(a, b)
    a = torch.ones(2, 3)
    f = torch.ones(2, 4)
    q = torch.zeros(2, 9)
    available = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    for model in models:
        first, weights = model(a, f, q, available)
        second, _ = model(a, f * 1e9, q, available)
        torch.testing.assert_close(first, second)
        assert (weights[:, 1] == 0).all()
