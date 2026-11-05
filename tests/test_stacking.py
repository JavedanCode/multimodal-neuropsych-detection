import numpy as np
import pandas as pd
from src.models.stacking import design, crossfit


def test_missing_modality_cannot_affect_quality_design():
    s = np.array([[1.0, 2.0, 3.0]])
    q = np.ones((1, 11))
    a = np.ones((1, 3))
    before = design(s, q, a, "quality", "missing_audio")
    s[:, 1] = 1e6
    q[:, :6] = 1e6
    np.testing.assert_array_equal(before, design(s, q, a, "quality", "missing_audio"))
    assert design(s, q, a, "text", "missing_both")[0, 0] == 1


def test_base_crossfit_never_trains_on_its_prediction_participants():
    rng = np.random.default_rng(1)
    ids = list(range(300, 312)) + list(range(600, 606))
    y = pd.Series([0] * 6 + [1] * 6 + [0] * 3 + [1] * 3, index=ids)
    data = {
        "text": pd.DataFrame(
            rng.normal(size=(18, 8)),
            index=ids,
            columns=[f"text__{i}" for i in range(8)],
        ),
        "audio": rng.normal(size=(18, 3)),
        "face": rng.normal(size=(18, 3)),
        "quality": np.ones((18, 11)),
        "available": np.ones((18, 3)),
    }
    s, records = crossfit(data, y, 42)
    assert s.shape == (18, 3) and np.isfinite(s).all()
    held = []
    for r in records:
        assert set(r["fit_ids"]).isdisjoint(r["heldout_ids"])
        held += r["heldout_ids"]
    assert sorted(held) == sorted(ids)
