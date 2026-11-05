import numpy as np
import pandas as pd
import pytest
from src.selection.hho import binary_hho, decode
from src.selection.fitness import TrainingFitness, select_l1
from src.dataset.preprocessing import ParticipantPreprocessor


def data():
    rng = np.random.default_rng(31)
    X = pd.DataFrame(
        rng.normal(size=(24, 8)),
        index=pd.Index(range(1, 25), name="Participant_ID"),
        columns=[f"audio__x{i}" for i in range(4)]
        + [f"facial__x{i}" for i in range(4)],
    )
    y = pd.Series(np.tile([0, 1], 12), index=X.index)
    return X, y


def test_binary_hho_reproducible_repaired_and_monotone():
    groups = [np.arange(4), np.arange(4, 8)]
    seen = []

    def objective(mask):
        assert mask.dtype == bool and mask[:4].any() and mask[4:].any()
        seen.append(mask.copy())
        return float(mask.sum() / 8)

    a = binary_hho(objective, 8, groups, 42, 8, 12)
    b = binary_hho(objective, 8, groups, 42, 8, 12)
    assert a == b
    assert a["fitness"] == sum(a["mask"]) / 8
    assert all(
        x["best_fitness"] >= y["best_fitness"]
        for x, y in zip(a["history"], a["history"][1:])
    )
    assert all(count > 0 for count in a["branches"].values())
    repaired = decode(np.full(8, -6.0), np.random.default_rng(0), groups)
    assert repaired[:4].any() and repaired[4:].any()
    with pytest.raises(ValueError):
        binary_hho(objective, 8, [np.arange(3)], 42)


def test_cv_fitness_fits_within_fold_and_cache(monkeypatch):
    X, y = data()
    observed = []
    original = ParticipantPreprocessor.fit

    def track(self, frame, target=None):
        observed.append(set(frame.index))
        return original(self, frame, target)

    monkeypatch.setattr(ParticipantPreprocessor, "fit", track)
    fitness = TrainingFitness(X, y, folds=3)
    mask = np.ones(8, dtype=bool)
    value = fitness(mask)
    assert len(observed) == 3
    assert all(len(ids) == 16 for ids in observed)
    for ids, (a, b) in zip(observed, fitness.splits):
        assert ids == set(X.index[a]) and not ids.intersection(X.index[b])
    record = fitness.records[0]
    assert value == pytest.approx(0.95 * (1 - record["mean_cv_f1"]) + 0.05)
    assert fitness(mask) == value and len(observed) == 3
    assert fitness(np.zeros(8, dtype=bool)) == 2
    with pytest.raises(ValueError):
        fitness(np.full(8, 2))
    with pytest.raises(ValueError):
        TrainingFitness(X, y, alpha=0.9, beta=0.9)


def test_l1_reproducible_and_modality_fallback():
    X, y = data()
    a = select_l1(X, y, C=1e-8)
    b = select_l1(X, y, C=1e-8)
    assert a["mask"] == b["mask"]
    assert len(a["forced_modality_features"]) == 2
    assert sum(a["mask"]) == 2
    assert any(c.startswith("audio__") for c in a["selected_features"])
    assert any(c.startswith("facial__") for c in a["selected_features"])


def test_fixed_selected_model_is_dev_independent_and_restorable(tmp_path):
    torch = pytest.importorskip("torch")
    from src.training.feature_selection import fit_selected
    from src.training.neural import predict_checkpoint

    X, y = data()
    train = X.iloc[:18]
    dev = X.iloc[18:]
    cols = ["audio__x0", "audio__x1", "facial__x0", "facial__x1"]
    parameters = {
        "hidden_dim": 4,
        "dropout": 0.2,
        "epochs": 3,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
    }
    a, r, p = fit_selected(
        train, y.iloc[:18], dev, y.iloc[18:], cols, parameters, 42, "cpu"
    )
    b, _, _ = fit_selected(
        train, y.iloc[:18], dev, 1 - y.iloc[18:], cols, parameters, 42, "cpu"
    )
    for k in a["state_dict"]:
        torch.testing.assert_close(
            a["state_dict"][k], b["state_dict"][k], rtol=0, atol=0
        )
    path = tmp_path / "model.pt"
    torch.save(a, path)
    restored = predict_checkpoint(path, dev[cols])
    for c in restored:
        np.testing.assert_array_equal(restored[c], p[c])
    assert r["selected_feature_count"] == 4 and r["original_feature_count"] == 8
    assert a["train_participant_ids"] == list(train.index)
    with pytest.raises(ValueError):
        fit_selected(
            train, y.iloc[:18], train, y.iloc[:18], cols, parameters, 42, "cpu"
        )
