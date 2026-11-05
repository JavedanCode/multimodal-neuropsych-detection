from src.training.checkpoints import load_checkpoint
from src.training.provenance import matches_hash

"""Independent membership, checkpoint, metric and selection checks for v2 runs."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from threadpoolctl import threadpool_limits
from src.training.validation import sha256
from src.dataset.text_inputs import inputs
from src.training.splits import splits, transfer
from src.training.classical import SEEDS


def verify(project, out):
    manifest = json.loads((out / "manifest.json").read_text())
    for name, h in manifest["hashes"].items():
        assert matches_hash(project / name, h), name
    for name, h in manifest["model_hashes"].items():
        assert matches_hash(out / "models" / name, h), name
    sets, y = inputs(project)
    features = sets["mpnet"]
    pred = pd.read_parquet(out / "predictions.parquet")
    assert not pred.duplicated(["condition", "seed", "participant_id"]).any()
    np.testing.assert_array_equal(pred.target, y.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction, (pred.score >= 0).astype(int))
    metrics = pd.read_csv(out / "repeat_metrics.csv").set_index(["condition", "seed"])
    summary = pd.read_csv(out / "summary.csv").set_index("condition")
    ap = pd.read_csv(out / "fold_average_precision.csv").set_index(
        ["condition", "seed", "fold"]
    )
    for (name, seed), g in pred.groupby(["condition", "seed"]):
        expected = set(y.index[y.index >= 600]) if name == "transfer" else set(y.index)
        assert set(g.participant_id) == expected
        tn, fp, fn, tp = confusion_matrix(g.target, g.prediction, labels=[0, 1]).ravel()
        values = {
            "f1": f1_score(g.target, g.prediction),
            "precision": precision_score(g.target, g.prediction, zero_division=0),
            "recall": recall_score(g.target, g.prediction),
            "accuracy": accuracy_score(g.target, g.prediction),
            "balanced_accuracy": balanced_accuracy_score(g.target, g.prediction),
            "specificity": tn / (tn + fp),
            "mean_fold_auc": np.mean(
                [roc_auc_score(v.target, v.score) for _, v in g.groupby("fold")]
            ),
        }
        for k, v in values.items():
            np.testing.assert_allclose(metrics.loc[(name, seed), k], v, atol=1e-12)
        for fold, v in g.groupby("fold"):
            np.testing.assert_allclose(
                ap.loc[(name, seed, fold), "average_precision"],
                average_precision_score(v.target, v.score),
                atol=1e-12,
            )
    for name, g in metrics.reset_index().groupby("condition"):
        for k in [
            "f1",
            "precision",
            "recall",
            "specificity",
            "accuracy",
            "balanced_accuracy",
            "mean_fold_auc",
        ]:
            np.testing.assert_allclose(
                summary.loc[name, k + "_mean"], g[k].mean(), atol=1e-12
            )
            np.testing.assert_allclose(
                summary.loc[name, k + "_sd"], g[k].std(ddof=0), atol=1e-12
            )
    count = 0
    for seed in SEEDS:
        modes = ["joint", "transfer"]
        for mode in modes:
            outer = splits(y, 5, seed) if mode == "joint" else [transfer(y)]
            for fold, (a, b) in enumerate(outer):
                audit = json.loads((out / f"{mode}_{seed}_{fold}.json").read_text())
                target = y.iloc[a]
                held = y.iloc[b]
                assert audit["fit_ids"] == list(target.index) and audit[
                    "heldout_ids"
                ] == list(held.index)
                for j, (ia, ib) in enumerate(splits(target, 3, seed + fold + 1)):
                    assert audit["inner_ids"][j] == {
                        "fit": list(target.index[ia]),
                        "heldout": list(target.index[ib]),
                    }
                names = [mode]
                assert audit["selected"] == max(
                    audit["candidates"], key=lambda r: r["selection_f1"]
                )
                for name in names:
                    model = load_checkpoint(
                        out / "models" / f"{name}_{seed}_{fold}.joblib"
                    )
                    np.testing.assert_allclose(
                        model["preprocess"].scale_.mean_,
                        features.iloc[a].to_numpy(dtype=np.float64).mean(0),
                        atol=1e-12,
                    )
                    scores = model.decision_function(features.loc[held.index])
                    g = (
                        pred[
                            (pred.condition == name)
                            & (pred.seed == seed)
                            & (pred.fold == fold)
                        ]
                        .set_index("participant_id")
                        .loc[held.index]
                    )
                    np.testing.assert_allclose(scores, g.score, atol=1e-12, rtol=1e-12)
                    count += 1
    result = {
        "models_restored": count,
        "predictions_checked": len(pred),
        "metrics_rows": len(metrics),
        "folds_hashes_selection_scalers_verified": True,
        "test_evaluated": False,
    }
    (out / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(verify(Path(__file__).resolve().parents[2], Path(sys.argv[1])))
