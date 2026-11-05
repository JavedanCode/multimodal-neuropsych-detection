"""Replay nested choices, primary predictions and observable fallback decisions."""

from src.training.checkpoints import load_checkpoint

from src.training.provenance import matches_hash
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from src.dataset.aligned_inputs import load
from src.experiments.fallback import inner_candidates, CONDITIONS
from src.training.splits import splits
from src.training.classical import SEEDS
from src.evaluation.crossfit import check_crossfit
from src.models.stacking import take
from src.models.fallback import predict, SCENARIOS


def independent_route(data, quality, fallback, scenario):
    q = data["quality"]
    a = data["available"].astype(bool).copy()
    if quality:
        a[:, 1] &= (q[:, 1] >= 0.95) & (q[:, 2] <= 0.001) & (q[:, 5] <= 0.01)
        a[:, 2] &= (q[:, 6] >= 0.8) & (q[:, 7] >= 0.8)
    if scenario in ["missing_audio", "missing_both"]:
        a[:, 1] = False
    if scenario in ["missing_face", "missing_both"]:
        a[:, 2] = False
    return ~(a[:, 1] | a[:, 2]) if fallback else np.zeros(len(a), bool)


def verify(project, out):
    m = json.loads((out / "manifest.json").read_text())
    for p, h in m["hashes"].items():
        assert matches_hash(project / p, h)
    for p, h in m["models"].items():
        assert matches_hash(out / "models" / p, h)
    data, y, _ = load(project)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert not pred.duplicated(
        ["condition", "scenario", "seed", "participant_id"]
    ).any()
    np.testing.assert_array_equal(pred.target, y.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction, (pred.score >= 0).astype(int))
    metrics = pd.read_csv(out / "repeat_metrics.csv").set_index(
        ["condition", "scenario", "seed"]
    )
    summary = pd.read_csv(out / "summary.csv").set_index(["condition", "scenario"])
    ap = pd.read_csv(out / "fold_average_precision.csv").set_index(
        ["condition", "scenario", "seed", "fold"]
    )
    domain = pd.read_csv(out / "domain_metrics.csv").set_index(
        ["condition", "scenario", "seed", "domain"]
    )
    routing = pd.read_csv(out / "routing.csv").set_index(
        ["condition", "scenario", "seed"]
    )
    for key, g in pred.groupby(["condition", "scenario", "seed"]):
        assert len(g) == len(y) and set(g.participant_id) == set(y.index)
        for field in ["fallback_used", "text_used"]:
            np.testing.assert_allclose(
                routing.loc[key, field], g[field].mean(), atol=1e-12
            )
        parts = [(g, metrics.loc[key])] + [
            (v, domain.loc[(*key, d)])
            for d, v in g.groupby(np.where(g.participant_id >= 600, "AI", "WoZ"))
        ]
        for v, row in parts:
            tn, fp, fn, tp = confusion_matrix(
                v.target, v.prediction, labels=[0, 1]
            ).ravel()
            expected = {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "precision": tp / max(1, tp + fp),
                "recall": tp / (tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                "accuracy": (tp + tn) / len(v),
                "specificity": tn / (tn + fp),
                "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
            }
            for name, value in expected.items():
                np.testing.assert_allclose(row[name], value, atol=1e-12)
        auc = []
        for fold, v in g.groupby("fold"):
            auc.append(roc_auc_score(v.target, v.score))
            np.testing.assert_allclose(
                ap.loc[(*key, fold), "average_precision"],
                average_precision_score(v.target, v.score),
                atol=1e-12,
            )
        np.testing.assert_allclose(
            metrics.loc[key, "mean_fold_auc"], np.mean(auc), atol=1e-12
        )
    for key, g in metrics.reset_index().groupby(["condition", "scenario"]):
        for field in [
            "f1",
            "precision",
            "recall",
            "accuracy",
            "specificity",
            "balanced_accuracy",
            "mean_fold_auc",
        ]:
            np.testing.assert_allclose(
                summary.loc[key, field + "_mean"], g[field].mean(), atol=1e-12
            )
            np.testing.assert_allclose(
                summary.loc[key, field + "_sd"], g[field].std(ddof=0), atol=1e-12
            )
    restored = replayed = 0
    max_score_error = 0.0
    for seed in SEEDS:
        for fold, (a, b) in enumerate(splits(y, 5, seed)):
            target, held = y.iloc[a], y.iloc[b]
            train, valid = take(data, a), take(data, b)
            audit = json.loads((out / f"fold_{seed}_{fold}.json").read_text())
            assert audit["fit_ids"] == list(target.index) and audit[
                "heldout_ids"
            ] == list(held.index)
            selected, again = inner_candidates(train, target, seed, fold)
            assert audit["inner_ids"] == again["inner_ids"]
            for j, (ia, ib) in enumerate(splits(target, 3, seed + fold + 1)):
                check_crossfit(
                    audit["crossfit"][j],
                    target.index[ia],
                    set(target.index[ib]) | set(held.index),
                )
            check_crossfit(audit["outer_crossfit"], target.index, set(held.index))
            for old, new in zip(audit["candidates"], again["candidates"]):
                np.testing.assert_allclose(old["scores"], new["scores"], atol=1e-12)
                assert (
                    old["routes"] == new["routes"]
                    and old["inner_f1"] == new["inner_f1"]
                )
                replayed += 1
            for name in list(CONDITIONS) + ["fixed_text", "retained_reference"]:
                bundle = load_checkpoint(
                    out / "models" / f"{name}_{seed}_{fold}.joblib"
                )
                if name in CONDITIONS:
                    assert (
                        audit["selected"][name] == selected[name]
                        and bundle["meta"]["model"].C == selected[name]["C"]
                    )
                    assert (bundle["quality"], bundle["fallback"]) == CONDITIONS[name]
                for scenario in SCENARIOS:
                    g = (
                        pred[
                            (pred.condition == name)
                            & (pred.scenario == scenario)
                            & (pred.seed == seed)
                            & (pred.fold == fold)
                        ]
                        .set_index("participant_id")
                        .loc[held.index]
                    )
                    if name in CONDITIONS:
                        scores, routes = predict(bundle, valid, scenario)
                        np.testing.assert_array_equal(
                            routes,
                            independent_route(valid, *CONDITIONS[name], scenario),
                        )
                        assert bundle["meta"]["model"].n_features_in_ == (
                            13 if bundle["quality"] else 4
                        )
                    else:
                        scores = bundle["model"].decision_function(valid["text"])
                        routes = np.zeros(len(held), bool)
                    # Float32 PCA inputs can shift by ~1e-7 after serialization
                    # changes array layout. Bound absolute score drift and require
                    # identical classification, rather than relaxing that decision.
                    error = float(np.max(np.abs(scores - g.score.to_numpy())))
                    max_score_error = max(max_score_error, error)
                    np.testing.assert_allclose(scores, g.score, rtol=0, atol=1e-6)
                    np.testing.assert_array_equal(
                        scores >= 0, g.prediction.astype(bool)
                    )
                    np.testing.assert_array_equal(routes, g.fallback_used)
                    np.testing.assert_array_equal(
                        routes | (name not in CONDITIONS), g.text_used
                    )
                restored += 1
    old = (
        pd.read_parquet(Path(m["baseline"]) / "predictions.parquet")
        .query('condition=="joint"')
        .set_index(["seed", "fold", "participant_id"])
        .sort_index()
    )
    current = (
        pred.query('condition=="retained_reference" and scenario=="clean"')
        .set_index(["seed", "fold", "participant_id"])
        .sort_index()
    )
    assert old.index.equals(current.index)
    np.testing.assert_allclose(old.score, current.score, atol=1e-12)
    comparison = json.loads((out / "comparison.json").read_text())
    for entry in comparison["against_reference"] + comparison["fallback_ablation"]:
        delta = (
            metrics.loc[(entry["candidate"], entry["scenario"])]
            - metrics.loc[(entry["control"], entry["scenario"])]
        )
        means = delta.mean(numeric_only=True)
        for name, value in entry["mean_differences"].items():
            np.testing.assert_allclose(value, means[name], atol=1e-12)
        assert entry["passes_clean_screen"] == bool(
            means.f1 >= 0.03
            and means.balanced_accuracy >= 0
            and means.mean_fold_auc >= 0
            and means.recall >= -0.05
            and (delta.f1 > 0).sum() >= 2
        )
    result = {
        "models_restored": restored,
        "prediction_rows": len(pred),
        "inner_candidate_vectors_replayed": replayed,
        "routing_metrics_nested_folds_hashes_verified": True,
        "primary_text_features": False,
        "maximum_restored_score_error": max_score_error,
        "restored_classes_identical": True,
        "test_evaluated": False,
    }
    (out / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(verify(Path(__file__).resolve().parents[2], Path(sys.argv[1])))
