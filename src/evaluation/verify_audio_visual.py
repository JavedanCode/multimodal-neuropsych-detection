"""Replay branch selection, neural fitting, checkpoints and reported metrics."""

from src.training.checkpoints import load_checkpoint

from src.training.provenance import matches_hash
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from src.experiments.audio_visual import load, inner_candidates, predict, NAMES
from src.training.splits import splits
from src.training.classical import SEEDS
from src.models.fusion import fit_network


def check_preprocessing(prep, x):
    assert prep.columns_ == list(x)
    raw = x.to_numpy(float)[:, prep.keep_]
    raw[~np.isfinite(raw)] = np.nan
    med = np.nanmedian(raw, axis=0)
    filled = np.where(np.isnan(raw), med, raw)
    np.testing.assert_allclose(prep.medians_, med, atol=1e-12)
    np.testing.assert_allclose(prep.scale_.mean_, filled.mean(0), atol=1e-12)
    np.testing.assert_allclose(prep.scale_.var_, filled.var(0), atol=1e-12)


def verify(project, out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("study") == "audio_preprocessing":
        return verify_preprocessing(project, out)
    if manifest.get("study") == "sparse_fusion":
        from src.evaluation.verify_sparse_fusion import verify as verify_sparse

        return verify_sparse(project, out)
    if manifest.get("study") == "stacked_fusion":
        from src.evaluation.verify_stacked_study import verify as verify_stacked

        return verify_stacked(project, out)
    if manifest.get("study") == "participant_pooled":
        from src.evaluation.verify_pooled_study import verify as verify_pooled

        return verify_pooled(project, out)
    if manifest.get("study") == "score_functionals_domain":
        from src.evaluation.verify_score_study import verify as verify_score

        return verify_score(project, out)
    if manifest.get("study") == "corrected_fusion":
        from src.evaluation.verify_corrected import verify as verify_corrected

        return verify_corrected(project, out)
    for p, h in manifest["hashes"].items():
        assert matches_hash(project / p, h)
    for p, h in manifest["models"].items():
        assert matches_hash(out / "models" / p, h)
    sets, y, q, available = load(project)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert not pred.duplicated(["condition", "seed", "participant_id"]).any()
    np.testing.assert_array_equal(pred.target, y.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction, (pred.score >= 0).astype(int))
    metrics = check_metrics(pred, y, out)
    restored = replayed = neural_retrained = 0
    maximum_error = 0.0
    for seed in SEEDS:
        for fold, (a, b) in enumerate(splits(y, 5, seed)):
            train = {n: x.iloc[a] for n, x in sets.items()}
            valid = {n: x.iloc[b] for n, x in sets.items()}
            target = y.iloc[a]
            held = y.iloc[b]
            audit = json.loads((out / f"fold_{seed}_{fold}.json").read_text())
            chosen, again = inner_candidates(train, target, seed, fold)
            assert (
                audit["fit_ids"] == list(target.index)
                and audit["heldout_ids"] == list(held.index)
                and audit["inner_ids"] == again["inner_ids"]
            )
            for name in NAMES:
                for old, new in zip(
                    audit["candidates"][name], again["candidates"][name], strict=True
                ):
                    np.testing.assert_allclose(old["scores"], new["scores"], atol=1e-10)
                    assert old["inner_f1"] == new["inner_f1"]
                    replayed += 1
            for name, spec in chosen.items():
                assert audit["selected"][name] == spec
            selected_features = [
                chosen["selected_audio"]["feature"],
                chosen["selected_face"]["feature"],
            ]
            names = list(chosen) + [
                "original_concat",
                "selected_concat",
                "neural_equal",
                "neural_adaptive",
                "neural_quality",
            ]
            for name in names:
                bundle = load_checkpoint(
                    out / "models" / f"{name}_{seed}_{fold}.joblib"
                )
                if name in chosen:
                    spec = chosen[name]
                    assert bundle["features"] == [spec["feature"]]
                    assert (
                        bundle["model"]["model"].C == spec["C"]
                        and bundle["model"]["model"].class_weight == "balanced"
                    )
                elif name == "original_concat":
                    assert bundle["features"] == ["original_audio", "original_face"]
                else:
                    assert bundle["features"] == selected_features
                if bundle["kind"] in ["branch", "concat"]:
                    x = pd.concat([train[n] for n in bundle["features"]], axis=1)
                    check_preprocessing(bundle["model"]["prep"], x)
                else:
                    pa, pf, pq = (
                        bundle["audio_prep"],
                        bundle["face_prep"],
                        bundle["quality_prep"],
                    )
                    check_preprocessing(pa, train[selected_features[0]])
                    check_preprocessing(pf, train[selected_features[1]])
                    np.testing.assert_allclose(pq.mean_, q[a].mean(0), atol=1e-12)
                    state, history = fit_network(
                        pa.transform(train[selected_features[0]]),
                        pf.transform(train[selected_features[1]]),
                        pq.transform(q[a]),
                        available[a],
                        target,
                        bundle["mode"],
                        seed,
                    )
                    for k, v in state.items():
                        np.testing.assert_allclose(v, bundle["state"][k], atol=1e-7)
                    np.testing.assert_allclose(
                        history, audit["learning_curves"][name], atol=1e-7
                    )
                    neural_retrained += 1
                scores = predict(bundle, valid, q[b], available[b])
                g = (
                    pred[
                        (pred.condition == name)
                        & (pred.seed == seed)
                        & (pred.fold == fold)
                    ]
                    .set_index("participant_id")
                    .loc[held.index]
                )
                error = float(np.max(np.abs(scores - g.score.to_numpy())))
                maximum_error = max(maximum_error, error)
                np.testing.assert_allclose(scores, g.score, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(scores >= 0, g.prediction.astype(bool))
                restored += 1
    for name, value in [("always_negative", -1.0), ("always_positive", 1.0)]:
        assert (pred.loc[pred.condition == name, "score"] == value).all()
    for entry in json.loads((out / "comparison.json").read_text()):
        delta = (
            metrics.loc[(entry["candidate"], "clean")]
            - metrics.loc[(entry["control"], "clean")]
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
        "neural_models_retrained": neural_retrained,
        "maximum_restored_score_error": maximum_error,
        "hashes_folds_selection_preprocessing_metrics_verified": True,
        "test_evaluated": False,
    }
    (out / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result


def check_metrics(pred, y, out):
    """Independently reconstruct confusion, ranking, domain and summary metrics."""
    metrics = pd.read_csv(out / "repeat_metrics.csv").set_index(
        ["condition", "scenario", "seed"]
    )
    summary = pd.read_csv(out / "summary.csv").set_index(["condition", "scenario"])
    domain = pd.read_csv(out / "domain_metrics.csv").set_index(
        ["condition", "scenario", "seed", "domain"]
    )
    ap = pd.read_csv(out / "fold_average_precision.csv").set_index(
        ["condition", "seed", "fold"]
    )
    for key, g in pred.groupby(["condition", "scenario", "seed"]):
        assert len(g) == len(y) and set(g.participant_id) == set(y.index)
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
                ap.loc[(key[0], key[2], fold), "average_precision"],
                average_precision_score(v.target, v.score),
                atol=1e-12,
            )
        np.testing.assert_allclose(
            metrics.loc[key, "mean_fold_auc"], np.mean(auc), atol=1e-12
        )
    for key, g in metrics.reset_index().groupby(["condition", "scenario"]):
        for name in [
            "precision",
            "recall",
            "f1",
            "specificity",
            "accuracy",
            "balanced_accuracy",
            "mean_fold_auc",
        ]:
            np.testing.assert_allclose(
                summary.loc[key, name + "_mean"], g[name].mean(), atol=1e-12
            )
            np.testing.assert_allclose(
                summary.loc[key, name + "_sd"], g[name].std(ddof=0), atol=1e-12
            )
    return metrics


def verify_preprocessing(project, out):
    """Restore and retrain every fixed model without accepting changed source/data."""
    from src.experiments.audio_visual import preprocessing_inputs, fit_classical
    from src.training.validation import sha256

    manifest = json.loads((out / "manifest.json").read_text())
    assert not manifest["test_evaluated"] and not manifest["development_evaluated"]
    for name, digest in manifest["hashes"].items():
        assert matches_hash(project / name, digest), name
    for name, digest in manifest["models"].items():
        assert sha256(out / "models" / name) == digest, name
    conditions, labels = preprocessing_inputs(project)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert len(pred) == len(labels) * len(SEEDS) * (len(conditions) + 2)
    assert not pred.duplicated(["condition", "seed", "participant_id"]).any()
    np.testing.assert_array_equal(pred.target, labels.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction, (pred.score >= 0).astype(int))
    assert set(pred.condition) == set(conditions) | {
        "always_negative",
        "always_positive",
    }
    metrics = check_metrics(pred, labels, out)
    restored, maximum_error = 0, 0.0
    for seed in SEEDS:
        for fold, (fit, heldout) in enumerate(splits(labels, 5, seed)):
            audit = json.loads((out / f"fold_{seed}_{fold}.json").read_text())
            assert audit == {
                "fit_ids": list(labels.index[fit]),
                "heldout_ids": list(labels.index[heldout]),
            }
            for name, features in conditions.items():
                model = load_checkpoint(out / "models" / f"{name}_{seed}_{fold}.joblib")
                check_preprocessing(model["prep"], features.iloc[fit])
                assert (
                    model["model"].C == 0.1
                    and model["model"].class_weight == "balanced"
                )
                saved = pred[
                    (pred.condition == name) & (pred.seed == seed) & (pred.fold == fold)
                ]
                saved = saved.set_index("participant_id").loc[labels.index[heldout]]
                scores = model.decision_function(features.iloc[heldout])
                np.testing.assert_allclose(scores, saved.score, atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(scores >= 0, saved.prediction)
                refit = fit_classical(
                    features.iloc[fit],
                    labels.iloc[fit],
                    {"family": "logistic", "C": 0.1},
                    seed,
                )
                np.testing.assert_array_equal(refit["prep"].keep_, model["prep"].keep_)
                np.testing.assert_allclose(
                    refit.decision_function(features.iloc[heldout]),
                    scores,
                    atol=1e-12,
                    rtol=1e-12,
                )
                maximum_error = max(
                    maximum_error, float(np.max(abs(scores - saved.score)))
                )
                restored += 1
    for name, value in [("always_negative", -1.0), ("always_positive", 1.0)]:
        assert (pred.loc[pred.condition == name, "score"] == value).all()
    for entry in json.loads((out / "comparison.json").read_text()):
        indexed = metrics.reset_index().set_index(["condition", "seed"])
        difference = indexed.loc[entry["candidate"]].select_dtypes(
            "number"
        ) - indexed.loc[entry["control"]].select_dtypes("number")
        means = difference.mean()
        for key, value in entry["mean_differences"].items():
            np.testing.assert_allclose(means[key], value, atol=1e-12)
        expected = bool(
            means.f1 >= 0.03
            and means.balanced_accuracy >= 0
            and means.mean_fold_auc >= 0
            and means.recall >= -0.05
            and (difference.f1 > 0).sum() >= 2
        )
        assert entry["passes_clean_screen"] == expected
    historical = (
        project
        / "experiments/runs/branch_audit_v6_20261104T202915295134Z/predictions.parquet"
    )
    previous = pd.read_parquet(historical)
    keys = ["seed", "fold", "participant_id"]
    previous = (
        previous[previous.condition == "original_concat"].set_index(keys).sort_index()
    )
    current = pred[pred.condition == "original_concat"].set_index(keys).sort_index()
    pd.testing.assert_index_equal(previous.index, current.index)
    np.testing.assert_array_equal(previous.target, current.target)
    np.testing.assert_allclose(previous.score, current.score, atol=1e-12, rtol=1e-12)
    result = {
        "models_restored_and_retrained": restored,
        "prediction_rows": len(pred),
        "maximum_restored_score_error": maximum_error,
        "historical_concat_reproduced": True,
        "hashes_folds_preprocessing_metrics_verified": True,
        "test_evaluated": False,
    }
    (out / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(verify(Path(__file__).resolve().parents[2], Path(sys.argv[1])))
