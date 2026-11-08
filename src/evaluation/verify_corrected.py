"""Independently check the corrected voicing / common-scale fusion run."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.verify_audio_visual import check_metrics, check_preprocessing
from src.models.sparse_fusion import branch_scores
from src.training.checkpoints import load_checkpoint
from src.training.classical import SEEDS
from src.training.corrected_study import (
    WEIGHTS,
    comparison_pairs,
    fit_policies,
    inner_audit,
    load_inputs,
    predict,
)
from src.training.provenance import matches_hash
from src.training.splits import splits
from src.training.validation import sha256


def independent_fusion(model, audio, face, quality, available):
    """Recompute fused scores element by element, without fuse_scores."""
    result = []
    for a, f, q, present in zip(audio, face, quality, available, strict=True):
        wa, wf = model["weight"], 1 - model["weight"]
        if model["reliability"]:
            wa *= 0.5 if q[2] > 0.001 or q[5] > 0.01 or q[1] < 0.95 else 1
            wf *= 0.5 if q[6] < 0.8 or q[7] < 0.8 else 1
        wa *= bool(present[0])
        wf *= bool(present[1])
        if not wa + wf:
            prior = model["prior"]
            result.append(
                np.log(prior / (1 - prior)) if model["scales"] else prior - 0.5
            )
        elif model["scales"]:
            sa, sf = model["scales"]
            result.append((wa * a / sa + wf * f / sf) / (wa + wf))
        else:
            ea, ef = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-f))
            result.append((wa * ea + wf * ef) / (wa + wf) - 0.5)
    return np.array(result)


def verify(project, output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["test_evaluated"] and not manifest["development_evaluated"]
    for name, digest in manifest["hashes"].items():
        assert matches_hash(project / name, digest), name
    for name, digest in manifest["models"].items():
        assert sha256(output / "models" / name) == digest, name
    features, labels, quality, available = load_inputs(project)
    pred = pd.read_parquet(output / "predictions.parquet")
    assert len(pred) == len(labels) * len(SEEDS) * 11
    assert not pred.duplicated(["condition", "seed", "participant_id"]).any()
    np.testing.assert_array_equal(pred.target, labels.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction, pred.score >= 0)
    metrics = check_metrics(pred, labels, output)
    restored = scales_checked = 0
    maximum_error = 0.0
    for seed in SEEDS:
        for fold, (a, b) in enumerate(splits(labels, 5, seed)):
            train = {n: f.iloc[a] for n, f in features.items()}
            valid = {n: f.iloc[b] for n, f in features.items()}
            audit = json.loads((output / f"fold_{seed}_{fold}.json").read_text())
            assert audit["fit_ids"] == list(labels.index[a])
            assert audit["heldout_ids"] == list(labels.index[b])
            assert not set(audit["fit_ids"]) & set(audit["heldout_ids"])
            for inner in audit["inner_ids"]:
                assert set(inner["fit"]) | set(inner["heldout"]) <= set(audit["fit_ids"])
            repeated = inner_audit(train, labels.iloc[a], quality[a], available[a], seed, fold)
            assert audit["inner_ids"] == repeated["inner_ids"]
            # Scales: recompute RMS of saved OOF margins independently.
            for position, name in enumerate(["voiced", "face"]):
                saved = np.asarray(audit["inner_margins"][name])
                np.testing.assert_allclose(saved, repeated["inner_margins"][name], atol=1e-12)
                rms = float(np.sqrt(np.mean(saved**2))) or 1.0
                np.testing.assert_allclose(audit["scales"][position], rms, rtol=1e-12)
                scales_checked += 1
            # Weight: recompute inner F1 from saved scores, first maximum in order.
            assert [r["weight"] for r in audit["weight_candidates"]] == WEIGHTS
            y_inner = labels.iloc[a].to_numpy()
            f1s = [f1_score(y_inner, np.asarray(r["scores"]) >= 0) for r in audit["weight_candidates"]]
            for record, value in zip(audit["weight_candidates"], f1s, strict=True):
                assert record["inner_f1"] == value
            assert audit["weight"] == WEIGHTS[int(np.argmax(f1s))] == repeated["weight"]
            policies = fit_policies(train, labels.iloc[a], repeated, seed)
            assert set(pred.condition) == set(policies) | {"always_negative", "always_positive"}
            for name, refit in policies.items():
                model = load_checkpoint(output / "models" / f"{name}_{seed}_{fold}.joblib")
                saved = (
                    pred[(pred.condition == name) & (pred.seed == seed) & (pred.fold == fold)]
                    .set_index("participant_id")
                    .loc[labels.index[b]]
                )
                scores = predict(model, valid, quality[b], available[b])
                np.testing.assert_allclose(scores, saved.score, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(
                    predict(refit, valid, quality[b], available[b]), scores,
                    atol=1e-12, rtol=1e-12,
                )
                if model["kind"] == "single":
                    check_preprocessing(model["model"]["preprocess"], train[model["input"]])
                    assert model["model"]["spec"] == {"selection": "all", "C": 0.1}
                else:
                    check_preprocessing(model["audio"]["preprocess"], train["voiced"])
                    check_preprocessing(model["face"]["preprocess"], train["face"])
                    independent = independent_fusion(
                        model,
                        branch_scores(model["audio"], valid["voiced"]),
                        branch_scores(model["face"], valid["face"]),
                        quality[b],
                        available[b],
                    )
                    np.testing.assert_allclose(independent, scores, atol=1e-10, rtol=1e-10)
                maximum_error = max(maximum_error, float(np.max(abs(scores - saved.score))))
                restored += 1
    for name, score in [("always_negative", -1.0), ("always_positive", 1.0)]:
        assert (pred.loc[pred.condition == name, "score"] == score).all()
    comparisons = json.loads((output / "comparison.json").read_text())
    assert [(r["candidate"], r["control"]) for r in comparisons] == comparison_pairs()
    indexed = metrics.reset_index().set_index(["condition", "seed"])
    for row in comparisons:
        differences = indexed.loc[row["candidate"]].select_dtypes("number") - indexed.loc[
            row["control"]
        ].select_dtypes("number")
        means = differences.mean()
        passes = (
            means.f1 >= 0.03
            and means.balanced_accuracy >= 0
            and means.mean_fold_auc >= 0
            and means.recall >= -0.05
            and (differences.f1 > 0).sum() >= 2
        )
        assert bool(passes) == row["passes_clean_screen"]
    historical = pd.read_parquet(
        project / "experiments/runs/branch_audit_v6_20261104T202915295134Z/predictions.parquet"
    )
    keys = ["seed", "fold", "participant_id"]
    old = historical[historical.condition == "original_concat"].set_index(keys).sort_index()
    new = pred[pred.condition == "original_concat"].set_index(keys).sort_index()
    pd.testing.assert_index_equal(old.index, new.index)
    np.testing.assert_allclose(old.score, new.score, atol=1e-12, rtol=1e-12)
    result = {
        "policy_bundles_restored_and_refitted": restored,
        "prediction_rows": len(pred),
        "branch_scales_recomputed": scales_checked,
        "fusion_weights_recomputed_from_saved_scores": len(SEEDS) * 5,
        "maximum_restored_score_error": maximum_error,
        "independent_fusion_verified": True,
        "historical_concat_reproduced": True,
        "test_evaluated": False,
    }
    (output / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result
