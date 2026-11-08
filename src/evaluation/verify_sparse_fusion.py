"""Independently check sparse selection, matched fusion and all saved outcomes."""

import json

import numpy as np
import pandas as pd
from scipy.special import expit

from src.evaluation.verify_audio_visual import check_metrics, check_preprocessing
from src.models.sparse_fusion import branch_scores, predict_policy
from src.training.checkpoints import load_checkpoint
from src.training.classical import SEEDS
from src.training.fusion_study import (
    load_inputs,
    select_candidates,
    fit_policies,
    comparison_pairs,
)
from src.training.splits import splits
from src.training.validation import sha256


def verify(project, output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["test_evaluated"] and not manifest["development_evaluated"]
    for name, digest in manifest["hashes"].items():
        assert sha256(project / name) == digest, name
    for name, digest in manifest["models"].items():
        assert sha256(output / "models" / name) == digest, name
    features, labels, quality, available = load_inputs(project)
    pred = pd.read_parquet(output / "predictions.parquet")
    assert len(pred) == len(labels) * 3 * 14
    assert not pred.duplicated(["condition", "seed", "participant_id"]).any()
    np.testing.assert_array_equal(pred.target, labels.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction, pred.score >= 0)
    metrics = check_metrics(pred, labels, output)
    restored = replayed = weights_replayed = 0
    maximum_error = 0.0
    for seed in SEEDS:
        for fold, (a, b) in enumerate(splits(labels, 5, seed)):
            train = {name: frame.iloc[a] for name, frame in features.items()}
            valid = {name: frame.iloc[b] for name, frame in features.items()}
            audit = json.loads((output / f"fold_{seed}_{fold}.json").read_text())
            assert audit["fit_ids"] == list(labels.index[a])
            assert audit["heldout_ids"] == list(labels.index[b])
            repeated = select_candidates(
                train, labels.iloc[a], quality[a], available[a], seed, fold
            )
            assert audit["inner_ids"] == repeated["inner_ids"]
            assert (
                audit["choices"] == repeated["choices"]
                and audit["weight"] == repeated["weight"]
            )
            for name in features:
                for original, replay in zip(
                    audit["candidates"][name], repeated["candidates"][name], strict=True
                ):
                    assert original["spec"] == replay["spec"]
                    assert original["inner_f1"] == replay["inner_f1"]
                    assert (
                        original["inner_selected_counts"]
                        == replay["inner_selected_counts"]
                    )
                    np.testing.assert_allclose(
                        original["scores"], replay["scores"], atol=1e-12, rtol=1e-12
                    )
                    replayed += 1
            for original, replay in zip(
                audit["weight_candidates"], repeated["weight_candidates"], strict=True
            ):
                assert (
                    original["weight"] == replay["weight"]
                    and original["inner_f1"] == replay["inner_f1"]
                )
                np.testing.assert_allclose(
                    original["scores"], replay["scores"], atol=1e-12, rtol=1e-12
                )
                weights_replayed += 1
            policies = fit_policies(train, labels.iloc[a], repeated, seed)
            assert set(pred.condition) == set(policies) | {
                "always_negative",
                "always_positive",
            }
            for name, refit in policies.items():
                model = load_checkpoint(
                    output / "models" / f"{name}_{seed}_{fold}.joblib"
                )
                saved = (
                    pred[
                        (pred.condition == name)
                        & (pred.seed == seed)
                        & (pred.fold == fold)
                    ]
                    .set_index("participant_id")
                    .loc[labels.index[b]]
                )
                scores = predict_policy(model, valid, quality[b], available[b])
                np.testing.assert_allclose(scores, saved.score, atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(scores >= 0, saved.prediction)
                np.testing.assert_allclose(
                    predict_policy(refit, valid, quality[b], available[b]),
                    scores,
                    atol=1e-12,
                    rtol=1e-12,
                )
                branches = (
                    [model["input"]] if model["kind"] == "single" else ["audio", "face"]
                )
                for branch in branches:
                    fitted = (
                        model["model"] if model["kind"] == "single" else model[branch]
                    )
                    fresh = (
                        refit["model"] if model["kind"] == "single" else refit[branch]
                    )
                    check_preprocessing(fitted["preprocess"], train[branch])
                    assert fitted["selected_features"] == fresh["selected_features"]
                    np.testing.assert_array_equal(fitted["indices"], fresh["indices"])
                if model["kind"] == "fusion":
                    assert (
                        model["weight"] == refit["weight"]
                        and model["reliability"] == refit["reliability"]
                    )
                    independent = []
                    audio = branch_scores(model["audio"], valid["audio"])
                    face = branch_scores(model["face"], valid["face"])
                    for position, (q, present) in enumerate(
                        zip(quality[b], available[b])
                    ):
                        wa, wf = model["weight"], 1 - model["weight"]
                        if model["reliability"]:
                            wa *= (
                                0.5 if q[2] > 0.001 or q[5] > 0.01 or q[1] < 0.95 else 1
                            )
                            wf *= 0.5 if q[6] < 0.8 or q[7] < 0.8 else 1
                        wa *= bool(present[0])
                        wf *= bool(present[1])
                        probability = (
                            (wa * expit(audio[position]) + wf * expit(face[position]))
                            / (wa + wf)
                            if wa + wf
                            else model["prior"]
                        )
                        independent.append(probability - 0.5)
                    np.testing.assert_allclose(
                        independent, scores, atol=1e-12, rtol=1e-12
                    )
                maximum_error = max(
                    maximum_error, float(np.max(abs(scores - saved.score)))
                )
                restored += 1
    for name, score in [("always_negative", -1.0), ("always_positive", 1.0)]:
        assert (pred.loc[pred.condition == name, "score"] == score).all()
    comparisons = json.loads((output / "comparison.json").read_text())
    assert [(r["candidate"], r["control"]) for r in comparisons] == comparison_pairs()
    indexed = metrics.reset_index().set_index(["condition", "seed"])
    for row in comparisons:
        differences = indexed.loc[row["candidate"]].select_dtypes(
            "number"
        ) - indexed.loc[row["control"]].select_dtypes("number")
        means = differences.mean()
        for name, value in row["mean_differences"].items():
            np.testing.assert_allclose(means[name], value, atol=1e-12)
        passes = (
            means.f1 >= 0.03
            and means.balanced_accuracy >= 0
            and means.mean_fold_auc >= 0
            and means.recall >= -0.05
            and (differences.f1 > 0).sum() >= 2
        )
        assert bool(passes) == row["passes_clean_screen"]
    historical = pd.read_parquet(
        project
        / "experiments/runs/branch_audit_v6_20261104T202915295134Z/predictions.parquet"
    )
    keys = ["seed", "fold", "participant_id"]
    old = (
        historical[historical.condition == "original_concat"]
        .set_index(keys)
        .sort_index()
    )
    new = pred[pred.condition == "original_concat"].set_index(keys).sort_index()
    pd.testing.assert_index_equal(old.index, new.index)
    np.testing.assert_allclose(old.score, new.score, atol=1e-12, rtol=1e-12)
    result = {
        "policy_bundles_restored_and_refitted": restored,
        "prediction_rows": len(pred),
        "inner_candidate_vectors_replayed": replayed,
        "inner_weight_vectors_replayed": weights_replayed,
        "maximum_restored_score_error": maximum_error,
        "independent_quality_weights_verified": True,
        "hashes_folds_preprocessing_selection_metrics_verified": True,
        "historical_concat_reproduced": True,
        "test_evaluated": False,
    }
    (output / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result
