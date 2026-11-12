"""Independently check the stacked-fusion run under all three fold schemes."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.dataset.contracts import domain
from src.evaluation.verify_audio_visual import check_preprocessing
from src.models.sparse_fusion import branch_scores
from src.training.checkpoints import load_checkpoint
from src.training.classical import SEEDS
from src.training.provenance import matches_hash
from src.training.stacked_study import (
    BRANCHES,
    SCHEMES,
    STACKS,
    comparison_pairs,
    fit_fusion,
    load_inputs,
    outer_splits,
    predict,
)
from src.training.validation import sha256


def verify(project, output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["test_evaluated"] and not manifest["development_evaluated"]
    for name, digest in manifest["hashes"].items():
        assert matches_hash(project / name, digest), name
    for name, digest in manifest["models"].items():
        assert sha256(output / "models" / name) == digest, name
    features, labels, gender = load_inputs(project)
    official = pd.read_parquet(project / "data/processed/labels.parquet")
    assert (official.loc[labels.index, "split"] == "train").all()
    np.testing.assert_array_equal(labels, official.loc[labels.index, "PHQ_Binary"])
    pred = pd.read_parquet(output / "predictions.parquet")
    audits = json.loads((output / "folds.json").read_text())
    np.testing.assert_array_equal(pred.target, labels.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction, pred.score >= 0)
    assert not pred.duplicated(["condition", "seed", "participant_id"]).any()
    assert len(pred) == len(labels) * len(SEEDS) * len(SCHEMES) * 7
    restored = 0
    maximum_error = 0.0
    for scheme in SCHEMES:
        for seed in SEEDS:
            folds = outer_splits(labels, gender, scheme, seed)
            covered = np.concatenate([b for _, b in folds])
            assert sorted(covered) == list(range(len(labels)))  # each person held out once
            for fold, (a, b) in enumerate(folds):
                audit = audits[f"{scheme}_{seed}_{fold}"]
                assert audit["fit_ids"] == list(labels.index[a])
                assert audit["heldout_ids"] == list(labels.index[b])
                assert not set(audit["fit_ids"]) & set(audit["heldout_ids"])
                train = {n: f.iloc[a] for n, f in features.items()}
                valid = {n: f.iloc[b] for n, f in features.items()}
                bundle = load_checkpoint(output / "models" / f"{scheme}_{seed}_{fold}.joblib")
                refit = fit_fusion(train, labels.iloc[a], seed, fold)
                scores, _ = predict(bundle, valid)
                again, _ = predict(refit, valid)
                for name, frame in BRANCHES.items():
                    check_preprocessing(bundle["branches"][name]["preprocess"], train[frame])
                # Independent recomputation of the combiners from saved parts.
                text = bundle["text"].decision_function(valid["text"]) / bundle["scales"]["text"]
                scaled = {"text": text}
                for name, frame in BRANCHES.items():
                    scaled[name] = (
                        branch_scores(bundle["branches"][name], valid[frame]) / bundle["scales"][name]
                    )
                independent = {}
                for name, inputs in STACKS.items():
                    stacker = bundle["stackers"][name]
                    independent[name] = stacker.intercept_[0] + sum(
                        c * scaled[i] for c, i in zip(stacker.coef_[0], inputs)
                    )
                stacker = bundle["stackers"]["stack3"]
                independent["gate3"] = np.where(
                    np.abs(text) >= audit["gate"],
                    stacker.intercept_[0] + stacker.coef_[0][0] * text,
                    independent["stack3"],
                )
                for name, values in scores.items():
                    saved = (
                        pred[(pred.condition == f"{name}_{scheme}") & (pred.seed == seed) & (pred.fold == fold)]
                        .set_index("participant_id").loc[labels.index[b]].score.to_numpy()
                    )
                    np.testing.assert_allclose(values, saved, atol=1e-12)
                    np.testing.assert_allclose(again[name], values, atol=1e-8)
                    if name in independent:
                        np.testing.assert_allclose(independent[name], values, atol=1e-10)
                    maximum_error = max(maximum_error, float(np.max(abs(values - saved))))
                restored += 1
    metrics = pd.read_csv(output / "repeat_metrics.csv").set_index(["condition", "seed"])
    for (name, seed), g in pred.groupby(["condition", "seed"]):
        tn, fp, fn, tp = confusion_matrix(g.target, g.prediction, labels=[0, 1]).ravel()
        row = metrics.loc[(name, seed)]
        np.testing.assert_allclose(row.f1, 2 * tp / max(1, 2 * tp + fp + fn), atol=1e-12)
        np.testing.assert_allclose(row.accuracy, (tp + tn) / len(g), atol=1e-12)
        aucs = [roc_auc_score(f.target, f.score) for _, f in g.groupby("fold")]
        np.testing.assert_allclose(row.mean_fold_auc, np.mean(aucs), atol=1e-12)
    comparisons = json.loads((output / "comparison.json").read_text())
    assert [(r["candidate"], r["control"]) for r in comparisons] == comparison_pairs()
    numeric = metrics.select_dtypes("number")
    for row in comparisons:
        differences = numeric.loc[row["candidate"]] - numeric.loc[row["control"]]
        means = differences.mean()
        passes = (
            means.f1 >= 0.03
            and means.balanced_accuracy >= 0
            and means.mean_fold_auc >= 0
            and means.recall >= -0.05
            and (differences.f1 > 0).sum() >= 2
        )
        assert bool(passes) == row["passes_clean_screen"]
    # Scheme A must reproduce the retained text reference and the earlier best AV model.
    keys = ["seed", "fold", "participant_id"]
    for run, old_name, new_name in [
        ("baseline_v2_20261105T071535380455Z", "joint", "text_A"),
        ("participant_pooled_20261112T071357784484Z", "segment_concat", "av_A"),
    ]:
        history = pd.read_parquet(project / "experiments/runs" / run / "predictions.parquet")
        old = history[history.condition == old_name].set_index(keys).sort_index()
        new = pred[pred.condition == new_name].set_index(keys).sort_index()
        pd.testing.assert_index_equal(old.index, new.index)
        np.testing.assert_allclose(old.score, new.score, atol=1e-12, rtol=1e-12)
    # Gender balance achieved by scheme B (descriptive check that it is applied).
    spread = {}
    for scheme in ["A", "B"]:
        shares = [
            (gender.iloc[b] == "female").mean()
            for seed in SEEDS
            for _, b in outer_splits(labels, gender, scheme, seed)
        ]
        spread[scheme] = float(np.std(shares))
    assert spread["B"] <= spread["A"]
    assert set(domain(labels.index)) == {"WoZ", "AI"}
    result = {
        "fold_bundles_restored_and_refitted": restored,
        "prediction_rows": len(pred),
        "maximum_restored_score_error": maximum_error,
        "combiners_and_gate_recomputed": True,
        "text_reference_and_segment_concat_reproduced_in_scheme_A": True,
        "female_share_sd_across_folds": spread,
        "test_evaluated": False,
    }
    (output / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result
