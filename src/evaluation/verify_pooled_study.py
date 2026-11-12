"""Independently check the participant-speech / pooled-cohort / text-assisted run."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from src.evaluation.verify_audio_visual import check_preprocessing
from src.models.sparse_fusion import branch_scores, fit_branch
from src.training.checkpoints import load_checkpoint
from src.training.classical import SEEDS
from src.training.pooled_study import (
    P217,
    SPEC,
    T161,
    WEIGHTS,
    comparison_pairs,
    load_inputs,
    text_av_audit,
)
from src.training.provenance import matches_hash
from src.training.splits import splits
from src.training.validation import sha256


def verify(project, output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["test_evaluated"]
    for name, digest in manifest["hashes"].items():
        assert matches_hash(project / name, digest), name
    for name, digest in manifest["models"].items():
        assert sha256(output / "models" / name) == digest, name
    data = load_inputs(project)
    official = pd.read_parquet(project / "data/processed/labels.parquet")
    for cohort in ["t161", "p217"]:
        labels = data[cohort][1]
        assert not (official.loc[labels.index, "split"] == "test").any()
        np.testing.assert_array_equal(labels, official.loc[labels.index, "PHQ_Binary"])
    pred = pd.read_parquet(output / "predictions.parquet")
    audits = json.loads((output / "folds.json").read_text())
    np.testing.assert_array_equal(pred.prediction, pred.score >= 0)
    assert not pred.duplicated(["condition", "seed", "participant_id"]).any()
    restored = fused = 0
    maximum_error = 0.0
    for cohort, policies in [("t161", T161), ("p217", P217)]:
        features, labels = data[cohort]
        np.testing.assert_array_equal(
            pred[pred.condition.isin(policies)].target,
            labels.loc[pred[pred.condition.isin(policies)].participant_id],
        )
        for seed in SEEDS:
            for fold, (a, b) in enumerate(splits(labels, 5, seed)):
                audit = audits[f"{cohort}_{seed}_{fold}"]
                assert audit["fit_ids"] == list(labels.index[a])
                assert audit["heldout_ids"] == list(labels.index[b])
                assert not set(audit["fit_ids"]) & set(audit["heldout_ids"])

                def saved(name):
                    rows = pred[(pred.condition == name) & (pred.seed == seed) & (pred.fold == fold)]
                    return rows.set_index("participant_id").loc[labels.index[b]].score.to_numpy()

                margins = {}
                for name, frame in policies.items():
                    bundle = load_checkpoint(output / "models" / f"{name}_{seed}_{fold}.joblib")
                    assert bundle["input"] == frame and bundle["model"]["spec"] == SPEC
                    check_preprocessing(bundle["model"]["preprocess"], features[frame].iloc[a])
                    scores = branch_scores(bundle["model"], features[frame].iloc[b])
                    refit = fit_branch(features[frame].iloc[a], labels.iloc[a], SPEC, seed)
                    np.testing.assert_allclose(
                        branch_scores(refit, features[frame].iloc[b]), scores, atol=1e-12
                    )
                    np.testing.assert_allclose(scores, saved(name), atol=1e-12)
                    maximum_error = max(maximum_error, float(np.max(abs(scores - saved(name)))))
                    margins[name] = scores
                    restored += 1
                if cohort == "t161":
                    text_model = load_checkpoint(output / "models" / f"text_{seed}_{fold}.joblib")
                    text = text_model.decision_function(features["text"].iloc[b])
                    np.testing.assert_allclose(text, saved("text"), atol=1e-12)
                    train = {k: features[k].iloc[a] for k in ["text", "concat"]}
                    replay = text_av_audit(
                        train, labels.iloc[a], seed, fold, audit["text_av"]["text_parameters"]
                    )
                    recorded = audit["text_av"]
                    np.testing.assert_allclose(recorded["scales"], replay["scales"], rtol=1e-12)
                    f1s = [r["inner_f1"] for r in recorded["weight_candidates"]]
                    assert [r["weight"] for r in recorded["weight_candidates"]] == WEIGHTS
                    assert recorded["weight"] == WEIGHTS[int(np.argmax(f1s))] == replay["weight"]
                    st, sa = recorded["scales"]
                    for name, weight in [("text_av_equal", 0.5), ("text_av_weighted", recorded["weight"])]:
                        independent = weight * text / st + (1 - weight) * margins["original_concat"] / sa
                        np.testing.assert_allclose(independent, saved(name), atol=1e-10)
                        fused += 1
                    restored += 1
    # Metrics recomputed from predictions.
    metrics = pd.read_csv(output / "repeat_metrics.csv").set_index(["condition", "seed"])
    for (name, seed), g in pred.groupby(["condition", "seed"]):
        tn, fp, fn, tp = confusion_matrix(g.target, g.prediction, labels=[0, 1]).ravel()
        row = metrics.loc[(name, seed)]
        np.testing.assert_allclose(row.f1, 2 * tp / max(1, 2 * tp + fp + fn), atol=1e-12)
        np.testing.assert_allclose(row.accuracy, (tp + tn) / len(g), atol=1e-12)
        np.testing.assert_allclose(
            row.balanced_accuracy, (tp / (tp + fn) + tn / (tn + fp)) / 2, atol=1e-12
        )
        aucs = [roc_auc_score(f.target, f.score) for _, f in g.groupby("fold")]
        np.testing.assert_allclose(row.mean_fold_auc, np.mean(aucs), atol=1e-12)
    comparisons = json.loads((output / "comparison.json").read_text())
    assert [(r["candidate"], r["control"]) for r in comparisons] == comparison_pairs()
    for row in comparisons:
        numeric = metrics.select_dtypes("number")
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
    # Restricted P217 table and abstention table recomputed.
    train_ids = set(data["t161"][1].index)
    restricted = pd.read_csv(output / "p217_on_train161.csv").set_index(["condition", "seed"])
    subset = pred[pred.condition.str.endswith("_p217") & pred.participant_id.isin(train_ids)]
    for (name, seed), g in subset.groupby(["condition", "seed"]):
        np.testing.assert_allclose(
            restricted.loc[(name, seed), "f1"], f1_score(g.target, g.prediction), atol=1e-12
        )
    abstain = pd.read_csv(output / "abstention.csv")
    for _, row in abstain[abstain.subset.str.startswith("top_")].iterrows():
        g = pred[(pred.condition == row.condition) & (pred.seed == row.seed)]
        order = g.reindex(g.score.abs().sort_values(ascending=False, kind="stable").index)
        kept = order.iloc[: int(round(int(row.subset[4:]) / 100 * len(order)))]
        assert len(kept) == row.n
        np.testing.assert_allclose(row.accuracy, (kept.target == kept.prediction).mean(), atol=1e-12)
    # Historical controls reproduced exactly.
    keys = ["seed", "fold", "participant_id"]
    for run, old_name, new_name in [
        ("branch_audit_v6_20261104T202915295134Z", "original_concat", "original_concat"),
        ("baseline_v2_20261105T071535380455Z", "joint", "text"),
    ]:
        history = pd.read_parquet(project / "experiments/runs" / run / "predictions.parquet")
        old = history[history.condition == old_name].set_index(keys).sort_index()
        new = pred[pred.condition == new_name].set_index(keys).sort_index()
        pd.testing.assert_index_equal(old.index, new.index)
        np.testing.assert_allclose(old.score, new.score, atol=1e-12, rtol=1e-12)
    result = {
        "branch_models_restored_and_refitted": restored,
        "prediction_rows": len(pred),
        "text_av_fusions_recomputed": fused,
        "maximum_restored_score_error": maximum_error,
        "metrics_comparisons_abstention_recomputed": True,
        "historical_concat_and_text_reference_reproduced": True,
        "test_evaluated": False,
    }
    (output / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result
