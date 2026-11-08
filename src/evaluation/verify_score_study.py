"""Independently check the PHQ-score / functionals / domain-scaling run."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.dataset.contracts import domain
from src.evaluation.verify_audio_visual import check_metrics, check_preprocessing
from src.models.score_domain import ALPHAS, SHRINKAGE, THRESHOLD
from src.training.checkpoints import load_checkpoint
from src.training.classical import SEEDS
from src.training.provenance import matches_hash
from src.training.score_study import (
    POLICIES,
    bootstrap,
    bootstrap_draws,
    comparison_pairs,
    fit_policies,
    inner_audit,
    load_inputs,
)
from src.models.score_domain import model_scores
from src.training.splits import splits
from src.training.validation import sha256

BOOTSTRAP_CHECKS = 25


def independent_domain_values(model, frame):
    """Recompute domain-standardized values from training rows, without DomainScaler."""
    values = model["preprocess"].transform(frame)
    if model["domain_scaler"] is None:
        return values
    groups = domain(frame.index)
    result = np.empty_like(values)
    for name, (mean, scale, n) in model["domain_scaler"].stats_.items():
        rows = groups == name
        result[rows] = (values[rows] - mean) / scale
    return result


def check_domain_stats(model, train):
    values = model["preprocess"].transform(train)
    groups = domain(train.index)
    for name, (mean, scale, n) in model["domain_scaler"].stats_.items():
        rows = values[groups == name]
        assert n == len(rows)
        np.testing.assert_allclose(mean, len(rows) * rows.mean(0) / (len(rows) + SHRINKAGE), atol=1e-12)
        np.testing.assert_allclose(
            scale ** 2, (len(rows) * rows.var(0) + SHRINKAGE) / (len(rows) + SHRINKAGE), atol=1e-12
        )


def check_ridge(model, train, labels, phq):
    """Closed-form weighted ridge with unpenalized intercept."""
    x = independent_domain_values(model, train)
    y = phq.to_numpy(float) - THRESHOLD
    counts = np.bincount(labels.to_numpy(int), minlength=2)
    w = len(labels) / (2.0 * counts[labels.to_numpy(int)])
    xm = (w[:, None] * x).sum(0) / w.sum()
    ym = (w * y).sum() / w.sum()
    xc, yc = x - xm, y - ym
    alpha = model["spec"]["alpha"]
    beta = np.linalg.solve(xc.T @ (w[:, None] * xc) + alpha * np.eye(x.shape[1]), xc.T @ (w * yc))
    np.testing.assert_allclose(model["estimator"].coef_, beta, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(model["estimator"].intercept_, ym - xm @ beta, atol=1e-7)


def verify(project, output):
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["test_evaluated"] and not manifest["development_evaluated"]
    for name, digest in manifest["hashes"].items():
        assert matches_hash(project / name, digest), name
    for name, digest in manifest["models"].items():
        assert sha256(output / "models" / name) == digest, name
    features, labels, phq, quality, available = load_inputs(project)
    pred = pd.read_parquet(output / "predictions.parquet")
    assert len(pred) == len(labels) * len(SEEDS) * (len(POLICIES) + 2)
    assert not pred.duplicated(["condition", "seed", "participant_id"]).any()
    np.testing.assert_array_equal(pred.target, labels.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction, pred.score >= 0)
    metrics = check_metrics(pred, labels, output)
    restored = alphas_checked = ridge_checked = domain_checked = 0
    maximum_error = 0.0
    for seed in SEEDS:
        for fold, (a, b) in enumerate(splits(labels, 5, seed)):
            train = {n: f.iloc[a] for n, f in features.items()}
            valid = {n: f.iloc[b] for n, f in features.items()}
            audit = json.loads((output / f"fold_{seed}_{fold}.json").read_text())
            assert audit["fit_ids"] == list(labels.index[a])
            assert audit["heldout_ids"] == list(labels.index[b])
            assert not set(audit["fit_ids"]) & set(audit["heldout_ids"])
            repeated = inner_audit(train, labels.iloc[a], phq.iloc[a], seed, fold)
            assert audit["inner_ids"] == repeated["inner_ids"]
            target = phq.iloc[a].to_numpy() - THRESHOLD
            counts = np.bincount(labels.iloc[a].to_numpy(int), minlength=2)
            w = len(a) / (2.0 * counts[labels.iloc[a].to_numpy(int)])
            for name, records in audit["alpha_candidates"].items():
                assert [r["alpha"] for r in records] == ALPHAS
                errors = []
                for record, replay in zip(records, repeated["alpha_candidates"][name], strict=True):
                    np.testing.assert_allclose(record["scores"], replay["scores"], atol=1e-10)
                    mse = float(np.sum(w * (np.asarray(record["scores"]) - target) ** 2) / w.sum())
                    np.testing.assert_allclose(record["weighted_mse"], mse, rtol=1e-12)
                    errors.append(mse)
                assert audit["alphas"][name] == ALPHAS[int(np.argmin(errors))] == repeated["alphas"][name]
                alphas_checked += 1
            policies = fit_policies(train, labels.iloc[a], phq.iloc[a], repeated, seed)
            assert set(pred.condition) == set(policies) | {"always_negative", "always_positive"}
            for name, refit in policies.items():
                bundle = load_checkpoint(output / "models" / f"{name}_{seed}_{fold}.joblib")
                frame = bundle["input"]
                assert frame == POLICIES[name][0]
                model = bundle["model"]
                saved = (
                    pred[(pred.condition == name) & (pred.seed == seed) & (pred.fold == fold)]
                    .set_index("participant_id")
                    .loc[labels.index[b]]
                )
                scores = model_scores(model, valid[frame])
                np.testing.assert_allclose(scores, saved.score, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(
                    model_scores(refit["model"], valid[frame]), scores, atol=1e-12, rtol=1e-12
                )
                preprocess = model["branch"]["preprocess"] if "branch" in model else model["preprocess"]
                check_preprocessing(preprocess, train[frame])
                if "branch" not in model:
                    if model["domain_scaler"] is not None:
                        check_domain_stats(model, train[frame])
                        domain_checked += 1
                    values = independent_domain_values(model, valid[frame])
                    if model["spec"]["target"] == "score":
                        check_ridge(model, train[frame], labels.iloc[a], phq.iloc[a])
                        independent = values @ model["estimator"].coef_ + model["estimator"].intercept_
                        ridge_checked += 1
                    else:
                        independent = model["estimator"].decision_function(values)
                    np.testing.assert_allclose(independent, scores, atol=1e-9)
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
    saved_bootstrap = json.loads((output / "bootstrap.json").read_text())
    assert saved_bootstrap == json.loads(json.dumps(bootstrap(pred, labels)))
    # Independent sklearn cross-check of the vectorized bootstrap on early resamples.
    draws = bootstrap_draws(labels)
    ids = labels.index.to_numpy()
    for record in saved_bootstrap:
        assert record["resamples"] == len(record["f1_deltas"]) == len(draws)
        assert record["f1_delta_ci95"] == np.percentile(record["f1_deltas"], [2.5, 97.5]).tolist()
        for i in range(BOOTSTRAP_CHECKS):
            rows = ids[draws[i]]
            f1s, aucs = [], []
            for seed in SEEDS:
                a, b = (
                    pred[(pred.condition == c) & (pred.seed == seed)].set_index("participant_id").loc[rows]
                    for c in (record["candidate"], record["control"])
                )
                f1s.append(f1_score(a.target, a.prediction) - f1_score(b.target, b.prediction))
                aucs.append(roc_auc_score(a.target, a.score) - roc_auc_score(b.target, b.score))
            np.testing.assert_allclose(record["f1_deltas"][i], np.mean(f1s), atol=1e-12)
            np.testing.assert_allclose(record["pooled_auc_deltas"][i], np.mean(aucs), atol=1e-12)
    # Regression error and score/label disagreement tables, recomputed from predictions.
    regression = pd.read_csv(output / "score_regression.csv")
    expected = []
    for (name, seed, fold), g in pred[pred.condition.str.startswith("score_")].groupby(
        ["condition", "seed", "fold"]
    ):
        predicted = g.score.to_numpy() + THRESHOLD
        truth = phq.loc[g.participant_id].to_numpy()
        audit = json.loads((output / f"fold_{seed}_{fold}.json").read_text())
        expected.append(
            {
                "condition": name, "seed": seed, "fold": fold,
                "alpha": audit["alphas"][name],
                "rmse": np.sqrt(np.mean((predicted - truth) ** 2)),
                "mae": np.mean(np.abs(predicted - truth)),
                "pearson_r": np.corrcoef(predicted, truth)[0, 1],
            }
        )
    expected = pd.DataFrame(expected).set_index(["condition", "seed", "fold"]).sort_index()
    saved_regression = regression.set_index(["condition", "seed", "fold"]).sort_index()
    pd.testing.assert_index_equal(expected.index, saved_regression.index)
    np.testing.assert_allclose(saved_regression[expected.columns].values, expected.values, atol=1e-12)
    disagreeing = labels.index[(phq.to_numpy() >= 10) & (labels.to_numpy() == 0)]
    assert manifest["score_label_disagreements"] == len(disagreeing) == 12
    table = pd.read_csv(output / "score_label_disagreements.csv", index_col=0).iloc[:, 0]
    recomputed = pred[pred.participant_id.isin(disagreeing)].groupby("condition").prediction.mean()
    pd.testing.assert_series_equal(table.sort_index(), recomputed.sort_index(), check_names=False)
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
        "alpha_choices_recomputed_from_saved_scores": alphas_checked,
        "ridge_closed_form_checks": ridge_checked,
        "domain_statistics_checks": domain_checked,
        "maximum_restored_score_error": maximum_error,
        "bootstrap_reproduced": True,
        "historical_concat_reproduced": True,
        "test_evaluated": False,
    }
    (output / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result
