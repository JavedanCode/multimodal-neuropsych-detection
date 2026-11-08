"""PHQ-score target, eGeMAPS functionals and domain scaling (one fixed series).

Protocol: docs/methods/score_functionals_domain.md. The only inner-fold choice
is the ridge alpha, chosen on squared error of the released training score.
"""

from datetime import datetime, timezone
import json
import logging
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from src.dataset.egemaps_functionals import load as load_functionals
from src.dataset.inventory import load_split_labels
from src.evaluation.comparison import paired
from src.evaluation.predictions import save_predictions, summarize
from src.models.score_domain import ALPHAS, THRESHOLD, balanced_weights, fit_model, model_scores
from src.training.classical import SEEDS
from src.training.corrected_study import load_inputs as load_corrected
from src.training.provenance import source_hashes
from src.training.splits import splits
from src.training.validation import sha256

BOOTSTRAP_SEED = 20261108
BOOTSTRAP_RESAMPLES = 2000
POLICIES = {
    "original_audio": ("audio", "binary", False),
    "voiced_audio": ("voiced", "binary", False),
    "func_audio": ("func", "binary", False),
    "face": ("face", "binary", False),
    "original_concat": ("concat", "binary", False),
    "voiced_concat": ("voiced_concat", "binary", False),
    "func_concat": ("func_concat", "binary", False),
    "voiced_concat_domain": ("voiced_concat", "binary", True),
    "func_concat_domain": ("func_concat", "binary", True),
    "score_voiced_concat": ("voiced_concat", "score", False),
    "score_func_concat": ("func_concat", "score", False),
    "score_func_concat_domain": ("func_concat", "score", True),
}
PRIMARY = [
    "func_concat",
    "voiced_concat_domain",
    "func_concat_domain",
    "score_voiced_concat",
    "score_func_concat",
    "score_func_concat_domain",
]


def load_phq(project, labels):
    frame, info = load_split_labels(project / "data/raw/train_split.csv")
    if info.duplicate_ids:
        raise ValueError("Duplicate training participants")
    frame = frame.set_index("Participant_ID").loc[labels.index]
    if not (frame["PHQ_Binary"].astype(int) == labels).all():
        raise ValueError("Released binary labels changed")
    phq = frame["PHQ_Score"].astype(float)
    if phq.isna().any() or not phq.between(0, 24).all():
        raise ValueError("Released PHQ score missing or out of range")
    return phq


def load_inputs(project):
    features, labels, quality, available = load_corrected(project)
    features["func"] = load_functionals(project, labels)
    features["func_concat"] = pd.concat([features["func"], features["face"]], axis=1)
    phq = load_phq(project, labels)
    for frame in features.values():
        assert frame.index.equals(labels.index)
    return features, labels, phq, quality, available


def select_alpha(frame, labels, phq, use_domain, seed, fold):
    """Pooled inner-OOF weighted squared error on the training score only."""
    inner = splits(labels, 3, seed + fold + 1)
    weights = balanced_weights(labels)
    records = []
    for alpha in ALPHAS:
        spec = {"target": "score", "domain": use_domain, "alpha": alpha}
        predictions = np.full(len(labels), np.nan)
        for fitting, validation in inner:
            model = fit_model(
                frame.iloc[fitting], labels.iloc[fitting], phq.iloc[fitting], spec, seed
            )
            predictions[validation] = model_scores(model, frame.iloc[validation])
        error = (predictions - (phq.to_numpy() - THRESHOLD)) ** 2
        records.append(
            {
                "alpha": alpha,
                "weighted_mse": float(np.sum(weights * error) / np.sum(weights)),
                "scores": predictions.tolist(),
            }
        )
    best = min(records, key=lambda r: r["weighted_mse"])["alpha"]
    return best, records


def inner_audit(features, labels, phq, seed, fold):
    choices, records = {}, {}
    for name, (frame, target, use_domain) in POLICIES.items():
        if target == "score":
            choices[name], records[name] = select_alpha(
                features[frame], labels, phq, use_domain, seed, fold
            )
    return {
        "alphas": choices,
        "alpha_candidates": records,
        "inner_ids": [
            {"fit": list(map(int, labels.index[a])), "heldout": list(map(int, labels.index[b]))}
            for a, b in splits(labels, 3, seed + fold + 1)
        ],
    }


def spec_for(name, audit):
    _, target, use_domain = POLICIES[name]
    return {
        "target": target,
        "domain": use_domain,
        "alpha": audit["alphas"].get(name) if target == "score" else None,
    }


def fit_policies(features, labels, phq, audit, seed):
    return {
        name: {
            "input": POLICIES[name][0],
            "model": fit_model(features[POLICIES[name][0]], labels, phq, spec_for(name, audit), seed),
        }
        for name in POLICIES
    }


def comparison_pairs():
    return [(name, "original_concat") for name in PRIMARY] + [
        ("func_audio", "original_audio"),
        ("func_audio", "voiced_audio"),
        ("func_concat", "voiced_concat"),
        ("voiced_concat_domain", "voiced_concat"),
        ("func_concat_domain", "func_concat"),
        ("score_voiced_concat", "voiced_concat"),
        ("score_func_concat", "func_concat"),
        ("score_func_concat_domain", "score_func_concat"),
    ]


def bootstrap_draws(labels):
    """Row positions (into labels.index) of label-stratified resamples."""
    random = np.random.default_rng(BOOTSTRAP_SEED)
    y = labels.to_numpy()
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.stack(
        [
            np.concatenate(
                [random.choice(positives, len(positives)), random.choice(negatives, len(negatives))]
            )
            for _ in range(BOOTSTRAP_RESAMPLES)
        ]
    )


def _f1(y, predicted):
    tp = (y & predicted).sum(1)
    return 2 * tp / np.maximum(1, 2 * tp + (~y & predicted).sum(1) + (y & ~predicted).sum(1))


def _auc(y, scores):
    """Mann-Whitney AUC per row, ties counted as one half."""
    ranks = rankdata(scores, axis=1)
    n_pos = y.sum(1)
    n_neg = y.shape[1] - n_pos
    return ((ranks * y).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap(pred, labels):
    """Paired label-stratified participant bootstrap of mean-over-repeat deltas.

    F1 at the fixed threshold, and AUC pooled over each repeat's held-out folds.
    """
    ids = labels.index.to_numpy()
    draws = bootstrap_draws(labels)
    y = labels.to_numpy().astype(bool)[draws]
    table = {
        (c, s): g.set_index("participant_id").loc[ids]
        for (c, s), g in pred.groupby(["condition", "seed"])
    }

    def deltas(candidate, control):
        f1s, aucs = [], []
        for seed in SEEDS:
            a, b = table[(candidate, seed)], table[(control, seed)]
            f1s.append(
                _f1(y, a.prediction.to_numpy().astype(bool)[draws])
                - _f1(y, b.prediction.to_numpy().astype(bool)[draws])
            )
            aucs.append(
                _auc(y, a.score.to_numpy()[draws]) - _auc(y, b.score.to_numpy()[draws])
            )
        return np.mean(f1s, 0), np.mean(aucs, 0)

    results = []
    for candidate, control in [(n, "original_concat") for n in PRIMARY]:
        f1_delta, auc_delta = deltas(candidate, control)
        results.append(
            {
                "candidate": candidate,
                "control": control,
                "resamples": BOOTSTRAP_RESAMPLES,
                "f1_delta_ci95": np.percentile(f1_delta, [2.5, 97.5]).tolist(),
                "f1_delta_fraction_positive": float(np.mean(f1_delta > 0)),
                "pooled_auc_delta_ci95": np.percentile(auc_delta, [2.5, 97.5]).tolist(),
                "pooled_auc_delta_fraction_positive": float(np.mean(auc_delta > 0)),
                "f1_deltas": f1_delta.tolist(),
                "pooled_auc_deltas": auc_delta.tolist(),
            }
        )
    return results


def run(project):
    features, labels, phq, quality, available = load_inputs(project)
    output = project / "experiments/runs" / (
        "score_functionals_domain_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    (output / "models").mkdir(parents=True)
    hashes = source_hashes(project)
    for path in [
        "data/raw/train_split.csv",
        "data/processed/features.parquet",
        "data/processed/labels.parquet",
        "data/processed/aligned_v2/manifest.json",
        "data/processed/representation_study_v1/manifest.json",
        "data/processed/voiced_audio_v1/features.parquet",
        "data/processed/egemaps_functionals_v1/features.parquet",
        "data/processed/egemaps_functionals_v1/manifest.json",
    ]:
        hashes[path] = sha256(project / path)
    (output / "protocol.json").write_text(json.dumps({"hashes": hashes, "test_evaluated": False}, indent=2))
    predictions, average_precisions, regression = [], [], []
    for seed in SEEDS:
        for fold, (fitting, heldout) in enumerate(splits(labels, 5, seed)):
            logging.info("Score/functionals/domain seed %s fold %s", seed, fold)
            train = {n: v.iloc[fitting] for n, v in features.items()}
            valid = {n: v.iloc[heldout] for n, v in features.items()}
            audit = inner_audit(train, labels.iloc[fitting], phq.iloc[fitting], seed, fold)
            policies = fit_policies(train, labels.iloc[fitting], phq.iloc[fitting], audit, seed)
            held = labels.iloc[heldout]
            for name, bundle in policies.items():
                scores = model_scores(bundle["model"], valid[bundle["input"]])
                joblib.dump(bundle, output / "models" / f"{name}_{seed}_{fold}.joblib")
                save_predictions(predictions, name, "clean", seed, fold, held, scores)
                average_precisions.append(
                    {"condition": name, "seed": seed, "fold": fold,
                     "average_precision": average_precision_score(held, scores)}
                )
                if POLICIES[name][1] == "score":
                    truth = phq.iloc[heldout].to_numpy()
                    predicted = scores + THRESHOLD
                    regression.append(
                        {
                            "condition": name, "seed": seed, "fold": fold,
                            "alpha": audit["alphas"][name],
                            "rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
                            "mae": float(np.mean(np.abs(predicted - truth))),
                            "pearson_r": float(np.corrcoef(predicted, truth)[0, 1]),
                        }
                    )
            for name, value in [("always_negative", -1.0), ("always_positive", 1.0)]:
                scores = np.full(len(heldout), value)
                save_predictions(predictions, name, "clean", seed, fold, held, scores)
                average_precisions.append(
                    {"condition": name, "seed": seed, "fold": fold,
                     "average_precision": average_precision_score(held, scores)}
                )
            audit.update(
                fit_ids=list(map(int, labels.index[fitting])),
                heldout_ids=list(map(int, labels.index[heldout])),
            )
            (output / f"fold_{seed}_{fold}.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    frame = pd.DataFrame(predictions)
    frame.to_parquet(output / "predictions.parquet", index=False)
    metrics = summarize(frame, output)
    pd.DataFrame(average_precisions).to_csv(output / "fold_average_precision.csv", index=False)
    pd.DataFrame(regression).to_csv(output / "score_regression.csv", index=False)
    disagreement = labels.index[(phq >= 10).to_numpy() & (labels == 0).to_numpy()]
    rows = frame[frame.participant_id.isin(disagreement)]
    rows.groupby("condition").prediction.mean().rename("fraction_predicted_positive").to_csv(
        output / "score_label_disagreements.csv"
    )
    (output / "comparison.json").write_text(
        json.dumps([paired(metrics, a, b) for a, b in comparison_pairs()], indent=2)
    )
    (output / "bootstrap.json").write_text(json.dumps(bootstrap(frame, labels), indent=2))
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    for path, digest in hashes.items():
        assert sha256(project / path) == digest, path
    manifest = {
        "study": "score_functionals_domain",
        "hashes": hashes,
        "models": {p.name: sha256(p) for p in (output / "models").glob("*.joblib")},
        "participants": len(labels),
        "positives": int(labels.sum()),
        "score_label_disagreements": len(disagreement),
        "test_evaluated": False,
        "development_evaluated": False,
        "feature_counts": {name: f.shape[1] for name, f in features.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output
