"""Stacked text + voice + face fusion under three outer fold schemes.

Protocol: docs/methods/stacked_fusion.md. Training cohort only.
"""

from datetime import datetime, timezone
import json
import logging
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.dataset.contracts import domain
from src.dataset.inventory import load_split_labels
from src.evaluation.comparison import paired
from src.evaluation.predictions import save_predictions, summarize
from src.experiments.reference import fit_fold as fit_text
from src.models.sparse_fusion import branch_scores, fit_branch, margin_scale
from src.models.text_preprocessing import TextPreprocessor
from src.training.classical import SEEDS, estimator, metric_row
from src.training.pooled_study import load_inputs as load_pooled
from src.training.provenance import source_hashes
from src.training.score_study import _auc, _f1
from src.training.splits import splits
from src.training.validation import sha256

SPEC = {"selection": "all", "C": 0.1}
SCHEMES = ["A", "B", "C"]
BRANCHES = {"voice": "segment", "face": "face", "av": "segment_concat"}
STACKS = {"stack3": ["text", "voice", "face"], "stack2": ["text", "av"]}
FUSIONS = ["stack3", "stack2", "gate3"]
BOOTSTRAP_SEED = 20261112
BOOTSTRAP_RESAMPLES = 2000
COVERAGES = [1.0, 0.8, 0.6, 0.4]


def load_inputs(project):
    features, labels = load_pooled(project)["t161"]
    frame, _ = load_split_labels(project / "data/raw/train_split.csv")
    gender = frame.set_index("Participant_ID").loc[labels.index, "Gender"].astype(str)
    gender = gender.str.strip().str.lower()
    if gender.isna().any():
        raise ValueError("Gender required for fold assignment")
    keep = {name: features[name] for name in ["text", "segment", "face", "segment_concat"]}
    return keep, labels, gender


def outer_splits(labels, gender, scheme, seed):
    """Outer folds; gender only assigns folds and is never a model input."""
    if scheme == "A":
        return splits(labels, 5, seed)
    base = np.array([f"{d}_{int(t)}" for d, t in zip(domain(labels.index), labels)])
    if scheme == "B":
        n_splits = 5
        strata = np.array([f"{b}_{g}" for b, g in zip(base, gender)])
        names, counts = np.unique(strata, return_counts=True)
        # If any gender stratum of a label x domain group is too small, the
        # whole group falls back to label x domain.
        rare = {name.rsplit("_", 1)[0] for name in names[counts < n_splits]}
        strata = np.where(np.isin(base, list(rare)), base, strata)
    elif scheme == "C":
        n_splits = 10
        names, counts = np.unique(base, return_counts=True)
        rare = set(names[counts < n_splits])
        merged = np.array([f"any_{int(t)}" for t in labels])
        # A rare stratum joins every same-label participant in one stratum.
        rare_labels = {name.split("_")[1] for name in rare}
        strata = np.where(np.isin([b.split("_")[1] for b in base], list(rare_labels)), merged, base)
    else:
        raise ValueError("Unknown fold scheme")
    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < n_splits:
        raise ValueError("Insufficient people for stratification")
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return list(folds.split(np.zeros(len(labels)), strata))


def fit_fusion(features, labels, seed, fold):
    """Fit branches and combiners on these (outer-training) participants only."""
    text_model, audit = fit_text(features["text"], labels, np.arange(len(labels)), [], seed, fold)
    parameters = audit["selected"]["parameters"]
    inner = splits(labels, 3, seed + fold + 1)
    oof = {name: np.full(len(labels), np.nan) for name in ["text", *BRANCHES]}
    for fitting, validation in inner:
        target = labels.iloc[fitting]
        model = Pipeline(
            [("preprocess", TextPreprocessor()), ("classifier", estimator("linear", parameters, seed))]
        ).fit(features["text"].iloc[fitting], target)
        oof["text"][validation] = model.decision_function(features["text"].iloc[validation])
        for name, frame in BRANCHES.items():
            branch = fit_branch(features[frame].iloc[fitting], target, SPEC, seed)
            oof[name][validation] = branch_scores(branch, features[frame].iloc[validation])
    scales = {name: margin_scale(values) for name, values in oof.items()}
    stackers = {}
    for name, inputs in STACKS.items():
        design = np.column_stack([oof[i] / scales[i] for i in inputs])
        stackers[name] = LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000
        ).fit(design, labels)
    return {
        "text": text_model,
        "branches": {
            name: fit_branch(features[frame], labels, SPEC, seed) for name, frame in BRANCHES.items()
        },
        "scales": scales,
        "stackers": stackers,
        "gate": float(np.median(np.abs(oof["text"] / scales["text"]))),
        "text_parameters": parameters,
    }


def predict(bundle, features):
    """Zero-threshold scores for every policy on the given participants."""
    margins = {"text": bundle["text"].decision_function(features["text"])}
    for name, frame in BRANCHES.items():
        margins[name] = branch_scores(bundle["branches"][name], features[frame])
    scaled = {name: margins[name] / bundle["scales"][name] for name in margins}
    scores = {"text": margins["text"], "av": margins["av"]}
    for name, inputs in STACKS.items():
        scores[name] = bundle["stackers"][name].decision_function(
            np.column_stack([scaled[i] for i in inputs])
        )
    stacker = bundle["stackers"]["stack3"]
    text_only = stacker.intercept_[0] + stacker.coef_[0][0] * scaled["text"]
    confident = np.abs(scaled["text"]) >= bundle["gate"]
    scores["gate3"] = np.where(confident, text_only, scores["stack3"])
    return scores, confident


def comparison_pairs():
    return [(f"{name}_{scheme}", f"text_{scheme}") for scheme in SCHEMES for name in FUSIONS]


def bootstrap(pred, labels):
    random = np.random.default_rng(BOOTSTRAP_SEED)
    y = labels.to_numpy()
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = np.stack(
        [
            np.concatenate(
                [random.choice(positives, len(positives)), random.choice(negatives, len(negatives))]
            )
            for _ in range(BOOTSTRAP_RESAMPLES)
        ]
    )
    truth = y.astype(bool)[draws]
    table = {
        key: g.set_index("participant_id").loc[labels.index]
        for key, g in pred.groupby(["condition", "seed"])
    }
    results = []
    for candidate, control in comparison_pairs():
        f1s, aucs = [], []
        for seed in SEEDS:
            a, b = table[(candidate, seed)], table[(control, seed)]
            f1s.append(
                _f1(truth, a.prediction.to_numpy().astype(bool)[draws])
                - _f1(truth, b.prediction.to_numpy().astype(bool)[draws])
            )
            aucs.append(_auc(truth, a.score.to_numpy()[draws]) - _auc(truth, b.score.to_numpy()[draws]))
        f1_delta, auc_delta = np.mean(f1s, 0), np.mean(aucs, 0)
        results.append(
            {
                "candidate": candidate,
                "control": control,
                "resamples": BOOTSTRAP_RESAMPLES,
                "f1_delta_ci95": np.percentile(f1_delta, [2.5, 97.5]).tolist(),
                "f1_delta_fraction_positive": float(np.mean(f1_delta > 0)),
                "pooled_auc_delta_ci95": np.percentile(auc_delta, [2.5, 97.5]).tolist(),
                "pooled_auc_delta_fraction_positive": float(np.mean(auc_delta > 0)),
            }
        )
    return results


def abstention(pred):
    rows = []
    for name in ["text_A", *[f"{f}_A" for f in FUSIONS]]:
        for seed, g in pred[pred.condition == name].groupby("seed"):
            order = g.reindex(g.score.abs().sort_values(ascending=False, kind="stable").index)
            for coverage in COVERAGES:
                kept = order.iloc[: int(round(coverage * len(order)))]
                rows.append(
                    {"condition": name, "seed": seed, "subset": f"top_{int(coverage * 100)}",
                     "n": len(kept), "positives": int(kept.target.sum()),
                     **metric_row(kept.target, kept.prediction)}
                )
    return pd.DataFrame(rows)


def run(project):
    features, labels, gender = load_inputs(project)
    output = project / "experiments/runs" / (
        "stacked_fusion_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    (output / "models").mkdir(parents=True)
    hashes = source_hashes(project)
    for path in [
        "data/raw/train_split.csv",
        "data/processed/features.parquet",
        "data/processed/labels.parquet",
        "data/processed/participant_speech_v1/voiced.parquet",
        "data/processed/mpnet_v1/features.parquet",
    ]:
        hashes[path] = sha256(project / path)
    predictions, precisions, audits, weights = [], [], {}, []
    for scheme in SCHEMES:
        for seed in SEEDS:
            for fold, (fitting, heldout) in enumerate(outer_splits(labels, gender, scheme, seed)):
                logging.info("Stacked fusion scheme %s seed %s fold %s", scheme, seed, fold)
                train = {name: frame.iloc[fitting] for name, frame in features.items()}
                valid = {name: frame.iloc[heldout] for name, frame in features.items()}
                bundle = fit_fusion(train, labels.iloc[fitting], seed, fold)
                joblib.dump(bundle, output / "models" / f"{scheme}_{seed}_{fold}.joblib")
                scores, confident = predict(bundle, valid)
                held = labels.iloc[heldout]
                scores["always_positive"] = np.full(len(held), 1.0)
                scores["always_negative"] = np.full(len(held), -1.0)
                for name, values in scores.items():
                    condition = f"{name}_{scheme}"
                    save_predictions(predictions, condition, "clean", seed, fold, held, values)
                    precisions.append(
                        {"condition": condition, "seed": seed, "fold": fold,
                         "average_precision": average_precision_score(held, values)}
                    )
                for name, inputs in STACKS.items():
                    stacker = bundle["stackers"][name]
                    weights.append(
                        {"scheme": scheme, "seed": seed, "fold": fold, "stacker": name,
                         "intercept": float(stacker.intercept_[0]),
                         **{f"coef_{i}": float(c) for i, c in zip(inputs, stacker.coef_[0])}}
                    )
                audits[f"{scheme}_{seed}_{fold}"] = {
                    "fit_ids": list(map(int, labels.index[fitting])),
                    "heldout_ids": list(map(int, held.index)),
                    "scales": bundle["scales"],
                    "gate": bundle["gate"],
                    "text_parameters": bundle["text_parameters"],
                    "heldout_text_confident_fraction": float(confident.mean()),
                }
    (output / "folds.json").write_text(json.dumps(audits, indent=2, allow_nan=False))
    frame = pd.DataFrame(predictions)
    frame.to_parquet(output / "predictions.parquet", index=False)
    metrics = summarize(frame, output)
    pd.DataFrame(precisions).to_csv(output / "fold_average_precision.csv", index=False)
    pd.DataFrame(weights).to_csv(output / "stacker_weights.csv", index=False)
    (output / "comparison.json").write_text(
        json.dumps([paired(metrics, a, b) for a, b in comparison_pairs()], indent=2)
    )
    (output / "bootstrap.json").write_text(json.dumps(bootstrap(frame, labels), indent=2))
    abstention(frame).to_csv(output / "abstention.csv", index=False)
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    for path, digest in hashes.items():
        assert sha256(project / path) == digest, path
    manifest = {
        "study": "stacked_fusion",
        "hashes": hashes,
        "models": {p.name: sha256(p) for p in (output / "models").glob("*.joblib")},
        "participants": len(labels),
        "positives": int(labels.sum()),
        "test_evaluated": False,
        "development_evaluated": False,
        "fold_schemes": {"A": "5-fold label x domain", "B": "5-fold label x domain x gender", "C": "10-fold label x domain"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output
