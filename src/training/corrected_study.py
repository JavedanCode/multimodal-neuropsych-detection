"""Corrected voicing summaries and common-scale late fusion (one fixed series).

Protocol: docs/methods/corrected_fusion.md. All branches use the matched
historical recipe (all features, balanced L2 logistic, C=.1); the only inner
fold work is estimating branch margin scales and choosing one fusion weight.
"""

from datetime import datetime, timezone
import json
import logging
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from src.dataset.voiced_audio import load as load_voiced
from src.evaluation.comparison import paired
from src.evaluation.predictions import save_predictions, summarize
from src.models.sparse_fusion import (
    branch_scores,
    fit_branch,
    fuse_scores,
    fusion_weights,
    margin_scale,
    predict_policy,
)
from src.training.classical import SEEDS
from src.training.fusion_study import load_inputs as load_original
from src.training.provenance import source_hashes
from src.training.splits import splits
from src.training.validation import sha256

SPEC = {"selection": "all", "C": 0.1}
WEIGHTS = [0.5, 0.25, 0.75]
SINGLE = {
    "original_audio": "audio",
    "voiced_audio": "voiced",
    "face": "face",
    "original_concat": "concat",
    "voiced_concat": "voiced_concat",
}


def load_inputs(project):
    features, labels, quality, available = load_original(project)
    voiced = load_voiced(project, labels)
    features["voiced"] = voiced
    features["voiced_concat"] = pd.concat([voiced, features["face"]], axis=1)
    for frame in features.values():
        assert frame.index.equals(labels.index)
    return features, labels, quality, available


def inner_audit(features, labels, quality, available, seed, fold):
    """Out-of-fold branch margins on outer-training people only."""
    inner = splits(labels, 3, seed + fold + 1)
    priors = np.empty(len(labels))
    margins = {name: np.full(len(labels), np.nan) for name in ["voiced", "face"]}
    for fitting, validation in inner:
        priors[validation] = float(labels.iloc[fitting].mean())
        for name in margins:
            model = fit_branch(features[name].iloc[fitting], labels.iloc[fitting], SPEC, seed)
            margins[name][validation] = branch_scores(model, features[name].iloc[validation])
    scales = [margin_scale(margins["voiced"]), margin_scale(margins["face"])]
    weight_records = []
    for weight in WEIGHTS:
        scores = fuse_scores(
            margins["voiced"], margins["face"], quality, available, weight, priors,
            scales=scales,
        )
        weight_records.append(
            {
                "weight": weight,
                "inner_f1": float(f1_score(labels, scores >= 0)),
                "scores": scores.tolist(),
            }
        )
    return {
        "scales": scales,
        "weight": max(weight_records, key=lambda r: r["inner_f1"])["weight"],
        "weight_candidates": weight_records,
        "inner_margins": {name: values.tolist() for name, values in margins.items()},
        "inner_ids": [
            {
                "fit": list(map(int, labels.index[a])),
                "heldout": list(map(int, labels.index[b])),
            }
            for a, b in inner
        ],
    }


def fit_policies(features, labels, audit, seed):
    models = {
        name: fit_branch(features[name], labels, SPEC, seed)
        for name in ["audio", "voiced", "face", "concat", "voiced_concat"]
    }
    policies = {
        policy: {"kind": "single", "input": name, "model": models[name]}
        for policy, name in SINGLE.items()
    }
    for policy, weight, reliability, scales in [
        ("late_equal_raw", 0.5, False, None),
        ("late_equal_scaled", 0.5, False, audit["scales"]),
        ("late_weighted_scaled", audit["weight"], False, audit["scales"]),
        ("late_quality_scaled", audit["weight"], True, audit["scales"]),
    ]:
        policies[policy] = {
            "kind": "fusion",
            "audio": models["voiced"],
            "face": models["face"],
            "audio_input": "voiced",
            "weight": weight,
            "reliability": reliability,
            "scales": scales,
            "prior": float(labels.mean()),
        }
    return policies


def predict(bundle, features, quality, available):
    if bundle["kind"] == "fusion":
        features = {"audio": features[bundle["audio_input"]], "face": features["face"]}
    return predict_policy(bundle, features, quality, available)


def comparison_pairs():
    return [
        (name, "original_concat")
        for name in [
            "voiced_concat",
            "late_equal_scaled",
            "late_weighted_scaled",
            "late_quality_scaled",
        ]
    ] + [
        ("voiced_audio", "original_audio"),
        ("late_equal_scaled", "late_equal_raw"),
        ("late_weighted_scaled", "late_equal_scaled"),
        ("late_quality_scaled", "late_weighted_scaled"),
        ("late_weighted_scaled", "face"),
        ("late_weighted_scaled", "voiced_audio"),
    ]


def run(project):
    features, labels, quality, available = load_inputs(project)
    output = (
        project
        / "experiments/runs"
        / ("corrected_fusion_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    (output / "models").mkdir(parents=True)
    hashes = source_hashes(project)
    for path in [
        "data/processed/features.parquet",
        "data/processed/labels.parquet",
        "data/processed/aligned_v2/manifest.json",
        "data/processed/representation_study_v1/manifest.json",
        "data/processed/voiced_audio_v1/manifest.json",
        "data/processed/voiced_audio_v1/features.parquet",
    ]:
        hashes[path] = sha256(project / path)
    (output / "protocol.json").write_text(
        json.dumps({"hashes": hashes, "test_evaluated": False}, indent=2)
    )
    predictions, average_precisions, weight_logs = [], [], []
    for seed in SEEDS:
        for fold, (fitting, heldout) in enumerate(splits(labels, 5, seed)):
            logging.info("Corrected fusion seed %s fold %s", seed, fold)
            train = {n: v.iloc[fitting] for n, v in features.items()}
            valid = {n: v.iloc[heldout] for n, v in features.items()}
            audit = inner_audit(
                train, labels.iloc[fitting], quality[fitting], available[fitting], seed, fold
            )
            policies = fit_policies(train, labels.iloc[fitting], audit, seed)
            held = labels.iloc[heldout]
            for name, bundle in policies.items():
                scores = predict(bundle, valid, quality[heldout], available[heldout])
                joblib.dump(bundle, output / "models" / f"{name}_{seed}_{fold}.joblib")
                save_predictions(predictions, name, "clean", seed, fold, held, scores)
                average_precisions.append(
                    {
                        "condition": name,
                        "seed": seed,
                        "fold": fold,
                        "average_precision": average_precision_score(held, scores),
                    }
                )
                if bundle["kind"] == "fusion":
                    weights = fusion_weights(
                        quality[heldout], available[heldout], bundle["weight"],
                        bundle["reliability"],
                    )
                    audio = branch_scores(bundle["audio"], valid["voiced"])
                    face = branch_scores(bundle["face"], valid["face"])
                    scales = bundle["scales"] or [1.0, 1.0]
                    signs = np.sign(scores)
                    weight_logs.append(
                        {
                            "condition": name,
                            "seed": seed,
                            "fold": fold,
                            "base_audio_weight": bundle["weight"],
                            "audio_scale": scales[0],
                            "face_scale": scales[1],
                            "heldout_audio_margin_rms": margin_scale(audio),
                            "heldout_face_margin_rms": margin_scale(face),
                            "mean_audio_weight": float(weights[:, 0].mean()),
                            "mean_face_weight": float(weights[:, 1].mean()),
                            "sign_agrees_audio": float(np.mean(signs == np.sign(audio))),
                            "sign_agrees_face": float(np.mean(signs == np.sign(face))),
                        }
                    )
            for name, value in [("always_negative", -1.0), ("always_positive", 1.0)]:
                scores = np.full(len(heldout), value)
                save_predictions(predictions, name, "clean", seed, fold, held, scores)
                average_precisions.append(
                    {
                        "condition": name,
                        "seed": seed,
                        "fold": fold,
                        "average_precision": average_precision_score(held, scores),
                    }
                )
            audit.update(
                fit_ids=list(map(int, labels.index[fitting])),
                heldout_ids=list(map(int, labels.index[heldout])),
            )
            (output / f"fold_{seed}_{fold}.json").write_text(
                json.dumps(audit, indent=2, allow_nan=False)
            )
    frame = pd.DataFrame(predictions)
    frame.to_parquet(output / "predictions.parquet", index=False)
    metrics = summarize(frame, output)
    pd.DataFrame(average_precisions).to_csv(output / "fold_average_precision.csv", index=False)
    pd.DataFrame(weight_logs).to_csv(output / "fusion_weights.csv", index=False)
    (output / "comparison.json").write_text(
        json.dumps([paired(metrics, a, b) for a, b in comparison_pairs()], indent=2)
    )
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    for path, digest in hashes.items():
        assert sha256(project / path) == digest, path
    manifest = {
        "study": "corrected_fusion",
        "hashes": hashes,
        "models": {p.name: sha256(p) for p in (output / "models").glob("*.joblib")},
        "participants": len(labels),
        "positives": int(labels.sum()),
        "test_evaluated": False,
        "development_evaluated": False,
        "feature_counts": {name: f.shape[1] for name, f in features.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output
