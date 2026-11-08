"""Bounded inner selection and execution for the sparse/late-fusion study."""

from datetime import datetime, timezone
import json
import logging
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from src.evaluation.comparison import paired
from src.evaluation.predictions import save_predictions, summarize
from src.models.sparse_fusion import (
    branch_scores,
    fit_branch,
    specifications,
    fuse_scores,
    fusion_weights,
    predict_policy,
)
from src.training.classical import SEEDS
from src.training.provenance import source_hashes
from src.training.splits import splits
from src.training.validation import sha256


def load_inputs(project):
    from src.experiments.audio_visual import load

    sets, labels, quality, available = load(project)
    features = {"audio": sets["original_audio"], "face": sets["original_face"]}
    features["concat"] = pd.concat([features["audio"], features["face"]], axis=1)
    return features, labels, quality, available


def select_candidates(features, labels, quality, available, seed, fold):
    """Keep every candidate score and choice, using inner held-out people only."""
    inner = splits(labels, 3, seed + fold + 1)
    inner_priors = np.empty(len(labels))
    for fitting, validation in inner:
        inner_priors[validation] = float(labels.iloc[fitting].mean())
    records, choices = {}, {}
    for name, frame in features.items():
        candidates = []
        for spec in specifications():
            scores = np.full(len(labels), np.nan)
            counts = []
            for fitting, validation in inner:
                model = fit_branch(
                    frame.iloc[fitting], labels.iloc[fitting], spec, seed
                )
                scores[validation] = branch_scores(model, frame.iloc[validation])
                counts.append(len(model["indices"]))
            assert np.isfinite(scores).all()
            candidates.append(
                {
                    "spec": spec,
                    "scores": scores.tolist(),
                    "inner_f1": float(f1_score(labels, scores >= 0)),
                    "inner_selected_counts": counts,
                }
            )
        records[name] = candidates
        choices[name + "_all"] = max(
            (r for r in candidates if r["spec"]["selection"] == "all"),
            key=lambda r: r["inner_f1"],
        )
        choices[name + "_sparse"] = max(
            (r for r in candidates if r["spec"]["selection"] != "all"),
            key=lambda r: r["inner_f1"],
        )
        choices["selected_" + name] = max(candidates, key=lambda r: r["inner_f1"])
    weight_records = []
    for weight in [0.5, 0.25, 0.75]:
        scores = fuse_scores(
            choices["selected_audio"]["scores"],
            choices["selected_face"]["scores"],
            quality,
            available,
            weight,
            inner_priors,
        )
        weight_records.append(
            {
                "weight": weight,
                "inner_f1": float(f1_score(labels, scores >= 0)),
                "scores": scores.tolist(),
            }
        )
    selected_weight = max(weight_records, key=lambda r: r["inner_f1"])["weight"]
    return {
        "candidates": records,
        "choices": {name: row["spec"] for name, row in choices.items()},
        "weight_candidates": weight_records,
        "weight": selected_weight,
        "inner_ids": [
            {
                "fit": list(map(int, labels.index[a])),
                "heldout": list(map(int, labels.index[b])),
            }
            for a, b in inner
        ],
    }


def fit_policies(features, labels, audit, seed):
    """Fit selected policies once; shared branch fits are reused by fusion controls."""
    fitted = {}

    def get_model(name, spec):
        key = (name, spec["selection"], spec["C"])
        if key not in fitted:
            fitted[key] = fit_branch(features[name], labels, spec, seed)
        return fitted[key]

    policies = {
        "original_concat": {
            "kind": "single",
            "input": "concat",
            "model": get_model("concat", {"selection": "all", "C": 0.1}),
        }
    }
    for name in features:
        for selection in ["all", "sparse"]:
            policy = name + "_" + selection
            policies[policy] = {
                "kind": "single",
                "input": name,
                "model": get_model(name, audit["choices"][policy]),
            }
    for name in ["audio", "face"]:
        policies["selected_" + name] = {
            "kind": "single",
            "input": name,
            "model": get_model(name, audit["choices"]["selected_" + name]),
        }
    for name, weight, reliability in [
        ("late_equal", 0.5, False),
        ("late_weighted", audit["weight"], False),
        ("late_quality", audit["weight"], True),
    ]:
        policies[name] = {
            "kind": "fusion",
            "audio": policies["selected_audio"]["model"],
            "face": policies["selected_face"]["model"],
            "weight": weight,
            "reliability": reliability,
            "prior": float(labels.mean()),
        }
    return policies


def comparison_pairs():
    return (
        [
            (name, "original_concat")
            for name in [
                "concat_all",
                "concat_sparse",
                "late_equal",
                "late_weighted",
                "late_quality",
            ]
        ]
        + [(name + "_sparse", name + "_all") for name in ["audio", "face", "concat"]]
        + [
            ("late_weighted", "late_equal"),
            ("late_quality", "late_weighted"),
            ("late_weighted", "selected_audio"),
            ("late_weighted", "selected_face"),
        ]
    )


def run(project):
    features, labels, quality, available = load_inputs(project)
    output = (
        project
        / "experiments/runs"
        / ("sparse_fusion_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    (output / "models").mkdir(parents=True)
    hashes = source_hashes(project)
    for path in [
        "data/processed/features.parquet",
        "data/processed/labels.parquet",
        "data/processed/aligned_v2/manifest.json",
        "data/processed/mpnet_v1/manifest.json",
        "data/processed/representation_study_v1/manifest.json",
        "data/processed/audio_study_v1/manifest.json",
    ]:
        hashes[path] = sha256(project / path)
    (output / "protocol.json").write_text(
        json.dumps({"hashes": hashes, "test_evaluated": False}, indent=2)
    )
    predictions, average_precisions, counts, weight_logs, timings = [], [], [], [], []
    for seed in SEEDS:
        for fold, (fitting, heldout) in enumerate(splits(labels, 5, seed)):
            logging.info("Sparse fusion seed %s fold %s", seed, fold)
            train = {name: values.iloc[fitting] for name, values in features.items()}
            valid = {name: values.iloc[heldout] for name, values in features.items()}
            started = time.perf_counter()
            audit = select_candidates(
                train,
                labels.iloc[fitting],
                quality[fitting],
                available[fitting],
                seed,
                fold,
            )
            selection_seconds = time.perf_counter() - started
            policies = fit_policies(train, labels.iloc[fitting], audit, seed)
            for name, bundle in policies.items():
                scores = predict_policy(
                    bundle, valid, quality[heldout], available[heldout]
                )
                joblib.dump(bundle, output / "models" / f"{name}_{seed}_{fold}.joblib")
                save_predictions(
                    predictions, name, "clean", seed, fold, labels.iloc[heldout], scores
                )
                average_precisions.append(
                    {
                        "condition": name,
                        "seed": seed,
                        "fold": fold,
                        "average_precision": average_precision_score(
                            labels.iloc[heldout], scores
                        ),
                    }
                )
                branches = (
                    {bundle["input"]: bundle["model"]}
                    if bundle["kind"] == "single"
                    else {branch: bundle[branch] for branch in ["audio", "face"]}
                )
                for branch, model in branches.items():
                    counts.append(
                        {
                            "condition": name,
                            "seed": seed,
                            "fold": fold,
                            "branch": branch,
                            "selection": model["spec"]["selection"],
                            "C": model["spec"]["C"],
                            "selected_count": len(model["indices"]),
                            "features": model["selected_features"],
                        }
                    )
                if bundle["kind"] == "fusion":
                    weights = fusion_weights(
                        quality[heldout],
                        available[heldout],
                        bundle["weight"],
                        bundle["reliability"],
                    )
                    weight_logs.append(
                        {
                            "condition": name,
                            "seed": seed,
                            "fold": fold,
                            "base_audio_weight": bundle["weight"],
                            "mean_audio_weight": float(weights[:, 0].mean()),
                            "mean_face_weight": float(weights[:, 1].mean()),
                            "no_input_count": int((weights.sum(1) == 0).sum()),
                        }
                    )
            for name, value in [("always_negative", -1.0), ("always_positive", 1.0)]:
                scores = np.full(len(heldout), value)
                save_predictions(
                    predictions, name, "clean", seed, fold, labels.iloc[heldout], scores
                )
                average_precisions.append(
                    {
                        "condition": name,
                        "seed": seed,
                        "fold": fold,
                        "average_precision": average_precision_score(
                            labels.iloc[heldout], scores
                        ),
                    }
                )
            audit.update(
                fit_ids=list(map(int, labels.index[fitting])),
                heldout_ids=list(map(int, labels.index[heldout])),
            )
            (output / f"fold_{seed}_{fold}.json").write_text(
                json.dumps(audit, indent=2, allow_nan=False)
            )
            timings.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "selection_seconds": selection_seconds,
                    "total_seconds": time.perf_counter() - started,
                }
            )
    frame = pd.DataFrame(predictions)
    frame.to_parquet(output / "predictions.parquet", index=False)
    metrics = summarize(frame, output)
    pd.DataFrame(average_precisions).to_csv(
        output / "fold_average_precision.csv", index=False
    )
    pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)
    pd.DataFrame(weight_logs).to_csv(output / "fusion_weights.csv", index=False)
    (output / "selected_features.json").write_text(json.dumps(counts, indent=2))
    (output / "comparison.json").write_text(
        json.dumps([paired(metrics, a, b) for a, b in comparison_pairs()], indent=2)
    )
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    for path, digest in hashes.items():
        assert sha256(project / path) == digest, path
    manifest = {
        "study": "sparse_fusion",
        "hashes": hashes,
        "models": {p.name: sha256(p) for p in (output / "models").glob("*.joblib")},
        "participants": len(labels),
        "positives": int(labels.sum()),
        "test_evaluated": False,
        "development_evaluated": False,
        "feature_counts": {name: frame.shape[1] for name, frame in features.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output
