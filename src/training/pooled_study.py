"""Participant speech, pooled train+dev cohort, text-assisted fusion, abstention.

Protocol: docs/methods/participant_pooled.md. Fixed recipes only; the single
inner-fold choice is the text weight of the text-assisted late fusion.
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
from sklearn.pipeline import Pipeline

from src.dataset.participant_speech import load as load_speech
from src.dataset.text_inputs import inputs as text_inputs
from src.evaluation.comparison import paired
from src.evaluation.predictions import save_predictions, summarize
from src.experiments.reference import fit_fold as fit_text
from src.models.text_preprocessing import TextPreprocessor
from src.models.sparse_fusion import branch_scores, fit_branch, fuse_scores, margin_scale
from src.training.classical import SEEDS, estimator, metric_row
from src.training.corrected_study import load_inputs as load_corrected
from src.training.provenance import source_hashes
from src.training.splits import splits
from src.training.validation import sha256

SPEC = {"selection": "all", "C": 0.1}
WEIGHTS = [0.5, 0.25, 0.75]
COVERAGES = [1.0, 0.8, 0.6, 0.4]
T161 = {
    "original_audio": "audio",
    "voiced_audio": "voiced",
    "segment_audio": "segment",
    "participant_audio": "participant",
    "face": "face",
    "original_concat": "concat",
    "voiced_concat": "voiced_concat",
    "segment_concat": "segment_concat",
    "participant_concat": "participant_concat",
    "original_concat_nopose": "concat_nopose",
    "participant_concat_nopose": "participant_concat_nopose",
}
P217 = {
    "original_audio_p217": "audio",
    "segment_audio_p217": "segment",
    "participant_audio_p217": "participant",
    "face_p217": "face",
    "original_concat_p217": "concat",
    "segment_concat_p217": "segment_concat",
    "participant_concat_p217": "participant_concat",
}
ABSTAIN = ["original_concat", "participant_concat", "text", "text_av_weighted"]


def drop_pose(frame):
    return frame.loc[:, [c for c in frame if not c.startswith("facial__pose_T")]]


def add_derived(features):
    for name in ["segment", "participant"]:
        features[name + "_concat"] = pd.concat([features[name], features["face"]], axis=1)
    features["concat_nopose"] = drop_pose(features["concat"])
    features["participant_concat_nopose"] = drop_pose(features["participant_concat"])
    return features


def load_inputs(project):
    """Return T161 and P217 feature sets with labels; never reads the test split."""
    train, labels, quality, available = load_corrected(project)
    segment, participant, speech_labels = load_speech(project)
    if set(speech_labels.split) != {"train", "dev"}:
        raise ValueError("Only train/dev participants are allowed")
    train["segment"], train["participant"] = segment.loc[labels.index], participant.loc[labels.index]
    train = add_derived(train)
    text, text_labels = text_inputs(project)
    pd.testing.assert_series_equal(text_labels, labels)
    train["text"] = text["mpnet"]
    dev = speech_labels.index[speech_labels.split == "dev"]
    ids = labels.index.append(dev).sort_values()
    pooled_labels = pd.concat([labels, speech_labels.loc[dev, "PHQ_Binary"]]).loc[ids].astype(int)
    supplied = pd.read_parquet(project / "data/processed/features.parquet")
    face = pd.concat([train["face"], supplied.loc[dev].filter(like="facial__")]).loc[ids]
    audio = pd.concat([train["audio"], supplied.loc[dev].filter(like="audio__")]).loc[ids]
    pooled = add_derived(
        {
            "audio": audio,
            "face": face,
            "concat": pd.concat([audio, face], axis=1),
            "segment": segment.loc[ids],
            "participant": participant.loc[ids],
        }
    )
    for frame in train.values():
        assert frame.index.equals(labels.index)
    for frame in pooled.values():
        assert frame.index.equals(pooled_labels.index)
    return {
        "t161": (train, labels),
        "p217": (pooled, pooled_labels),
        "quality": quality,
        "available": available,
    }


def text_scores(frame, labels, fitting, heldout, seed, fold):
    model, audit = fit_text(frame, labels, fitting, heldout, seed, fold)
    return model, model.decision_function(frame.iloc[heldout]), audit["selected"]["parameters"]


def text_av_audit(features, labels, seed, fold, text_parameters):
    """Inner OOF margins of the text and original_concat branches (training only).

    The text branch reuses the parameters its own inner selection chose on these
    same outer-training participants, refitted within each inner partition.
    """
    inner = splits(labels, 3, seed + fold + 1)
    margins = {"text": np.full(len(labels), np.nan), "av": np.full(len(labels), np.nan)}
    for fitting, validation in inner:
        text_model = Pipeline(
            [
                ("preprocess", TextPreprocessor()),
                ("classifier", estimator("linear", text_parameters, seed)),
            ]
        ).fit(features["text"].iloc[fitting], labels.iloc[fitting])
        margins["text"][validation] = text_model.decision_function(
            features["text"].iloc[validation]
        )
        model = fit_branch(features["concat"].iloc[fitting], labels.iloc[fitting], SPEC, seed)
        margins["av"][validation] = branch_scores(model, features["concat"].iloc[validation])
    scales = [margin_scale(margins["text"]), margin_scale(margins["av"])]
    ones, present = np.ones((len(labels), 9)), np.ones((len(labels), 2), bool)
    records = []
    for weight in WEIGHTS:
        scores = fuse_scores(
            margins["text"], margins["av"], ones, present, weight, 0.5, scales=scales
        )
        records.append({"weight": weight, "inner_f1": float(f1_score(labels, scores >= 0))})
    return {
        "scales": scales,
        "weight": max(records, key=lambda r: r["inner_f1"])["weight"],
        "weight_candidates": records,
    }


def fuse_text_av(text, av, weight, scales):
    ones, present = np.ones((len(text), 9)), np.ones((len(text), 2), bool)
    return fuse_scores(text, av, ones, present, weight, 0.5, scales=scales)


def comparison_pairs():
    return (
        [
            (name, "original_concat")
            for name in [
                "participant_concat",
                "segment_concat",
                "original_concat_nopose",
                "participant_concat_nopose",
            ]
        ]
        + [
            ("participant_audio", "segment_audio"),
            ("participant_concat", "segment_concat"),
            ("participant_concat_nopose", "participant_concat"),
            ("text_av_equal", "text"),
            ("text_av_weighted", "text"),
            ("participant_concat_p217", "original_concat_p217"),
            ("segment_concat_p217", "original_concat_p217"),
        ]
    )


def abstention(pred, quality_flag):
    rows = []
    for name in ABSTAIN:
        for seed, g in pred[pred.condition == name].groupby("seed"):
            order = g.reindex(g.score.abs().sort_values(ascending=False, kind="stable").index)
            for coverage in COVERAGES:
                kept = order.iloc[: int(round(coverage * len(order)))]
                rows.append(
                    {"condition": name, "seed": seed, "subset": f"top_{int(coverage * 100)}",
                     "n": len(kept), "positives": int(kept.target.sum()),
                     **metric_row(kept.target, kept.prediction)}
                )
    control = pred[pred.condition == "original_concat"]
    flagged = control.participant_id.map(quality_flag)
    for seed, g in control.groupby("seed"):
        for label, part in [("quality_flagged", g[flagged.loc[g.index]]), ("quality_clean", g[~flagged.loc[g.index]])]:
            rows.append(
                {"condition": "original_concat", "seed": seed, "subset": label, "n": len(part),
                 "positives": int(part.target.sum()), **metric_row(part.target, part.prediction)}
            )
    return pd.DataFrame(rows)


def run(project):
    data = load_inputs(project)
    output = project / "experiments/runs" / (
        "participant_pooled_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    (output / "models").mkdir(parents=True)
    hashes = source_hashes(project)
    for path in [
        "data/processed/features.parquet",
        "data/processed/labels.parquet",
        "data/processed/voiced_audio_v1/features.parquet",
        "data/processed/participant_speech_v1/manifest.json",
        "data/processed/participant_speech_v1/voiced.parquet",
        "data/processed/participant_speech_v1/participant.parquet",
        "data/processed/mpnet_v1/features.parquet",
    ]:
        hashes[path] = sha256(project / path)
    predictions, precisions, audits = [], [], {}

    def record(name, seed, fold, held, scores):
        save_predictions(predictions, name, "clean", seed, fold, held, scores)
        precisions.append(
            {"condition": name, "seed": seed, "fold": fold,
             "average_precision": average_precision_score(held, scores)}
        )

    for cohort, policies in [("t161", T161), ("p217", P217)]:
        features, labels = data[cohort]
        for seed in SEEDS:
            for fold, (fitting, heldout) in enumerate(splits(labels, 5, seed)):
                logging.info("Participant/pooled %s seed %s fold %s", cohort, seed, fold)
                target, held = labels.iloc[fitting], labels.iloc[heldout]
                margins = {}
                for name, frame in policies.items():
                    model = fit_branch(features[frame].iloc[fitting], target, SPEC, seed)
                    margins[name] = branch_scores(model, features[frame].iloc[heldout])
                    joblib.dump(
                        {"input": frame, "model": model},
                        output / "models" / f"{name}_{seed}_{fold}.joblib",
                    )
                    record(name, seed, fold, held, margins[name])
                for name, value in [("always_negative", -1.0), ("always_positive", 1.0)]:
                    record(name + ("_p217" if cohort == "p217" else ""), seed, fold, held,
                           np.full(len(held), value))
                audit = {
                    "cohort": cohort,
                    "fit_ids": list(map(int, target.index)),
                    "heldout_ids": list(map(int, held.index)),
                }
                if cohort == "t161":
                    text_model, text, parameters = text_scores(
                        features["text"], labels, fitting, heldout, seed, fold
                    )
                    joblib.dump(text_model, output / "models" / f"text_{seed}_{fold}.joblib")
                    record("text", seed, fold, held, text)
                    train = {k: features[k].iloc[fitting] for k in ["text", "concat"]}
                    fusion = text_av_audit(train, target, seed, fold, parameters)
                    fusion["text_parameters"] = parameters
                    audit["text_av"] = fusion
                    for name, weight in [("text_av_equal", 0.5), ("text_av_weighted", fusion["weight"])]:
                        record(name, seed, fold, held,
                               fuse_text_av(text, margins["original_concat"], weight, fusion["scales"]))
                audits[f"{cohort}_{seed}_{fold}"] = audit
    (output / "folds.json").write_text(json.dumps(audits, indent=2, allow_nan=False))
    frame = pd.DataFrame(predictions)
    frame.to_parquet(output / "predictions.parquet", index=False)
    metrics = summarize(frame, output)
    pd.DataFrame(precisions).to_csv(output / "fold_average_precision.csv", index=False)
    (output / "comparison.json").write_text(
        json.dumps([paired(metrics, a, b) for a, b in comparison_pairs()], indent=2)
    )
    # P217 policies restricted to the 161 training participants (descriptive).
    train_ids = set(data["t161"][1].index)
    subset = frame[frame.condition.str.endswith("_p217") & frame.participant_id.isin(train_ids)]
    rows = [
        {"condition": name, "seed": seed, **metric_row(g.target, g.prediction)}
        for (name, seed), g in subset.groupby(["condition", "seed"])
    ]
    pd.DataFrame(rows).to_csv(output / "p217_on_train161.csv", index=False)
    quality = data["quality"]
    poor = (
        (quality[:, 2] > 0.001) | (quality[:, 5] > 0.01) | (quality[:, 1] < 0.95)
        | (quality[:, 6] < 0.8) | (quality[:, 7] < 0.8)
    )
    flags = pd.Series(poor, index=data["t161"][1].index)
    abstention(frame, flags).to_csv(output / "abstention.csv", index=False)
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    for path, digest in hashes.items():
        assert sha256(project / path) == digest, path
    manifest = {
        "study": "participant_pooled",
        "hashes": hashes,
        "models": {p.name: sha256(p) for p in (output / "models").glob("*.joblib")},
        "participants": {k: len(data[k][1]) for k in ["t161", "p217"]},
        "positives": {k: int(data[k][1].sum()) for k in ["t161", "p217"]},
        "test_evaluated": False,
        "development_evaluated": True,
        "development_use": "pooled into cross-validation as extra participants; not a holdout",
        "feature_counts": {k: {n: f.shape[1] for n, f in data[k][0].items()} for k in ["t161", "p217"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output
