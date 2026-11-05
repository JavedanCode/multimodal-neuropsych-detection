"""Finite existing-data experiment series, fixed before inspecting outcomes."""

from src.training.provenance import matches_hash, source_hashes
from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from src.training.validation import sha256
from src.training.classical import candidates, SEEDS, estimator
from src.training.splits import splits
from src.dataset.aligned_inputs import load as load_aligned
from src.evaluation.predictions import save_predictions, summarize
from src.evaluation.comparison import paired
from src.models.text_preprocessing import TextPreprocessor
from src.models.stacking import take
from src.models.text_candidates import (
    measures,
    choose,
    fit_text,
    fit_aux,
    residual,
    predict,
    SCENARIOS,
)

FEATURES = ("text", "balanced", "concatenated")


def load(project):
    data, y, _ = load_aligned(project)
    root = project / "data/processed/topics_v3"
    m = json.loads((root / "manifest.json").read_text())
    for p, h in m["hashes"].items():
        assert matches_hash(project / p, h)
    for p, h in m["outputs"].items():
        assert matches_hash(root / p, h)
    pd.testing.assert_series_equal(
        y, pd.read_parquet(root / "labels.parquet").PHQ_Binary
    )
    for name in FEATURES[1:]:
        data[name] = pd.read_parquet(root / f"{name}.parquet")
        assert data[name].index.equals(y.index)
    return data, y


def record(feature, params, alpha, threshold, scores, y):
    return {
        "feature": feature,
        "parameters": params,
        "alpha": alpha,
        "threshold": threshold,
        "scores": np.asarray(scores).tolist(),
        "metrics": measures(y, scores),
    }


def inner_candidates(data, y, seed, fold):
    inner = splits(y, 3, seed + fold + 1)
    raw = {}
    aux_scores = np.full(len(y), np.nan)
    for feature in FEATURES:
        prepared = []
        for a, b in inner:
            prep = TextPreprocessor().fit(data[feature].iloc[a])
            prepared.append(
                (
                    a,
                    b,
                    prep.transform(data[feature].iloc[a]),
                    prep.transform(data[feature].iloc[b]),
                )
            )
        for pi, params in enumerate(candidates("linear")):
            scores = np.full(len(y), np.nan)
            for a, b, x, v in prepared:
                scores[b] = (
                    estimator("linear", params, seed)
                    .fit(x, y.iloc[a])
                    .decision_function(v)
                )
            raw[feature, pi] = scores
    ref_records = [
        record("text", p, 0.0, 0.0, raw["text", i], y)
        for i, p in enumerate(candidates("linear"))
    ]
    reference = max(ref_records, key=lambda r: r["metrics"]["f1"])
    for a, b in inner:
        aux, scales = fit_aux(take(data, a), y.iloc[a], seed)
        aux_scores[b] = residual(aux, scales, take(data, b))
    groups = {
        k: [reference]
        for k in ["threshold", "topic_balanced", "topic_concat", "quality_residual"]
    }
    for t in [0.0, 0.1, 0.2, 0.3]:
        groups["threshold"].append(
            record(
                "text",
                reference["parameters"],
                0.0,
                t,
                np.array(reference["scores"]) - t,
                y,
            )
        )
        for alpha in [0.0, 0.1, 0.25]:
            groups["quality_residual"].append(
                record(
                    "text",
                    reference["parameters"],
                    alpha,
                    t,
                    np.array(reference["scores"]) + alpha * aux_scores - t,
                    y,
                )
            )
        for feature, name in [
            ("balanced", "topic_balanced"),
            ("concatenated", "topic_concat"),
        ]:
            for i, params in enumerate(candidates("linear")):
                groups[name].append(
                    record(feature, params, 0.0, t, raw[feature, i] - t, y)
                )
    selected = {name: choose(records, reference) for name, records in groups.items()}
    selected["selected"] = choose(
        [reference] + [r for records in groups.values() for r in records], reference
    )
    selected["reference"] = reference
    return selected, {
        "reference_candidates": ref_records,
        "groups": groups,
        "selected": selected,
        "inner_ids": [
            {"fit": list(map(int, y.index[a])), "heldout": list(map(int, y.index[b]))}
            for a, b in inner
        ],
    }


def precision_screen(metrics, name):
    a = metrics[(metrics.condition == name) & (metrics.scenario == "clean")].set_index(
        "seed"
    )
    b = metrics[
        (metrics.condition == "reference") & (metrics.scenario == "clean")
    ].set_index("seed")
    d = a.select_dtypes("number") - b.select_dtypes("number")
    v = d.mean()
    checks = {
        "precision_gain_5pp": v.precision >= 0.05,
        "recall_floor_65": a.recall.mean() >= 0.65,
        "recall_loss_max_5pp": v.recall >= -0.05,
        "f1_loss_max_2pp": v.f1 >= -0.02,
        "precision_gain_two_repeats": (d.precision > 0).sum() >= 2,
    }
    return {
        "checks": {k: bool(v) for k, v in checks.items()},
        "passes": bool(all(checks.values())),
        "mean_precision_difference": float(v.precision),
        "mean_f1_difference": float(v.f1),
    }


def run(project, baseline):
    baseline = baseline.resolve()
    data, y = load(project)
    bm = json.loads((baseline / "manifest.json").read_text())
    for p, h in bm["hashes"].items():
        assert matches_hash(project / p, h)
    paths = [
        "src/experiments/text_candidates.py",
        "src/models/text_candidates.py",
        "src/dataset/topics.py",
        "src/training/splits.py",
        "src/models/text_preprocessing.py",
        "src/training/classical.py",
        "src/evaluation/predictions.py",
        "src/evaluation/metrics.py",
        "docs/methods/text_candidates.md",
        "data/processed/topics_v3/manifest.json",
        "data/processed/aligned_v2/manifest.json",
    ]
    hashes = {p: sha256(project / p) for p in paths}
    hashes.update(source_hashes(project))
    out = (
        project
        / "experiments/runs"
        / ("bounded_v3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    out.mkdir()
    (out / "models").mkdir()
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "hashes": hashes,
                "baseline": str(baseline),
                "development_evaluated": False,
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    predictions = []
    aps = []
    for seed in SEEDS:
        for fold, (a, b) in enumerate(splits(y, 5, seed)):
            logging.info("Bounded v3 seed%s fold%s", seed, fold)
            train, valid = take(data, a), take(data, b)
            target, held = y.iloc[a], y.iloc[b]
            selected, audit = inner_candidates(train, target, seed, fold)
            aux, scales = fit_aux(train, target, seed)
            for name, spec in selected.items():
                bundle = {
                    "text": fit_text(
                        train[spec["feature"]], target, spec["parameters"], seed
                    ),
                    "aux": aux,
                    "scales": scales,
                    "feature": spec["feature"],
                    "alpha": spec["alpha"],
                    "threshold": spec["threshold"],
                }
                joblib.dump(bundle, out / "models" / f"{name}_{seed}_{fold}.joblib")
                for scenario in SCENARIOS:
                    scores = predict(bundle, valid, scenario)
                    save_predictions(
                        predictions, name, scenario, seed, fold, held, scores
                    )
                    aps.append(
                        {
                            "condition": name,
                            "scenario": scenario,
                            "seed": seed,
                            "fold": fold,
                            "average_precision": average_precision_score(held, scores),
                        }
                    )
            audit.update(
                fit_ids=list(map(int, target.index)),
                heldout_ids=list(map(int, held.index)),
            )
            (out / f"fold_{seed}_{fold}.json").write_text(
                json.dumps(audit, indent=2, allow_nan=False)
            )
    pred = pd.DataFrame(predictions)
    pred.to_parquet(out / "predictions.parquet", index=False)
    metrics = summarize(pred, out)
    pd.DataFrame(aps).to_csv(out / "fold_average_precision.csv", index=False)
    names = [n for n in selected if n != "reference"]
    (out / "comparison.json").write_text(
        json.dumps(
            {
                "precision": {n: precision_screen(metrics, n) for n in names},
                "f1": [paired(metrics, n, "reference") for n in names],
            },
            indent=2,
        )
    )
    old = (
        pd.read_parquet(baseline / "predictions.parquet")
        .query('condition=="joint"')
        .set_index(["seed", "fold", "participant_id"])
        .sort_index()
    )
    new = (
        pred.query('condition=="reference" and scenario=="clean"')
        .set_index(["seed", "fold", "participant_id"])
        .sort_index()
    )
    assert old.index.equals(new.index)
    np.testing.assert_allclose(old.score, new.score, atol=1e-12, rtol=1e-12)
    for p, h in hashes.items():
        assert matches_hash(project / p, h)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "hashes": hashes,
                "baseline": str(baseline),
                "models": {p.name: sha256(p) for p in (out / "models").iterdir()},
                "test_evaluated": False,
                "development_evaluated": False,
            },
            indent=2,
        )
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with threadpool_limits(limits=1):
        print(run(Path(__file__).resolve().parents[2], Path(sys.argv[1])))
