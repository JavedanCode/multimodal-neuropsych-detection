"""Matched baseline under domain-balanced folds and WoZ-to-AI transfer."""

from src.training.checkpoints import load_checkpoint

from src.training.provenance import matches_hash, source_hashes
from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from src.dataset.text_inputs import inputs
from src.training.classical import SEEDS, select_inner, estimator
from src.training.validation import sha256
from src.training.splits import splits, transfer
from src.evaluation.predictions import save_predictions, summarize
from src.models.text_preprocessing import TextPreprocessor


def fit_fold(x, y, a, b, seed, fold):
    fit = x.iloc[a]
    target = y.iloc[a]
    inner = splits(target, 3, seed + fold + 1)
    prepared = []
    for ia, ib in inner:
        prep = TextPreprocessor().fit(fit.iloc[ia])
        prepared.append(
            (ia, ib, prep.transform(fit.iloc[ia]), prep.transform(fit.iloc[ib]))
        )
    selected, _, records = select_inner(fit, target, "linear", prepared, seed, [0.0])
    model = Pipeline(
        [
            ("preprocess", TextPreprocessor()),
            ("classifier", estimator("linear", selected["parameters"], seed)),
        ]
    ).fit(fit, target)
    audit = {
        "fit_ids": list(map(int, target.index)),
        "heldout_ids": list(map(int, y.index[b])),
        "inner_ids": [
            {
                "fit": list(map(int, target.index[ia])),
                "heldout": list(map(int, target.index[ib])),
            }
            for ia, ib in inner
        ],
        "selected": selected,
        "candidates": records,
    }
    return model, audit


def run(project):
    sets, y = inputs(project)
    x = sets["mpnet"]
    out = (
        project
        / "experiments/runs"
        / ("baseline_v2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    out.mkdir()
    (out / "models").mkdir()
    records = []
    aps = []
    sources = [
        "src/experiments/reference.py",
        "src/training/splits.py",
        "src/dataset/contracts.py",
        "src/models/text_preprocessing.py",
        "src/training/classical.py",
        "docs/methods/reference.md",
        "data/processed/mpnet_v1/features.parquet",
        "data/processed/mpnet_v1/labels.parquet",
    ]
    hashes = {p: sha256(project / p) for p in sources}
    hashes.update(source_hashes(project))
    for mode in ["joint", "transfer"]:
        for seed in SEEDS:
            folds = splits(y, 5, seed) if mode == "joint" else [transfer(y)]
            for fold, (a, b) in enumerate(folds):
                logging.info("%s seed%s fold%s", mode, seed, fold)
                model, audit = fit_fold(x, y, a, b, seed, fold)
                scores = model.decision_function(x.iloc[b])
                path = out / "models" / f"{mode}_{seed}_{fold}.joblib"
                joblib.dump(model, path)
                np.testing.assert_array_equal(
                    scores, load_checkpoint(path).decision_function(x.iloc[b])
                )
                save_predictions(records, mode, "clean", seed, fold, y.iloc[b], scores)
                aps.append(
                    {
                        "condition": mode,
                        "seed": seed,
                        "fold": fold,
                        "average_precision": average_precision_score(y.iloc[b], scores),
                    }
                )
                (out / f"{mode}_{seed}_{fold}.json").write_text(
                    json.dumps(audit, indent=2)
                )
    pred = pd.DataFrame(records)
    pred.to_parquet(out / "predictions.parquet", index=False)
    summarize(pred, out)
    pd.DataFrame(aps).to_csv(out / "fold_average_precision.csv", index=False)
    for p, h in hashes.items():
        assert matches_hash(project / p, h)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "hashes": hashes,
                "participants": list(map(int, y.index)),
                "model_hashes": {p.name: sha256(p) for p in (out / "models").iterdir()},
                "test_evaluated": False,
                "development_evaluated": False,
                "transfer_uses_training_partition_only": True,
            },
            indent=2,
        )
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with threadpool_limits(limits=1):
        print(run(Path(__file__).resolve().parents[2]))
