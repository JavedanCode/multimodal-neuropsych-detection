"""Collect held-out predictions and summarize repeat and domain metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.training.classical import metric_row


def summarize(pred, out):
    rows, domains = [], []
    for (condition, scenario, seed), g in pred.groupby(
        ["condition", "scenario", "seed"]
    ):
        if not g.participant_id.is_unique:
            raise ValueError("Repeated participant in repeat")
        aucs = [roc_auc_score(f.target, f.score) for _, f in g.groupby("fold")]
        rows.append(
            {
                "condition": condition,
                "scenario": scenario,
                "seed": seed,
                **metric_row(g.target, g.prediction),
                "mean_fold_auc": float(np.mean(aucs)),
            }
        )
        for domain, d in g.groupby(np.where(g.participant_id >= 600, "AI", "WoZ")):
            domains.append(
                {
                    "condition": condition,
                    "scenario": scenario,
                    "seed": seed,
                    "domain": domain,
                    "n": len(d),
                    "positives": int(d.target.sum()),
                    **metric_row(d.target, d.prediction),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "repeat_metrics.csv", index=False)
    pd.DataFrame(domains).to_csv(out / "domain_metrics.csv", index=False)
    columns = [
        "f1",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "mean_fold_auc",
    ]
    summary = metrics.groupby(["condition", "scenario"])[columns].agg(
        ["mean", lambda x: x.std(ddof=0)]
    )
    summary.columns = [
        a + ("_mean" if b == "mean" else "_sd") for a, b in summary.columns
    ]
    summary.to_csv(out / "summary.csv")
    return metrics


def save_predictions(records, condition, scenario, seed, fold, held, scores):
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction")
    records.extend(
        {
            "condition": condition,
            "scenario": scenario,
            "seed": seed,
            "fold": fold,
            "participant_id": int(pid),
            "target": int(t),
            "score": float(s),
            "prediction": int(s >= 0),
        }
        for pid, t, s in zip(held.index, held, scores)
    )
