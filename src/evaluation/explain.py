"""Stage 8: frozen-checkpoint descriptor IG and reference perturbation tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import logging
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch

from src.dataset.preprocessing import ParticipantPreprocessor
from src.dataset.quality import QUALITY_COLUMNS, validate_quality
from src.training.validation import load_development_inputs, sha256, TARGET_POLICY
from src.models.adaptive import AdaptiveFusion
from src.models.reliability import ReliabilityFusion


def integrated_gradients(forward, x, steps=512):
    """Midpoint IG from zero; forward must return one independent logit per row."""
    if x.ndim != 2 or not torch.isfinite(x).all() or steps < 1:
        raise ValueError("Finite 2D inputs and positive integration steps required")
    gradient = torch.zeros_like(x)
    for i in range(steps):
        point = (x.detach() * ((i + 0.5) / steps)).requires_grad_(True)
        gradient += torch.autograd.grad(forward(point).sum(), point)[0].detach()
    attribution = x.detach() * gradient / steps
    with torch.no_grad():
        output, reference = forward(x), forward(torch.zeros_like(x))
    residual = attribution.sum(1) - (output - reference)
    if not torch.isfinite(attribution).all() or not torch.isfinite(residual).all():
        raise ValueError("Nonfinite attribution or completeness residual")
    return attribution, output, reference, residual


def perturbation(
    forward, x, attribution, fractions=(0.1, 0.25, 0.5), repeats=32, seed=42
):
    """Return per-person changes; random trials are averaged within person."""
    if (
        x.shape != attribution.shape
        or repeats < 1
        or any(not 0 < f <= 1 for f in fractions)
    ):
        raise ValueError("Invalid perturbation inputs")
    rng = np.random.default_rng(seed)
    ranking = np.argsort(-np.abs(attribution.detach().numpy()), axis=1, kind="stable")
    rows = np.arange(len(x))[:, None]
    records = []
    with torch.no_grad():
        original = forward(x)
        for fraction in fractions:
            k = int(np.ceil(x.shape[1] * fraction))
            for method in ["top", "bottom", "random"]:
                trials = []
                for _ in range(repeats if method == "random" else 1):
                    order = (
                        np.argsort(rng.random(x.shape), axis=1)
                        if method == "random"
                        else (
                            ranking
                            if method == "top"
                            else np.argsort(
                                np.abs(attribution.detach().numpy()),
                                axis=1,
                                kind="stable",
                            )
                        )
                    )
                    changed = x.clone()
                    changed[rows, order[:, :k].copy()] = 0
                    logits = forward(changed)
                    trials.append(
                        torch.stack(
                            [
                                (logits - original).abs(),
                                (logits.sigmoid() - original.sigmoid()).abs(),
                                ((logits >= 0) != (original >= 0)).float(),
                            ],
                            dim=1,
                        ).numpy()
                    )
                values = np.mean(trials, axis=0)
                for i, value in enumerate(values):
                    records.append(
                        {
                            "row": i,
                            "fraction": fraction,
                            "count": k,
                            "method": method,
                            "absolute_logit_change": float(value[0]),
                            "absolute_probability_change": float(value[1]),
                            "flip_rate": float(value[2]),
                        }
                    )
    return pd.DataFrame(records)


def ranking_agreement(first, second):
    names = sorted(set(first.index) | set(second.index))
    a, b = first.reindex(names, fill_value=0), second.reindex(names, fill_value=0)
    correlation = a.rank().corr(b.rank())
    top_a = set(first.sort_values(ascending=False, kind="stable").head(10).index)
    top_b = set(second.sort_values(ascending=False, kind="stable").head(10).index)
    return {
        "spearman_union_zero_absent": (
            float(correlation) if pd.notna(correlation) else None
        ),
        "top10_jaccard": len(top_a & top_b) / len(top_a | top_b),
    }


def run(processed, source, output, project):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    previous = json.loads((source / "results.json").read_text(encoding="utf-8"))
    hashes = {
        n: sha256(processed / n)
        for n in ["features.parquet", "labels.parquet", "quality.parquet"]
    }
    if hashes != previous["input_sha256"]:
        raise ValueError("Stage 7 input hashes differ")
    train, _, dev, _ = load_development_inputs(processed)
    quality = pd.read_parquet(processed / "quality.parquet").loc[
        dev.index, QUALITY_COLUMNS
    ]
    validate_quality(quality, dev.index)
    expected = {
        f"{s}_{m}_seed{seed}"
        for s in ["all", "l1", "hho"]
        for m in ["adaptive", "quality"]
        for seed in [42, 137, 2026]
    }
    if {p.stem for p in source.glob("*.pt")} != expected:
        raise ValueError("Expected all 18 Stage 7 checkpoints")
    if output.resolve().is_relative_to((project / "data/raw").resolve()):
        raise ValueError("Cannot write into raw data")
    output.mkdir(parents=True, exist_ok=False)
    results, importance, all_faithfulness = [], {}, []
    for name in sorted(expected):
        logging.info("Explaining %s", name)
        path = source / f"{name}.pt"
        cp = torch.load(path, map_location="cpu", weights_only=True)
        if cp["target_policy"] != TARGET_POLICY or cp["train_participant_ids"] != list(
            map(int, train.index)
        ):
            raise ValueError("Checkpoint target/training participant mismatch")
        prep = ParticipantPreprocessor.from_state(cp["preprocessing"])
        frame = prep.transform(dev[list(prep.feature_names_in_)])
        is_quality = "quality_mean" in cp["model_config"]
        if is_quality and cp.get("quality_columns") != QUALITY_COLUMNS:
            raise ValueError("Checkpoint quality schema mismatch")
        model = (ReliabilityFusion if is_quality else AdaptiveFusion)(
            **cp["model_config"]
        )
        model.load_state_dict(cp["state_dict"])
        model.eval()
        audio = [i for i, c in enumerate(frame) if c.startswith("audio__")]
        facial = [i for i, c in enumerate(frame) if c.startswith("facial__")]
        x = torch.tensor(frame.to_numpy(), dtype=torch.float32)
        q = torch.tensor(quality.to_numpy(), dtype=torch.float32)

        def forward(values):
            args = (values[:, audio], values[:, facial])
            return model(*(args + (q,) if is_quality else args))[0]

        coarse, _, _, _ = integrated_gradients(forward, x, 256)
        attribution, logits, reference, residual = integrated_gradients(forward, x, 512)
        old = pd.read_parquet(source / f"{name}_dev_predictions.parquet")
        if not old.index.equals(dev.index):
            raise ValueError("Saved development prediction alignment mismatch")
        np.testing.assert_allclose(
            logits.sigmoid().numpy(), old.probability_class_1, rtol=1e-5, atol=1e-6
        )
        values = attribution.numpy()
        local = pd.DataFrame(values, index=dev.index, columns=frame.columns)
        local.to_parquet(output / f"{name}_attributions.parquet")
        global_table = pd.DataFrame(
            {
                "mean_absolute_ig": local.abs().mean(),
                "mean_signed_ig": local.mean(),
                "modality": [c.split("__")[0] for c in frame],
            }
        )
        global_table.index.name = "feature"
        global_table.sort_values(
            "mean_absolute_ig", ascending=False, kind="stable"
        ).to_csv(output / f"{name}_global.csv")
        importance[name] = global_table.mean_absolute_ig
        diagnostic = pd.DataFrame(
            {
                "logit": logits.numpy(),
                "probability": logits.sigmoid().numpy(),
                "reference_logit": reference.numpy(),
                "completeness_residual": residual.numpy(),
                "numerical_l1_difference": (attribution - coarse).abs().sum(1).numpy(),
                "audio_signed_ig": values[:, audio].sum(1),
                "facial_signed_ig": values[:, facial].sum(1),
                "audio_absolute_ig": np.abs(values[:, audio]).sum(1),
                "facial_absolute_ig": np.abs(values[:, facial]).sum(1),
            },
            index=dev.index,
        )
        if is_quality:
            neutral = model.quality_mean.expand_as(q).clone()
            neutral[:, [0, 3]] = q[:, [0, 3]]
            with torch.no_grad():
                neutral_logit = model(x[:, audio], x[:, facial], neutral)[0]
            diagnostic["quality_mean_replacement_logit_change"] = (
                neutral_logit - logits
            ).numpy()
        diagnostic.to_csv(output / f"{name}_participants.csv")
        faith = perturbation(forward, x, attribution, seed=cp["seed"])
        faith["participant_id"] = [int(dev.index[i]) for i in faith.pop("row")]
        faith["model"] = name
        faith.to_csv(output / f"{name}_perturbations.csv", index=False)
        all_faithfulness.append(faith)
        failed = residual.abs() > 0.001 + 0.01 * (logits - reference).abs()
        results.append(
            {
                "name": name,
                "checkpoint_sha256": sha256(path),
                "prediction_verified": True,
                "features": len(frame.columns),
                "completeness_flagged_rows": int(failed.sum()),
                "max_absolute_completeness_residual": float(residual.abs().max()),
                "mean_numerical_l1_difference": float(
                    (attribution - coarse).abs().sum(1).mean()
                ),
            }
        )
        lines = [
            f"# {name}: development participant explanations",
            "",
            "Signed IG explains the positive-class logit relative to training-mean descriptors.",
            "Quality is held fixed. These are model contributions, not clinical findings.",
            "",
        ]
        for participant, row in local.iterrows():
            top = row.abs().sort_values(ascending=False, kind="stable").head(5).index
            lines += [
                f"## Participant {participant}",
                "",
                f'Model probability: {diagnostic.loc[participant,"probability"]:.4f}.',
                "",
                *[f"- `{c}`: {row[c]:+.5f} logit contribution." for c in top],
                "",
            ]
        (output / f"{name}_explanations.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    comparisons = []
    for a, b in combinations(sorted(expected), 2):
        sa, ma, seeda = a.split("_")
        sb, mb, seedb = b.split("_")
        if sa == sb and (
            (ma == mb and seeda != seedb) or (seeda == seedb and ma != mb)
        ):
            comparisons.append(
                {
                    "first": a,
                    "second": b,
                    **ranking_agreement(importance[a], importance[b]),
                }
            )
    pd.DataFrame(comparisons).to_csv(output / "ranking_stability.csv", index=False)
    combined = pd.concat(all_faithfulness)
    summary = (
        combined.groupby(["model", "fraction", "method"])[
            ["absolute_logit_change", "absolute_probability_change", "flip_rate"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(output / "faithfulness_summary.csv", index=False)
    report = {
        "test_evaluated": False,
        "target_policy": TARGET_POLICY,
        "source_run": str(source),
        "input_sha256": hashes,
        "models": results,
        "protocol_sha256": sha256(project / "docs/methods/explainability.md"),
        "source_sha256": {
            str(p.relative_to(project)): sha256(p)
            for p in [
                Path(__file__),
                project / "src/models/adaptive.py",
                project / "src/models/reliability.py",
                project / "src/dataset/preprocessing.py",
            ]
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
        },
        "settings": {
            "steps": [256, 512],
            "random_repeats": 32,
            "fractions": [0.1, 0.25, 0.5],
            "reference": "training mean",
            "quality": "held fixed",
        },
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Stage 8: exploratory descriptor explanations",
        "",
        "All 18 frozen Stage 7 models; all 56 development participants; test not evaluated.",
        "Reference: training-mean descriptors. Quality fixed. Positive-class logit IG.",
        "",
        "| Model | Top 25% change | Random 25% change | Bottom 25% change | Flagged completeness rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        subset = summary[
            (summary.model == r["name"]) & (summary.fraction == 0.25)
        ].set_index("method")
        lines.append(
            f"| {r['name']} | {subset.loc['top','absolute_logit_change']:.4f} | {subset.loc['random','absolute_logit_change']:.4f} | {subset.loc['bottom','absolute_logit_change']:.4f} | {r['completeness_flagged_rows']} |"
        )
    lines += [
        "",
        "Changes are mean absolute logit changes, not F1 improvements or causal effects.",
        "Global rankings, signed local explanations, numerical diagnostics and seed stability are saved separately.",
        "Reference replacement can create unrealistic combinations; correlated features and interactions limit interpretation.",
        "Quality conditioning is not explained by descriptor IG. See separate quality-mean replacement sensitivity.",
        "Released-label semantics remain unresolved. No clinical symptom interpretations.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    output = (
        project
        / "experiments/runs"
        / ("stage8_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(project / "data/processed", args.source_run.resolve(), output, project)
    logging.info("Completed: %s", output)


if __name__ == "__main__":
    main()
