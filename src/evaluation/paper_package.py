"""Copy a run's aggregate, non-identifying outputs into a paper-writer folder.

Excludes participant predictions, fold files (they list participant IDs),
model binaries, recordings and transcripts. Usage:

    python -m src.evaluation.paper_package <run_dir> <destination_dir>
"""

import json
from pathlib import Path
import shutil
import sys

AGGREGATE = [
    "summary.csv",
    "repeat_metrics.csv",
    "domain_metrics.csv",
    "fold_average_precision.csv",
    "comparison.json",
    "fusion_weights.csv",
    "score_regression.csv",
    "score_label_disagreements.csv",
    "environment.txt",
    "verification_current.json",
]


def export(run, destination):
    run, destination = Path(run), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in AGGREGATE:
        if (run / name).exists():
            shutil.copy2(run / name, destination / name)
            copied.append(name)
    if (run / "bootstrap.json").exists():
        summary = [
            {k: v for k, v in row.items() if not k.endswith("_deltas")}
            for row in json.loads((run / "bootstrap.json").read_text())
        ]
        (destination / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2))
        copied.append("bootstrap_summary.json")
    manifest = json.loads((run / "manifest.json").read_text())
    provenance = {
        "run": run.name,
        "study": manifest["study"],
        "participants": manifest["participants"],
        "positives": manifest["positives"],
        "test_evaluated": manifest["test_evaluated"],
        "development_evaluated": manifest["development_evaluated"],
        "feature_counts": manifest.get("feature_counts"),
        "source_and_input_hashes": manifest["hashes"],
        "saved_model_count": len(manifest["models"]),
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2))
    copied.append("provenance.json")
    if "verification_current.json" in copied:
        (destination / "verification_current.json").rename(destination / "verification.json")
        copied[copied.index("verification_current.json")] = "verification.json"
    return copied


if __name__ == "__main__":
    print(export(sys.argv[1], sys.argv[2]))
