"""Load and validate cached enriched audio summaries."""

from __future__ import annotations
from src.training.provenance import matches_hash
import json
import pandas as pd
from src.training.validation import validate_partition


def load_inputs(project):
    directory = project / "data/processed/audio_study_v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["partition"] != "train":
        raise ValueError("Training-only enriched summaries required")
    for name, digest in manifest["output_sha256"].items():
        if not matches_hash(directory / name, digest):
            raise ValueError("Enriched table provenance mismatch")
    if not matches_hash(
        project / "src/dataset/audio_summary.py", manifest["contract"]["source"]
    ):
        raise ValueError("Extractor changed; rebuild summaries")
    x = pd.read_parquet(directory / "features.parquet")
    y = pd.read_parquet(directory / "labels.parquet").PHQ_Binary
    official = pd.read_parquet(project / "data/processed/labels.parquet")
    train_ids = official.index[official.split == "train"]
    if not x.index.equals(train_ids) or not y.equals(
        official.loc[train_ids, "PHQ_Binary"]
    ):
        raise ValueError(
            "Study must contain unchanged official training participants/targets only"
        )
    validate_partition(x, y)
    return x, y, manifest["baseline_columns"]
