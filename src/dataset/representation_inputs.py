"""Load and validate original and richer acoustic/facial representations."""

from src.training.provenance import matches_hash
import json
import pandas as pd
from src.training.validation import validate_partition


def load_inputs(project):
    root = project / "data/processed"
    directory = root / "representation_study_v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["partition"] != "train":
        raise ValueError("Training-only features required")
    if not matches_hash(
        project / "src/dataset/representations.py", manifest["contract"]["source"]
    ):
        raise ValueError("Extractor changed")
    for name, digest in manifest["output_hashes"].items():
        if not matches_hash(directory / name, digest):
            raise ValueError("Output provenance mismatch")
    for name, digest in manifest["source_tables"].items():
        if not matches_hash(root / name, digest):
            raise ValueError("Source table changed")
    official = pd.read_parquet(root / "labels.parquet")
    y = pd.read_parquet(directory / "labels.parquet").PHQ_Binary
    previous = pd.read_parquet(root / "segmentation_study_v1/labels.parquet").PHQ_Binary
    ids = official.index[
        (official.split == "train") & ~official.index.isin(manifest["excluded_ids"])
    ]
    if not y.equals(official.loc[ids, "PHQ_Binary"]) or not y.equals(previous):
        raise ValueError("Unchanged training cohort required")
    base = pd.read_parquet(root / "features.parquet").loc[ids]
    dense = pd.read_parquet(directory / "dense.parquet")
    temporal = pd.read_parquet(directory / "facial.parquet")
    if not dense.index.equals(ids) or not temporal.index.equals(ids):
        raise ValueError("Participant order mismatch")
    audio = base.filter(like="audio__")
    facial = base.filter(like="facial__")
    sets = {
        "base_audio": audio,
        "base_facial": facial,
        "base_concat": base,
        "temporal_facial": temporal,
        "dense_audio": dense,
        "dense_base_facial": pd.concat([dense, facial], axis=1),
        "dense_temporal_facial": pd.concat([dense, temporal], axis=1),
    }
    for x in sets.values():
        validate_partition(x, y)
    return sets, y, manifest
