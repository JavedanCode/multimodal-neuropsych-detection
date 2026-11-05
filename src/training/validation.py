"""Validate participant tables and hash artifacts without altering them."""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.dataset.preprocessing import ParticipantPreprocessor

TARGET_POLICY = "released_binary_provisional"


def validate_partition(X: pd.DataFrame, y: pd.Series) -> None:
    ParticipantPreprocessor._validate(X)
    if not X.index.is_unique or not X.index.equals(y.index) or len(X) == 0:
        raise ValueError(
            "Participant IDs and labels must be unique, nonempty and aligned"
        )
    ids = pd.to_numeric(X.index, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError("Invalid participant IDs")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Target must contain only nonmissing binary labels")


def load_development_inputs(processed: Path) -> tuple:
    """Read raw summaries, never the globally normalized official-split tables."""
    features = pd.read_parquet(processed / "features.parquet")
    labels = pd.read_parquet(processed / "labels.parquet")
    if (
        not features.index.is_unique
        or not labels.index.is_unique
        or not features.index.equals(labels.index)
    ):
        raise ValueError("Feature/label participant alignment failure")
    if set(labels.columns) != {"PHQ_Binary", "split"} or set(labels["split"]) != {
        "train",
        "dev",
        "test",
    }:
        raise ValueError("Unexpected labels/split schema")
    train_ids = labels.index[labels["split"] == "train"]
    dev_ids = labels.index[labels["split"] == "dev"]
    # Test feature values and targets are not passed into the experiment API.
    return (
        features.loc[train_ids],
        labels.loc[train_ids, "PHQ_Binary"],
        features.loc[dev_ids],
        labels.loc[dev_ids, "PHQ_Binary"],
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
