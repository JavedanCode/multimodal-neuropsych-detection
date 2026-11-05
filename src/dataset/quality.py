"""Reusable helpers extracted from the recorded research experiments."""

from __future__ import annotations
import numpy as np
import pandas as pd

QUALITY_COLUMNS = [
    "audio__available",
    "audio__invalid_fraction",
    "audio__retained_fraction",
    "facial__available",
    "facial__invalid_fraction",
    "facial__retained_fraction",
    "facial__confidence_mean",
    "facial__confidence_valid_fraction",
]


def validate_quality(quality: pd.DataFrame, index: pd.Index) -> None:
    if not quality.index.is_unique or not quality.index.equals(index):
        raise ValueError("Quality participant IDs must be unique and aligned")
    if list(quality.columns) != QUALITY_COLUMNS:
        raise ValueError(
            "Quality schema must exactly match the approved label-independent fields"
        )
    values = quality.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Quality values must be finite and in [0,1]")
    available = quality[["audio__available", "facial__available"]]
    if not available.isin([0, 1]).all().all() or (available.sum(axis=1) == 0).any():
        raise ValueError(
            "Availability must be binary with at least one usable modality"
        )
