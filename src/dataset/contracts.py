"""Versioned screening and timing contracts; historical extractors stay frozen."""

import re
import numpy as np
import pandas as pd

DISCLOSURE = re.compile(
    r"\b(?:phq\s*[- ]?\s*[89]?|questionnaire|diagnos\w*|depression\s+(?:score|label)|not at all|several days|more than half the days|nearly every day)\b",
    re.I,
)


def screened_rows(frame, duration):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid duration")
    if not {"Start_Time", "End_Time", "Text"} <= set(frame):
        raise ValueError("Missing transcript columns")
    starts = pd.to_numeric(frame.Start_Time, errors="coerce").to_numpy(float)
    ends = pd.to_numeric(frame.End_Time, errors="coerce").to_numpy(float)
    text = frame.Text.fillna("").astype(str).str.strip()
    valid = (
        np.isfinite(starts)
        & np.isfinite(ends)
        & (starts >= 0)
        & (ends > starts)
        & (ends <= duration + 0.1)
        & text.ne("").to_numpy()
    )
    hit = text.str.contains(DISCLOSURE).to_numpy()
    excluded = hit.copy()
    excluded[1:] |= hit[:-1]
    excluded[:-1] |= hit[1:]
    raw = pd.to_numeric(
        frame.get("Confidence", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy(float)
    known = np.isfinite(raw) & (raw >= 0) & (raw <= 1)
    keep = valid & ~excluded
    if not keep.any():
        raise ValueError("No screened utterances")
    return {
        "text": text[keep].tolist(),
        "starts": starts[keep],
        "ends": ends[keep],
        "row_indices": np.flatnonzero(keep),
        "confidence": np.where(known, raw, 0)[keep],
        "confidence_known": known[keep],
        "speaker_validated": False,
        "audit": {
            "rows": len(frame),
            "retained": int(keep.sum()),
            "excluded_with_neighbors": int(excluded.sum()),
            "unknown_confidence": int((~known).sum()),
            "direct_disclosures": int(hit.sum()),
            "backward_start_steps": int((np.diff(starts[keep]) < 0).sum()),
        },
    }


def domain(ids):
    ids = np.asarray(ids)
    if not (((ids >= 300) & (ids <= 492)) | ((ids >= 600) & (ids <= 718))).all():
        raise ValueError("Unknown interview domain")
    return np.where(ids >= 600, "AI", "WoZ")


def interval_summary(values, times, starts, ends, valid=None):
    values = np.asarray(values, float)
    times = np.asarray(times, float)
    if (
        values.ndim != 2
        or len(values) != len(times)
        or not np.isfinite(times).all()
        or (np.diff(times) < 0).any()
    ):
        raise ValueError("Invalid feature timeline")
    valid = np.ones(len(times), bool) if valid is None else np.asarray(valid, bool)
    valid = valid & np.isfinite(values).all(1)
    out = np.zeros((len(starts), 2 * values.shape[1]), np.float32)
    available = np.zeros(len(starts), bool)
    fraction = np.zeros(len(starts), np.float32)
    count = np.zeros(len(starts), int)
    for i, (start, end) in enumerate(zip(starts, ends)):
        lo, hi = np.searchsorted(times, [start, end], side="left")
        count[i] = hi - lo
        selected = values[lo:hi][valid[lo:hi]]
        fraction[i] = len(selected) / max(1, hi - lo)
        if len(selected):
            out[i] = np.r_[selected.mean(0), selected.std(0)]
            available[i] = True
    return out, available, fraction, count
