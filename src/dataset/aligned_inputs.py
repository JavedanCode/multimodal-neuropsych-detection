"""Aggregate aligned utterances into primary features and quality indicators."""

from src.training.provenance import matches_hash
import json
import numpy as np
import pandas as pd
from src.dataset.text_inputs import inputs


def load(project):
    sets, y = inputs(project)
    root = project / "data/processed/aligned_v2"
    m = json.loads((root / "manifest.json").read_text())
    assert m["partition"] == "train" and set(m["participants"]) == set(
        map(str, y.index)
    )
    for p, h in m["sources"].items():
        assert matches_hash(project / p, h)
    assert matches_hash(root / "labels.parquet", m["labels_hash"])
    pd.testing.assert_series_equal(
        y, pd.read_parquet(root / "labels.parquet").PHQ_Binary
    )
    values = {"audio": [], "face": [], "quality": [], "available": []}
    for pid in y.index:
        path = root / "participant_cache" / f"{pid}.npz"
        assert matches_hash(path, m["participants"][str(pid)]["hash"])
        with np.load(path) as z:
            for i, name in [(1, "audio"), (2, "face")]:
                a = z[name][z["masks"][:, i]]
                values[name].append(
                    np.r_[a.mean(0), a.std(0)]
                    if len(a)
                    else np.zeros(2 * z[name].shape[1])
                )
            values["quality"].append(z["quality"].mean(0))
            values["available"].append(z["masks"].any(0))
    data = {k: np.asarray(v, dtype=float) for k, v in values.items()}
    data["text"] = sets["mpnet"]
    return data, y, m
