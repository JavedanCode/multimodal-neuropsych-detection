"""Load validated MPNet features for the unchanged training cohort."""

from src.training.provenance import matches_hash
import json
import numpy as np
import pandas as pd
from src.dataset.sequence_inputs import load


def inputs(project):
    old, _, y = load(project)
    root = project / "data/processed/mpnet_v1"
    m = json.loads((root / "manifest.json").read_text())
    if m["partition"] != "train" or not matches_hash(
        project / "src/dataset/mpnet.py", m["contract"]["source"]
    ):
        raise ValueError("Encoder provenance mismatch")
    if not matches_hash(
        project / "src/dataset/text_temporal.py", m["contract"]["encoding_source"]
    ):
        raise ValueError("Encoding changed")
    if not matches_hash(
        project / "src/dataset/lexical.py", m["contract"]["reader_source"]
    ):
        raise ValueError("Reader changed")
    if not matches_hash(
        project / "data/processed/text_temporal_v1/manifest.json",
        m["previous_manifest_hash"],
    ):
        raise ValueError("Previous inputs changed")
    for name, digest in m["output_hashes"].items():
        if not matches_hash(root / name, digest):
            raise ValueError("Feature table changed")
    pd.testing.assert_series_equal(
        y, pd.read_parquet(root / "labels.parquet").PHQ_Binary
    )
    x = pd.read_parquet(root / "features.parquet")
    if (
        not x.index.equals(y.index)
        or x.shape != (len(y), 768)
        or not np.isfinite(x.to_numpy()).all()
    ):
        raise ValueError("Invalid encoder features")
    return {"minilm": old["text"], "mpnet": x}, y
