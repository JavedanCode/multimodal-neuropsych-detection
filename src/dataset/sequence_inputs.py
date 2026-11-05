"""Read validated utterance caches and participant-level text summaries."""

from src.training.provenance import matches_hash
import json
import numpy as np
import pandas as pd


def load(project):
    root = project / "data/processed"
    directory = root / "text_temporal_v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["partition"] != "train" or not matches_hash(
        project / "src/dataset/text_temporal.py", manifest["source"]
    ):
        raise ValueError("Extractor provenance mismatch")
    for name, digest in manifest["output_hashes"].items():
        if not matches_hash(directory / name, digest):
            raise ValueError("Input changed")
    y = pd.read_parquet(directory / "labels.parquet").PHQ_Binary
    official = pd.read_parquet(root / "labels.parquet")
    previous = pd.read_parquet(root / "segmentation_study_v1/labels.parquet").PHQ_Binary
    if (
        not y.equals(previous)
        or not y.equals(official.loc[y.index, "PHQ_Binary"])
        or not (official.loc[y.index, "split"] == "train").all()
    ):
        raise ValueError("Training cohort changed")
    with np.load(directory / "sequences.npz") as data:
        values = {k: data[k] for k in data.files}
    np.testing.assert_array_equal(values.pop("ids"), y.index)
    for x in values.values():
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite features")
    text = pd.DataFrame(
        values["text_mean"], index=y.index, columns=[f"text__{i}" for i in range(384)]
    )
    base = pd.read_parquet(root / "features.parquet").loc[y.index]
    sets = {
        "base_concat": base,
        "text": text,
        "text_concat": pd.concat([text, base], axis=1),
        "audio_only": base.filter(like="audio__"),
        "face_only": base.filter(like="facial__"),
    }
    return sets, values, y
