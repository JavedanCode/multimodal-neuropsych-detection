"""Voiced-frame eGeMAPS summaries from the supplied low-level descriptor files.

Label-free, per-participant computation (docs/methods/corrected_fusion.md). Each
participant's supplied descriptor file is read once; the legacy all-frame summary
is recomputed from the same bytes and must reproduce ``features.parquet``.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import logging
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from src.dataset.inventory import _is_unsafe_member
from src.dataset.tables import aggregate
from src.training.provenance import matches_hash
from src.training.validation import sha256

DIRECTORY = "data/processed/voiced_audio_v1"
SUFFIX = "OpenSMILE2.3.0_egemaps.csv"
LEGACY_TOLERANCE = 1e-9


def descriptor_bytes(archive_path):
    wanted = f"{int(archive_path.name.split('_')[0])}_{SUFFIX}"
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member in {archive_path.name}")
            if Path(member.name).name == wanted:
                if not member.isfile():
                    raise ValueError("Nonregular descriptor member")
                payload = archive.extractfile(member).read()
                if len(payload) != member.size:
                    raise ValueError("Truncated descriptor member")
                return payload
    raise ValueError(f"Missing {SUFFIX} in {archive_path.name}")


def summarize(payload, columns, chunk_rows=10000):
    voiced, _, diagnostics = aggregate(
        io.BytesIO(payload), columns, "audio", chunk_rows, voiced_only=True
    )
    legacy, _, _ = aggregate(io.BytesIO(payload), columns, "audio", chunk_rows)
    return {"voiced": voiced, "legacy": legacy, "rows": diagnostics["rows"]}


def build(project, workers=3):
    from src.dataset.representation_inputs import load_inputs

    _, labels, _ = load_inputs(project)
    root = project / "data/processed"
    columns = json.loads((root / "stage3_report.json").read_text())["schemas"][
        "audio"
    ]["columns"]
    output = project / DIRECTORY
    cache = output / "participants"
    cache.mkdir(parents=True, exist_ok=True)
    source = sha256(project / "src/dataset/voiced_audio.py")
    contract = hashlib.sha256(
        json.dumps(
            {
                "columns": columns,
                "tables": sha256(project / "src/dataset/tables.py"),
                "source": source,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    def one(pid):
        archive = project / "data/raw" / f"{pid}_P.tar.gz"
        stat = archive.stat()
        key = {"contract": contract, "size": stat.st_size, "mtime": stat.st_mtime_ns}
        target = cache / f"{pid}.json"
        if target.exists():
            saved = json.loads(target.read_text())
            if saved["key"] == key:
                return pid, saved["data"]
        data = summarize(descriptor_bytes(archive), columns)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"key": key, "data": data}, allow_nan=False))
        temporary.replace(target)
        return pid, data

    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (pid, data) in enumerate(pool.map(one, labels.index), 1):
            rows[pid] = data
            logging.info("Voiced audio summaries %d/%d", i, len(labels))
    frames = {
        kind: pd.DataFrame.from_dict(
            {pid: rows[pid][kind] for pid in labels.index}, orient="index"
        ).astype(float)
        for kind in ("voiced", "legacy")
    }
    for frame in frames.values():
        frame.index.name = "Participant_ID"
    supplied = pd.read_parquet(root / "features.parquet").loc[labels.index]
    supplied = supplied.filter(like="audio__")
    if list(frames["legacy"]) != list(supplied):
        raise ValueError("Legacy schema differs from supplied features")
    difference = float(np.nanmax(np.abs(frames["legacy"].values - supplied.values)))
    if difference > LEGACY_TOLERANCE or not (
        frames["legacy"].isna().values == supplied.isna().values
    ).all():
        raise ValueError(f"Legacy recomputation mismatch {difference}")
    voiced = frames["voiced"]
    voiced.to_parquet(output / "features.parquet")
    labels.to_frame().to_parquet(output / "labels.parquet")
    nz = [c for c in voiced if "_sma3nz__" in c]
    manifest = {
        "partition": "train",
        "participants": len(voiced),
        "features": voiced.shape[1],
        "voicing_columns": len(nz),
        "frames": int(sum(rows[pid]["rows"] for pid in labels.index)),
        "voiced_fraction_median": float(voiced["audio__voiced_fraction"].median()),
        "missing_cells": int(voiced.isna().sum().sum()),
        "legacy_max_abs_difference": difference,
        "contract": contract,
        "source": source,
        "tables_source": sha256(project / "src/dataset/tables.py"),
        "outputs": {
            name: sha256(output / name) for name in ["features.parquet", "labels.parquet"]
        },
        "inputs": {"features.parquet": sha256(root / "features.parquet")},
        "rule": "sma3nz descriptors over F0>0 frames; -201 missing; sma3 all frames",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(project, labels):
    """Return validated voiced summaries aligned to ``labels``."""
    output = project / DIRECTORY
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["partition"] != "train":
        raise ValueError("Training-only features required")
    for path, digest in [
        (project / "src/dataset/voiced_audio.py", manifest["source"]),
        (project / "src/dataset/tables.py", manifest["tables_source"]),
        *[(output / name, digest) for name, digest in manifest["outputs"].items()],
        (project / "data/processed/features.parquet", manifest["inputs"]["features.parquet"]),
    ]:
        if not matches_hash(path, digest):
            raise ValueError(f"Voiced audio provenance mismatch: {path}")
    features = pd.read_parquet(output / "features.parquet")
    saved = pd.read_parquet(output / "labels.parquet").iloc[:, 0]
    if not features.index.equals(labels.index) or not saved.equals(labels):
        raise ValueError("Voiced audio cohort differs")
    return features
