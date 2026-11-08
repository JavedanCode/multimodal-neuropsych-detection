"""Official eGeMAPSv02 functionals (88 features) from untreated recordings.

Protocol: docs/methods/score_functionals_domain.md. Label-free per-participant
extraction on the training cohort only; resumable per-participant cache.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.dataset.audio_decode import RATE
from src.dataset.audio_preprocessing import clean_waveform, read_waveform
from src.training.provenance import matches_hash
from src.training.validation import sha256

DIRECTORY = "data/processed/egemaps_functionals_v1"
FEATURES = 88


def contract(project):
    import opensmile

    if opensmile.__version__ != "2.6.0":
        raise ValueError("Recorded extraction requires openSMILE Python 2.6.0")
    package = Path(opensmile.__file__).parent
    return {
        "extractor": "eGeMAPSv02_Functionals",
        "opensmile": opensmile.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source": sha256(project / "src/dataset/egemaps_functionals.py"),
        "waveform": sha256(project / "src/dataset/audio_preprocessing.py"),
        "decoder": sha256(project / "src/dataset/audio_decode.py"),
        "package_files": {
            str(p.relative_to(package)): sha256(p)
            for p in sorted(package.rglob("*"))
            if p.is_file() and p.suffix in {".conf", ".inc", ".dll"}
        },
    }


def extract(project, participant, destination, key_contract):
    import opensmile

    started = time.perf_counter()
    archive = project / "data/raw" / f"{participant}_P.tar.gz"
    stat = archive.stat()
    key = {"contract": key_contract, "size": stat.st_size, "mtime": stat.st_mtime_ns}
    target = destination / "participants" / f"{participant}.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["key"] == key:
            return saved, True
    signal, metadata, raw_hash = read_waveform(archive)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=1,
    )
    frame = smile.process_signal(clean_waveform(signal, "raw"), RATE)
    if frame.shape != (1, FEATURES):
        raise ValueError("Unexpected eGeMAPSv02 functional schema")
    values = frame.iloc[0].astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite functional for {participant}")
    saved = {
        "participant": int(participant),
        "key": key,
        "source_waveform_sha256": raw_hash,
        "recording": metadata,
        "duration_seconds": len(signal) / RATE,
        "features": {f"audio__fn__{name}": float(v) for name, v in values.items()},
        "seconds": time.perf_counter() - started,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, indent=2, allow_nan=False))
    temporary.replace(target)
    return saved, False


def build(project, workers=6):
    from src.dataset.representation_inputs import load_inputs

    _, labels, _ = load_inputs(project)
    official = pd.read_parquet(project / "data/processed/labels.parquet")
    if not (official.loc[labels.index, "split"] == "train").all():
        raise ValueError("Training participants only")
    destination = project / DIRECTORY
    (destination / "participants").mkdir(parents=True, exist_ok=True)
    key_contract = contract(project)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = pool.map(
            lambda pid: extract(project, pid, destination, key_contract), labels.index
        )
        for i, (record, cached) in enumerate(tasks, 1):
            records.append(record)
            logging.info(
                "Functionals %d/%d: %s (%s, %.1fs)", i, len(labels),
                record["participant"], "cached" if cached else "extracted",
                record["seconds"],
            )
    features = pd.DataFrame([r["features"] for r in records], index=labels.index)
    if features.shape != (len(labels), FEATURES) or not np.isfinite(features).all().all():
        raise ValueError("Incomplete functional table")
    features.to_parquet(destination / "features.parquet")
    labels.to_frame().to_parquet(destination / "labels.parquet")
    manifest = {
        "partition": "train",
        "participants": len(labels),
        "features": FEATURES,
        "hours": float(sum(r["duration_seconds"] for r in records) / 3600),
        "extraction_seconds": float(sum(r["seconds"] for r in records)),
        "contract": key_contract,
        "outputs": {
            name: sha256(destination / name)
            for name in ["features.parquet", "labels.parquet"]
        },
        "records": {
            str(r["participant"]): sha256(
                destination / "participants" / f"{r['participant']}.json"
            )
            for r in records
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(project, labels):
    destination = project / DIRECTORY
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest["partition"] != "train":
        raise ValueError("Training-only features required")
    for path, digest in [
        (project / "src/dataset/egemaps_functionals.py", manifest["contract"]["source"]),
        (project / "src/dataset/audio_preprocessing.py", manifest["contract"]["waveform"]),
        *[(destination / n, d) for n, d in manifest["outputs"].items()],
    ]:
        if not matches_hash(path, digest):
            raise ValueError(f"Functional provenance mismatch: {path}")
    features = pd.read_parquet(destination / "features.parquet")
    saved = pd.read_parquet(destination / "labels.parquet").iloc[:, 0]
    if not features.index.equals(labels.index) or not saved.equals(labels):
        raise ValueError("Functional cohort differs")
    return features
