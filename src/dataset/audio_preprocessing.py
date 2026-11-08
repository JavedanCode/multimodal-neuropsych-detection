"""Matched waveform cleanup and resumable openSMILE feature extraction."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import tarfile
import time

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, sosfiltfilt

from src.dataset.audio_decode import RATE, decode_audio
from src.dataset.inventory import _is_unsafe_member
from src.training.validation import sha256

VARIANTS = ("raw", "dc", "highpass")


def clean_waveform(audio, variant):
    """Preserve timing and gain while applying one fixed cleanup operation."""
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1 or len(signal) <= RATE or not np.isfinite(signal).all():
        raise ValueError("Expected a finite mono recording longer than one second")
    if variant == "raw":
        return signal.astype(np.float32)
    signal = signal - signal.mean()
    if variant == "highpass":
        sections = butter(2, 50, btype="highpass", fs=RATE, output="sos")
        signal = sosfiltfilt(sections, signal)
    elif variant != "dc":
        raise ValueError("Unknown audio preprocessing condition")
    return signal.astype(np.float32)


def waveform_metrics(signal):
    values = np.asarray(signal, dtype=np.float64)
    return {
        "samples": len(values),
        "duration_seconds": len(values) / RATE,
        "dc_offset": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values**2))),
        "clipped_fraction": float(np.mean(np.abs(values) >= 0.999)),
        "peak": float(np.max(np.abs(values))),
    }


def summarize_descriptors(frame):
    """Preserve missing descriptors for training-only imputation later."""
    result = {}
    for name in frame:
        values = frame[name].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        result[f"audio__{name}__mean"] = float(values.mean()) if len(values) else None
        result[f"audio__{name}__std"] = float(values.std()) if len(values) else None
    return result


def read_waveform(archive_path):
    participant = archive_path.name.split("_")[0]
    wanted = f"{participant}_AUDIO.wav"
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            if Path(member.name).name != wanted:
                continue
            if not member.isfile() or member.size > 768 * 1024**2:
                raise ValueError("Invalid or oversized audio member")
            payload = archive.extractfile(member).read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError("Truncated audio member")
            signal, metadata = decode_audio(payload)
            return signal, metadata, hashlib.sha256(payload).hexdigest()
    raise ValueError("Missing audio recording")


def extract_participant(project, participant, destination, contract):
    import opensmile

    started = time.perf_counter()
    archive = project / "data/raw" / f"{participant}_P.tar.gz"
    stat = archive.stat()
    key = {
        "contract": contract,
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
    }
    target = destination / "participants" / f"{participant}.json"
    checksum = target.with_suffix(".sha256")
    if target.exists() and checksum.exists() and sha256(target) == checksum.read_text():
        saved = json.loads(target.read_text())
        if saved["key"] == key:
            return saved, True
    signal, metadata, raw_hash = read_waveform(archive)
    extractor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_workers=1,
    )
    features, audit = {}, {}
    for variant in VARIANTS:
        cleaned = clean_waveform(signal, variant)
        frame = extractor.process_signal(cleaned, RATE)
        if frame.shape[1] != 25 or not len(frame):
            raise ValueError("Unexpected eGeMAPSv02 descriptor schema")
        features[variant] = summarize_descriptors(frame)
        audit[variant] = {**waveform_metrics(cleaned), "descriptor_frames": len(frame)}
    saved = {
        "participant": int(participant),
        "key": key,
        "source_waveform_sha256": raw_hash,
        "recording": metadata,
        "features": features,
        "audit": audit,
        "seconds": time.perf_counter() - started,
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, indent=2, allow_nan=False))
    temporary.replace(target)
    checksum.write_text(sha256(target))
    return saved, False


def build(project, labels, workers=4):
    """Extract only the caller's validated training cohort; never load test audio."""
    import opensmile

    official = pd.read_parquet(project / "data/processed/labels.parquet")
    if (
        not labels.index.is_unique
        or not (official.loc[labels.index, "split"] == "train").all()
    ):
        raise ValueError("Unique official training participants required")
    pd.testing.assert_series_equal(labels, official.loc[labels.index, "PHQ_Binary"])
    if opensmile.__version__ != "2.6.0":
        raise ValueError("This recorded extraction requires openSMILE Python 2.6.0")
    destination = project / "data/processed/audio_preprocessing"
    (destination / "participants").mkdir(parents=True, exist_ok=True)
    package = Path(opensmile.__file__).parent
    contract = {
        "extractor": "eGeMAPSv02_LLD_mean_std",
        "opensmile": opensmile.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "source": sha256(Path(__file__)),
        "decoder": sha256(project / "src/dataset/audio_decode.py"),
        "protocol": sha256(project / "docs/methods/audio_preprocessing.md"),
        "package_files": {
            str(p.relative_to(package)): sha256(p)
            for p in sorted(package.rglob("*"))
            if p.is_file() and p.suffix in {".conf", ".inc", ".dll"}
        },
    }
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = pool.map(
            lambda pid: extract_participant(project, pid, destination, contract),
            labels.index,
        )
        for i, (record, cached) in enumerate(tasks, 1):
            records.append(record)
            logging.info(
                "Audio extraction %d/%d: %s (%s, %.1fs)",
                i,
                len(labels),
                record["participant"],
                "cached" if cached else "extracted",
                record["seconds"],
            )
    for variant in VARIANTS:
        frame = pd.DataFrame(
            [r["features"][variant] for r in records], index=labels.index
        )
        if frame.shape != (len(labels), 50):
            raise ValueError("Incomplete feature table")
        frame.to_parquet(destination / f"{variant}.parquet")
    labels.to_frame().to_parquet(destination / "labels.parquet")
    manifest = {
        "contract": contract,
        "partition": "train",
        "participants": list(map(int, labels.index)),
        "outputs": {p.name: sha256(p) for p in destination.glob("*.parquet")},
        "records": {
            str(r["participant"]): sha256(
                destination / "participants" / f"{r['participant']}.json"
            )
            for r in records
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return destination
