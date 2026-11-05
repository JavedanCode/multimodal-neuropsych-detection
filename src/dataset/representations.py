"""Bounded extraction of supplied deep audio and facial temporal summaries."""

from src.training.provenance import matches_hash
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
from src.training.validation import sha256

MAX_BYTES = 256 * 1024 * 1024
MAX_ROWS = 500000


def timeline(frame, column):
    t = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if (
        not len(t)
        or len(t) > MAX_ROWS
        or not np.isfinite(t).all()
        or (t < 0).any()
        or (np.diff(t) < 0).any()
    ):
        raise ValueError("Invalid feature timeline")
    return t


def facial_summary(frame, columns):
    t = timeline(frame, "timestamp")
    windows = np.floor(t / 10).astype(np.int64)
    confidence = pd.to_numeric(frame.confidence, errors="coerce").to_numpy(float)
    success = pd.to_numeric(frame.success, errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(confidence)
        & (confidence >= 0.8)
        & (confidence <= 1)
        & (success == 1)
    )
    features = {}
    for c in columns:
        values = pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
        finite = np.isfinite(values) & keep
        v = values[finite]
        stats = dict.fromkeys(
            [
                "mean",
                "std",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "iqr",
                "adjacent_change",
                "window_mean_std",
            ]
        )
        if len(v):
            q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
            adj = finite[:-1] & finite[1:] & (np.diff(t) <= 0.1)
            delta = np.abs(values[1:][adj] - values[:-1][adj])
            means = pd.Series(v).groupby(windows[finite]).mean().to_numpy()
            stats.update(
                mean=float(v.mean()),
                std=float(v.std()),
                **{k: float(n) for k, n in zip(["p10", "p25", "p50", "p75", "p90"], q)},
                iqr=float(q[3] - q[1]),
                adjacent_change=float(delta.mean()) if len(delta) else None,
                window_mean_std=float(means.std()),
            )
        features.update({f"facial__{c}__{s}": v for s, v in stats.items()})
    return features, {
        "rows": len(t),
        "retained_fraction": float(keep.mean()),
        "frame_start": float(t[0]),
        "frame_end": float(t[-1]),
    }


def dense_summary(frame, expected_neurons=1920):
    t = timeline(frame, "timeStamp")
    cols = [f"neuron_{i}" for i in range(expected_neurons)]
    if set(frame.columns) != {"name", "timeStamp", *cols}:
        raise ValueError("Unexpected DenseNet schema")
    values = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    means = np.divide(
        np.where(valid, values, 0).sum(axis=0),
        count,
        out=np.zeros(len(cols)),
        where=count > 0,
    )
    squared = np.where(valid, values - means, 0) ** 2
    std = np.sqrt(
        np.divide(squared.sum(axis=0), count, out=np.zeros(len(cols)), where=count > 0)
    )
    features = {}
    for i, c in enumerate(cols):
        features[f"audio__dense_{c}__mean"] = float(means[i]) if count[i] else None
        features[f"audio__dense_{c}__std"] = float(std[i]) if count[i] else None
    return features, {
        "rows": len(t),
        "frame_start": float(t[0]),
        "frame_end": float(t[-1]),
        "invalid_fraction": float(1 - valid.mean()),
    }


def extract(path, columns):
    pid = int(path.name.split("_")[0])
    result = {}
    wanted = {
        f"{pid}_densenet201.csv": "dense",
        f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv": "facial",
    }
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            name = Path(member.name).name
            if name not in wanted:
                continue
            kind = wanted[name]
            if kind in result or not member.isfile() or member.size > MAX_BYTES:
                raise ValueError("Invalid/oversized member")
            raw = archive.extractfile(member).read(MAX_BYTES + 1)
            if len(raw) != member.size:
                raise ValueError("Member size mismatch")
            frame = pd.read_csv(io.BytesIO(raw), nrows=MAX_ROWS + 1)
            frame.columns = frame.columns.str.strip()
            if frame.columns.duplicated().any():
                raise ValueError("Duplicate feature columns")
            features, diagnostics = (
                dense_summary(frame)
                if kind == "dense"
                else facial_summary(frame, columns)
            )
            result[kind] = {
                "features": features,
                "diagnostics": diagnostics,
                "member_sha256": hashlib.sha256(raw).hexdigest(),
            }
            if len(result) == 2:
                break
    if len(result) != 2:
        raise ValueError("Missing requested feature member")
    return result


def build(project):
    root = project / "data/processed"
    labels = pd.read_parquet(root / "labels.parquet")
    prior = json.loads((root / "segmentation_study_v1/manifest.json").read_text())
    for name, digest in prior["output_hashes"].items():
        if not matches_hash(root / "segmentation_study_v1" / name, digest):
            raise ValueError("Prior cohort table changed")
    previous = pd.read_parquet(root / "segmentation_study_v1/labels.parquet")
    ids = labels.index[
        (labels.split == "train") & ~labels.index.isin(prior["excluded_ids"])
    ]
    if not previous.equals(labels.loc[ids, ["PHQ_Binary"]]):
        raise ValueError("Prior cohort mismatch")
    columns = json.loads((root / "stage3_report.json").read_text())["schemas"][
        "facial"
    ]["columns"]
    out = root / "representation_study_v1"
    out.mkdir(exist_ok=True)
    cache = out / "participant_cache"
    cache.mkdir(exist_ok=True)
    contract = {
        "source": sha256(Path(__file__)),
        "facial_columns": columns,
        "max_bytes": MAX_BYTES,
        "max_rows": MAX_ROWS,
    }

    def one(pid):
        path = project / "data/raw" / f"{pid}_P.tar.gz"
        stat = path.stat()
        key = {
            **contract,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "root": str(path.parent.resolve()),
        }
        target = cache / f"{pid}.json"
        saved = json.loads(target.read_text()) if target.exists() else {}
        if saved.get("key") != key:
            saved = {"key": key, "data": extract(path, columns)}
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(saved, allow_nan=False), encoding="utf-8")
            tmp.replace(target)
        return int(pid), saved["data"]

    records = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (pid, data) in enumerate(pool.map(one, ids), 1):
            records[pid] = data
            logging.info("Deep audio/facial summaries %s/%s", i, len(ids))
    original = (
        pd.read_parquet(root / "features.parquet").loc[ids].filter(like="facial__")
    )
    for kind in ["dense", "facial"]:
        x = pd.DataFrame.from_dict(
            {pid: r[kind]["features"] for pid, r in records.items()}, orient="index"
        ).reindex(ids)
        x.index.name = labels.index.name
        if kind == "facial":
            np.testing.assert_allclose(
                x[original.columns], original, rtol=1e-8, atol=1e-8, equal_nan=True
            )
        x.to_parquet(out / f"{kind}.parquet")
    labels.loc[ids, ["PHQ_Binary"]].to_parquet(out / "labels.parquet")
    manifest = {
        "contract": contract,
        "partition": "train",
        "excluded_ids": prior["excluded_ids"],
        "facial_baseline_verified": True,
        "participant_ids": list(map(int, ids)),
        "participants": {
            str(pid): {k: {n: v for n, v in r[k].items() if n != "features"} for k in r}
            for pid, r in records.items()
        },
        "source_tables": {
            n: sha256(root / n)
            for n in [
                "features.parquet",
                "labels.parquet",
                "stage3_report.json",
                "segmentation_study_v1/manifest.json",
            ]
        },
        "output_hashes": {
            n: sha256(out / n)
            for n in ["dense.parquet", "facial.parquet", "labels.parquet"]
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(build(Path(__file__).resolve().parents[2]))
