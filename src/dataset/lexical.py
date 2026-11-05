"""Screened training transcripts to fixed sparse hashed counts, without text export."""

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
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from src.dataset.inventory import _is_unsafe_member
from src.dataset.text_temporal import safe_rows
from src.training.validation import sha256

DIMENSIONS = 65536


def counts(texts):
    vectorizer = HashingVectorizer(
        n_features=DIMENSIONS, alternate_sign=False, norm=None, ngram_range=(1, 2)
    )
    return sparse.csr_matrix(vectorizer.transform(texts).sum(axis=0))


def read_transcript(path):
    wanted = path.name.split("_")[0] + "_Transcript.csv"
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError("Unsafe archive")
            if Path(member.name).name != wanted:
                continue
            if not member.isfile() or member.size > 16 * 1024**2:
                raise ValueError("Invalid transcript member")
            raw = archive.extractfile(member).read(member.size + 1)
            if len(raw) != member.size:
                raise ValueError("Truncated transcript")
            frame = pd.read_csv(io.BytesIO(raw), nrows=50001)
            if len(frame) > 50000:
                raise ValueError("Too many transcript rows")
            return frame, hashlib.sha256(raw).hexdigest()
    raise ValueError("Missing transcript")


def build(project):
    root = project / "data/processed"
    previous = json.loads((root / "text_temporal_v1/manifest.json").read_text())
    y = pd.read_parquet(root / "text_temporal_v1/labels.parquet")
    official = pd.read_parquet(root / "labels.parquet")
    pd.testing.assert_frame_equal(y, official.loc[y.index, ["PHQ_Binary"]])
    assert (official.loc[y.index, "split"] == "train").all()
    for name, digest in previous["output_hashes"].items():
        if not matches_hash(root / "text_temporal_v1" / name, digest):
            raise ValueError("Previous data changed")
    out = root / "lexical_v1"
    out.mkdir(exist_ok=True)
    cache = out / "participant_cache"
    cache.mkdir(exist_ok=True)
    contract = {
        "source": sha256(Path(__file__)),
        "screen_source": sha256(project / "src/dataset/text_temporal.py"),
        "dimensions": DIMENSIONS,
    }

    def one(pid):
        path = project / "data/raw" / f"{pid}_P.tar.gz"
        stat = path.stat()
        prior = previous["participants"][str(pid)]
        if (
            stat.st_size != prior["key"]["archive_size"]
            or stat.st_mtime_ns != prior["key"]["mtime"]
        ):
            raise ValueError("Archive changed")
        audit_path = project / "reports/data_audit/label_speech_cache" / f"{pid}.json"
        if not matches_hash(audit_path, prior["key"]["audit"]):
            raise ValueError("Timing audit changed")
        duration = json.loads(audit_path.read_text())["result"]["audio_seconds"]
        key = {
            **contract,
            "prior": prior["key"],
            "transcript_hash": prior["member_hashes"]["text"],
        }
        meta = cache / f"{pid}.json"
        target = cache / f"{pid}.npz"
        saved = json.loads(meta.read_text()) if meta.exists() else {}
        if (
            saved.get("key") != key
            or not target.exists()
            or not matches_hash(target, saved.get("hash"))
        ):
            frame, digest = read_transcript(path)
            if digest != key["transcript_hash"]:
                raise ValueError("Transcript changed")
            texts, _, _, screen = safe_rows(frame, duration)
            for k, v in screen.items():
                if prior["audit"][k] != v:
                    raise ValueError("Disclosure screen changed")
            matrix = counts(texts)
            if matrix.nnz == 0 or not np.isfinite(matrix.data).all():
                raise ValueError("Empty/invalid lexical counts")
            sparse.save_npz(target, matrix)
            saved = {
                "key": key,
                "audit": screen,
                "hash": sha256(target),
                "nonzero_bins": matrix.nnz,
                "total_ngrams": int(matrix.sum()),
            }
            meta.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        return int(pid), saved

    records = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (pid, saved) in enumerate(pool.map(one, y.index), 1):
            records[str(pid)] = saved
            logging.info("Lexical counts %s/%s", i, len(y))
    matrix = sparse.vstack(
        [sparse.load_npz(cache / f"{pid}.npz") for pid in y.index], format="csr"
    )
    sparse.save_npz(out / "counts.npz", matrix)
    y.to_parquet(out / "labels.parquet")
    manifest = {
        "contract": contract,
        "partition": "train",
        "participants": records,
        "source_manifest": sha256(root / "text_temporal_v1/manifest.json"),
        "output_hashes": {
            name: sha256(out / name) for name in ["counts.npz", "labels.parquet"]
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(build(Path(__file__).resolve().parents[2]))
