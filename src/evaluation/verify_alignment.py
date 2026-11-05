"""Validate every aligned artifact and independently replay source intervals."""

from src.training.provenance import matches_hash
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.training.validation import sha256
from src.dataset.alignment import read
from src.dataset.contracts import screened_rows


def verify(project):
    root = project / "data/processed/aligned_v2"
    m = json.loads((root / "manifest.json").read_text())
    for p, h in m["sources"].items():
        assert matches_hash(project / p, h)
    y = pd.read_parquet(root / "labels.parquet").PHQ_Binary
    official = pd.read_parquet(project / "data/processed/labels.parquet")
    assert (official.loc[y.index, "split"] == "train").all()
    pd.testing.assert_series_equal(y, official.loc[y.index, "PHQ_Binary"])
    assert matches_hash(root / "labels.parquet", m["labels_hash"])
    counts = []
    qualities = []
    backward = 0
    quarantined = 0
    for pid in y.index:
        p = root / "participant_cache" / f"{pid}.npz"
        meta = m["participants"][str(pid)]
        archive = project / "data/raw" / f"{pid}_P.tar.gz"
        stat = archive.stat()
        assert meta["key"] == {
            "sources": m["sources"],
            "size": stat.st_size,
            "mtime": stat.st_mtime_ns,
            "embedding_hash": sha256(
                project / "data/processed/mpnet_v1/participant_cache" / f"{pid}.npz"
            ),
        }
        assert matches_hash(p, meta["hash"])
        backward += meta["audit"]["backward_start_steps"]
        quarantined += meta["facial_timing_quarantined"]
        with np.load(p) as z:
            n = len(z["text"])
            counts.append(n)
            qualities.append(z["quality"])
            assert (
                z["text"].shape == (n, 768)
                and z["audio"].shape == (n, 46)
                and z["face"].shape == (n, 98)
            )
            assert z["masks"].shape == (n, 3) and z["quality"].shape == (n, 11)
            assert (np.diff(z["starts"]) >= 0).all() and (z["ends"] > z["starts"]).all()
            for name in ["text", "audio", "face", "quality", "starts", "ends"]:
                assert np.isfinite(z[name]).all()
            assert (z["quality"][:, 10] >= 0).all() and (z["quality"][:, 10] <= 1).all()
            assert (z["quality"][z["quality"][:, 10] == 0, 9] == 0).all()
            for i, name in [(1, "audio"), (2, "face")]:
                assert (z[name][~z["masks"][:, i]] == 0).all()
    replay = [int(y.index[0]), int(y.index[y.index >= 600][0])]
    schemas = json.loads((project / "data/processed/stage3_report.json").read_text())[
        "schemas"
    ]
    for pid in replay:
        data, hashes = read(project / "data/raw" / f"{pid}_P.tar.gz")
        assert hashes == m["participants"][str(pid)]["member_hashes"]
        rows = screened_rows(data["text"], data["wave_meta"]["duration_seconds"])
        order = np.argsort(rows["starts"], kind="stable")
        starts = rows["starts"][order]
        ends = rows["ends"][order]
        with (
            np.load(root / "participant_cache" / f"{pid}.npz") as z,
            np.load(
                project / "data/processed/mpnet_v1/participant_cache" / f"{pid}.npz"
            ) as old,
        ):
            np.testing.assert_array_equal(z["text"], old["embeddings"][order])
            np.testing.assert_array_equal(z["starts"], starts)
            for name, timecol, mod in [
                ("audio", "frameTime", "audio"),
                ("face", "timestamp", "facial"),
            ]:
                if (
                    name == "face"
                    and m["participants"][str(pid)]["facial_timing_quarantined"]
                ):
                    continue
                f = data[name]
                values = f[schemas[mod]["columns"]].to_numpy(float)
                valid = np.isfinite(values).all(1)
                if name == "face":
                    valid &= (f.success == 1) & f.confidence.between(0.8, 1)
                for i, (start, end) in enumerate(zip(starts, ends)):
                    selected = values[
                        (f[timecol].to_numpy() >= start)
                        & (f[timecol].to_numpy() < end)
                        & valid
                    ]
                    expected = (
                        np.r_[selected.mean(0), selected.std(0)]
                        if len(selected)
                        else np.zeros(values.shape[1] * 2)
                    )
                    np.testing.assert_allclose(
                        z[name][i], expected, atol=1e-5, rtol=1e-6
                    )
    quality = np.concatenate(qualities)
    result = {
        "participants": len(y),
        "utterances": sum(counts),
        "source_interval_replays": len(replay),
        "backward_steps_reordered": backward,
        "face_quarantines": int(quarantined),
        "quality_std": dict(zip(m["quality_columns"], map(float, quality.std(0)))),
        "speaker_validated": False,
        "hashes_alignment_and_confidence_verified": True,
    }
    (root / "verification_current.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(verify(Path(__file__).resolve().parents[2]))
