"""Local training-only transcript embeddings and aligned coarse sequences."""

from src.training.provenance import matches_hash
import hashlib
import io
import json
import logging
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.inventory import _is_unsafe_member
from src.training.validation import sha256

BINS = 32
# Successful caches from the initial strict version are identical. The amendment
# only masks previously rejected face sequences; retain their original source hash.
COMPATIBLE_SOURCE = "f4db4d9414b8cbb5ea90e0dc37a4da8d56d81b665da9a3b7db60c12ab3396577"
DISCLOSURE = re.compile(
    r"\b(phq\s*[- ]?\s*[89]?|questionnaire|diagnos\w*|depression\s+(score|label)|"
    r"not at all|several days|more than half the days|nearly every day)\b",
    re.I,
)


def safe_rows(frame, duration):
    if not {"Start_Time", "End_Time", "Text"} <= set(frame):
        raise ValueError("Unexpected transcript schema")
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
    confidence = pd.to_numeric(
        frame.get("Confidence", pd.Series(np.ones(len(frame)))), errors="coerce"
    ).to_numpy(float)
    unknown = ~np.isfinite(confidence) | (confidence < 0) | (confidence > 1)
    confidence[unknown] = 1
    keep = valid & ~excluded
    if not keep.any():
        raise ValueError("No screened text")
    return (
        text[keep].tolist(),
        (starts[keep] + ends[keep]) / 2,
        confidence[keep],
        {
            "rows": len(frame),
            "invalid_or_empty": int((~valid).sum()),
            "direct_disclosures": int(hit.sum()),
            "excluded_with_neighbors": int(excluded.sum()),
            "retained": int(keep.sum()),
            "unknown_confidence": int(unknown.sum()),
            "speaker_columns": [c for c in frame if "speaker" in c.lower()],
        },
    )


def bin_means(values, times, duration, valid=None):
    values = np.asarray(values, float)
    times = np.asarray(times, float)
    if (
        len(values) != len(times)
        or not np.isfinite(times).all()
        or (times < 0).any()
        or (times > duration + 1).any()
    ):
        raise ValueError("Invalid alignment")
    which = np.minimum((times / duration * BINS).astype(int), BINS - 1)
    valid = np.ones(len(values), bool) if valid is None else np.asarray(valid, bool)
    out = np.zeros((BINS, values.shape[1]), np.float32)
    quality = np.zeros(BINS, np.float32)
    mask = np.zeros(BINS, bool)
    for b in range(BINS):
        total = which == b
        selected = total & valid & np.isfinite(values).all(axis=1)
        quality[b] = selected.sum() / max(1, total.sum())
        if selected.any():
            out[b] = values[selected].mean(axis=0)
            mask[b] = True
    return out, mask, quality


def read_members(path):
    pid = path.name.split("_")[0]
    wanted = {
        f"{pid}_Transcript.csv": ("text", ","),
        f"{pid}_OpenSMILE2.3.0_egemaps.csv": ("audio", ";"),
        f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv": ("face", ","),
    }
    found, hashes = {}, {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError("Unsafe archive")
            name = Path(member.name).name
            if name not in wanted:
                continue
            kind, sep = wanted[name]
            if kind in found or not member.isfile() or member.size > 256 * 1024**2:
                raise ValueError("Invalid member")
            raw = archive.extractfile(member).read(member.size + 1)
            if len(raw) != member.size:
                raise ValueError("Truncated member")
            frame = pd.read_csv(io.BytesIO(raw), sep=sep, nrows=500001)
            frame.columns = frame.columns.str.strip()
            if len(frame) > 500000 or frame.columns.duplicated().any():
                raise ValueError("Invalid table")
            found[kind] = frame
            hashes[kind] = hashlib.sha256(raw).hexdigest()
            if len(found) == 3:
                break
    if len(found) != 3:
        raise ValueError("Missing members")
    return found, hashes


def encode(texts, tokenizer, model):
    import torch

    chunks, owners = [], []
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        for offset in range(0, max(1, len(ids)), 254):
            chunks.append(
                tokenizer.prepare_for_model(
                    ids[offset : offset + 254],
                    add_special_tokens=True,
                    return_attention_mask=True,
                )
            )
            owners.append(i)
    vectors = []
    with torch.inference_mode():
        for offset in range(0, len(chunks), 32):
            inputs = tokenizer.pad(
                chunks[offset : offset + 32], padding=True, return_tensors="pt"
            ).to("cuda")
            hidden = model(**inputs).last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.extend(
                torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            )
    vectors = np.asarray(vectors)
    return np.stack(
        [vectors[np.asarray(owners) == i].mean(0) for i in range(len(texts))]
    )


def facial_bins(frame, columns, duration):
    times = pd.to_numeric(frame.timestamp, errors="coerce").to_numpy(float)
    if (
        not len(times)
        or not np.isfinite(times).all()
        or (times < 0).any()
        or (np.diff(times) < 0).any()
    ):
        raise ValueError("Invalid facial timeline")
    gap = float(times[-1] - duration)
    if abs(gap) > 1:
        return (
            np.zeros((BINS, len(columns)), np.float32),
            np.zeros(BINS, bool),
            np.zeros(BINS, np.float32),
            gap,
        )
    valid = (frame.success == 1) & frame.confidence.between(0.8, 1)
    return (*bin_means(frame[columns].to_numpy(float), times, duration, valid), gap)


def build(project):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = project / "data/processed"
    labels = pd.read_parquet(root / "segmentation_study_v1/labels.parquet")
    official = pd.read_parquet(root / "labels.parquet")
    assert (official.loc[labels.index, "split"] == "train").all()
    pd.testing.assert_frame_equal(labels, official.loc[labels.index, ["PHQ_Binary"]])
    schemas = json.loads((root / "stage3_report.json").read_text())["schemas"]
    model_dir = project / "models/pretrained/all-MiniLM-L6-v2"
    model_hashes = {p.name: sha256(p) for p in model_dir.iterdir() if p.is_file()}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True
        )
        .eval()
        .to("cuda")
    )
    model.requires_grad_(False)
    out = root / "text_temporal_v1"
    out.mkdir(exist_ok=True)
    cache = out / "participant_cache"
    cache.mkdir(exist_ok=True)
    records, arrays = {}, []
    for number, pid in enumerate(labels.index, 1):
        archive = project / "data/raw" / f"{pid}_P.tar.gz"
        audit_file = project / "reports/data_audit/label_speech_cache" / f"{pid}.json"
        audit = json.loads(audit_file.read_text())
        duration = audit["result"]["audio_seconds"]
        stat = archive.stat()
        if (
            audit["key"]["size"] != stat.st_size
            or audit["key"]["mtime_ns"] != stat.st_mtime_ns
        ):
            raise ValueError("Stale timing audit")
        key = {
            "source": sha256(Path(__file__)),
            "models": model_hashes,
            "archive_size": stat.st_size,
            "mtime": stat.st_mtime_ns,
            "audit": sha256(audit_file),
        }
        metadata, target = cache / f"{pid}.json", cache / f"{pid}.npz"
        saved = json.loads(metadata.read_text()) if metadata.exists() else {}
        old_key = saved.get("key", {})
        compatible = (
            old_key.get("source") in [key["source"], COMPATIBLE_SOURCE]
            and {**old_key, "source": key["source"]} == key
        )
        if (
            not compatible
            or not target.exists()
            or not matches_hash(target, saved.get("sha256"))
        ):
            tables, hashes = read_members(archive)
            if (
                hashes["text"] != audit["result"]["transcript_sha256"]
                or hashes["audio"] != audit["result"]["egemaps_sha256"]
            ):
                raise ValueError("Transcript/acoustic audit mismatch")
            texts, times, confidence, stats = safe_rows(tables["text"], duration)
            embeddings = encode(texts, tokenizer, model)
            text, tm, tq = bin_means(embeddings, times, duration)
            conf, _, _ = bin_means(confidence[:, None], times, duration)
            tq = conf[:, 0] * tm
            audio = tables["audio"]
            av, am, aq = bin_means(
                audio[schemas["audio"]["columns"]].to_numpy(float),
                audio.frameTime,
                duration,
            )
            face = tables["face"]
            fv, fm, fq, gap = facial_bins(face, schemas["facial"]["columns"], duration)
            stats["facial_timing_quarantined"] = abs(gap) > 1
            stats["facial_end_gap_seconds"] = gap
            np.savez_compressed(
                target,
                text=text,
                audio=av,
                face=fv,
                masks=np.stack([tm, am, fm]),
                quality=np.stack([tq, aq, fq]),
                text_mean=embeddings.mean(0),
            )
            saved = {
                "key": key,
                "audit": stats,
                "member_hashes": hashes,
                "sha256": sha256(target),
            }
            metadata.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        records[str(pid)] = saved
        with np.load(target) as data:
            arrays.append({k: data[k] for k in data.files})
        logging.info("Text and sequence extraction %s/%s", number, len(labels))
    np.savez_compressed(
        out / "sequences.npz",
        **{k: np.stack([r[k] for r in arrays]) for k in arrays[0]},
        ids=labels.index.to_numpy(),
    )
    labels.to_parquet(out / "labels.parquet")
    manifest = {
        "partition": "train",
        "source": sha256(Path(__file__)),
        "models": model_hashes,
        "compatible_cache_source": COMPATIBLE_SOURCE,
        "revision": json.loads((model_dir / "revision.json").read_text()),
        "participants": records,
        "output_hashes": {
            n: sha256(out / n) for n in ["sequences.npz", "labels.parquet"]
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(build(Path(__file__).resolve().parents[2]))
