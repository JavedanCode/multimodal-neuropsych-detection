"""Frozen local MPNet embeddings with identical screening and pooling to MiniLM."""

from src.training.provenance import matches_hash
import json
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.dataset.lexical import read_transcript
from src.dataset.text_temporal import safe_rows, encode
from src.training.validation import sha256


def build(project, limit=None):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required by extraction protocol")
    root = project / "data/processed"
    prior = json.loads((root / "text_temporal_v1/manifest.json").read_text())
    y = pd.read_parquet(root / "text_temporal_v1/labels.parquet")
    official = pd.read_parquet(root / "labels.parquet")
    pd.testing.assert_frame_equal(y, official.loc[y.index, ["PHQ_Binary"]])
    assert (official.loc[y.index, "split"] == "train").all()
    for name, digest in prior["output_hashes"].items():
        if not matches_hash(root / "text_temporal_v1" / name, digest):
            raise ValueError("Previous features changed")
    model_dir = project / "models/pretrained/all-mpnet-base-v2"
    model_hashes = {p.name: sha256(p) for p in model_dir.iterdir() if p.is_file()}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model, loading = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
        add_pooling_layer=False,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
        raise ValueError("Incomplete encoder weights")
    model = model.eval().to("cuda")
    model.requires_grad_(False)
    out = root / "mpnet_v1"
    out.mkdir(exist_ok=True)
    cache = out / "participant_cache"
    cache.mkdir(exist_ok=True)
    contract = {
        "source": sha256(Path(__file__)),
        "encoding_source": sha256(project / "src/dataset/text_temporal.py"),
        "reader_source": sha256(project / "src/dataset/lexical.py"),
        "models": model_hashes,
        "revision": json.loads((model_dir / "revision.json").read_text()),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "loading_info": loading,
        "batch_size": 32,
        "dtype": "float32",
        "attention": "eager",
    }
    records = {}
    means = []
    for i, pid in enumerate(y.index[:limit] if limit else y.index, 1):
        previous = prior["participants"][str(pid)]
        path = project / "data/raw" / f"{pid}_P.tar.gz"
        stat = path.stat()
        if (
            stat.st_size != previous["key"]["archive_size"]
            or stat.st_mtime_ns != previous["key"]["mtime"]
        ):
            raise ValueError("Archive changed")
        audit_file = project / "reports/data_audit/label_speech_cache" / f"{pid}.json"
        if not matches_hash(audit_file, previous["key"]["audit"]):
            raise ValueError("Timing audit changed")
        duration = json.loads(audit_file.read_text())["result"]["audio_seconds"]
        key = {
            **contract,
            "previous": previous["key"],
            "transcript_hash": previous["member_hashes"]["text"],
        }
        target = cache / f"{pid}.npz"
        meta = cache / f"{pid}.json"
        saved = json.loads(meta.read_text()) if meta.exists() else {}
        if (
            saved.get("key") != key
            or not target.exists()
            or not matches_hash(target, saved.get("hash"))
        ):
            started = time.perf_counter()
            frame, digest = read_transcript(path)
            if digest != key["transcript_hash"]:
                raise ValueError("Transcript changed")
            texts, _, _, audit = safe_rows(frame, duration)
            for k, v in audit.items():
                if previous["audit"][k] != v:
                    raise ValueError("Screen changed")
            torch.cuda.reset_peak_memory_stats()
            embeddings = encode(texts, tokenizer, model)
            if (
                embeddings.shape != (len(texts), 768)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("Invalid embeddings")
            np.savez_compressed(target, embeddings=embeddings, mean=embeddings.mean(0))
            saved = {
                "key": key,
                "hash": sha256(target),
                "audit": audit,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            meta.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        records[str(pid)] = saved
        with np.load(target) as data:
            means.append(data["mean"])
        logging.info(
            "MPNet %s/%s: %.1fs, %.2f GiB",
            i,
            len(y),
            saved["elapsed_seconds"],
            saved["peak_cuda_bytes"] / 1024**3,
        )
    if limit:
        return out
    pd.DataFrame(
        means, index=y.index, columns=[f"text__{i}" for i in range(768)]
    ).to_parquet(out / "features.parquet")
    y.to_parquet(out / "labels.parquet")
    manifest = {
        "partition": "train",
        "contract": contract,
        "participants": records,
        "previous_manifest_hash": sha256(root / "text_temporal_v1/manifest.json"),
        "output_hashes": {
            n: sha256(out / n) for n in ["features.parquet", "labels.parquet"]
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(build(Path(__file__).resolve().parents[2], args.limit))
