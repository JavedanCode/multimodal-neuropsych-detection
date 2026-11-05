"""Read-only label reconciliation and audio/transcript feasibility investigation."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import tarfile
import wave

import numpy as np
import pandas as pd

from src.dataset.inventory import _is_unsafe_member
from src.training.validation import sha256


def transcript_intervals(frame, duration):
    required = {"Start_Time", "End_Time", "Text"}
    if not required.issubset(frame):
        raise ValueError("Unexpected transcript schema")
    start = pd.to_numeric(frame.Start_Time, errors="coerce").to_numpy(float)
    end = pd.to_numeric(frame.End_Time, errors="coerce").to_numpy(float)
    nonempty = frame.Text.fillna("").astype(str).str.strip().ne("").to_numpy()
    valid = (
        np.isfinite(start)
        & np.isfinite(end)
        & (start >= 0)
        & (end > start)
        & (end <= duration + 0.1)
        & nonempty
    )
    intervals = []
    for left, right in sorted(zip(start[valid], np.minimum(end[valid], duration))):
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(right, intervals[-1][1])
        else:
            intervals.append([float(left), float(right)])
    return intervals, {
        "transcript_rows": len(frame),
        "invalid_or_empty_intervals": int((~valid).sum()),
        "speaker_column_present": any(
            c.lower() in ["speaker", "speaker_id", "role"] for c in frame
        ),
        "transcript_columns": list(frame),
        "transcribed_seconds": sum(b - a for a, b in intervals),
        "transcribed_duration_fraction": sum(b - a for a, b in intervals) / duration,
    }


def covered_frames(times, intervals):
    mask = np.zeros(len(times), dtype=bool)
    for left, right in intervals:
        mask |= (times >= left) & (times < right)
    return mask


def scan(path, inspect_features):
    pid = int(path.name.split("_")[0])
    audio = None
    transcript = None
    features = None
    hashes = {}
    names = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError("Unsafe member")
            name = Path(member.name).name
            if name not in [
                f"{pid}_AUDIO.wav",
                f"{pid}_Transcript.csv",
                f"{pid}_OpenSMILE2.3.0_egemaps.csv",
            ]:
                continue
            if not member.isfile():
                raise ValueError("Expected regular data member")
            names.append(name)
            if name.endswith("_AUDIO.wav"):
                # Read only the header; do not export recordings or speech content.
                with wave.open(
                    io.BytesIO(archive.extractfile(member).read(65536))
                ) as wav:
                    audio = {
                        "channels": wav.getnchannels(),
                        "sample_rate": wav.getframerate(),
                        "sample_width": wav.getsampwidth(),
                        "audio_seconds": wav.getnframes() / wav.getframerate(),
                    }
            elif name.endswith("_Transcript.csv"):
                if member.size > 16 * 1024 * 1024:
                    raise ValueError("Oversized transcript")
                raw = archive.extractfile(member).read()
                hashes["transcript_sha256"] = (
                    __import__("hashlib").sha256(raw).hexdigest()
                )
                transcript = pd.read_csv(io.BytesIO(raw))
            elif inspect_features:
                if member.size > 256 * 1024 * 1024:
                    raise ValueError("Oversized features")
                raw = archive.extractfile(member).read()
                hashes["egemaps_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
                features = pd.read_csv(
                    io.BytesIO(raw),
                    sep=";",
                    usecols=["frameTime", "F0semitoneFrom27.5Hz_sma3nz"],
                )
            if (
                audio is not None
                and transcript is not None
                and (not inspect_features or features is not None)
            ):
                break
    if audio is None or transcript is None or (inspect_features and features is None):
        raise ValueError(f"Missing requested file in {path.name}")
    intervals, stats = transcript_intervals(transcript, audio["audio_seconds"])
    result = {
        "participant_id": pid,
        **audio,
        **stats,
        **hashes,
        "inspected_members": names,
    }
    if inspect_features:
        times = pd.to_numeric(features.frameTime, errors="coerce").to_numpy(float)
        if (
            len(times) < 2
            or not np.isfinite(times).all()
            or (times < 0).any()
            or (np.diff(times) < 0).any()
        ):
            raise ValueError("Invalid feature timeline")
        mask = covered_frames(times, intervals)
        f0 = pd.to_numeric(
            features["F0semitoneFrom27.5Hz_sma3nz"], errors="coerce"
        ).to_numpy(float)
        result.update(
            feature_rows=len(times),
            frame_start=float(times[0]),
            frame_end=float(times[-1]),
            median_step=float(np.median(np.diff(times))),
            max_step=float(np.max(np.diff(times))),
            repeated_timestamps=int((np.diff(times) == 0).sum()),
            end_gap_seconds=float(audio["audio_seconds"] - times[-1]),
            transcript_frame_fraction=float(mask.mean()),
            zero_f0_fraction=float((f0 == 0).mean()),
            zero_f0_inside_transcript=(
                float((f0[mask] == 0).mean()) if mask.any() else None
            ),
            zero_f0_outside_transcript=(
                float((f0[~mask] == 0).mean()) if (~mask).any() else None
            ),
        )
    return result


def label_audit(raw, output):
    splits = pd.concat(
        [
            pd.read_csv(raw / f"{s}_split.csv").assign(split=s)
            for s in ["train", "dev", "test"]
        ]
    ).set_index("Participant_ID")
    detail = pd.read_csv(raw / "detailed_lables.csv").set_index("Participant")
    legacy = pd.read_csv(raw / "Detailed_PHQ8_Labels.csv").set_index("Participant_ID")
    mapped = pd.read_csv(raw / "metadata_mapped.csv").set_index("Participant_ID")
    for f in [splits, detail, legacy, mapped]:
        if not f.index.is_unique:
            raise ValueError("Duplicate label IDs")
    columns = [c for c in detail if c.startswith("PHQ8_")]
    if len(columns) != 8 or not detail[columns].isin([0, 1, 2, 3]).all().all():
        raise ValueError("Invalid questionnaire items")
    result = splits[["split", "PHQ_Binary", "PHQ_Score"]].copy()
    result["item_sum"] = detail[columns].sum(axis=1).reindex(result.index)
    result["master_binary"] = detail.Depression_label.reindex(result.index)
    result["score_ge10"] = (result.PHQ_Score >= 10).astype(int)
    result["discordant"] = result.PHQ_Binary != result.score_ge10
    result["interview_condition"] = np.where(result.index >= 600, "AI", "WoZ")
    # Participant-level records remain in ignored reports, never in tracked docs.
    result.to_csv(output / "label_reconciliation.csv")
    totals = []
    for split, g in result.groupby("split"):
        totals.append(
            {
                "split": split,
                "n": len(g),
                "released_positive": int(g.PHQ_Binary.sum()),
                "score_ge10_positive": int(g.score_ge10.sum()),
                "discordant": int(g.discordant.sum()),
                "WoZ": int((g.interview_condition == "WoZ").sum()),
                "AI": int((g.interview_condition == "AI").sum()),
            }
        )
    c = result.index.intersection(mapped.index)
    l = result.index.intersection(legacy.index)
    discrepancies = legacy.loc[l, "PHQ_8Total"] != result.loc[l, "PHQ_Score"]
    legacy.loc[l[discrepancies]].to_csv(
        output / "legacy_questionnaire_discrepancies.csv"
    )
    summary = {
        "by_split": totals,
        "master_binary_disagreements": int(
            (result.master_binary != result.PHQ_Binary).sum()
        ),
        "master_item_sum_disagreements": int(
            (result.item_sum != result.PHQ_Score).sum()
        ),
        "master_split_disagreements": int(
            (detail["split"].reindex(result.index) != result["split"]).sum()
        ),
        "metadata_overlap": len(c),
        "metadata_binary_disagreements": int(
            (mapped.loc[c, "PHQ_Binary"] != result.loc[c, "PHQ_Binary"]).sum()
        ),
        "metadata_score_disagreements": int(
            (mapped.loc[c, "PHQ_Score"] != result.loc[c, "PHQ_Score"]).sum()
        ),
        "legacy_overlap": len(l),
        "legacy_total_vs_split_disagreements": int(discrepancies.sum()),
        "legacy_internal_sum_disagreements": int(
            (legacy.drop(columns="PHQ_8Total").sum(axis=1) != legacy.PHQ_8Total).sum()
        ),
        "discordant_score_histogram": {
            str(k): int(v)
            for k, v in result.loc[result.discordant, "PHQ_Score"]
            .value_counts()
            .sort_index()
            .items()
        },
        "positive_below10": int(
            ((result.PHQ_Binary == 1) & (result.PHQ_Score < 10)).sum()
        ),
    }
    return result, summary


def run(project):
    raw = project / "data/raw"
    output = (
        project
        / "reports/data_audit"
        / ("label_speech_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    output.mkdir(parents=True)
    labels, label_summary = label_audit(raw, output)
    cache = project / "reports/data_audit/label_speech_cache"
    cache.mkdir(exist_ok=True)

    def one(pid):
        path = raw / f"{pid}_P.tar.gz"
        stat = path.stat()
        inspect_features = labels.loc[pid, "split"] == "train"
        key = {
            "source": sha256(Path(__file__)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "features": inspect_features,
        }
        target = cache / f"{pid}.json"
        if target.exists():
            saved = json.loads(target.read_text())
            if saved["key"] == key:
                return saved["result"]
        result = scan(path, inspect_features)
        target.write_text(
            json.dumps({"key": key, "result": result}, allow_nan=False),
            encoding="utf-8",
        )
        return result

    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(one, labels.index), 1):
            records.append(result)
            logging.info("Audited audio/transcript %s/%s", i, len(labels))
    table = (
        pd.DataFrame(records)
        .set_index("participant_id")
        .join(labels[["split", "interview_condition"]])
    )
    table.to_csv(output / "speech_timing_audit.csv")
    train = table[table["split"] == "train"]
    report = {
        "scope": "Label-file reconciliation across splits; file headers/transcript timing across splits; acoustic diagnostics on train only.",
        "labels": label_summary,
        "archives_checked": len(table),
        "speaker_column_count": int(table.speaker_column_present.sum()),
        "transcript_schema_counts": table.transcript_columns.map(lambda v: ",".join(v))
        .value_counts()
        .to_dict(),
        "audio_format_counts": table.groupby(
            ["channels", "sample_rate", "sample_width"]
        )
        .size()
        .rename_axis(["channels", "rate", "bytes"])
        .reset_index(name="n")
        .to_dict("records"),
        "invalid_or_empty_transcript_intervals": int(
            table.invalid_or_empty_intervals.sum()
        ),
        "training_timing_summary": train[
            [
                "audio_seconds",
                "feature_rows",
                "median_step",
                "max_step",
                "repeated_timestamps",
                "end_gap_seconds",
                "transcript_frame_fraction",
                "zero_f0_fraction",
                "zero_f0_inside_transcript",
                "zero_f0_outside_transcript",
            ]
        ]
        .agg(["min", "median", "max"])
        .to_dict(),
        "source_sha256": sha256(Path(__file__)),
        "input_sha256": {p.name: sha256(p) for p in raw.glob("*.csv")},
        "models_trained": False,
        "labels_modified": False,
        "test_predictions_generated": False,
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(output)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(Path(__file__).resolve().parents[2])
