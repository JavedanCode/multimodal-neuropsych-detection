"""Label-free participant/interviewer separation on supplied eGeMAPS descriptors.

Protocol: docs/methods/participant_pooled.md. The virtual interviewer has one
voice per interview domain, so a domain template is learned from sessions where
the participant's pitch is clearly lower, then every voiced segment is assigned
to the interviewer template or a session-specific participant model. No
questionnaire label is read. Transcripts are used only to measure agreement
with scripted prompts; they never influence the assignment.
"""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd

from src.dataset.contracts import domain
from src.dataset.inventory import _is_unsafe_member
from src.training.provenance import matches_hash
from src.training.validation import sha256

DIRECTORY = "data/processed/participant_speech_v1"
F0 = "F0semitoneFrom27.5Hz_sma3nz"
SEGMENT_COLUMNS = [
    F0,
    "HNRdBACF_sma3nz",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "Loudness_sma3",
]
MAX_GAP_FRAMES = 20  # 0.2 s at the supplied 10 ms hop
MIN_VOICED_FRAMES = 20
ANCHOR_MAX_PITCH = 27.0  # session median segment pitch (semitones from 27.5 Hz)
INTERVIEWER_PITCH = (36.0, 45.0)
MIN_ANCHOR_SEGMENTS = 150
ITERATIONS = 10
MIN_PARTICIPANT_SEGMENTS = 20
PROMPT_MIN_SESSIONS = 15
PLACEHOLDER = -201.0


def read_session(archive_path):
    pid = int(archive_path.name.split("_")[0])
    wanted = {
        f"{pid}_opensmile2.3.0_egemaps.csv": "audio",
        f"{pid}_transcript.csv": "transcript",
    }
    found = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member in {archive_path.name}")
            kind = wanted.get(Path(member.name).name.lower())
            if kind and member.isfile():
                found[kind] = archive.extractfile(member).read()
            if len(found) == 2:
                break
    if "audio" not in found:
        raise ValueError(f"Missing descriptors in {archive_path.name}")
    audio = pd.read_csv(io.BytesIO(found["audio"]), sep=";")
    audio.columns = audio.columns.str.strip()
    transcript = None
    if "transcript" in found:
        transcript = pd.read_csv(io.BytesIO(found["transcript"]))
    return audio, transcript


def voiced_segments(voiced):
    index = np.flatnonzero(voiced)
    if not len(index):
        return np.zeros((0, 2), int)
    cuts = np.flatnonzero(np.diff(index) > MAX_GAP_FRAMES)
    starts = np.r_[index[0], index[cuts + 1]]
    ends = np.r_[index[cuts], index[-1]]
    keep = [voiced[s : e + 1].sum() >= MIN_VOICED_FRAMES for s, e in zip(starts, ends)]
    return np.column_stack([starts, ends])[np.array(keep, bool)]


def session_statistics(audio, columns):
    """Per-segment medians for assignment and moment sums for later summaries."""
    values = audio[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values[values == PLACEHOLDER] = np.nan
    voiced = audio[F0].to_numpy(float) > 0
    segments = voiced_segments(voiced)
    positions = [columns.index(c) for c in SEGMENT_COLUMNS]
    medians, counts, sums, squares = [], [], [], []
    for start, end in segments:
        block = values[start : end + 1][voiced[start : end + 1]]
        finite = np.isfinite(block)
        medians.append(np.nanmedian(block[:, positions], axis=0))
        counts.append(finite.sum(0))
        sums.append(np.where(finite, block, 0).sum(0))
        squares.append(np.where(finite, block, 0).__pow__(2).sum(0))
    shape = (-1, len(columns))
    return {
        "segments": segments,
        "medians": np.array(medians).reshape(-1, len(SEGMENT_COLUMNS)),
        "counts": np.array(counts).reshape(shape),
        "sums": np.array(sums).reshape(shape),
        "squares": np.array(squares).reshape(shape),
        "frames": len(audio),
        "voiced_frames": int(voiced.sum()),
        "times": audio["frameTime"].to_numpy(float)[segments].reshape(-1, 2),
    }


def assignment_features(medians):
    """Voice-quality features; loudness is taken relative to the session."""
    features = np.array(medians, float)
    loudness = np.log(features[:, -1] + 1e-4)
    features[:, -1] = loudness - np.median(loudness)
    return features


def fit_template(sessions):
    """Interviewer mean/variance from sessions with clearly lower participant pitch."""
    rows = []
    for stats in sessions:
        features = assignment_features(stats["medians"])
        if len(features) and np.median(features[:, 0]) < ANCHOR_MAX_PITCH:
            pitch = features[:, 0]
            rows.append(features[(pitch >= INTERVIEWER_PITCH[0]) & (pitch <= INTERVIEWER_PITCH[1])])
    rows = np.vstack(rows) if rows else np.zeros((0, len(SEGMENT_COLUMNS)))
    rows = rows[np.isfinite(rows).all(1)]
    if len(rows) < MIN_ANCHOR_SEGMENTS:
        return None
    return {"mean": rows.mean(0), "var": rows.var(0) + 1e-6, "segments": len(rows)}


def log_density(features, mean, var):
    return -0.5 * (np.log(2 * np.pi * var) + (features - mean) ** 2 / var).sum(1)


def assign(stats, template):
    """True where a segment is assigned to the interviewer."""
    features = assignment_features(stats["medians"])
    usable = np.isfinite(features).all(1)
    interviewer = np.zeros(len(features), bool)
    if template is None or usable.sum() < 2 * MIN_PARTICIPANT_SEGMENTS:
        return interviewer, {"separated": False, "reason": "no template or too few segments"}
    x = features[usable]
    current = log_density(x, template["mean"], template["var"]) > log_density(
        x, np.median(x, 0), x.var(0) + 1e-6
    )
    for _ in range(ITERATIONS):
        if (~current).sum() < MIN_PARTICIPANT_SEGMENTS:
            return interviewer, {"separated": False, "reason": "participant model collapsed"}
        mean, var = x[~current].mean(0), x[~current].var(0) + 1e-6
        prior = np.clip(current.mean(), 0.02, 0.6)
        updated = (
            log_density(x, template["mean"], template["var"]) + np.log(prior)
            > log_density(x, mean, var) + np.log(1 - prior)
        )
        if (updated == current).all():
            break
        current = updated
    interviewer[np.flatnonzero(usable)] = current
    return interviewer, {"separated": True, "interviewer_segment_fraction": float(current.mean())}


def summarize(stats, keep, columns):
    """Mean/std over kept segments' voiced frames, plus timing features."""
    count = stats["counts"][keep].sum(0)
    total = stats["sums"][keep].sum(0)
    square = stats["squares"][keep].sum(0)
    features = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean**2, 0))
    for i, column in enumerate(columns):
        features[f"audio__{column}__mean"] = float(mean[i]) if count[i] else None
        features[f"audio__{column}__std"] = float(std[i]) if count[i] else None
    segments = stats["segments"][keep]
    voiced = int(stats["counts"][keep][:, columns.index(F0)].sum())
    features["audio__voiced_fraction"] = voiced / stats["frames"]
    features["audio__segments_per_minute"] = len(segments) / (stats["frames"] / 6000)
    lengths = (segments[:, 1] - segments[:, 0] + 1) / 100
    features["audio__segment_seconds_mean"] = float(lengths.mean()) if len(lengths) else None
    return features


def normalize_text(text):
    return re.sub(r"[^a-z' ]", "", str(text).lower()).strip()


def prompt_agreement(sessions, assignments, ids):
    """Proxy check: scripted prompts should be interviewer, long unique rows participant."""
    seen = {}
    for pid in ids:
        transcript = sessions[pid]["transcript"]
        if transcript is not None:
            for text in {normalize_text(t) for t in transcript["Text"]}:
                seen[text] = seen.get(text, 0) + 1
    rows = []
    for pid in ids:
        transcript, stats = sessions[pid]["transcript"], sessions[pid]["stats"]
        if transcript is None or not len(stats["segments"]):
            continue
        times, weight = stats["times"], stats["counts"][:, 0]
        for start, end, text in zip(transcript["Start_Time"], transcript["End_Time"], transcript["Text"]):
            text = normalize_text(text)
            words = len(text.split())
            if words >= 3 and seen[text] >= PROMPT_MIN_SESSIONS:
                kind = "scripted_prompt"
            elif words >= 10 and seen[text] == 1:
                kind = "long_unique"
            else:
                continue
            inside = (times[:, 0] < end) & (times[:, 1] > start)
            if not weight[inside].sum():
                continue
            share = float((weight[inside] * assignments[pid][inside]).sum() / weight[inside].sum())
            rows.append({"participant": pid, "kind": kind, "interviewer_share": share})
    return pd.DataFrame(rows)


def build(project, workers=4):
    labels = pd.read_parquet(project / "data/processed/labels.parquet")
    ids = labels.index[labels.split.isin(["train", "dev"])].tolist()
    columns = json.loads((project / "data/processed/stage3_report.json").read_text())[
        "schemas"
    ]["audio"]["columns"]
    output = project / DIRECTORY
    cache = output / "sessions"
    cache.mkdir(parents=True, exist_ok=True)

    def one(pid):
        target = cache / f"{pid}.npz"
        archive = project / "data/raw" / f"{pid}_P.tar.gz"
        key = f"{archive.stat().st_size}_{archive.stat().st_mtime_ns}"
        if target.exists():
            saved = np.load(target, allow_pickle=True)
            if str(saved["key"]) == key:
                transcript = (
                    pd.read_parquet(cache / f"{pid}_transcript.parquet")
                    if (cache / f"{pid}_transcript.parquet").exists()
                    else None
                )
                return pid, {k: saved[k] for k in saved.files if k != "key"}, transcript
        audio, transcript = read_session(archive)
        stats = session_statistics(audio, columns)
        np.savez(target, key=key, **stats)
        if transcript is not None:
            transcript[["Start_Time", "End_Time", "Text"]].to_parquet(
                cache / f"{pid}_transcript.parquet"
            )
        return pid, stats, transcript

    sessions = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (pid, stats, transcript) in enumerate(pool.map(one, ids), 1):
            stats = {k: (int(v) if np.ndim(v) == 0 else v) for k, v in stats.items()}
            sessions[pid] = {"stats": stats, "transcript": transcript}
            logging.info("Participant speech %d/%d", i, len(ids))
    domains = dict(zip(ids, domain(np.array(ids))))
    shared = fit_template([sessions[p]["stats"] for p in ids])
    templates = {
        # A domain with too few anchor segments falls back to the shared template.
        name: fit_template([sessions[p]["stats"] for p in ids if domains[p] == name]) or shared
        for name in ("WoZ", "AI")
    }
    assignments, audits, rows = {}, {}, {"voiced": {}, "participant": {}}
    for pid in ids:
        stats = sessions[pid]["stats"]
        interviewer, audit = assign(stats, templates[domains[pid]])
        assignments[pid], audits[pid] = interviewer, audit
        everything = np.ones(len(interviewer), bool)
        rows["voiced"][pid] = summarize(stats, everything, columns)
        rows["participant"][pid] = summarize(stats, ~interviewer, columns)
    agreement = prompt_agreement(sessions, assignments, ids)
    tables = {}
    for name, data in rows.items():
        table = pd.DataFrame.from_dict(data, orient="index").astype(float)
        table.index.name = "Participant_ID"
        table.to_parquet(output / f"{name}.parquet")
        tables[name] = table
    labels.loc[ids, ["PHQ_Binary", "split"]].to_parquet(output / "labels.parquet")
    summary = agreement.assign(
        correct=lambda d: np.where(
            d.kind == "scripted_prompt", d.interviewer_share >= 0.5, d.interviewer_share < 0.5
        ),
        domain=lambda d: d.participant.map(domains),
    )
    manifest = {
        "partitions": ["train", "dev"],
        "participants": len(ids),
        "features": tables["participant"].shape[1],
        "templates": {
            k: None if v is None else {"segments": v["segments"], "mean_pitch": float(v["mean"][0])}
            for k, v in templates.items()
        },
        "separated_sessions": int(sum(a["separated"] for a in audits.values())),
        "interviewer_segment_fraction_percentiles": dict(
            zip(
                ["p10", "p50", "p90", "max"],
                np.percentile(
                    [a.get("interviewer_segment_fraction", 0.0) for a in audits.values()],
                    [10, 50, 90, 100],
                ).tolist(),
            )
        ),
        "sessions_over_5_percent_interviewer": int(
            sum(a.get("interviewer_segment_fraction", 0.0) > 0.05 for a in audits.values())
        ),
        "proxy_agreement": {
            f"{d}_{k}": {"rows": int(len(g)), "agreement": float(g.correct.mean())}
            for (d, k), g in summary.groupby(["domain", "kind"])
        },
        "source": sha256(project / "src/dataset/participant_speech.py"),
        "outputs": {
            n: sha256(output / n) for n in ["voiced.parquet", "participant.parquet", "labels.parquet"]
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "session_audit.json").write_text(json.dumps(audits, indent=2))
    return manifest


def load(project):
    output = project / DIRECTORY
    manifest = json.loads((output / "manifest.json").read_text())
    if not matches_hash(project / "src/dataset/participant_speech.py", manifest["source"]):
        raise ValueError("Participant speech extractor changed")
    for name, digest in manifest["outputs"].items():
        if not matches_hash(output / name, digest):
            raise ValueError(f"Participant speech output changed: {name}")
    return (
        pd.read_parquet(output / "voiced.parquet"),
        pd.read_parquet(output / "participant.parquet"),
        pd.read_parquet(output / "labels.parquet"),
    )
