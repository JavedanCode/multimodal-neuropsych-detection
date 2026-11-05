"""Validate local human review without promoting a sample into dataset truth."""

from src.training.provenance import matches_hash
from pathlib import Path
import json
from collections import Counter
import sys
from src.training.validation import sha256

SPEAKERS = {"Participant", "Interviewer", "Mixed", "Uncertain"}
TIMING = {"Matches", "Does not match", "Not applicable", "Uncertain"}


def validate(rows, expected):
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Expected one annotation per review clip")
    keys = [r.get("review_key") for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("Missing, duplicate, or unknown clip")
    for r in rows:
        source = expected[r["review_key"]]
        for field in ["file", "kind", "start", "end", "transcript"]:
            if r.get(field) != source[field]:
                raise ValueError("Review clip provenance changed")
        if (
            r.get("speaker") not in SPEAKERS
            or r.get("timing") not in TIMING
            or r.get("reviewed") is not True
        ):
            raise ValueError(
                "Incomplete review: use Uncertain rather than inventing answers"
            )
    covered = [r for r in rows if r["kind"] == "transcript_covered"]
    return {
        "reviewed_clips": len(rows),
        "covered_clips": len(covered),
        "covered_speaker_counts": dict(Counter(r["speaker"] for r in covered)),
        "covered_timing_counts": dict(Counter(r["timing"] for r in covered)),
        "covered_clips_needing_followup": sum(
            r["speaker"] != "Participant" or r["timing"] != "Matches" for r in covered
        ),
        "whole_dataset_speaker_validated": False,
        "labels_changed": False,
    }


def run(project):
    folder = project / "data/processed/aligned_v2/speaker_review"
    path = folder / "annotations.json"
    if not path.exists():
        return {
            "status": "awaiting_human_review",
            "worksheet": str(folder / "index.html"),
        }
    expected = {
        r["file"]: r
        for p in folder.glob("*.json")
        if p.stem.isdigit()
        for r in json.loads(p.read_text(encoding="utf-8"))
    }
    report = validate(json.loads(path.read_text(encoding="utf-8-sig")), expected)
    report.update(
        annotation_hash=sha256(path),
        clip_hashes={name: sha256(folder / name) for name in expected},
        source_hash=sha256(Path(__file__)),
    )
    out = project / "reports/data_audit/speaker_review_v3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def run_context(project):
    folder = project / "data/processed/aligned_v2/speaker_review/context_v3"
    path = folder / "annotations_context.json"
    if not path.exists():
        return {
            "status": "awaiting_context_review",
            "worksheet": str(folder / "index.html"),
        }
    manifest = json.loads((folder / "manifest.json").read_text())
    assert matches_hash(
        folder.parent / "annotations.json", manifest["initial_annotations_hash"]
    )
    for name, h in manifest["files"].items():
        assert matches_hash(folder / name, h)
    expected = {
        r["file"]: r
        for r in json.loads((folder / "clips.json").read_text(encoding="utf-8"))
    }
    result = validate(json.loads(path.read_text(encoding="utf-8-sig")), expected)
    result.update(
        annotation_hash=sha256(path),
        source_hash=sha256(Path(__file__)),
        scope="Follow-up target utterances only",
    )
    out = project / "reports/data_audit/speaker_context_review_v3.json"
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    print(
        json.dumps(
            run_context(project) if "--context" in sys.argv else run(project), indent=2
        )
    )
