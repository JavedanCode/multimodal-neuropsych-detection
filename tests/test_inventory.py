"""Tests for the E-DAIC dataset inventory tool (Stage B).

All fixtures are synthetic (created in pytest ``tmp_path``); no real
dataset files are required or committed.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.dataset import inventory as inv


# ----------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------
def make_tar(path: Path, members: dict[str, bytes]) -> None:
    """Create a ``.tar.gz`` at ``path`` with the given member name -> content."""
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


SPLIT_COLUMNS = [
    "Participant_ID", "Gender", "PHQ_Binary", "PHQ_Score",
    "PCL-C (PTSD)", "PTSD Severity",
]


def split_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=SPLIT_COLUMNS).to_csv(path, index=False)


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    make_tar(root / "300_P.tar.gz", {"300_P/openSMILE_features.csv": b"a,b\n1,2\n"})
    make_tar(root / "301_P.tar.gz", {"301_P/openSMILE_features.csv": b"a,b\n3,4\n"})
    split_csv(root / "train_split.csv", [
        {"Participant_ID": 300, "Gender": "male", "PHQ_Binary": 0,
         "PHQ_Score": 4, "PCL-C (PTSD)": 0, "PTSD Severity": 28},
        {"Participant_ID": 301, "Gender": "female ", "PHQ_Binary": 1,
         "PHQ_Score": 22, "PCL-C (PTSD)": 1, "PTSD Severity": 67},
    ])
    split_csv(root / "dev_split.csv", [
        {"Participant_ID": 302, "Gender": "male", "PHQ_Binary": 0,
         "PHQ_Score": 12, "PCL-C (PTSD)": 0, "PTSD Severity": 33},
    ])
    split_csv(root / "test_split.csv", [
        {"Participant_ID": 303, "Gender": "female", "PHQ_Binary": 1,
         "PHQ_Score": 10, "PCL-C (PTSD)": 1, "PTSD Severity": 51},
    ])
    return root


# ----------------------------------------------------------------------
# Unit: naming / scanning
# ----------------------------------------------------------------------
def test_parse_archive_name_valid():
    assert inv.parse_archive_name("300_P.tar.gz") == 300
    assert inv.parse_archive_name("718_P.tar.gz") == 718


@pytest.mark.parametrize("name", ["300_p.tar.gz", "300.tar.gz", "abc_P.tar.gz",
                                  "300_P.tgz", "notes.txt"])
def test_parse_archive_name_invalid(name):
    assert inv.parse_archive_name(name) is None


def test_scan_directory_classifies(dataset_root: Path):
    archives, labels = inv.scan_directory(dataset_root)
    assert [a.participant_id for a in archives] == [300, 301]
    assert set(labels) == {"train_split.csv", "dev_split.csv", "test_split.csv"}
    assert all(a.size_bytes > 0 for a in archives)


# ----------------------------------------------------------------------
# Unit: path-traversal safety
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "../evil.txt", "a/../../evil", "/abs/path", "C:/windows/x", "..",
])
def test_unsafe_members_detected(name):
    assert inv._is_unsafe_member(name) is True


@pytest.mark.parametrize("name", ["300_P/feature.csv", "dir/sub/file.txt", "./ok.csv"])
def test_safe_members_accepted(name):
    assert inv._is_unsafe_member(name) is False


def test_inspect_archive_lists_without_extraction(tmp_path: Path):
    path = tmp_path / "300_P.tar.gz"
    make_tar(path, {"good.csv": b"x", "../evil.txt": b"x"})
    size, mtime = path.stat().st_size, "unused"
    before = sorted(p.name for p in tmp_path.iterdir())
    result = inv.inspect_archive(path, size, mtime)
    assert result.error is None
    assert "good.csv" in result.members
    assert result.unsafe_members == ["../evil.txt"]
    # nothing extracted: directory contents unchanged, archive untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert path.stat().st_size == size


def test_inspect_archive_corrupt_reports_error(tmp_path: Path):
    path = tmp_path / "bad_P.tar.gz"
    path.write_bytes(b"not a tarball at all")
    result = inv.inspect_archive(path, path.stat().st_size, "t")
    assert result.error is not None
    assert result.members == []


# ----------------------------------------------------------------------
# Unit: label validation
# ----------------------------------------------------------------------
def test_load_split_labels_rejects_missing_target(tmp_path: Path):
    p = tmp_path / "train_split.csv"
    pd.DataFrame({"Participant_ID": [1], "PHQ_Score": [5]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="PHQ_Binary"):
        inv.load_split_labels(p)


def test_load_split_labels_reports_balance_duplicates_whitespace(tmp_path: Path):
    p = tmp_path / "train_split.csv"
    split_csv(p, [
        {"Participant_ID": 300, "Gender": "male ", "PHQ_Binary": 0,
         "PHQ_Score": 4, "PCL-C (PTSD)": 0, "PTSD Severity": 28},
        {"Participant_ID": 301, "Gender": "female", "PHQ_Binary": 1,
         "PHQ_Score": 22, "PCL-C (PTSD)": 1, "PTSD Severity": None},
        {"Participant_ID": 301, "Gender": "male", "PHQ_Binary": 0,
         "PHQ_Score": 3, "PCL-C (PTSD)": 0, "PTSD Severity": 17},
    ])
    df, info = inv.load_split_labels(p)
    assert info.class_balance == {"0": 2, "1": 1}
    assert info.duplicate_ids == [301]
    assert info.whitespace_cells == {"Gender": 1}
    assert info.null_counts == {"PTSD Severity": 1}
    assert len(df) == 3


def test_cross_split_overlap_detection():
    frames = {
        "train_split.csv": pd.DataFrame({inv.ID_COLUMN: [300, 301]}),
        "test_split.csv": pd.DataFrame({inv.ID_COLUMN: [301, 302]}),
    }
    cross = inv.check_label_consistency(frames)
    assert cross["ids_in_multiple_splits"] == {"301": ["train_split.csv", "test_split.csv"]}
    assert cross["duplicate_ids_within_split"] == {}


def test_binary_vs_score_counter():
    # Mirrors real E-DAIC rows: 335 (score 12 -> 0), 352 (score 10 -> 0)
    # disagree with the >=10 rule; a score 4 -> 0 row agrees.
    df = pd.DataFrame({
        inv.ID_COLUMN: [1, 2, 3],
        "PHQ_Binary": [0, 0, 0],
        "PHQ_Score": [12, 10, 4],
    })
    chk = inv.check_binary_vs_score(df, "train_split.csv")
    assert chk["n_compared"] == 3
    assert chk["n_mismatch_vs_score_ge_10"] == 2
    assert {e["participant_id"] for e in chk["examples"]} == {1, 2}


# ----------------------------------------------------------------------
# Integration: manifest, audit, cache
# ----------------------------------------------------------------------
def test_manifest_and_audit_end_to_end(dataset_root: Path, tmp_path: Path):
    rc = inv.main([
        "--root", str(dataset_root),
        "--out", str(tmp_path / "manifest.json"),
        "--report", str(tmp_path / "audit.md"),
    ])
    assert rc == 1  # fixture has labels without matching archives
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["totals"]["n_archives"] == 2
    assert manifest["totals"]["n_labeled_participants"] == 4  # 300, 301, 302, 303
    assert manifest["scan_mode"] == "shallow"
    # archives 300, 301 labeled; 302, 303 labeled but no archive
    assert manifest["cross_checks"]["labels_without_archives"] == [302, 303]
    assert manifest["cross_checks"]["archives_without_labels"] == []

    audit = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "# E-DAIC Dataset Audit Report" in audit
    assert "PHQ_Binary" in audit
    assert "Archives without labels: **0**" in audit


def test_deep_scan_cache_reused(dataset_root: Path, tmp_path: Path):
    cache = tmp_path / "members.json"
    archives, _ = inv.scan_directory(dataset_root)
    first = inv.deep_scan_archives(archives, dataset_root, cache)
    assert set(first) == {"300_P.tar.gz", "301_P.tar.gz"}
    assert cache.exists()
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert "300_P.tar.gz" in cached
    # second call must be served from cache (results equal, file not rewritten
    # with new keys)
    second = inv.deep_scan_archives(archives, dataset_root, cache)
    assert second["300_P.tar.gz"].members == first["300_P.tar.gz"].members


def test_deep_manifest_contains_members(dataset_root: Path, tmp_path: Path):
    archives, labels = inv.scan_directory(dataset_root)
    results = inv.deep_scan_archives(archives, dataset_root, tmp_path / "c.json")
    manifest = inv.build_manifest(dataset_root, archives, labels,
                                  deep=True, deep_results=results)
    assert "300_P/openSMILE_features.csv" in manifest["archive_contents"]["300_P.tar.gz"]["members"]
    assert manifest["cross_checks"]["unsafe_archive_members"] == {}


# ----------------------------------------------------------------------
# CLI error handling
# ----------------------------------------------------------------------
def test_cli_missing_root_exits_2(tmp_path: Path):
    assert inv.main(["--root", str(tmp_path / "nope")]) == 2


def test_cli_empty_directory_exits_2(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert inv.main(["--root", str(empty)]) == 2
