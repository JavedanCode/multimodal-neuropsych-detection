"""Dataset inventory and audit tool for the local E-DAIC release (Stage B).

Inspects a local directory of participant ``*.tar.gz`` archives and official
label CSVs **without extracting the corpus**. Produces:

* a machine-readable manifest (``dataset_manifest.json``)
* a human-readable audit report (``reports/data_audit/audit.md``)

Design notes
------------
* Shallow mode (default) only stats files and reads the small label CSVs in
  full (they are a few KB, so this is cheap and enables class-balance and
  ID cross-checks).
* Deep mode (``--deep``) additionally lists every member inside each
  ``.tar.gz``. gzip streams are not seekable, so this decompresses each
  archive (~106 GB total for the full release); results are cached in
  ``archive_members.json`` keyed by file size/mtime so each archive is
  streamed at most once across runs.
* Nothing is ever written inside ``data/raw``; raw data stays untouched.

Usage
-----
    python -m src.dataset.inventory --root data/raw
    python -m src.dataset.inventory --root data/raw --deep
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

ARCHIVE_RE = re.compile(r"^(\d+)_P\.tar\.gz$")

EXPECTED_SPLIT_FILES = ("train_split.csv", "dev_split.csv", "test_split.csv")
EXPECTED_LABEL_COLUMNS = ("Participant_ID", "PHQ_Binary", "PHQ_Score")
REQUIRED_TARGET_COLUMN = "PHQ_Binary"
ID_COLUMN = "Participant_ID"
# Official split member: the file name is a split file if its stem is one of these.
SPLIT_STEMS = {name.removesuffix(".csv") for name in EXPECTED_SPLIT_FILES}


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------
@dataclass
class ArchiveEntry:
    """One participant archive found on disk."""

    participant_id: int
    filename: str
    size_bytes: int
    mtime_utc: str


@dataclass
class ArchiveMembers:
    """Member listing of one archive (deep scan only)."""

    filename: str
    size_bytes: int
    mtime_utc: str
    members: list[str] = field(default_factory=list)
    member_sizes: dict[str, int] = field(default_factory=dict)
    unsafe_members: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class SplitLabels:
    """Validated contents of one official split label file."""

    filename: str
    columns: list[str]
    n_rows: int
    participant_ids: list[int]
    class_balance: dict[str, int]
    null_counts: dict[str, int]
    whitespace_cells: dict[str, int]
    duplicate_ids: list[int]


# ----------------------------------------------------------------------
# Directory scanning (shallow)
# ----------------------------------------------------------------------
def parse_archive_name(filename: str) -> int | None:
    """Return the participant ID from ``NNN_P.tar.gz``, else ``None``."""
    m = ARCHIVE_RE.match(filename)
    return int(m.group(1)) if m else None


def scan_directory(root: Path) -> tuple[list[ArchiveEntry], dict[str, Path]]:
    """List participant archives and label CSVs under ``root`` (top level).

    Returns ``(archives, label_files)`` where ``label_files`` maps file
    names to paths for CSV files that look like official label/split files.
    """
    archives: list[ArchiveEntry] = []
    label_files: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        pid = parse_archive_name(path.name)
        if pid is not None:
            archives.append(
                ArchiveEntry(pid, path.name, stat.st_size, mtime)
            )
        elif path.suffix.lower() == ".csv" and path.stem.lower() in {
            s.lower().removesuffix(".csv") for s in EXPECTED_SPLIT_FILES
        } | {"detailed_phq8_labels", "detailed_lables", "metadata_mapped"}:
            label_files[path.name] = path
    return archives, label_files


def find_unlabeled_archives(
    archives: Iterable[ArchiveEntry], labeled_ids: set[int]
) -> list[int]:
    return sorted(a.participant_id for a in archives if a.participant_id not in labeled_ids)


def find_archives_missing_for_labels(
    archives: Iterable[ArchiveEntry], labeled_ids: set[int]
) -> list[int]:
    archive_ids = {a.participant_id for a in archives}
    return sorted(labeled_ids - archive_ids)


# ----------------------------------------------------------------------
# Label validation
# ----------------------------------------------------------------------
def load_split_labels(path: Path) -> tuple[pd.DataFrame, SplitLabels]:
    """Read one official split CSV in full (tiny files) and validate it.

    Raises ``ValueError`` if the required ID/target columns are absent.
    """
    df = pd.read_csv(path)
    missing = [c for c in (ID_COLUMN, REQUIRED_TARGET_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required column(s) {missing}; found {list(df.columns)}")

    ids_raw = df[ID_COLUMN]
    ws = {
        col: int(df[col].astype(str).ne(df[col].astype(str).str.strip()).sum())
        for col in df.select_dtypes(include="object").columns
    }
    ws = {k: v for k, v in ws.items() if v}
    ids = pd.to_numeric(ids_raw, errors="coerce")
    if ids.isna().any() or not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any() or (ids >= 2**63).any():
        raise ValueError(f"{path.name}: participant IDs must be finite positive integers")
    target = pd.to_numeric(df[REQUIRED_TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(f"{path.name}: PHQ_Binary must contain only nonmissing 0 or 1")
    df[ID_COLUMN] = ids.astype("int64")
    df[REQUIRED_TARGET_COLUMN] = target.astype("int64")
    duplicate_ids = sorted(int(i) for i in ids[ids.duplicated()].dropna().unique())
    balance = (
        df[REQUIRED_TARGET_COLUMN].astype("Int64").value_counts(dropna=False).sort_index()
    )
    class_balance = {str(k): int(v) for k, v in balance.items()}
    null_counts = {c: int(df[c].isna().sum()) for c in df.columns}
    null_counts = {k: v for k, v in null_counts.items() if v}

    info = SplitLabels(
        filename=path.name,
        columns=list(df.columns),
        n_rows=int(len(df)),
        participant_ids=[int(i) for i in ids.dropna()],
        class_balance=class_balance,
        null_counts=null_counts,
        whitespace_cells=ws,
        duplicate_ids=duplicate_ids,
    )
    return df, info


def check_label_consistency(split_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Cross-split ID checks: duplicates inside a split, IDs shared across splits."""
    per_split_dups: dict[str, list[int]] = {}
    owners: dict[int, list[str]] = {}
    for name, df in split_frames.items():
        ids = pd.to_numeric(df[ID_COLUMN], errors="coerce").dropna().astype(int)
        dups = sorted(ids[ids.duplicated()].unique().tolist())
        if dups:
            per_split_dups[name] = dups
        for i in ids.unique():
            owners.setdefault(int(i), []).append(name)
    overlaps = {str(i): s for i, s in sorted(owners.items()) if len(s) > 1}
    return {"duplicate_ids_within_split": per_split_dups, "ids_in_multiple_splits": overlaps}


def check_binary_vs_score(df: pd.DataFrame, source: str, examples: int = 5) -> dict[str, Any]:
    """Informational: how often ``PHQ_Binary`` disagrees with ``PHQ_Score >= 10``.

    Local label files contain discrepancies against the >=10 PHQ-8 threshold.
    Their cause and the exact target derivation remain unverified. This counter is recorded in
    the audit so the discrepancy is documented rather than silently ignored;
    the label is used exactly as released.
    """
    if not {"PHQ_Binary", "PHQ_Score"}.issubset(df.columns):
        return {"available": False}
    sub = df[[ID_COLUMN, "PHQ_Binary", "PHQ_Score"]].copy()
    sub["PHQ_Score"] = pd.to_numeric(sub["PHQ_Score"], errors="coerce")
    sub["PHQ_Binary"] = pd.to_numeric(sub["PHQ_Binary"], errors="coerce")
    sub = sub.dropna()
    mismatch = sub[(sub["PHQ_Binary"] != (sub["PHQ_Score"] >= 10).astype(int))]
    return {
        "available": True,
        "source": source,
        "n_compared": int(len(sub)),
        "n_mismatch_vs_score_ge_10": int(len(mismatch)),
        "examples": [
            {
                "participant_id": int(r[ID_COLUMN]),
                "PHQ_Binary": int(r["PHQ_Binary"]),
                "PHQ_Score": int(r["PHQ_Score"]),
            }
            for _, r in mismatch.head(examples).iterrows()
        ],
    }


def check_phq8_totals(split_frames: dict[str, pd.DataFrame], phq8_path: Path) -> dict[str, Any]:
    """Verify ``Detailed_PHQ8_Labels.PHQ_8Total`` equals ``PHQ_Score`` when merged."""
    if not phq8_path.exists():
        return {"available": False}
    phq8 = pd.read_csv(phq8_path)
    if ID_COLUMN not in phq8.columns or "PHQ_8Total" not in phq8.columns:
        return {"available": False, "reason": "unexpected schema", "columns": list(phq8.columns)}
    all_splits = pd.concat(split_frames.values(), ignore_index=True)
    if "PHQ_Score" not in all_splits.columns:
        return {"available": False, "reason": "PHQ_Score absent from split files"}
    merged = all_splits[[ID_COLUMN, "PHQ_Score"]].merge(
        phq8[[ID_COLUMN, "PHQ_8Total"]], on=ID_COLUMN, how="inner"
    )
    merged["PHQ_Score"] = pd.to_numeric(merged["PHQ_Score"], errors="coerce")
    mismatch = merged[merged["PHQ_Score"] != merged["PHQ_8Total"]]
    return {
        "available": True,
        "n_merged": int(len(merged)),
        "n_mismatch_score_vs_phq8_total": int(len(mismatch)),
        "examples": mismatch.head(5).astype(str).to_dict(orient="records"),
    }


# ----------------------------------------------------------------------
# Archive deep scan (no extraction)
# ----------------------------------------------------------------------
def _is_unsafe_member(name: str) -> bool:
    """True for absolute paths, drive letters, or ``..`` traversal components."""
    if not name:
        return True
    norm = name.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        return True
    return any(part == ".." for part in norm.split("/"))


def inspect_archive(path: Path, size_bytes: int, mtime_utc: str) -> ArchiveMembers:
    """List members of a ``.tar.gz`` without extracting anything.

    Uses ``tarfile`` in stream mode; decompresses the archive sequentially
    (expensive — deep scans are cached by the caller).
    """
    result = ArchiveMembers(filename=path.name, size_bytes=size_bytes, mtime_utc=mtime_utc)
    try:
        with tarfile.open(path, "r|gz") as tar:
            for member in tar:
                result.members.append(member.name)
                if member.isfile():
                    result.member_sizes[member.name] = int(member.size)
                if _is_unsafe_member(member.name):
                    result.unsafe_members.append(member.name)
    except (tarfile.TarError, OSError, EOFError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.error("Failed to read archive %s: %s", path.name, exc)
    return result


def load_member_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Ignoring unreadable member cache %s: %s", cache_path, exc)
    return {}


def save_member_cache(cache: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def deep_scan_archives(
    archives: list[ArchiveEntry],
    root: Path,
    cache_path: Path,
    force: bool = False,
) -> dict[str, ArchiveMembers]:
    """List members of every archive, using/updating the on-disk cache.

    An archive is re-streamed only if it is absent from the cache, its
    size/mtime changed, or ``force`` is set.
    """
    cache = load_member_cache(cache_path)
    results: dict[str, ArchiveMembers] = {}
    n_total = len(archives)
    for idx, entry in enumerate(archives, start=1):
        cached = cache.get(entry.filename)
        if (
            not force
            and cached
            and cached.get("size_bytes") == entry.size_bytes
            and cached.get("mtime_utc") == entry.mtime_utc
        ):
            results[entry.filename] = ArchiveMembers(**cached)
            continue
        logger.info("Deep scan [%d/%d] %s", idx, n_total, entry.filename)
        listing = inspect_archive(root / entry.filename, entry.size_bytes, entry.mtime_utc)
        results[entry.filename] = listing
        cache[entry.filename] = asdict(listing)
        save_member_cache(cache, cache_path)
    return results


# ----------------------------------------------------------------------
# Manifest + audit report
# ----------------------------------------------------------------------
def build_manifest(
    root: Path,
    archives: list[ArchiveEntry],
    label_files: dict[str, Path],
    deep: bool = False,
    deep_results: dict[str, ArchiveMembers] | None = None,
) -> dict[str, Any]:
    """Assemble the full inventory manifest."""
    split_frames: dict[str, pd.DataFrame] = {}
    split_infos: dict[str, SplitLabels] = {}
    other_schemas: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    for name, path in sorted(label_files.items()):
        stem = path.stem
        try:
            if stem.lower() in SPLIT_STEMS:
                df, info = load_split_labels(path)
                split_frames[info.filename] = df
                split_infos[info.filename] = info
            else:
                header = pd.read_csv(path, nrows=5)
                other_schemas[name] = {
                    "columns": list(header.columns),
                    "sample_rows": int(len(header)),
                }
        except ValueError as exc:
            errors[name] = str(exc)
            logger.error("%s", exc)

    labeled_ids: set[int] = set()
    for info in split_infos.values():
        labeled_ids.update(info.participant_ids)

    archive_ids = [a.participant_id for a in archives]
    duplicate_archives = sorted({i for i in archive_ids if archive_ids.count(i) > 1})

    cross = check_label_consistency(split_frames) if split_frames else {}
    phq8_path = next(
        (p for n, p in label_files.items() if n.lower() == "detailed_phq8_labels.csv"),
        root / "Detailed_PHQ8_Labels.csv",
    )
    phq8_check = check_phq8_totals(split_frames, phq8_path) if split_frames else {"available": False}
    binary_checks = {
        name: check_binary_vs_score(df, name) for name, df in split_frames.items()
    }

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "scan_mode": "deep" if deep else "shallow",
        "totals": {
            "n_archives": len(archives),
            "archive_bytes": int(sum(a.size_bytes for a in archives)),
            "n_label_files": len(label_files),
            "n_labeled_participants": len(labeled_ids),
            "n_splits": len(split_infos),
        },
        "archives": [asdict(a) for a in archives],
        "duplicate_archive_participant_ids": duplicate_archives,
        "splits": {name: asdict(info) for name, info in split_infos.items()},
        "other_label_file_schemas": other_schemas,
        "cross_checks": {
            **cross,
            "archives_without_labels": find_unlabeled_archives(archives, labeled_ids),
            "labels_without_archives": find_archives_missing_for_labels(archives, labeled_ids),
            "binary_vs_score_ge_10": binary_checks,
            "phq8_total_vs_score": phq8_check,
            "schema_errors": errors,
        },
    }
    if deep and deep_results is not None:
        manifest["archive_contents"] = {k: asdict(v) for k, v in deep_results.items()}
        unsafe = {
            k: v.unsafe_members for k, v in deep_results.items() if v.unsafe_members
        }
        unreadable = {
            k: v.error for k, v in deep_results.items() if v.error is not None
        }
        manifest["cross_checks"]["unsafe_archive_members"] = unsafe
        manifest["cross_checks"]["unreadable_archives"] = unreadable
    return manifest


def render_audit_markdown(manifest: dict[str, Any]) -> str:
    """Human-readable audit report derived from the manifest."""
    t = manifest["totals"]
    lines: list[str] = []
    add = lines.append
    add("# E-DAIC Dataset Audit Report")
    add("")
    add(f"- Generated (UTC): {manifest['generated_at_utc']}")
    add(f"- Root: `{manifest['root']}`")
    add(f"- Scan mode: **{manifest['scan_mode']}**")
    add("")
    add("## Cohort overview")
    add("")
    add(f"- Participant archives: **{t['n_archives']}** "
        f"({t['archive_bytes'] / 1024**3:.2f} GB)")
    add(f"- Labeled participants: **{t['n_labeled_participants']}** across "
        f"{t['n_splits']} official splits")
    add("")

    splits = manifest.get("splits", {})
    if splits:
        add("## Splits and class balance (`PHQ_Binary`)")
        add("")
        add("| Split file | N | Class 0 | Class 1 | Duplicates | Whitespace cells | Null columns |")
        add("|---|---|---|---|---|---|---|")
        for name, info in sorted(splits.items()):
            bal = info.get("class_balance", {})
            add(
                f"| {name} | {info['n_rows']} | {bal.get('0', 0)} | {bal.get('1', 0)} "
                f"| {len(info.get('duplicate_ids', []))} "
                f"| {sum(info.get('whitespace_cells', {}).values())} "
                f"| {len(info.get('null_counts', {}))} |"
            )
        add("")

    cc = manifest.get("cross_checks", {})
    add("## Cross-checks")
    add("")
    for key, label in (
        ("archives_without_labels", "Archives without labels"),
        ("labels_without_archives", "Labeled participants without archive"),
    ):
        vals = cc.get(key, [])
        add(f"- {label}: **{len(vals)}**" + (f" — {vals}" if vals else ""))
    overlaps = cc.get("ids_in_multiple_splits", {})
    add(f"- IDs appearing in multiple splits: **{len(overlaps)}**"
        + (f" — {overlaps}" if overlaps else ""))
    if manifest.get("duplicate_archive_participant_ids"):
        add(f"- Duplicate archive participant IDs: "
            f"{manifest['duplicate_archive_participant_ids']}")
    add("")

    add("## Label semantics notes")
    add("")
    for name, chk in sorted(cc.get("binary_vs_score_ge_10", {}).items()):
        if chk.get("available"):
            add(
                f"- `{name}`: {chk['n_mismatch_vs_score_ge_10']}/{chk['n_compared']} "
                f"rows where `PHQ_Binary != (PHQ_Score >= 10)` — **`PHQ_Binary` is used "
                f"as released; it is not recomputed.**"
            )
    phq8 = cc.get("phq8_total_vs_score", {})
    if phq8.get("available"):
        add(
            f"- `Detailed_PHQ8_Labels.PHQ_8Total` vs `PHQ_Score`: "
            f"{phq8['n_mismatch_score_vs_phq8_total']}/{phq8['n_merged']} mismatches."
        )
    add("")

    errors = cc.get("schema_errors", {})
    if errors:
        add("## Schema errors")
        add("")
        for name, msg in errors.items():
            add(f"- `{name}`: {msg}")
        add("")

    if manifest.get("scan_mode") == "deep":
        contents = manifest.get("archive_contents", {})
        n_members = sum(len(v.get("members", [])) for v in contents.values())
        add("## Archive contents (deep scan)")
        add("")
        add(f"- Archives listed: {len(contents)}, total members: {n_members}")
        unsafe = cc.get("unsafe_archive_members", {})
        add(f"- Archives with unsafe member paths: **{len(unsafe)}**"
            + (f" — {list(unsafe)}" if unsafe else ""))
        unreadable = cc.get("unreadable_archives", {})
        add(f"- Unreadable archives: **{len(unreadable)}**"
            + (f" — {list(unreadable)}" if unreadable else ""))
        add("")

    add("## Exclusions")
    add("")
    add("- No participants excluded by this tool. Cohort selection (both "
        "modalities present, valid labels) happens at feature-table build "
        "time (Stage D) and is reported there.")
    add("")
    return "\n".join(lines)


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    logger.info("Manifest written: %s", path)


def write_audit_report(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_audit_markdown(manifest), encoding="utf-8")
    logger.info("Audit report written: %s", path)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edaic-inventory",
        description="Inventory and audit the local E-DAIC release without extracting it.",
    )
    parser.add_argument("--config", type=Path, help="YAML configuration; relative paths resolve from the repository root.")
    parser.add_argument("--root", type=Path,
                        help="Directory containing NNN_P.tar.gz archives and label CSVs.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path for the JSON manifest.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Output path for the Markdown audit report.")
    parser.add_argument("--deep", action="store_true", default=None,
                        help="Also list every member inside each .tar.gz (slow; cached).")
    parser.add_argument("--force-deep", action="store_true",
                        help="Ignore the member cache and re-stream all archives.")
    parser.add_argument("--member-cache", type=Path,
                        default=None,
                        help="Cache file for deep-scan member listings.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    project = Path(__file__).resolve().parents[2]
    config_path = args.config or project / "configs/default.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a mapping")
        paths = config.get("paths", {})
        inventory = config.get("inventory", {})
        if not isinstance(paths, dict) or not isinstance(inventory, dict):
            raise ValueError("paths and inventory must be mappings")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.error("Cannot load configuration: %s", exc)
        return 2
    def configured_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project / path
    args.root = args.root or configured_path(paths.get("data_root", "data/raw"))
    args.out = args.out or configured_path(paths.get("manifest", "data/manifests/dataset_manifest.json"))
    args.report = args.report or configured_path(paths.get("audit_report", "reports/data_audit/audit.md"))
    args.member_cache = args.member_cache or configured_path(inventory.get("deep_scan_cache", "data/manifests/archive_members.json"))
    args.deep = args.deep if args.deep is not None else inventory.get("deep_scan", False)
    root: Path = args.root
    # Prevent configured outputs from modifying the source dataset.
    for output in (args.out, args.report, args.member_cache):
        if output.resolve().is_relative_to(root.resolve()):
            logger.error("Output must be outside the raw dataset: %s", output)
            return 2
    if not root.exists():
        logger.error("Root directory does not exist: %s", root)
        return 2
    if not root.is_dir():
        logger.error("Root is not a directory: %s", root)
        return 2

    archives, label_files = scan_directory(root)
    if not archives and not label_files:
        logger.error(
            "No participant archives (*.tar.gz) and no label CSVs found in %s. "
            "Point --root at the downloaded E-DAIC directory.", root,
        )
        return 2
    if not archives:
        logger.warning("No NNN_P.tar.gz archives found in %s (labels only?).", root)
    if not label_files:
        logger.warning("No label CSVs found in %s (archives only?).", root)

    deep_results: dict[str, ArchiveMembers] | None = None
    if args.deep and archives:
        deep_results = deep_scan_archives(
            archives, root, args.member_cache, force=args.force_deep
        )

    manifest = build_manifest(root, archives, label_files, deep=args.deep,
                              deep_results=deep_results)
    write_manifest(manifest, args.out)
    write_audit_report(manifest, args.report)

    t = manifest["totals"]
    logger.info(
        "Inventory complete: %d archives (%.2f GB), %d labeled participants, "
        "%d splits.",
        t["n_archives"], t["archive_bytes"] / 1024**3,
        t["n_labeled_participants"], t["n_splits"],
    )
    checks = manifest["cross_checks"]
    invalid = any(checks.get(key) for key in (
        "schema_errors", "duplicate_ids_within_split", "ids_in_multiple_splits",
        "labels_without_archives", "archives_without_labels", "unreadable_archives",
        "unsafe_archive_members"))
    invalid = invalid or bool(manifest["duplicate_archive_participant_ids"])
    invalid = invalid or len(manifest["splits"]) != len(EXPECTED_SPLIT_FILES)
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
