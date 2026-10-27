"""Bounded-memory, read-only feature schema inventory; no feature tables are fitted."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import tarfile
from pathlib import Path

from src.dataset.inventory import scan_directory, _is_unsafe_member

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
HEADER_LIMIT = 65536


def inspect_features(path: Path) -> dict:
    """Stream an archive and inspect only headers of named OpenSMILE/OpenFace CSVs.

    Never extract files or deserialize model/pickle objects. Other feature files
    are inventoried by name and size, without guessing their column semantics.
    """
    result = {"files": [], "errors": []}
    try:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                    result["errors"].append(f"Unsafe or linked member: {member.name}")
                    continue
                if not member.isfile() or "features" not in Path(member.name).parts:
                    continue
                record = {"name": member.name, "bytes": member.size}
                name = Path(member.name).name.lower()
                if "opensmile2" in name or "openface2" in name:
                    stream = archive.extractfile(member)
                    assert stream is not None
                    header = stream.readline(HEADER_LIMIT + 1)
                    if len(header) > HEADER_LIMIT:
                        result["errors"].append(f"Oversized header: {member.name}")
                    else:
                        delimiter = ";" if "opensmile2" in name else ","
                        columns = [c.strip() for c in next(csv.reader([header.decode("utf-8-sig")], delimiter=delimiter), [])]
                        record.update(delimiter=delimiter, columns=columns,
                                      modality="audio" if delimiter == ";" else "facial",
                                      quality_columns=[c for c in columns if c in ("confidence", "success")])
                        if not columns or len(columns) != len(set(columns)):
                            result["errors"].append(f"Empty or duplicate columns: {member.name}")
                result["files"].append(record)
    except (tarfile.TarError, OSError, EOFError, UnicodeError, csv.Error) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def audit_features(root: Path, output: Path) -> dict:
    """Cache successful inspections by resolved root, version, file size and mtime."""
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Feature inventory output must be outside raw data")
    archives, _ = scan_directory(root)
    if not archives:
        raise ValueError("No participant archives found")
    previous = {}
    if output.exists():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            LOGGER.warning("Ignoring unreadable inventory cache")
    cached = previous.get("archives", {}) if previous.get("root") == str(root) and previous.get("schema_version") == SCHEMA_VERSION else {}
    report = {"schema_version": SCHEMA_VERSION, "root": str(root),
              "scope": "File availability and CSV headers only; no row-level quality or alignment validation",
              "archives": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(archives, 1):
        old = cached.get(entry.filename, {})
        if old.get("size_bytes") == entry.size_bytes and old.get("mtime_utc") == entry.mtime_utc and not old.get("errors"):
            item = old
        else:
            LOGGER.info("Inspecting %d/%d %s", index, len(archives), entry.filename)
            item = inspect_features(root / entry.filename)
            item.update(participant_id=entry.participant_id, size_bytes=entry.size_bytes, mtime_utc=entry.mtime_utc)
        report["archives"][entry.filename] = item
        # Preserve unvisited cache entries for restartability until the final write.
        snapshot = {**report, "archives": {**cached, **report["archives"]}}
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        temporary.replace(output)
    report["summary"] = summarize(report)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def summarize(report: dict) -> dict:
    variants: dict[str, dict] = {}
    both = 0
    missing = []
    for filename, item in report["archives"].items():
        modalities = set()
        for feature in item["files"]:
            if "columns" not in feature:
                continue
            modalities.add(feature["modality"])
            key = json.dumps([feature["modality"], feature["columns"]])
            variant = variants.setdefault(key, {"modality": feature["modality"], "columns": feature["columns"], "files": 0})
            variant["files"] += 1
        present = {"audio", "facial"}.issubset(modalities)
        both += present
        if not present:
            missing.append(filename)
    return {"archives": len(report["archives"]), "archives_with_both_header_types": both,
            "archives_missing_header_types": missing,
            "archives_with_errors": sum(bool(i["errors"]) for i in report["archives"].values()),
            "schema_variants": list(variants.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/manifests/feature_inventory.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = audit_features(args.root, args.out)
    LOGGER.info("Complete: %s", {k: v for k, v in report["summary"].items() if k != "schema_variants"})
    return 1 if report["summary"]["archives_with_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
