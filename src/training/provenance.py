"""Check live artifacts and explicitly archived source without rewriting history.

Data and model files must always match on disk. Only source code and method
documents may match the recorded pre-cleanup Git revision. This establishes
which code produced an old cache; it does not certify new code as identical.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]


def source_hashes(project: Path) -> dict[str, str]:
    """Record all maintained code/configuration for newly generated runs."""
    paths = [
        *project.joinpath("src").rglob("*.py"),
        *project.joinpath("configs").glob("*.json"),
        project / "configs/default.yaml",
        project / "pyproject.toml",
        *project.joinpath("docs/methods").glob("*.md"),
    ]
    return {
        path.relative_to(project)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(paths)
    }


# Committed revisions that produced recorded runs after the cleanup snapshot
# (feadacb: audio preprocessing and sparse-fusion runs). Source files only.
RESULT_COMMITS = (
    "feadacba68bdd51440161d3ca0721b103ff2e241",
    # corrected_fusion run (voiced audio and common-scale fusion).
    "04df0840d3ef35b69c945d42f29c0794c8407324",
    # score_functionals_domain run.
    "83b3e9c68075860e78c5122cee502cd7d1f6f1d2",
)


@lru_cache(maxsize=256)
def archived_hashes(relative_path: str) -> frozenset[str]:
    """Return hashes for a source file in the explicitly recorded revisions."""
    if not relative_path.startswith(("src/", "docs/")):
        return frozenset()
    history = json.loads((PROJECT / "configs/history.json").read_text())
    original = history["historical_paths"].get(relative_path, relative_path)
    hashes = _revision_hashes(history["snapshot_commit"], original)
    for commit in RESULT_COMMITS:
        hashes |= _revision_hashes(commit, relative_path)
    return hashes


def _revision_hashes(commit: str, path: str) -> frozenset[str]:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return frozenset()
    # Git normalizes text line endings; earlier Windows runs hashed working files.
    normalized = result.stdout.replace(b"\r\n", b"\n")
    versions = (
        result.stdout,
        normalized,
        normalized.replace(b"\n", b"\r\n"),
        # The recorded adaptive model used LF except for its final CRLF.
        normalized[:-1] + b"\r\n" if normalized.endswith(b"\n") else normalized,
    )
    return frozenset(hashlib.sha256(value).hexdigest() for value in versions)


def matches_hash(path: Path, expected: str) -> bool:
    """Check an artifact exactly, with source-only Git provenance fallback."""
    path = Path(path).resolve()
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() == expected:
            return True
    try:
        relative = path.relative_to(PROJECT).as_posix()
    except ValueError:
        return False
    return expected in archived_hashes(relative)
