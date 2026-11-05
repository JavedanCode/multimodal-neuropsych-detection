"""Protect the consolidated entry points and historical-artifact contract."""

import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src.__main__ import main
from src.training.provenance import PROJECT, matches_hash


def test_data_corruption_never_falls_back_to_git(tmp_path):
    path = tmp_path / "features.parquet"
    path.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    assert matches_hash(path, digest)
    path.write_bytes(b"changed")
    assert not matches_hash(path, digest)


def test_recorded_source_hash_resolves_after_rename():
    digest = "e6299b7e89bf839d219a4bab620414b44847d35050f2d7e78c1834279ed1b326"
    assert matches_hash(PROJECT / "src/models/branch_audit_v6.py", digest)
    assert matches_hash(PROJECT / "src/models/fusion.py", digest)
    assert not matches_hash(PROJECT / "src/models/fusion.py", "0" * 64)


def test_cli_requires_reference_before_starting_fallback():
    with pytest.raises(SystemExit) as error:
        main(["run", "fallback"])
    assert error.value.code == 2


def test_only_maintained_studies_have_runners():
    runners = {
        path.stem
        for path in (PROJECT / "src/experiments").glob("*.py")
        if path.name != "__init__.py"
    }
    assert runners == {"reference", "text_candidates", "audio_visual", "fallback"}


def test_cli_imports_numerical_code_before_limiting_threads(monkeypatch, capsys):
    import src.__main__ as cli

    events = []

    def import_study(name):
        events.append("import")
        return SimpleNamespace(run=lambda project: events.append("run"))

    @contextmanager
    def limit_threads(limits):
        assert limits == 1
        events.append("limit")
        yield

    monkeypatch.setattr(cli.importlib, "import_module", import_study)
    monkeypatch.setattr(cli, "threadpool_limits", limit_threads)
    cli.main(["run", "reference"])
    assert events == ["import", "limit", "run"]
