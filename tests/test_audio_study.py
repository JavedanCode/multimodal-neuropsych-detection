import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.dataset.audio_summary import summarize, extract
from src.dataset.preprocessing import ParticipantPreprocessor
from src.training.classical import inner_data, select_inner, fit_outer, metric_row
from src.dataset.audio_inputs import load_inputs
from src.training.validation import sha256


def test_audio_summary_exact_quantiles_changes_and_windows():
    frame = pd.DataFrame(
        {
            "frameTime": [0.0, 1.0, 10.0, 11.0],
            "a": [0.0, 2.0, 4.0, 6.0],
            "b": [1.0, np.nan, 5.0, 7.0],
        }
    )
    r = summarize(frame, ["a", "b"])
    assert r["audio__a__mean"] == 3
    assert r["audio__a__std"] == pytest.approx(np.sqrt(5))
    assert r["audio__a__p10"] == pytest.approx(0.6)
    assert r["audio__a__iqr"] == 3
    assert r["audio__a__adjacent_change"] == 2
    assert r["audio__a__window_mean_std"] == 2
    assert r["audio__b__adjacent_change"] == 2  # Never bridge 1 -> missing -> 5.
    assert r["audio__b__window_mean_std"] == 2.5


def test_summary_missing_zero_and_bad_times():
    frame = pd.DataFrame(
        {"frameTime": [0.0, 0.1], "a": [np.nan, np.inf], "b": [0.0, 0.0]}
    )
    result = summarize(frame, ["a", "b"])
    assert all(v is None for k, v in result.items() if k.startswith("audio__a__"))
    assert all(v == 0 for k, v in result.items() if k.startswith("audio__b__"))
    with pytest.raises(ValueError, match="frameTime"):
        summarize(frame.assign(frameTime=[1.0, 0.0]), ["a"])


def test_streaming_archive_extraction_without_extracting_files(tmp_path):
    path = tmp_path / "300_P.tar.gz"
    data = b"name;frameTime;a\nx;0;1\nx;10;3\n"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("300_P/features/300_OpenSMILE2.3.0_egemaps.csv")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    result = extract(path, ["a"])
    assert result["features"]["audio__a__mean"] == 2 and result["rows"] == 2
    assert list(tmp_path.iterdir()) == [path]


def test_nested_preprocessing_only_sees_inner_fit_rows(monkeypatch):
    rng = np.random.default_rng(4)
    x = pd.DataFrame(
        rng.normal(size=(24, 3)),
        columns=["audio__a", "audio__b", "audio__c"],
        index=range(1, 25),
    )
    y = pd.Series([0, 1] * 12, index=x.index)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(x, y))
    observed = []
    original = ParticipantPreprocessor.fit

    def track(self, values, y=None):
        observed.append(list(values.index))
        return original(self, values, y)

    monkeypatch.setattr(ParticipantPreprocessor, "fit", track)
    prepared = inner_data(x, y, splits)
    assert observed == [list(x.index[a]) for a, b in splits]
    default, tuned, records = select_inner(x, y, "logistic", prepared, 42)
    assert (
        len(observed) == 3
    )  # Shared fold transforms, no full-data refit during selection.
    assert (
        default["threshold"] == 0 and tuned["selection_f1"] >= default["selection_f1"]
    )
    assert len(records) == 30
    # A held-out value change cannot modify a fit or selected inner settings.
    valid = x.iloc[:3].copy()
    valid.index = range(101, 104)
    a, _ = fit_outer(x, y, valid, "logistic", tuned, 42)
    b, _ = fit_outer(x, y, valid * 100, "logistic", tuned, 42)
    np.testing.assert_array_equal(
        a.named_steps["classifier"].coef_, b.named_steps["classifier"].coef_
    )


def test_margin_selection_reproducible_and_default_wins_ties():
    x = pd.DataFrame(
        {"audio__a": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] * 4, "audio__b": np.arange(24)}
    )
    y = pd.Series([0, 0, 0, 1, 1, 1] * 4)
    splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(x, y))
    prepared = inner_data(x, y, splits)
    a = select_inner(x, y, "linear", prepared, 42)
    b = select_inner(x, y, "linear", prepared, 42)
    assert a == b
    assert a[1]["threshold"] == 0 and a[1]["selection_f1"] == 1
    metrics = metric_row([0, 0, 1, 1], [0, 1, 0, 1])
    assert metrics["f1"] == 0.5 and metrics["balanced_accuracy"] == 0.5
    assert "roc_auc" not in metrics


def test_loader_rejects_nontraining_rows(tmp_path):
    directory = tmp_path / "data/processed/audio_study_v1"
    directory.mkdir(parents=True)
    source = tmp_path / "src/dataset/audio_summary.py"
    source.parent.mkdir(parents=True)
    source.write_text("source")
    x = pd.DataFrame(
        {"audio__a": [1.0, 2.0, 3.0]}, index=pd.Index([1, 2, 3], name="Participant_ID")
    )
    labels = pd.DataFrame(
        {"PHQ_Binary": [0, 1, 1], "split": ["train", "train", "test"]}, index=x.index
    )
    x.to_parquet(directory / "features.parquet")
    labels[["PHQ_Binary"]].to_parquet(directory / "labels.parquet")
    labels.to_parquet(directory.parent / "labels.parquet")
    manifest = {
        "partition": "train",
        "output_sha256": {
            n: sha256(directory / n) for n in ["features.parquet", "labels.parquet"]
        },
        "contract": {"source": sha256(source)},
        "baseline_columns": ["audio__a"],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training participants"):
        load_inputs(tmp_path)
