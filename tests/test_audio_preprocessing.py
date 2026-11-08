"""Check the intended filtering effect without changing duration or speech gain."""

import numpy as np
import pandas as pd
import pytest

from src.dataset.audio_decode import RATE
from src.dataset.audio_preprocessing import clean_waveform, summarize_descriptors


def test_cleanup_removes_offset_and_preserves_raw_samples():
    time = np.arange(RATE * 3) / RATE
    original = (0.1 + 0.2 * np.sin(2 * np.pi * 200 * time)).astype(np.float32)
    np.testing.assert_array_equal(clean_waveform(original, "raw"), original)
    centered = clean_waveform(original, "dc")
    assert len(centered) == len(original)
    assert abs(centered.mean()) < 1e-7
    assert centered.std() == pytest.approx(original.std(), rel=1e-6)


def test_highpass_attenuates_rumble_without_shifting_speech():
    time = np.arange(RATE * 4) / RATE
    rumble = np.sin(2 * np.pi * 10 * time)
    speech = np.sin(2 * np.pi * 200 * time)
    filtered = clean_waveform(0.2 * (rumble + speech), "highpass")
    middle = slice(RATE, -RATE)
    amplitude = lambda reference: 2 * np.mean(filtered[middle] * reference[middle])
    assert abs(amplitude(rumble)) < 0.002
    assert amplitude(speech) == pytest.approx(0.2, rel=0.01)
    assert len(filtered) == len(time)


def test_summary_leaves_missing_values_for_fold_imputation():
    frame = pd.DataFrame({"pitch": [1.0, 3.0, np.nan], "missing": [np.nan] * 3})
    assert summarize_descriptors(frame) == {
        "audio__pitch__mean": 2.0,
        "audio__pitch__std": 1.0,
        "audio__missing__mean": None,
        "audio__missing__std": None,
    }


def test_cli_rejects_audio_cleanup_for_unrelated_study():
    from src.__main__ import main

    with pytest.raises(SystemExit) as error:
        main(["run", "reference", "--preprocess-audio"])
    assert error.value.code == 2
