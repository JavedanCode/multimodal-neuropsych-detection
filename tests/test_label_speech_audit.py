import numpy as np
import pandas as pd
import pytest
import io
import tarfile
import wave
from src.dataset.label_speech_audit import transcript_intervals, covered_frames, scan


def test_intervals_merge_overlap_without_double_counting():
    frame = pd.DataFrame(
        {
            "Start_Time": [2.0, 1.0, 6.0],
            "End_Time": [4.0, 3.0, 8.0],
            "Text": ["a", "b", "c"],
            "Confidence": [1.0, 1.0, 1.0],
        }
    )
    intervals, stats = transcript_intervals(frame, 10.0)
    assert intervals == [[1.0, 4.0], [6.0, 8.0]]
    assert (
        stats["transcribed_seconds"] == 5
        and stats["transcribed_duration_fraction"] == 0.5
    )
    assert not stats["speaker_column_present"]
    mask = covered_frames(np.array([0.0, 1.0, 3.99, 4.0, 6.0, 8.0]), intervals)
    np.testing.assert_array_equal(mask, [False, True, True, False, True, False])


def test_bad_or_empty_transcript_intervals_excluded_and_counted():
    frame = pd.DataFrame(
        {
            "Start_Time": [0.0, -1.0, 2.0, 3.0, 4.0, np.nan],
            "End_Time": [1.0, 2.0, 2.0, 11.0, 5.0, 6.0],
            "Text": ["yes", "bad", "bad", "bad", " ", "bad"],
            "Speaker": ["participant"] * 6,
        }
    )
    intervals, stats = transcript_intervals(frame, 10.0)
    assert intervals == [[0.0, 1.0]]
    assert stats["invalid_or_empty_intervals"] == 5 and stats["speaker_column_present"]
    with pytest.raises(ValueError):
        transcript_intervals(frame.drop(columns="Start_Time"), 10.0)


def test_small_endpoint_rounding_is_clipped_to_audio():
    frame = pd.DataFrame({"Start_Time": [9.0], "End_Time": [10.05], "Text": ["a"]})
    intervals, stats = transcript_intervals(frame, 10.0)
    assert intervals == [[9.0, 10.0]] and stats["transcribed_seconds"] == 1


def test_archive_timing_audit_records_duplicate_times_without_changing_them(tmp_path):
    audio = io.BytesIO()
    with wave.open(audio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 32000)
    files = {
        "300_AUDIO.wav": audio.getvalue(),
        "300_Transcript.csv": b"Start_Time,End_Time,Text,Confidence\n0.1,0.4,example,1\n",
        "300_OpenSMILE2.3.0_egemaps.csv": b"frameTime;F0semitoneFrom27.5Hz_sma3nz\n0;0\n0.2;2\n0.2;3\n0.99;0\n",
    }
    path = tmp_path / "300_P.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo("300_P/" + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    result = scan(path, True)
    assert result["repeated_timestamps"] == 1 and result["feature_rows"] == 4
    assert result["audio_seconds"] == 1 and result["transcript_frame_fraction"] == 0.5
    assert (
        result["zero_f0_inside_transcript"] == 0
        and result["zero_f0_outside_transcript"] == 1
    )
    assert "example" not in str(result)
