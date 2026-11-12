"""Guard label-free participant/interviewer separation and pooled-study helpers."""

import numpy as np
import pandas as pd
import pytest

from src.dataset.participant_speech import (
    F0,
    SEGMENT_COLUMNS,
    assign,
    fit_template,
    session_statistics,
    summarize,
    voiced_segments,
)

COLUMNS = SEGMENT_COLUMNS + ["spectralFlux_sma3"]


def session(participant_pitch, seed, interviewer_blocks=40, participant_blocks=120):
    """Alternating 0.4 s voiced blocks separated by 0.4 s of silence."""
    random = np.random.default_rng(seed)
    rows = []
    speakers = ["i"] * interviewer_blocks + ["p"] * participant_blocks
    random.shuffle(speakers)
    for speaker in speakers:
        pitch = 40.0 if speaker == "i" else participant_pitch
        block = random.normal(size=(40, len(COLUMNS)))
        block[:, 0] = pitch + 0.5 * block[:, 0]
        block[:, 1] = (10.0 if speaker == "i" else 5.0) + 0.5 * block[:, 1]
        block[:, 2] = (20.0 if speaker == "i" else 32.0) + block[:, 2]
        block[:, COLUMNS.index("Loudness_sma3")] = 0.03 if speaker == "i" else 0.12
        silence = np.zeros((40, len(COLUMNS)))
        silence[:, COLUMNS.index("Loudness_sma3")] = 0.01
        rows += [block, silence]
    frame = pd.DataFrame(np.vstack(rows), columns=COLUMNS)
    frame["frameTime"] = np.arange(len(frame)) / 100
    return frame, np.array(speakers)


def test_segments_merge_short_gaps_and_drop_short_runs():
    voiced = np.zeros(200, bool)
    voiced[10:40] = voiced[50:70] = True  # 0.1 s gap: merged
    voiced[150:155] = True  # too short
    np.testing.assert_array_equal(voiced_segments(voiced), [[10, 69]])
    assert voiced_segments(np.zeros(5, bool)).shape == (0, 2)


def test_template_assignment_recovers_synthetic_interviewer():
    anchors = [session_statistics(session(23.0, s)[0], COLUMNS) for s in range(8)]
    template = fit_template(anchors)
    assert template is not None and abs(template["mean"][0] - 40) < 0.5
    frame, speakers = session(33.0, 99)  # higher-pitched participant
    stats = session_statistics(frame, COLUMNS)
    interviewer, audit = assign(stats, template)
    assert audit["separated"]
    assert np.mean(interviewer == (speakers == "i")) > 0.97
    everything = summarize(stats, np.ones(len(interviewer), bool), COLUMNS)
    participant = summarize(stats, ~interviewer, COLUMNS)
    assert participant[f"audio__{F0}__mean"] == pytest.approx(33.0, abs=0.3)
    assert everything[f"audio__{F0}__mean"] > 34
    assert participant["audio__voiced_fraction"] < everything["audio__voiced_fraction"]


def test_no_template_removes_nothing():
    stats = session_statistics(session(33.0, 1)[0], COLUMNS)
    interviewer, audit = assign(stats, None)
    assert not interviewer.any() and not audit["separated"]


def test_pose_translation_columns_are_dropped():
    from src.training.pooled_study import drop_pose

    frame = pd.DataFrame(columns=["facial__pose_Tx__mean", "facial__pose_Rx__mean", "audio__a"])
    assert list(drop_pose(frame)) == ["facial__pose_Rx__mean", "audio__a"]


def test_cli_pooled_study_runs_alone():
    from src.__main__ import main

    with pytest.raises(SystemExit):
        main(["run", "audio-visual", "--pooled-study", "--score-study"])
