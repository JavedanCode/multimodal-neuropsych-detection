import numpy as np
import pandas as pd
import pytest
from src.dataset.contracts import screened_rows, interval_summary
from src.training.splits import splits, transfer


@pytest.mark.parametrize("confidence", [None, [np.nan, -1, 2], [0.5, 1, 0]])
def test_confidence_unknown_never_becomes_perfect(confidence):
    f = pd.DataFrame(
        {"Start_Time": [0, 1, 2], "End_Time": [0.5, 1.5, 2.5], "Text": ["ordinary"] * 3}
    )
    if confidence is not None:
        f["Confidence"] = confidence
    r = screened_rows(f, 3)
    if confidence is None or confidence[0] != 0.5:
        assert not r["confidence_known"].any() and not r["confidence"].any()
    else:
        np.testing.assert_array_equal(r["confidence"], confidence)


def test_joint_folds_keep_people_and_domains_balanced():
    ids = list(range(300, 320)) + list(range(600, 610))
    y = pd.Series([0] * 10 + [1] * 10 + [0] * 5 + [1] * 5, index=ids)
    seen = []
    for a, b in splits(y, 5, 42):
        assert set(a).isdisjoint(b)
        seen.extend(b)
        assert ((y.index[b] >= 600) & (y.iloc[b].to_numpy() == 1)).sum() == 1
        splits(y.iloc[a], 3, 43)
    assert sorted(seen) == list(range(len(y)))
    a, b = transfer(y)
    assert (y.index[a] < 600).all() and (y.index[b] >= 600).all()
    with pytest.raises(ValueError):
        splits(y, 6, 42)


def test_out_of_order_transcript_retains_original_embedding_row_mapping():
    f = pd.DataFrame(
        {
            "Start_Time": [2, 0, 1],
            "End_Time": [2.5, 0.5, 1.5],
            "Text": ["late", "early", "middle"],
        }
    )
    r = screened_rows(f, 3)
    assert r["audit"]["backward_start_steps"] == 1
    np.testing.assert_array_equal(r["row_indices"], [0, 1, 2])
    order = np.argsort(r["starts"], kind="stable")
    assert [r["text"][i] for i in order] == ["early", "middle", "late"]


def test_interval_alignment_uses_real_timestamps_and_half_open_bounds():
    out, m, q, n = interval_summary(
        [[1], [3], [99], [5]],
        [0, 0.2, 0.2, 1.0],
        [0, 1],
        [0.3, 2],
        [True, True, False, True],
    )
    np.testing.assert_allclose(out, [[2, 1], [5, 0]])
    np.testing.assert_allclose(q, [2 / 3, 1])
    assert m.all()
    np.testing.assert_array_equal(n, [3, 1])


def test_measured_waveform_quality_detects_clipping_without_oracle_input():
    from src.dataset.alignment import waveform_quality

    clean = np.sin(np.arange(16000) * 0.1).astype(np.float32) * 0.1
    clipped = np.clip(clean * 100, -1, 1)
    a = waveform_quality(clean, [0], [1])
    b = waveform_quality(clipped, [0], [1])
    assert a[0, 0] == 0 and b[0, 0] > 0.5
    assert b[0, 1] > a[0, 1]
