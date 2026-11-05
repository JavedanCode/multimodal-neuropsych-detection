import pandas as pd
import pytest
from src.dataset.transcript_context import context


def test_context_uses_chronology_not_source_row_order():
    frame = pd.DataFrame(
        {
            "Start_Time": [4.0, 0.0, 2.0],
            "End_Time": [5.0, 1.0, 3.0],
            "Text": ["later", "earlier", "target"],
        }
    )
    rows = context(frame, 2.0, 3.0)
    assert [r["text"] for r in rows] == ["earlier", "target", "later"]
    assert rows[1]["target"] and rows[1]["source_row"] == 2
    assert sum(r["target"] for r in rows) == 1


def test_context_rejects_truncated_or_ambiguous_target():
    frame = pd.DataFrame(
        {"Start_Time": [1.0], "End_Time": [15.0], "Text": ["full utterance"]}
    )
    with pytest.raises(ValueError):
        context(frame, 1.0, 9.0)
    with pytest.raises(ValueError):
        context(pd.concat([frame, frame], ignore_index=True), 1.0, 15.0)
