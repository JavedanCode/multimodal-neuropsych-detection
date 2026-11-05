import numpy as np
import pandas as pd
import pytest
from src.dataset.text_temporal import safe_rows, bin_means, facial_bins


def test_disclosure_neighbors_and_missing_text():
    frame = pd.DataFrame(
        {
            "Start_Time": np.arange(7),
            "End_Time": np.arange(7) + 0.5,
            "Text": [
                "ordinary",
                "before",
                "my PHQ-8 score is 12",
                "after",
                "sleep is difficult",
                None,
                "last",
            ],
            "Confidence": [1] * 7,
        }
    )
    text, _, _, audit = safe_rows(frame, 8)
    assert text == ["ordinary", "sleep is difficult", "last"]
    assert audit["direct_disclosures"] == 1
    assert audit["excluded_with_neighbors"] == 3


def test_bin_alignment_masks_and_quality():
    x, mask, q = bin_means([[2], [100], [4]], [0, 0.2, 31.9], 32, [True, False, True])
    assert x[0, 0] == 2 and x[-1, 0] == 4
    assert mask.sum() == 2 and q[0] == 0.5
    with pytest.raises(ValueError):
        bin_means([[1]], [100], 32)


def test_facial_endpoint_mismatch_masks_without_rescaling():
    frame = pd.DataFrame(
        {
            "timestamp": [0.0, 40.0],
            "success": [1, 1],
            "confidence": [1.0, 1.0],
            "value": [1.0, 2.0],
        }
    )
    x, mask, q, gap = facial_bins(frame, ["value"], 32.0)
    assert gap == 8 and not mask.any() and not x.any() and not q.any()
