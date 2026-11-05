import numpy as np
import pytest
from src.dataset.topics import summarize_topics
from src.models.text_candidates import choose, residual, eligible_quality


def test_topic_pooling_is_order_invariant_and_empty_topic_uses_global_mean():
    x = np.array([[2.0, 0.0], [4.0, 0.0], [0.0, 2.0]])
    anchors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    b, c, n = summarize_topics(x, anchors)
    np.testing.assert_array_equal(n, [2, 1, 0])
    np.testing.assert_allclose(c.reshape(3, 2)[2], x.mean(0))
    rb, rc, rn = summarize_topics(x[::-1], anchors)
    np.testing.assert_allclose(rb, b)
    np.testing.assert_allclose(rc, c)
    np.testing.assert_array_equal(rn, n)
    with pytest.raises(ValueError):
        summarize_topics(np.array([[np.nan, 1]]), anchors)


def test_precision_policy_does_not_trade_away_declared_recall_or_f1():
    reference = {"metrics": {"precision": 0.5, "recall": 0.75, "f1": 0.60}}
    invalid = {"metrics": {"precision": 1.0, "recall": 0.2, "f1": 0.33}}
    valid = {"metrics": {"precision": 0.6, "recall": 0.7, "f1": 0.64}}
    assert choose([invalid, reference, valid], reference) is valid
    assert choose([invalid], reference) is reference
    tie = {"metrics": dict(reference["metrics"])}
    assert choose([reference, tie], reference) is reference


class FixedMargin:
    def __init__(self, value):
        self.value = value

    def decision_function(self, x):
        return np.full(len(x), self.value)


def test_quality_residual_ignores_unavailable_or_unreliable_inputs():
    q = np.ones((2, 11))
    q[:, 2] = 0.0
    q[:, 5] = 0.0
    data = {
        "audio": np.zeros((2, 2)),
        "face": np.zeros((2, 2)),
        "quality": q,
        "available": np.ones((2, 3)),
    }
    models = {"audio": FixedMargin(10), "face": FixedMargin(2)}
    np.testing.assert_allclose(residual(models, [2, 1], data), [3.5, 3.5])
    np.testing.assert_allclose(residual(models, [2, 1], data, "missing_both"), [0, 0])
    data["quality"][:, 2] = 0.5
    np.testing.assert_allclose(residual(models, [2, 1], data), [2, 2])
    data["available"][:, 2] = 0
    np.testing.assert_allclose(residual(models, [2, 1], data), [0, 0])
    with pytest.raises(ValueError):
        eligible_quality(data, "oracle_noise")


def test_bootstrap_preserves_strata_and_keeps_repeat_predictions_paired():
    from src.evaluation.uncertainty import bootstrap_weights, sampled_metrics

    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    d = np.array(["WoZ"] * 4 + ["AI"] * 4)
    w = bootstrap_weights(y, d, draws=20)
    np.testing.assert_array_equal(w.sum(1), 8)
    for domain in ["WoZ", "AI"]:
        for label in [0, 1]:
            np.testing.assert_array_equal(w[:, (d == domain) & (y == label)].sum(1), 2)
    perfect = np.repeat(y[:, None], 3, axis=1)
    for values in sampled_metrics(y, perfect, w).values():
        np.testing.assert_array_equal(values, np.ones(20))


def test_human_review_rejects_missing_answers_and_changed_clip_provenance():
    from src.dataset.review_annotations import validate

    source = {
        "file": "clip.wav",
        "kind": "transcript_covered",
        "start": 0,
        "end": 1,
        "transcript": "example",
    }
    row = {
        **source,
        "review_key": "clip.wav",
        "speaker": "Uncertain",
        "timing": "Matches",
        "reviewed": True,
    }
    result = validate([row], {"clip.wav": source})
    assert (
        result["covered_clips_needing_followup"] == 1
        and not result["whole_dataset_speaker_validated"]
    )
    with pytest.raises(ValueError):
        validate([{**row, "reviewed": False}], {"clip.wav": source})
    with pytest.raises(ValueError):
        validate([{**row, "end": 2}], {"clip.wav": source})
