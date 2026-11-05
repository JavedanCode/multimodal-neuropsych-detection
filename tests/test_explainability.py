import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.explain import (
    integrated_gradients,
    perturbation,
    ranking_agreement,
    run,
)
from src.models.reliability import ReliabilityFusion


def test_linear_ig_exact_signed_and_complete():
    x = torch.tensor([[2.0, 3.0, -1.0], [-2.0, 1.0, 4.0]])
    weight = torch.tensor([3.0, -2.0, 0.0])
    forward = lambda z: z @ weight + 7
    attribution, output, reference, residual = integrated_gradients(forward, x, 32)
    torch.testing.assert_close(attribution, x * weight)
    torch.testing.assert_close(reference, torch.full((2,), 7.0))
    torch.testing.assert_close(residual, torch.zeros(2))
    torch.testing.assert_close(attribution.sum(1), output - reference)


def test_nonlinear_ig_and_reference_zero():
    x = torch.tensor([[2.0, -3.0], [0.0, 0.0]])
    forward = lambda z: (z**2).sum(1)
    attribution, _, _, residual = integrated_gradients(forward, x, 64)
    torch.testing.assert_close(attribution, x**2)
    torch.testing.assert_close(residual, torch.zeros(2))
    with pytest.raises(ValueError):
        integrated_gradients(forward, x, 0)


def test_perturbation_finds_known_important_feature_and_is_reproducible():
    x = torch.ones((3, 10))
    forward = lambda z: z[:, 0] * 5 + z[:, 1] * 0.01
    attribution, *_ = integrated_gradients(forward, x, 16)
    a = perturbation(forward, x, attribution, fractions=(0.1, 1.0), seed=7)
    b = perturbation(forward, x, attribution, fractions=(0.1, 1.0), seed=7)
    pd.testing.assert_frame_equal(a, b)
    means = a[a.fraction == 0.1].groupby("method").absolute_logit_change.mean()
    assert means["top"] > means["random"] > means["bottom"]
    np.testing.assert_allclose(
        a[a.fraction == 1].absolute_logit_change, 5.01, rtol=1e-6
    )
    torch.testing.assert_close(x, torch.ones_like(x))


def test_quality_held_fixed_unavailable_modality_has_zero_attribution():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = ReliabilityFusion(2, 2).eval()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    q = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 1.0]])
    forward = lambda z: model(z[:, :2], z[:, 2:], q)[0]
    attribution, *_ = integrated_gradients(forward, x, 32)
    torch.testing.assert_close(attribution[:, 2:], torch.zeros((1, 2)))
    assert all(p.grad is None for p in model.parameters())


def test_ranking_alignment_and_missing_selected_features():
    a = pd.Series([3.0, 2.0, 1.0], index=["a", "b", "c"])
    assert ranking_agreement(a, a.iloc[::-1]) == {
        "spearman_union_zero_absent": 1.0,
        "top10_jaccard": 1.0,
    }
    b = pd.Series([3.0, 2.0], index=["a", "b"])
    result = ranking_agreement(a, b)
    assert result["top10_jaccard"] == pytest.approx(2 / 3)
    assert result["spearman_union_zero_absent"] == pytest.approx(1.0)


def test_input_provenance_mismatch_fails_before_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()
    (source / "results.json").write_text('{"input_sha256":{}}')
    for name in ["features.parquet", "labels.parquet", "quality.parquet"]:
        (processed / name).write_bytes(b"changed")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="hashes differ"):
        run(processed, source, output, tmp_path)
    assert not output.exists()
