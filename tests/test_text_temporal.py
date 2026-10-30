import numpy as np
import pandas as pd
import pytest
import torch
from src.dataset.text_temporal import safe_rows, bin_means, facial_bins
from src.models.text_preprocessing import TextPreprocessor
from src.models.temporal_fusion import SequenceScaler, TemporalFusion, augment, perturb


def test_disclosure_neighbors_and_missing_text():
    frame = pd.DataFrame({'Start_Time': np.arange(7), 'End_Time': np.arange(7)+.5,
                          'Text': ['ordinary', 'before', 'my PHQ-8 score is 12', 'after', 'sleep is difficult', None, 'last'],
                          'Confidence': [1]*7})
    text, _, _, audit = safe_rows(frame, 8)
    assert text == ['ordinary', 'sleep is difficult', 'last']
    assert audit['direct_disclosures'] == 1
    assert audit['excluded_with_neighbors'] == 3


def test_bin_alignment_masks_and_quality():
    x, mask, q = bin_means([[2], [100], [4]], [0, .2, 31.9], 32, [True, False, True])
    assert x[0, 0] == 2 and x[-1, 0] == 4
    assert mask.sum() == 2 and q[0] == .5
    with pytest.raises(ValueError):
        bin_means([[1]], [100], 32)


def test_fold_preprocessors_ignore_heldout_and_missing_bins():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(20, 40)), columns=[f'text__{i}' for i in range(40)])
    prep = TextPreprocessor().fit(x.iloc[:15])
    original = prep.transform(x.iloc[:15])
    x.iloc[15:] = 1e6
    np.testing.assert_array_equal(original, prep.transform(x.iloc[:15]))
    arrays = [rng.normal(size=(5, 32, d)).astype('float32') for d in [4, 3, 2]]
    masks = np.ones((5, 3, 32), bool)
    masks[:, :, 0] = False
    scale = SequenceScaler().fit(arrays, masks)
    for a in arrays:
        a[:, 0] = 1e9
    for transformed in scale.transform(arrays, masks):
        assert (transformed[:, 0] == 0).all()


def test_facial_endpoint_mismatch_masks_without_rescaling():
    frame = pd.DataFrame({'timestamp': [0., 40.], 'success': [1, 1], 'confidence': [1., 1.], 'value': [1., 2.]})
    x, mask, q, gap = facial_bins(frame, ['value'], 32.)
    assert gap == 8 and not mask.any() and not x.any() and not q.any()


def test_quality_missing_masks_and_blind_ablation():
    torch.manual_seed(4)
    xs = [torch.randn(4, 32, d) for d in [4, 3, 2]]
    m = torch.ones(4, 3, 32, dtype=torch.bool)
    q = torch.ones(4, 3, 32)
    model = TemporalFusion([4, 3, 2], quality=True).eval()
    _, clean = model(xs, m, q)
    q[:, 1] = .01
    _, aware = model(xs, m, q)
    _, blind = model(xs, m, q, blind=True)
    assert (aware[:, 1] < clean[:, 1]).all()
    torch.testing.assert_close(clean, blind)
    x2, m2, q2 = perturb(xs, m, q, 'missing_audio', 1)
    first, w = model(x2, m2, q2)
    x2[1].fill_(1e8)
    second, _ = model(x2, m2, q2)
    torch.testing.assert_close(first, second)
    assert (w[:, 1] == 0).all()
    for seed in range(40):
        _, ma, _ = augment(x2, m2, q2, seed)
        assert ma.any(2).any(1).all()


def test_pointwise_model_is_order_invariant():
    torch.manual_seed(5)
    xs = [torch.randn(3, 32, d) for d in [4, 3, 2]]
    m = torch.ones(3, 3, 32, dtype=torch.bool)
    q = torch.ones(3, 3, 32)
    order = torch.randperm(32)
    model = TemporalFusion([4, 3, 2], temporal=False).eval()
    torch.testing.assert_close(model(xs, m, q)[0], model([x[:, order] for x in xs], m[:, :, order], q[:, :, order])[0])
