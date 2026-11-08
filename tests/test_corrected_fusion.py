"""Guard voiced-frame aggregation and common-scale late fusion."""

import io

import numpy as np
import pytest

from src.dataset.tables import aggregate
from src.models.sparse_fusion import fuse_scores, margin_scale

COLUMNS = ["Loudness_sma3", "F0semitoneFrom27.5Hz_sma3nz", "F1amplitudeLogRelF0_sma3nz"]


def descriptor_csv(rows):
    lines = ["name;frameTime;" + ";".join(COLUMNS)]
    lines += [f"'x';{i * 0.01};" + ";".join(map(str, r)) for i, r in enumerate(rows)]
    return io.BytesIO("\n".join(lines).encode())


ROWS = [
    (0.2, 0.0, -201.0),  # unvoiced placeholders
    (0.4, 30.0, -10.0),
    (0.6, 20.0, -30.0),
    (0.8, 0.0, -201.0),
]


def test_voiced_only_excludes_unvoiced_placeholders():
    features, _, _ = aggregate(descriptor_csv(ROWS), COLUMNS, "audio", 2, voiced_only=True)
    assert features["audio__F0semitoneFrom27.5Hz_sma3nz__mean"] == pytest.approx(25.0)
    assert features["audio__F0semitoneFrom27.5Hz_sma3nz__std"] == pytest.approx(5.0)
    assert features["audio__F1amplitudeLogRelF0_sma3nz__mean"] == pytest.approx(-20.0)
    # Non-voicing descriptors keep every frame.
    assert features["audio__Loudness_sma3__mean"] == pytest.approx(0.5)
    assert features["audio__voiced_fraction"] == pytest.approx(0.5)


def test_legacy_default_still_averages_all_frames():
    features, _, _ = aggregate(descriptor_csv(ROWS), COLUMNS, "audio", 3)
    assert features["audio__F0semitoneFrom27.5Hz_sma3nz__mean"] == pytest.approx(12.5)
    assert "audio__voiced_fraction" not in features


def test_no_voiced_frames_gives_missing_values():
    rows = [(0.1, 0.0, -201.0), (0.2, 0.0, -201.0)]
    features, _, _ = aggregate(descriptor_csv(rows), COLUMNS, "audio", voiced_only=True)
    assert features["audio__F0semitoneFrom27.5Hz_sma3nz__mean"] is None
    assert features["audio__voiced_fraction"] == 0.0


def test_voicing_mask_is_audio_only():
    with pytest.raises(ValueError):
        aggregate(descriptor_csv(ROWS), COLUMNS, "facial", voiced_only=True)


def test_scaled_fusion_is_invariant_to_branch_margin_scale():
    random = np.random.default_rng(0)
    audio, face = random.normal(size=50), 30 * random.normal(size=50)
    quality = np.ones((50, 9))
    available = np.ones((50, 2), dtype=bool)
    scales = [margin_scale(audio), margin_scale(face)]
    base = fuse_scores(audio, face, quality, available, 0.5, 0.2, scales=scales)
    shrunk = fuse_scores(
        audio * 0.01, face, quality, available, 0.5, 0.2,
        scales=[scales[0] * 0.01, scales[1]],
    )
    np.testing.assert_allclose(base, shrunk)
    # Unscaled fusion follows the large-margin branch; scaled fusion does not.
    raw = fuse_scores(audio * 0.01, face, quality, available, 0.5, 0.2)
    assert np.mean(np.sign(raw) == np.sign(face)) > 0.95
    assert np.mean(np.sign(base) == np.sign(face)) < 0.95


def test_scaled_fusion_missing_inputs():
    quality = np.ones((3, 9))
    available = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    scores = fuse_scores([2.0, 2.0, 2.0], [-6.0, -6.0, -6.0], quality, available, 0.5,
                         0.2, scales=[2.0, 3.0])
    np.testing.assert_allclose(scores, [1.0, -2.0, np.log(0.25)])


def test_margin_scale_guards():
    assert margin_scale([0.0, 0.0]) == 1.0
    assert margin_scale([3.0, -4.0]) == pytest.approx(np.sqrt(12.5))
    with pytest.raises(ValueError):
        margin_scale([np.nan])


def test_cli_corrected_runs_alone():
    from src.__main__ import main

    with pytest.raises(SystemExit):
        main(["run", "audio-visual", "--corrected", "--sparse-fusion"])
