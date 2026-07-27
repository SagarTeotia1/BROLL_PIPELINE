"""Tests for the colour-grading *application* module."""

from __future__ import annotations

import numpy as np

from color_analyzer.analyzer.grading import ColorGrader, GradingParams


def _mid_image():
    return np.full((32, 32, 3), 0.5, dtype=np.float32)


def test_neutral_params_are_identity():
    img = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    out = ColorGrader().apply(img, GradingParams())
    assert np.allclose(out, img, atol=1e-5)


def test_warm_temperature_shifts_red_up_blue_down():
    out = ColorGrader().apply(_mid_image(), GradingParams(temperature=80))
    assert out[..., 0].mean() > 0.5   # red boosted
    assert out[..., 2].mean() < 0.5   # blue reduced


def test_exposure_brightens():
    dark = ColorGrader().apply(_mid_image(), GradingParams(exposure=-1.0))
    bright = ColorGrader().apply(_mid_image(), GradingParams(exposure=1.0))
    assert bright.mean() > 0.5 > dark.mean()


def test_contrast_expands_range():
    ramp = np.stack([np.linspace(0, 1, 64)[None, :].repeat(16, 0)] * 3, -1).astype(np.float32)
    graded = ColorGrader().apply(ramp, GradingParams(contrast=1.6))
    assert graded.std() > ramp.std()


def test_saturation_zero_is_grayscale():
    img = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    out = ColorGrader().apply(img, GradingParams(saturation=0.0))
    # all channels equal (achromatic) when saturation is removed.
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-3)
    assert np.allclose(out[..., 1], out[..., 2], atol=1e-3)


def test_output_is_clamped():
    out = ColorGrader().apply(_mid_image(), GradingParams(exposure=3.0, contrast=2.0))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_json_roundtrip_and_clamping():
    p = GradingParams(temperature=999, gamma=1.4, split_shadow_strength=0.5)
    assert p.temperature == 999  # dataclass stores raw; clamping happens on load
    loaded = GradingParams.from_dict(p.to_dict())
    assert loaded.temperature == 100.0  # clamped to range
    assert loaded.gamma == 1.4


def test_from_dict_ignores_unknown_and_partial():
    p = GradingParams.from_dict({"contrast": 1.5, "not_a_field": 7})
    assert p.contrast == 1.5
    assert p.exposure == 0.0  # untouched default


def test_is_grading_vs_analysis_dict():
    assert GradingParams.is_grading_dict({"temperature": 1, "contrast": 1, "gamma": 1})
    assert not GradingParams.is_grading_dict({"feature_vector": {}, "summary": {}})


def test_from_analysis_produces_valid_params():
    report = {
        "feature_vector": {
            "white_balance.color_temperature": 3500.0,
            "contrast.global_contrast": 0.5,
            "colorfulness.color_richness": 0.7,
            "tone_curve.gamma": 1.1,
            "split_toning.split_tone_confidence": 0.8,
            "split_toning.shadows.hue": 200.0,
            "split_toning.shadows.saturation": 0.3,
            "split_toning.highlights.hue": 40.0,
            "split_toning.highlights.saturation": 0.2,
        }
    }
    p = GradingParams.from_analysis(report)
    assert p.temperature > 0  # 3500K reference => warm control
    assert p.split_shadow_strength > 0
    ranges = GradingParams.ranges()
    for key, value in p.to_dict().items():
        lo, hi = ranges[key]
        assert lo <= value <= hi
