"""Tests for the professional grading-plan executor."""

from __future__ import annotations

import numpy as np

from color_analyzer.analyzer.grading_plan import GradingPlanExecutor, is_grading_plan


def _cool_portrait():
    """Warm-ish subject on left, cool teal background on right, low contrast."""
    x = np.full((120, 180, 3), 0.5, np.float32)
    x[:, :90] = (0.55, 0.45, 0.42)
    x[:, 90:] = (0.30, 0.42, 0.48)
    return (x * 0.85 + 0.1).astype(np.float32)


FULL_PLAN = {
    "grading_plan": [
        {"operation": "WHITE_BALANCE", "params": {"temperature": 5500, "tint": 12}},
        {"operation": "PRIMARY_CORRECTION",
         "params": {"exposure": 0.1, "contrast": -10, "highlights": -25,
                    "shadows": 20, "whites": 15, "blacks": -10}},
        {"operation": "COLOR_WHEELS",
         "params": {"shadows": {"hue": 215, "saturation": 15},
                    "midtones": {"hue": 35, "saturation": 5},
                    "highlights": {"hue": 42, "saturation": 20, "luminance": 5}}},
        {"operation": "HSL",
         "params": {"orange": {"hue": 8, "saturation": 10, "luminance": 10},
                    "blue": {"hue": -5, "saturation": 10, "luminance": -10}}},
        {"operation": "PRESENCE",
         "params": {"texture": 10, "clarity": 15, "dehaze": 5, "vibrance": 20}},
    ],
    "editor_notes": {"warnings": ["Protect skin highlights from clipping."]},
    "executor_settings": {"apply_order": [
        "white_balance", "primary", "tone_curve", "color_wheels", "hsl",
        "presence", "vignette", "grain"]},
}


def test_is_grading_plan_detection():
    assert is_grading_plan(FULL_PLAN)
    assert not is_grading_plan({"grading": {"temperature": 10}})
    assert not is_grading_plan({"feature_vector": {}})


def test_plan_applies_expected_stages_in_order():
    res = GradingPlanExecutor().apply(_cool_portrait(), FULL_PLAN)
    assert res.applied_steps == [
        "WHITE_BALANCE", "PRIMARY_CORRECTION", "COLOR_WHEELS", "HSL", "PRESENCE",
    ]
    # stages present in apply_order but absent from the plan are skipped.
    assert set(res.skipped_steps) == {"TONE_CURVE", "VIGNETTE", "GRAIN"}
    assert res.warnings  # passed through


def test_white_balance_warms_cool_image():
    img = _cool_portrait()
    res = GradingPlanExecutor().apply(img, FULL_PLAN)
    before_rb = float((img[..., 0] - img[..., 2]).mean())
    after_rb = float((res.image[..., 0] - res.image[..., 2]).mean())
    assert before_rb < 0 or after_rb > before_rb  # net warming
    assert after_rb > 0  # ends up warm overall


def test_subject_stays_warmer_than_background():
    res = GradingPlanExecutor().apply(_cool_portrait(), FULL_PLAN)
    skin = res.image[:, :90]
    bg = res.image[:, 90:]
    skin_warmth = float((skin[..., 0] - skin[..., 2]).mean())
    bg_warmth = float((bg[..., 0] - bg[..., 2]).mean())
    assert skin_warmth > bg_warmth  # teal-orange separation preserved


def test_output_clamped_and_shaped():
    res = GradingPlanExecutor().apply(_cool_portrait(), FULL_PLAN)
    assert res.image.shape == (120, 180, 3)
    assert float(res.image.min()) >= 0.0 and float(res.image.max()) <= 1.0


def test_partial_plan_only_white_balance():
    plan = {"grading_plan": [
        {"operation": "WHITE_BALANCE", "params": {"temperature": 8000, "tint": 0}}]}
    img = np.full((8, 8, 3), 0.5, np.float32)
    res = GradingPlanExecutor().apply(img, plan)
    assert res.applied_steps == ["WHITE_BALANCE"]
    # 8000K target (above neutral) cools the image: blue up, red down.
    assert res.image[..., 2].mean() > res.image[..., 0].mean()


def test_empty_plan_is_identity_ish():
    img = np.full((8, 8, 3), 0.4, np.float32)
    res = GradingPlanExecutor().apply(img, {"grading_plan": []})
    assert np.allclose(res.image, img, atol=1e-6)
    assert res.applied_steps == []
