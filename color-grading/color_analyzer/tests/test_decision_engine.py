"""Behaviour tests for the decision engine.

Shape is covered by ``test_schema.py``.  These check that the heuristics
actually point the right way — a cool frame gets warmed, clipped highlights get
recovered — because that is what silently regresses when the rules are retuned.
"""

from __future__ import annotations

import numpy as np

from color_analyzer import ColorGradingEngine
from color_analyzer.analyzer import schema
from color_analyzer.analyzer.decision_engine import DecisionEngine, to_executor_decision
from color_analyzer.analyzer.grading_plan import GradingPlanExecutor


def _cool_portrait() -> np.ndarray:
    """Skin-ish subject on a teal background, with an overall cool cast."""
    x = np.full((160, 240, 3), 0.5, np.float32)
    x[:, :120] = (0.62, 0.5, 0.44)   # warm-ish skin subject
    x[:, 120:] = (0.28, 0.42, 0.5)   # teal background
    x = np.clip(x * 0.8 + 0.12, 0, 1)
    x[..., 2] = np.clip(x[..., 2] * 1.08, 0, 1)  # cool/blue cast
    return x.astype(np.float32)


def _warm_image() -> np.ndarray:
    x = np.full((120, 160, 3), 0.5, np.float32)
    x[..., 0] *= 1.35  # strong red lift
    x[..., 2] *= 0.65  # blue pulled down
    return np.clip(x, 0, 1).astype(np.float32)


def _blown_highlights() -> np.ndarray:
    x = np.full((120, 160, 3), 0.55, np.float32)
    x[:, :60] = 1.0  # a third of the frame fully clipped
    return x


# -- structural invariants that hold for any image --------------------------
def test_key_set_is_identical_across_images():
    eng = ColorGradingEngine()
    a = eng.grade(_cool_portrait())
    b = eng.grade(np.full((64, 64, 3), 0.6, np.float32))
    assert a["grade"].keys() == b["grade"].keys()
    for name in a["grade"]:
        assert a["grade"][name].keys() == b["grade"][name].keys(), name


def test_grade_is_compact_while_analysis_stays_full():
    eng = ColorGradingEngine(deep=True)
    result = eng.analyze(_cool_portrait())
    doc = eng.decide(result)
    assert len(result.feature_vector) > 150   # internal analysis is still rich
    assert len(doc["grade"]) == 45            # the emitted contract is not


# -- white balance ----------------------------------------------------------
def test_cool_image_is_warmed():
    doc = ColorGradingEngine().grade(_cool_portrait())
    wb = doc["grade"]["white_balance.temperature"]
    # A blue cast reads as a high CCT; warming means recommending a lower one.
    assert wb["current"] > wb["recommended"]
    assert wb["delta"] < 0
    assert "Increase warmth" in doc["notes"]


def test_warm_image_is_cooled():
    doc = ColorGradingEngine().grade(_warm_image())
    wb = doc["grade"]["white_balance.temperature"]
    assert wb["current"] < wb["recommended"]
    assert wb["delta"] > 0
    assert "Reduce warmth" in doc["notes"]


def test_tint_is_pulled_toward_neutral():
    doc = ColorGradingEngine().grade(_cool_portrait())
    tint = doc["grade"]["white_balance.tint"]
    assert abs(tint["recommended"]) <= abs(tint["current"]) + 1e-6


# -- primary ----------------------------------------------------------------
def test_clipped_highlights_are_recovered():
    doc = ColorGradingEngine().grade(_blown_highlights())
    assert doc["grade"]["primary.highlights"]["delta"] < 0
    assert doc["grade"]["quality.highlight_clipping"]["current"] is True
    assert "Recover highlights" in doc["notes"]


def test_dark_image_gets_positive_exposure():
    dark = np.full((96, 96, 3), 0.12, np.float32)
    doc = ColorGradingEngine().grade(dark)
    assert doc["grade"]["primary.exposure"]["delta"] > 0
    assert doc["grade"]["quality.noise_risk"]["current"] == "high"


def test_exposure_target_lands_near_mid_grey():
    dark = np.full((96, 96, 3), 0.12, np.float32)
    doc = ColorGradingEngine().grade(dark)
    # Recommended exposure is measured in stops from mid-grey, so the target
    # should sit closer to zero than the measurement does.
    exposure = doc["grade"]["primary.exposure"]
    assert abs(exposure["recommended"]) < abs(exposure["current"])


# -- style ------------------------------------------------------------------
def test_style_target_is_a_known_style():
    from color_analyzer.analyzer.decision_engine import STYLE_TARGETS

    doc = ColorGradingEngine().grade(_cool_portrait())
    assert doc["style"]["target"] in STYLE_TARGETS


def test_creative_style_scores_in_unit_range():
    doc = ColorGradingEngine().grade(_cool_portrait())
    for param in schema.PARAMS:
        if param.group == "creative_style":
            assert 0.0 <= doc["grade"][param.name]["current"] <= 1.0, param.name


def test_notes_are_non_empty_and_unique():
    doc = ColorGradingEngine().grade(_cool_portrait())
    assert doc["notes"]
    assert len(doc["notes"]) == len(set(doc["notes"]))


# -- renderer bridge --------------------------------------------------------
def test_grade_renders_through_the_executor():
    img = _cool_portrait()
    doc = ColorGradingEngine().grade(img)
    plan = to_executor_decision(doc)
    result = GradingPlanExecutor().apply(img, plan)
    assert result.image.shape == img.shape
    assert np.isfinite(result.image).all()
    assert result.applied_steps[0] == "WHITE_BALANCE"


def test_executor_receives_deltas_not_states():
    doc = ColorGradingEngine().grade(_cool_portrait())
    plan = to_executor_decision(doc)
    # The renderer's white-balance convention is 6500K-relative.
    expected = 6500.0 + doc["grade"]["white_balance.temperature"]["delta"]
    assert plan["white_balance"]["temperature"] == expected
    assert plan["primary_corrections"]["contrast"] == doc["grade"]["primary.contrast"]["delta"]


def test_decision_engine_grade_matches_engine_decide():
    result = ColorGradingEngine().analyze(_cool_portrait())
    assert DecisionEngine().grade(result) == ColorGradingEngine().decide(result)
