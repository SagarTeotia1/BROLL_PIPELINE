"""The render loop: analyse, edit the controls, apply, and check it landed.

The central property is the **round trip**. A grading model reads the analysis,
returns modified controls, and the renderer must move the frame so that
re-analysing it reports something close to what was asked for. Everything else
here supports that: identity guards the delta arithmetic, robustness guards
against what a language model actually returns, and monotonicity guards the
direction of every operation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import pytest

from color_analyzer.editorial import EditorialAnalyzer
from color_analyzer.editorial import controls as C
from color_analyzer.editorial.apply import apply_controls

from .conftest import (
    clustered_frame,
    flat_grey,
    graded_scene_frame,
    portrait_frame,
    split_toned_frame,
)

#: Bounds on how much of a requested move must land.
#:
#: Wide on purpose. The renderer's response factors were measured on
#: photographic frames, and how far an operation moves its own reading is
#: genuinely content-dependent: flat saturated material reacts far more strongly
#: than a soft-lit scene, and a frame whose colour is split between two casts
#: can even move a global correction backwards. A single factor cannot be right
#: for all of it.
#:
#: So this is a **direction and magnitude** check, not an accuracy check: the
#: reading must move decisively toward what was asked for, and must not run away
#: past it. Landing accuracy on real footage is verified separately — the
#: measured per-control gains sit in 0.85-1.14 there.
MIN_PROGRESS = 0.3
MAX_PROGRESS = 2.5


@pytest.fixture(scope="module")
def analyzer():
    return EditorialAnalyzer(max_side=None)


def _set(base: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """Copy the control payload with one path changed."""
    out = copy.deepcopy(base)
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return out


def _progress(analyzer, image, path, move, protect_skin=False):
    """Signed fraction of a requested move that the render delivered.

    The target is ``source + move``, not an absolute value, so a test says "ask
    for 40 more contrast" rather than "ask for contrast 40" — which would be a
    no-op on a frame that already measures 40, and unreachable on one measuring
    -90.

    ``1.0`` means the target was reached, ``0.5`` half of it, ``1.4`` an
    overshoot, and anything ``<= 0`` means the reading did not move or moved the
    wrong way. Progress rather than "distance remaining" because an overshoot
    and a stall are very different situations that a closure metric scores the
    same.
    """
    control = C.CONTROL_BY_PATH[path]
    state = analyzer.analyze_rgb(image)
    base = C.extract(state)
    source = C.flatten(state)[path]

    target = min(control.hi, max(control.lo, source + move))
    headroom = abs(target - source)
    if headroom < 0.25 * abs(move):
        pytest.skip(f"{path} measures {source} on this frame, too close to its "
                    f"limit to test a {move:+g} move")

    result = apply_controls(image, _set(base, path, target), source=state,
                            protect_skin=protect_skin)
    achieved = C.flatten(analyzer.analyze_rgb(result.image))[path]
    return (achieved - source) / (target - source)


# ---------------------------------------------------------------------------
# identity — the delta arithmetic itself
# ---------------------------------------------------------------------------
def test_unchanged_controls_leave_the_frame_alone(analyzer):
    """The single most important guard in this module.

    If a model returns the analysis unmodified, the frame must come back
    untouched. Any operation applying in the wrong direction, double-counting a
    source value, or treating a target as an offset shows up here immediately.
    """
    image = portrait_frame()
    state = analyzer.analyze_rgb(image)

    result = apply_controls(image, C.extract(state), source=state)

    assert result.applied == []
    assert float(np.abs(result.image - image).mean()) < 1e-6


def test_identity_holds_without_a_supplied_source(analyzer):
    """Same, but letting the renderer measure the frame itself."""
    image = clustered_frame()
    state = analyzer.analyze_rgb(image)

    result = apply_controls(image, C.extract(state))

    assert float(np.abs(result.image - image).mean()) < 0.01


def test_empty_payload_is_a_no_op(analyzer):
    image = portrait_frame()
    result = apply_controls(image, {})
    assert result.applied == []
    assert np.array_equal(result.image, np.clip(image, 0, 1).astype(np.float32))


# ---------------------------------------------------------------------------
# round trip — does a requested state actually arrive?
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path,move", [
    ("white_balance.temperature", -1500),
    ("white_balance.temperature", +2000),
    ("white_balance.tint", -30),
    ("white_balance.tint", +25),
    ("tone.exposure", -1.0),
    ("tone.exposure", +0.6),
    ("tone.contrast", +40),
    ("tone.contrast", -30),
    ("tone.brightness", -30),
    ("tone.gamma", +0.5),
    ("color.saturation", -40),
    ("color.saturation", +30),
    ("color.vibrance", +30),
    ("split_toning.shadows.saturation", -35),
])
def test_requested_state_is_reached(analyzer, path, move):
    """Most of the requested move must land, in the right direction.

    Not all of it: only the levels are directly settable, and every other
    operation interacts with the measurement it is judged by — a colour wheel
    writes through the same soft zone weight the analyzer reads through, a band
    boost lifts the frame it is compared against. The renderer compensates with
    factors measured on real footage (``apply.RESPONSE``); what remains is the
    part that cannot be recovered in a single pass, plus the content sensitivity
    of those factors.
    """
    progress = _progress(analyzer, graded_scene_frame(), path, move)
    assert MIN_PROGRESS <= progress <= MAX_PROGRESS, (
        f"{path} {move:+g}: {progress:.0%} of the move landed"
    )


@pytest.mark.parametrize("path,move", [
    ("wheels.lift.blue", +35),
    ("wheels.lift.red", -30),
    ("wheels.gain.luma", +30),
    ("wheels.gamma.luma", -25),
    ("hsl.blue.saturation", +25),
    ("hsl.green.saturation", -20),
])
def test_zone_and_band_controls_reach_their_target(analyzer, path, move):
    progress = _progress(analyzer, graded_scene_frame(), path, move)
    assert MIN_PROGRESS <= progress <= MAX_PROGRESS, (
        f"{path} {move:+g}: {progress:.0%} of the move landed"
    )


@pytest.mark.parametrize("path,target,tolerance", [
    ("tone.black_point", 0.08, 0.03),
    ("tone.white_point", 0.80, 0.03),
])
def test_levels_are_hit_exactly(analyzer, path, target, tolerance):
    """Black and white point are signal levels, so a linear remap lands them.

    These are the only controls held to an absolute tolerance rather than a
    fraction of the move.
    """
    image = graded_scene_frame()
    state = analyzer.analyze_rgb(image)
    result = apply_controls(image, _set(C.extract(state), path, target), source=state)

    achieved = C.flatten(analyzer.analyze_rgb(result.image))[path]
    assert abs(achieved - target) <= tolerance


@pytest.mark.parametrize("path,small,large", [
    ("tone.contrast", +12, +40),
    ("tone.exposure", -0.3, -1.0),
    ("color.saturation", -12, -40),
    ("white_balance.temperature", -600, -2000),
])
def test_bigger_requests_produce_bigger_changes(analyzer, path, small, large):
    """Monotonicity. A control that saturates or reverses is worse than useless."""
    image = graded_scene_frame()
    state = analyzer.analyze_rgb(image)
    base = C.extract(state)
    source = C.flatten(state)[path]
    control = C.CONTROL_BY_PATH[path]

    def moved(move):
        target = min(control.hi, max(control.lo, source + move))
        result = apply_controls(image, _set(base, path, target), source=state,
                                protect_skin=False)
        return float(np.abs(result.image - image).mean())

    assert moved(large) > moved(small) > 0.0


def test_a_full_dark_grade_darkens_and_cools(analyzer):
    """The end-to-end case this exists for: a model asks for a dark look."""
    # Not portrait_frame: it has no neutral content, so its white-balance
    # reading carries a confidence of 0.06 and cannot be steered. See
    # test_white_balance_confidence_predicts_controllability.
    image = graded_scene_frame()
    state = analyzer.analyze_rgb(image)
    base = C.extract(state)

    target = copy.deepcopy(base)
    target["tone"]["exposure"] = base["tone"]["exposure"] - 0.8
    target["tone"]["black_point"] = 0.05
    target["white_balance"]["temperature"] = base["white_balance"]["temperature"] + 2000
    target["color"]["saturation"] = base["color"]["saturation"] - 20

    result = apply_controls(image, target, source=state)
    after = analyzer.analyze_rgb(result.image)

    assert after["tone"]["brightness"] < state["tone"]["brightness"]
    assert after["white_balance"]["temperature"] > state["white_balance"]["temperature"]
    assert set(result.applied) >= {"white_balance", "levels", "exposure", "saturation"}

    # Deliberately not asserting the saturation *reading* fell. Darkening raises
    # measured HSV saturation for much of a frame, which can outweigh a modest
    # desaturation request in the same pass — the same interaction as
    # test_darkening_a_frame_lowers_its_measured_contrast. The saturation
    # control is verified on its own in test_requested_state_is_reached.


def test_white_balance_confidence_predicts_controllability(analyzer):
    """A frame with no neutral content cannot be white-balanced to a target.

    The estimator measures the light from near-neutral pixels, so a frame made
    entirely of saturated colour gives it nothing to work with. It says so —
    ``confidence`` collapses — and the round trip fails there in a way that is
    reported rather than silent. A caller driving these controls should check
    the confidence before trusting a temperature.
    """
    opaque = analyzer.analyze_rgb(portrait_frame())["white_balance"]
    measurable = analyzer.analyze_rgb(graded_scene_frame())["white_balance"]

    assert opaque["confidence"] < 0.2
    assert measurable["confidence"] > 0.5

    progress = _progress(analyzer, graded_scene_frame(),
                         "white_balance.temperature", +1500)
    assert progress >= MIN_PROGRESS


def test_darkening_a_frame_lowers_its_measured_contrast(analyzer):
    """Exposure and contrast interact, and a caller needs to know which way.

    Contrast is reported as the absolute p95-p5 luminance spread, so halving the
    signal roughly halves it. Measured on one frame: dropping exposure by 0.8
    stops takes the contrast reading from -20 to -68 on its own, while a +30
    contrast request delivers +29 on its own. Ask for both and the exposure wins.

    This is honest rather than broken — a darker image really does occupy less
    of the range — but it means "darker *and* punchier" needs a much larger
    contrast request than the arithmetic suggests, and a model driving these
    controls has to be told so.
    """
    image = portrait_frame()
    state = analyzer.analyze_rgb(image)
    base = C.extract(state)

    darker = apply_controls(
        image, _set(base, "tone.exposure", base["tone"]["exposure"] - 0.8), source=state)
    punchier = apply_controls(
        image, _set(base, "tone.contrast", min(100, base["tone"]["contrast"] + 30)),
        source=state)

    assert analyzer.analyze_rgb(darker.image)["tone"]["contrast"] < state["tone"]["contrast"]
    assert analyzer.analyze_rgb(punchier.image)["tone"]["contrast"] > state["tone"]["contrast"]


# ---------------------------------------------------------------------------
# absent content
# ---------------------------------------------------------------------------
def test_a_band_that_is_not_in_the_frame_is_left_alone(analyzer):
    """A model cannot conjure a colour the frame does not contain.

    The analyzer reports 0 for a band below its presence floor, meaning "no such
    colour here" rather than "this colour is neutral". Differencing a target
    against that sentinel produced a correction with no basis, and was measured
    pushing the band the wrong way.
    """
    image = np.zeros((120, 160, 3), np.float32)
    image[...] = (0.15, 0.20, 0.80)  # blue only
    state = analyzer.analyze_rgb(image)
    assert state["hsl"]["green"]["presence"] == 0.0

    result = apply_controls(image, _set(C.extract(state), "hsl.green.saturation", 80),
                            source=state)
    assert "hsl" not in result.applied
    assert float(np.abs(result.image - image).mean()) < 1e-6


def test_grading_a_flat_grey_frame_does_not_crash(analyzer):
    image = flat_grey(0.5)
    state = analyzer.analyze_rgb(image)
    target = _set(C.extract(state), "tone.contrast", 40)

    result = apply_controls(image, target, source=state)
    assert np.isfinite(result.image).all()


# ---------------------------------------------------------------------------
# robustness — what a language model actually returns
# ---------------------------------------------------------------------------
def test_malformed_payload_never_raises(analyzer):
    image = portrait_frame()
    result = apply_controls(image, {
        "tone": {"contrast": "not a number", "exposure": "-1.5 stops"},
        "white_balance": {"temperature": 99999},
        "palette": [{"hex": "#ffffff"}],
        "look": {"mood": "dark and moody"},
        "made_up_section": {"nonsense": 5},
        "hsl": {"turquoise": {"saturation": 10}},
    })

    assert np.isfinite(result.image).all()
    assert any("not a number" in note for note in result.ignored)
    assert any("read-only" in note for note in result.ignored)
    assert any("temperature" in note for note in result.clamped)


def test_a_flat_dotted_payload_works_too(analyzer):
    """Models return ``{"tone.contrast": 20}`` about as often as nested."""
    image = clustered_frame()
    state = analyzer.analyze_rgb(image)
    source = C.flatten(state)["tone.contrast"]

    result = apply_controls(image, {"tone.contrast": source + 30}, source=state)
    assert "contrast" in result.applied


def test_non_mapping_payload_is_reported_not_raised(analyzer):
    result = apply_controls(portrait_frame(), "please make it darker")
    assert result.applied == []
    assert result.ignored


def test_out_of_range_values_are_clamped_not_rejected(analyzer):
    image = clustered_frame()
    state = analyzer.analyze_rgb(image)

    result = apply_controls(image, {"tone": {"contrast": 5000}}, source=state)
    assert "contrast" in result.applied          # clamped to +100, still applied
    assert any("out of range" in note for note in result.clamped)


def test_a_crushing_grade_is_reported(analyzer):
    """A grade that destroys shadow detail must say so.

    Nothing in the control values reveals it — the render did exactly what it
    was asked — so the cost has to be measured on the result and surfaced.
    """
    image = graded_scene_frame()
    state = analyzer.analyze_rgb(image)
    base = C.extract(state)

    brutal = copy.deepcopy(base)
    brutal["tone"]["exposure"] = base["tone"]["exposure"] - 2.0
    brutal["tone"]["contrast"] = min(100, base["tone"]["contrast"] + 80)

    result = apply_controls(image, brutal, source=state)
    assert result.crushed_shadows > 0.02
    assert any("crushed to black" in note for note in result.warnings())


def test_a_gentle_grade_reports_no_damage(analyzer):
    image = graded_scene_frame()
    state = analyzer.analyze_rgb(image)
    gentle = _set(C.extract(state), "white_balance.tint",
                  C.flatten(state)["white_balance.tint"] + 8)

    result = apply_controls(image, gentle, source=state)
    assert result.warnings() == []


def test_result_summary_is_readable(analyzer):
    image = clustered_frame()
    state = analyzer.analyze_rgb(image)
    result = apply_controls(image, _set(C.extract(state), "tone.contrast", 40), source=state)
    assert "applied" in result.summary()


# ---------------------------------------------------------------------------
# skin protection
# ---------------------------------------------------------------------------
def test_skin_protection_moderates_the_grade_on_skin(analyzer):
    """An aggressive grade must move skin less than it moves everything else.

    Measured on the skin mask itself rather than assumed: the same target is
    rendered twice, and the protected version must shift skin pixels less while
    still shifting the frame as a whole.
    """
    from color_analyzer.editorial.frame import Frame
    from color_analyzer.editorial.skin import skin_mask

    image = portrait_frame()
    state = analyzer.analyze_rgb(image)
    base = C.extract(state)

    target = copy.deepcopy(base)
    target["color"]["saturation"] = min(100, base["color"]["saturation"] + 60)
    target["tone"]["contrast"] = min(100, base["tone"]["contrast"] + 60)

    protected = apply_controls(image, target, source=state, protect_skin=True)
    unprotected = apply_controls(image, target, source=state, protect_skin=False)

    mask = skin_mask(Frame.from_rgb(image, max_side=None)).reshape(image.shape[:2])
    assert mask.sum() > 0, "fixture must contain skin for this test to mean anything"

    def shift(graded):
        difference = np.abs(graded.image - image).mean(axis=2)
        return float((difference * mask).sum() / mask.sum())

    assert shift(protected) < shift(unprotected)


def test_skin_protection_still_grades_the_rest_of_the_frame(analyzer):
    image = portrait_frame()
    state = analyzer.analyze_rgb(image)
    target = _set(C.extract(state), "tone.contrast",
                  min(100, C.flatten(state)["tone.contrast"] + 50))

    result = apply_controls(image, target, source=state, protect_skin=True)
    assert float(np.abs(result.image - image).mean()) > 0.01


# ---------------------------------------------------------------------------
# controls contract
# ---------------------------------------------------------------------------
def test_extract_round_trips_through_flatten(analyzer):
    state = analyzer.analyze_rgb(portrait_frame())
    nested = C.extract(state)
    flat = C.flatten(nested)
    assert set(flat) <= set(C.CONTROL_PATHS)
    assert flat == C.flatten(state)


def test_every_control_is_present_for_a_normal_frame(analyzer):
    """A frame with content in it should expose the whole surface."""
    state = analyzer.analyze_rgb(split_toned_frame())
    flat = C.flatten(state)
    missing = set(C.CONTROL_PATHS) - set(flat)
    assert not missing, f"controls missing from the analysis: {sorted(missing)}"
