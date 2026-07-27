"""Per-stage behaviour: does each reading point the right way?

The contract tests check the document's shape; these check it says something
true about the image.
"""

from __future__ import annotations

import numpy as np
import pytest

from color_analyzer.editorial import EditorialAnalyzer
from color_analyzer.editorial.color import analyze_color
from color_analyzer.editorial.frame import Frame
from color_analyzer.editorial.hsl import BAND_CENTRES, analyze_hsl
from color_analyzer.editorial.palette import analyze_palette
from color_analyzer.editorial.skin import analyze_skin
from color_analyzer.editorial.split_toning import analyze_split_toning
from color_analyzer.editorial.tone import analyze_tone
from color_analyzer.editorial.wheels import analyze_wheels
from color_analyzer.editorial.white_balance import analyze_white_balance

from .conftest import (
    cool_frame,
    flat_grey,
    hue_wheel_frame,
    portrait_frame,
    split_toned_frame,
    warm_frame,
)


def _frame(rgb: np.ndarray) -> Frame:
    return Frame.from_rgb(rgb, max_side=None)


# -- tone -------------------------------------------------------------------
def test_dark_frame_reads_dark_and_underexposed():
    tone = analyze_tone(_frame(flat_grey(0.12)))
    assert tone["brightness"] < -50
    assert tone["exposure"] < -1.0


def test_bright_frame_reads_bright_and_overexposed():
    tone = analyze_tone(_frame(flat_grey(0.88)))
    assert tone["brightness"] > 50
    assert tone["exposure"] > 0.7


def test_flat_frame_has_no_contrast():
    assert analyze_tone(_frame(flat_grey(0.5)))["contrast"] <= -95


def test_ramp_has_contrast_and_a_full_tonal_range():
    ramp = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None].repeat(64, 0).repeat(3, 2)
    tone = analyze_tone(_frame(ramp))
    assert tone["contrast"] > 50
    assert tone["black_point"] < 0.05
    assert tone["white_point"] > 0.95


def test_gamma_is_monotonic_in_the_applied_exponent():
    """More gamma must always read as more gamma.

    The range normalisation that makes this reading exposure-independent also
    compresses it toward 1.0, so the absolute value is approximate (a ramp
    raised to 0.45 reads about 0.7). What must hold is the ordering, because
    that is what a model adjusting the control relies on.
    """
    ramp = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None].repeat(96, 0).repeat(3, 2)
    readings = [analyze_tone(_frame(np.clip(ramp ** e, 0, 1)))["gamma"]
                for e in (0.45, 0.7, 1.0, 1.5, 2.2)]
    assert readings == sorted(readings), readings
    assert readings[0] < 1.0 < readings[-1]


def test_gamma_recovers_a_known_exponent_approximately():
    ramp = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None].repeat(96, 0).repeat(3, 2)
    assert analyze_tone(_frame(np.clip(ramp ** 2.2, 0, 1)))["gamma"] == pytest.approx(2.2, abs=0.3)
    assert analyze_tone(_frame(ramp))["gamma"] == pytest.approx(1.0, abs=0.2)


def test_gamma_separates_curve_shape_from_exposure():
    """A darkened frame with an intact curve must not read as a gamma change.

    Scaling every pixel changes level and range but not the distribution's shape
    within that range, so gamma should stay put while exposure moves.
    """
    base = portrait_frame()
    darker = np.clip(base * 0.55, 0, 1)

    reference = analyze_tone(_frame(base))
    dimmed = analyze_tone(_frame(darker))

    assert dimmed["exposure"] < reference["exposure"] - 0.4   # exposure moved
    assert abs(dimmed["gamma"] - reference["gamma"]) < 0.35   # shape did not


def test_clipping_is_reported():
    frame = flat_grey(0.5)
    frame[:20] = 1.0
    frame[-20:] = 0.0
    tone = analyze_tone(_frame(frame))
    assert tone["clipped_highlights"] > 0.05
    assert tone["clipped_shadows"] > 0.05


# -- white balance ----------------------------------------------------------
def test_warm_frame_reads_below_neutral_kelvin():
    assert analyze_white_balance(_frame(warm_frame()))["temperature"] < 5000


def test_cool_frame_reads_above_neutral_kelvin():
    assert analyze_white_balance(_frame(cool_frame()))["temperature"] > 7500


def test_neutral_frame_reads_near_d65():
    wb = analyze_white_balance(_frame(flat_grey(0.5)))
    assert 6000 <= wb["temperature"] <= 7000
    assert abs(wb["tint"]) <= 5


def test_confidence_drops_when_no_pixel_is_truly_neutral():
    """A saturated frame has no greys to measure the light with, and says so."""
    saturated = np.zeros((120, 160, 3), np.float32)
    saturated[..., 0] = 0.85          # a strongly red frame
    saturated[..., 1] = 0.15
    saturated[..., 2] = 0.10

    assert analyze_white_balance(_frame(saturated))["confidence"] < 0.5
    assert analyze_white_balance(_frame(flat_grey(0.5)))["confidence"] > 0.9


def test_neutral_pixels_drive_the_estimate_not_the_subject():
    """A coloured subject on neutral surroundings must not shift the reading.

    This is the failure grey-world has: it averages the subject in and reports a
    cast the light does not have.
    """
    frame = flat_grey(0.5)
    reference = analyze_white_balance(_frame(frame))["temperature"]

    frame[:, :60] = (0.05, 0.55, 0.15)  # a large saturated green object
    with_subject = analyze_white_balance(_frame(frame))["temperature"]

    assert abs(with_subject - reference) <= 500


# -- colour -----------------------------------------------------------------
def test_grey_frame_has_no_colour():
    color = analyze_color(_frame(flat_grey(0.5)))
    assert color["colorfulness"] == 0
    assert color["saturation"] <= -95


def test_saturated_frame_reads_colourful():
    assert analyze_color(_frame(hue_wheel_frame()))["colorfulness"] > 40


# -- wheels -----------------------------------------------------------------
def test_warm_frame_pushes_every_wheel_red():
    wheels = analyze_wheels(_frame(warm_frame()))
    for wheel in wheels.values():
        if wheel["coverage"] > 0.01:
            assert wheel["red"] > 0 > wheel["blue"]


def test_neutral_frame_leaves_the_wheels_balanced():
    for wheel in analyze_wheels(_frame(flat_grey(0.5))).values():
        if wheel["coverage"] > 0.01:
            assert abs(wheel["red"]) <= 2
            assert abs(wheel["blue"]) <= 2


def test_empty_zone_reports_neutral_rather_than_noise():
    """A frame with no highlights must not invent a highlight colour balance."""
    gain = analyze_wheels(_frame(flat_grey(0.15)))["gain"]
    assert gain["coverage"] == 0.0
    assert gain["red"] == gain["green"] == gain["blue"] == 0


# -- split toning -----------------------------------------------------------
def test_split_toned_frame_is_detected():
    split = analyze_split_toning(_frame(split_toned_frame()))
    assert split["separation"] > 60
    assert split["strength"] > 15


def test_uniform_cast_is_not_a_split_tone():
    """A warm frame is tinted, not split-toned; the strength must say so."""
    assert analyze_split_toning(_frame(warm_frame()))["strength"] <= 10


# -- palette ----------------------------------------------------------------
def test_palette_finds_the_planted_colours():
    frame = np.zeros((120, 300, 3), np.float32)
    frame[:, :100] = (0.85, 0.15, 0.15)   # red third
    frame[:, 100:200] = (0.15, 0.75, 0.20)  # green third
    frame[:, 200:] = (0.20, 0.25, 0.85)   # blue third

    swatches = analyze_palette(_frame(frame), colors=6)
    assert 3 <= len(swatches) <= 6
    assert sum(s["coverage"] for s in swatches) == pytest.approx(1.0, abs=0.02)

    channels = [int(np.argmax(s["rgb"])) for s in swatches[:3]]
    assert set(channels) == {0, 1, 2}


def test_palette_is_ordered_by_coverage():
    swatches = analyze_palette(_frame(portrait_frame()))
    coverages = [s["coverage"] for s in swatches]
    assert coverages == sorted(coverages, reverse=True)


def test_palette_merges_duplicate_swatches():
    """A two-colour frame must not be reported as six near-identical swatches."""
    frame = np.zeros((120, 200, 3), np.float32)
    frame[:, :100] = (0.9, 0.9, 0.9)
    frame[:, 100:] = (0.1, 0.1, 0.1)
    assert len(analyze_palette(_frame(frame), colors=6)) == 2


def test_small_saturated_region_is_labelled_an_accent():
    frame = flat_grey(0.45)
    frame[:, :14] = (0.95, 0.1, 0.05)  # ~7% of the frame, strongly saturated
    roles = {s["role"] for s in analyze_palette(_frame(frame), colors=6)}
    assert "accent" in roles


# -- skin -------------------------------------------------------------------
def test_skin_is_detected_and_described():
    skin = analyze_skin(_frame(portrait_frame()))
    assert skin["detected"] is True
    assert skin["coverage"] > 0.2
    assert 0 <= skin["hue"] <= 50
    assert skin["tone"] != "none"


def test_no_skin_in_a_grey_frame():
    assert analyze_skin(_frame(flat_grey(0.5)))["detected"] is False


def test_saturated_orange_is_not_mistaken_for_skin():
    """The RGB ordering rule must reject a traffic cone.

    A hue window alone admits any warm surface; skin also requires red > green >
    blue with a margin, which saturated orange fails.
    """
    cone = np.zeros((120, 160, 3), np.float32)
    cone[...] = (0.95, 0.42, 0.02)  # saturation ~0.98, above the skin ceiling
    assert analyze_skin(_frame(cone))["detected"] is False


# -- HSL --------------------------------------------------------------------
def test_every_band_is_found_in_a_hue_wheel():
    bands = analyze_hsl(_frame(hue_wheel_frame()))
    for name in BAND_CENTRES:
        assert bands[name]["presence"] > 0.02, name


def test_absent_bands_report_zero_rather_than_noise():
    """A frame containing one hue must not report offsets for the other six."""
    frame = np.zeros((120, 160, 3), np.float32)
    frame[...] = (0.15, 0.20, 0.80)  # blue only

    bands = analyze_hsl(_frame(frame))
    assert bands["blue"]["presence"] > 0.05
    for name in ("red", "orange", "yellow", "green"):
        assert bands[name]["presence"] == 0.0
        assert bands[name]["hue"] == 0
        assert bands[name]["saturation"] == 0


def test_band_membership_is_continuous_across_a_boundary():
    """A hue sweeping between two band centres must not make either jump.

    Hard bins would flip a pixel's band on a one-degree shift; the raised-cosine
    membership is what stops consecutive frames disagreeing about a colour that
    sits between two bands.
    """
    import cv2

    presences = []
    for hue in range(30, 62, 4):  # orange centre -> yellow centre
        hsv = np.zeros((60, 80, 3), np.float32)
        hsv[..., 0] = float(hue)
        hsv[..., 1] = 0.7
        hsv[..., 2] = 0.6
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        bands = analyze_hsl(_frame(rgb))
        presences.append(bands["yellow"]["presence"])

    steps = [abs(b - a) for a, b in zip(presences, presences[1:])]
    assert max(steps) < 0.25, f"yellow presence jumped: {presences}"


# -- look -------------------------------------------------------------------
def test_mood_reflects_the_frame():
    analyzer = EditorialAnalyzer()
    assert analyzer.analyze_rgb(warm_frame())["look"]["mood"] == "warm"
    assert analyzer.analyze_rgb(cool_frame())["look"]["mood"] == "cool"


def test_dark_muted_frame_reads_moody():
    dark = np.full((120, 160, 3), 0.14, np.float32)
    dark[..., 2] *= 1.25
    assert EditorialAnalyzer().analyze_rgb(dark)["look"]["mood"] == "moody"


def test_overall_look_is_a_sentence():
    look = EditorialAnalyzer().analyze_rgb(portrait_frame())["look"]
    assert isinstance(look["overall_look"], str) and "," in look["overall_look"]
    assert 0.0 <= look["confidence"] <= 1.0
