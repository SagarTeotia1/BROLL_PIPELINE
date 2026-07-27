"""Determinism, and stability across similar frames.

Two different promises:

* **Determinism** — the same input always gives byte-identical output. Easy, and
  mostly a matter of not seeding anything randomly.
* **Stability** — *similar* inputs give near-identical output. This is the one
  that matters in production. If consecutive frames of one shot produce readings
  that wobble, an automated grade driven by them flickers, and no amount of
  determinism helps.

The stability tests perturb a frame the way a camera does — sensor noise, a
sliver of reframing, a small resolution change — and assert the readings barely
move.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from color_analyzer.editorial import EditorialAnalyzer

from .conftest import (
    add_noise,
    clustered_frame,
    dark_scene_frame,
    flat_grey,
    portrait_frame,
    split_toned_frame,
)

#: Tolerance for a -100..100 slider.
SLIDER_TOLERANCE = 4

#: Saturation-derived readings are inherently the noisiest thing here, and no
#: amount of smoothing fixes it: HSV saturation is ``(max - min) / max`` over
#: three channels, so independent per-channel noise pushes ``max`` up and ``min``
#: down and biases the result upward. The effect scales with noise and with
#: darkness (the divisor shrinks), so a dark frame is the worst case.
#:
#: Measured on real footage: ~2 units at +/-1/255 of noise, ~9 at +/-2.5/255,
#: ~16 at +/-5/255. The alternative — blurring before measurement — would trade
#: this documented sensitivity for an undocumented bias in the readings
#: themselves, which is the worse deal.
SATURATION_TOLERANCE = 10

#: Fields that do not live on the slider scale need their own tolerance,
#: otherwise one number is being judged against a limit meant for another.
FIELD_TOLERANCES = {
    "saturation": SATURATION_TOLERANCE,
    "vibrance": SATURATION_TOLERANCE,
    "coverage": 0.03,      # 0..1 fractions
    "presence": 0.03,
    "neutral_coverage": 0.05,
    "confidence": 0.15,
    "black_point": 0.02,
    "white_point": 0.02,
    "clipped_shadows": 0.02,
    "clipped_highlights": 0.02,
    "mean_saturation": 0.03,
    "exposure": 0.15,      # stops
    "gamma": 0.15,
    "temperature": 300.0,  # Kelvin
    "hue": 8.0,            # degrees, or a hue-offset slider
    "rgb": 10.0,           # 0..255 colour chip
    "width": 0.0,
    "height": 0.0,
}


#: Fields holding an absolute angle in degrees, which must be compared the long
#: way round. ``split_toning.shadows.hue`` legitimately reads 357 on one frame
#: and 2 on the next — those are five degrees apart, not 355.
_CIRCULAR_FIELDS = ("split_toning.shadows.hue", "split_toning.highlights.hue",
                    "skin_tone.hue")


def _leaf(path: str) -> str:
    leaf = path.rsplit(".", 1)[-1]
    if leaf.isdigit():  # e.g. palette.0.rgb.2 -> judge as an rgb component
        leaf = path.rsplit(".", 2)[-2]
    return leaf


def _tolerance(path: str) -> float:
    """Tolerance for a flattened field path, by its leaf name."""
    return FIELD_TOLERANCES.get(_leaf(path), SLIDER_TOLERANCE)


def _difference(path: str, a: float, b: float) -> float:
    """Absolute difference, measured circularly for absolute hue angles.

    The HSL bands' ``hue`` is a signed offset from a band centre, not an
    absolute angle, so only the fields listed in :data:`_CIRCULAR_FIELDS` wrap.
    """
    if path in _CIRCULAR_FIELDS:
        return abs((a - b + 180.0) % 360.0 - 180.0)
    return abs(a - b)


def _numbers(document: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten every number in a document to ``{path: value}``."""
    out: Dict[str, float] = {}
    if isinstance(document, dict):
        items = document.items()
    elif isinstance(document, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(document))
    else:
        return out

    for key, value in items:
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[path] = float(value)
        else:
            out.update(_numbers(value, path))
    return out


def _drop_timing(document: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the wall-clock field, which is expected to differ between runs."""
    trimmed = {k: v for k, v in document.items() if k != "meta"}
    trimmed["meta"] = {k: v for k, v in document["meta"].items() if k != "elapsed_ms"}
    return trimmed


# -- determinism ------------------------------------------------------------
def test_same_frame_gives_identical_output():
    frame = portrait_frame()
    analyzer = EditorialAnalyzer()
    first = _drop_timing(analyzer.analyze_rgb(frame))
    second = _drop_timing(analyzer.analyze_rgb(frame))
    assert first == second


def test_fresh_analyzer_gives_identical_output():
    """No state accumulates between analyzer instances."""
    frame = portrait_frame()
    first = _drop_timing(EditorialAnalyzer().analyze_rgb(frame))
    second = _drop_timing(EditorialAnalyzer().analyze_rgb(frame))
    assert first == second


def test_palette_order_is_reproducible():
    frame = split_toned_frame()
    analyzer = EditorialAnalyzer()
    a = [s["hex"] for s in analyzer.analyze_rgb(frame)["palette"]]
    b = [s["hex"] for s in analyzer.analyze_rgb(frame)["palette"]]
    assert a == b


# -- stability across similar frames ----------------------------------------
@pytest.mark.parametrize("noise,slack", [
    (0.004, 1.0),   # +/-1/255: clean footage
    (0.01, 1.0),    # +/-2.5/255: a normal high-ISO frame
    (0.02, 2.0),    # +/-5/255: heavy noise, allowed twice the drift
])
@pytest.mark.parametrize("fixture", [clustered_frame, dark_scene_frame],
                         ids=["clustered", "dark-scene"])
def test_readings_survive_sensor_noise(noise, slack, fixture):
    """Noise of the magnitude a real sensor produces must not move the readings.

    ``slack`` widens the bounds for the heaviest case only. Drift is roughly
    linear in noise amplitude, so holding a single tolerance across a five-fold
    range would either be vacuous at the low end or unmeetable at the high one.
    """
    base = fixture()
    analyzer = EditorialAnalyzer()

    reference = _numbers(_drop_timing(analyzer.analyze_rgb(base)))
    for seed in (7, 8, 9):
        noisy = _numbers(_drop_timing(analyzer.analyze_rgb(add_noise(base, noise, seed=seed))))
        for key, expected in reference.items():
            assert key in noisy, key
            assert _difference(key, noisy[key], expected) <= _tolerance(key) * slack, (
                f"{key} moved {expected} -> {noisy[key]} under {noise} noise"
            )


def test_readings_survive_a_small_reframe():
    """A few pixels of pan must not change what the frame is made of."""
    base = portrait_frame()
    shifted = np.roll(base, shift=4, axis=1)
    analyzer = EditorialAnalyzer()

    reference = _numbers(_drop_timing(analyzer.analyze_rgb(base)))
    moved = _numbers(_drop_timing(analyzer.analyze_rgb(shifted)))

    for key, expected in reference.items():
        assert _difference(key, moved[key], expected) <= _tolerance(key), key


def test_readings_survive_a_resolution_change():
    """The same shot delivered at a different size reads the same.

    Both are capped to the analysis resolution first, so this checks the cap is
    doing its job rather than the analysis being resolution-invariant by luck.
    """
    import cv2

    base = portrait_frame()
    larger = cv2.resize(base, (base.shape[1] * 2, base.shape[0] * 2),
                        interpolation=cv2.INTER_LINEAR)
    analyzer = EditorialAnalyzer(max_side=256)

    reference = _numbers(_drop_timing(analyzer.analyze_rgb(base)))
    scaled = _numbers(_drop_timing(analyzer.analyze_rgb(larger)))

    for key, expected in reference.items():
        if key.startswith("meta."):
            continue
        assert _difference(key, scaled[key], expected) <= _tolerance(key), key


def test_palette_is_stable_across_similar_frames():
    """The dominant swatches of a shot must not churn between frames.

    This is what the deterministic grid seeding exists for: random k-means++
    initialisation is reproducible per frame but can land on a different local
    minimum for the next frame of the same shot, so the palette reorders and
    anything downstream sees it flicker.
    """
    base = clustered_frame()
    analyzer = EditorialAnalyzer()

    reference = analyzer.analyze_rgb(base)["palette"]
    for seed in range(4):
        similar = analyzer.analyze_rgb(add_noise(base, 0.012, seed=seed))["palette"]
        assert len(similar) == len(reference)
        for expected, actual in zip(reference, similar):
            distance = max(
                abs(a - b) for a, b in zip(expected["rgb"], actual["rgb"])
            )
            assert distance <= 8, f"swatch moved {expected['hex']} -> {actual['hex']}"
            assert abs(expected["coverage"] - actual["coverage"]) <= 0.05


def test_palette_on_a_smooth_gradient_is_documented_as_looser():
    """A continuous gradient has no clusters, so its swatches can drift.

    Not a defect to fix but a property of the problem: any partition of a
    continuum is arbitrary, and a small perturbation legitimately moves the
    boundaries. Pinned here so the limit is measured rather than assumed, and so
    a regression that made it much worse would still be caught.
    """
    base = portrait_frame()  # a smooth luminance ramp behind the subject
    analyzer = EditorialAnalyzer()

    reference = analyzer.analyze_rgb(base)["palette"]
    worst = 0
    for seed in range(4):
        similar = analyzer.analyze_rgb(add_noise(base, 0.012, seed=seed))["palette"]
        for expected, actual in zip(reference, similar):
            worst = max(worst, max(abs(a - b) for a, b in zip(expected["rgb"], actual["rgb"])))

    assert worst <= 45, f"gradient swatch drift regressed to {worst} levels"


def test_flat_frame_is_exactly_neutral():
    """A degenerate input must not produce noise in the readings."""
    document = EditorialAnalyzer().analyze_rgb(flat_grey(0.5))

    assert document["tone"]["contrast"] <= -95      # no tonal spread at all
    assert abs(document["color"]["saturation"]) >= 95  # no colour at all
    assert document["color"]["colorfulness"] == 0
    assert document["split_toning"]["strength"] == 0
    assert document["skin_tone"]["detected"] is False
    for band in document["hsl"].values():
        assert band["presence"] == 0.0
