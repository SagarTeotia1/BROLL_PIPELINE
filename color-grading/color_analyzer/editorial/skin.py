"""Skin-tone stage.

Skin is the one subject a grade must never break, so it gets its own reading:
how much of the frame is skin, where it sits on the hue/saturation/luminance
axes, and whether it currently leans warm or cool against the reference skin
line.

Detection uses HSV bounds intersected with an RGB ordering rule.  The classic
approach uses YCrCb, which this engine deliberately does not carry; the pair of
rules below covers the same chrominance region using planes already computed.
Requiring both keeps warm-coloured background — wood, sand, brick — from being
counted as skin, which a hue window alone would admit.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .frame import Frame
from .scales import clamp, degrees, label, ratio, slider

#: Hue window for skin, in degrees. Skin spans red-orange through yellow.
HUE_LOW, HUE_HIGH = 0.0, 50.0

#: Saturation window. Below this is grey, above it is a saturated object.
SATURATION_LOW, SATURATION_HIGH = 0.15, 0.70

#: Value window; excludes crushed shadow and blown highlight.
VALUE_LOW, VALUE_HIGH = 0.20, 0.98

#: Minimum red-over-green margin. Skin is always redder than it is green.
RED_GREEN_MARGIN = 0.035

#: Frame coverage below which a skin reading is not worth acting on.
MIN_COVERAGE = 0.005

#: Reference hue of neutral skin. Below it reads warm/red, above it yellow/cool.
REFERENCE_SKIN_HUE = 25.0


def analyze_skin(frame: Frame) -> Dict[str, Any]:
    """Measure the frame's skin tone, if any is present."""
    mask = skin_mask(frame)
    coverage = float(mask.sum()) / max(frame.pixels, 1)

    if coverage < MIN_COVERAGE:
        return {
            "detected": False,
            "coverage": ratio(coverage),
            "hue": 0,
            "saturation": 0,
            "luminance": 0,
            "tone": "none",
            "warmth": 0,
        }

    hue = frame.circular_hue_mean(mask)
    saturation = frame.masked_mean(frame.sat, mask)
    luminance = frame.masked_mean(frame.luma_flat, mask)

    return {
        "detected": True,
        "coverage": ratio(coverage),
        "hue": degrees(hue),
        "saturation": slider(saturation * 100.0, 0.0, 100.0),
        "luminance": slider(luminance * 100.0, 0.0, 100.0),
        "tone": _describe_tone(luminance, hue),
        # Negative reads yellow/cool for skin, positive reads red/warm.
        "warmth": slider((REFERENCE_SKIN_HUE - hue) / 20.0 * 100.0),
    }


def skin_mask(frame: Frame) -> np.ndarray:
    """Float32 0/1 mask of probable skin pixels, flattened.

    Public because the renderer needs the same mask the analyzer measured with:
    protecting skin during a grade only works if "skin" means the same pixels in
    both directions.
    """
    hue_ok = (frame.hue >= HUE_LOW) & (frame.hue <= HUE_HIGH)
    sat_ok = (frame.sat >= SATURATION_LOW) & (frame.sat <= SATURATION_HIGH)
    val_ok = (frame.val >= VALUE_LOW) & (frame.val <= VALUE_HIGH)

    red, green, blue = (frame.rgb_flat[:, c] for c in range(3))
    ordering_ok = (red > green) & (green > blue) & ((red - green) > RED_GREEN_MARGIN)

    return (hue_ok & sat_ok & val_ok & ordering_ok).astype(np.float32)


def _describe_tone(luminance: float, hue: float) -> str:
    """Depth and warmth of the detected skin, as one phrase."""
    depth = label(
        luminance,
        [(0.28, "deep"), (0.45, "medium-deep"), (0.62, "medium"), (0.78, "light")],
        "fair",
    )
    warmth = label(hue, [(18.0, "ruddy"), (30.0, "warm"), (40.0, "neutral")], "sallow")
    return f"{depth} {warmth}"
