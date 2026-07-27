"""Split-toning stage: the shadow and highlight tints, and the balance between them.

A split tone is two hues pulling in opposite directions at opposite ends of the
tonal range — the teal-shadow / orange-highlight look being the obvious case.
The stage reports each end's hue and strength, plus the tonal point the split
pivots around, which is what a split-toning panel's balance slider moves.

``separation`` distinguishes a deliberate split from a plain global cast: if
both ends carry the same hue the frame is simply tinted, and an editor should
reach for white balance instead.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .frame import Frame
from .scales import clamp, degrees, ratio, slider

#: Saturation that reads as a fully committed tint on the strength slider.
_FULL_TINT_SATURATION = 0.40


def analyze_split_toning(frame: Frame) -> Dict[str, Any]:
    """Measure the shadow tint, the highlight tint and their balance."""
    shadows = _tint(frame, "shadows")
    highlights = _tint(frame, "highlights")

    # Circular distance between the two hues; the wrap puts it in [0, 180].
    separation = abs((shadows["hue"] - highlights["hue"] + 180.0) % 360.0 - 180.0)

    return {
        "shadows": shadows,
        "highlights": highlights,
        "separation": int(round(separation)),
        "balance": _balance(frame),
        "strength": _strength(shadows, highlights, separation),
    }


def _tint(frame: Frame, zone: str) -> Dict[str, Any]:
    """Hue and strength of one end of the split."""
    mask = frame.zone_mask(zone)
    if float(mask.sum()) < 1.0:
        return {"hue": 0, "saturation": 0}
    return {
        "hue": degrees(frame.circular_hue_mean(mask)),
        "saturation": slider(
            frame.masked_mean(frame.sat, mask) / _FULL_TINT_SATURATION * 100.0, 0.0, 100.0
        ),
    }


def _balance(frame: Frame) -> int:
    """Where the tonal mass sits, as a -100 (shadow-heavy) .. 100 slider.

    A split-toning balance control shifts which tones count as shadows; the
    median luminance is the natural measurement of where that pivot currently
    falls.
    """
    (median,) = frame.percentiles(50.0)
    return slider((median - 0.5) / 0.5 * 100.0)


def _strength(shadows: Dict[str, Any], highlights: Dict[str, Any], separation: float) -> int:
    """How strongly the frame is split-toned, 0-100.

    Both ends must be tinted *and* the hues must differ.  Taking the weaker of
    the two ends means a single tinted zone does not read as a split, and
    scaling by hue separation means a uniform cast reads as zero however
    saturated it is.
    """
    both_ends = min(shadows.get("saturation", 0), highlights.get("saturation", 0))
    return slider(both_ends * clamp(separation / 120.0, 0.0, 1.0), 0.0, 100.0)
