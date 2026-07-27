"""White-balance stage: colour temperature and tint.

Temperature is estimated from the red/blue ratio of the frame's **near-neutral**
pixels, interpolated against a fixed daylight-locus table.  Tint is the
green-magenta residual on the same population.

Why not McCamy's CCT formula
----------------------------
The textbook route — average the whole frame, convert to chromaticity, apply
McCamy's cubic — assumes the average of a scene is grey.  It is not: a forest
averages green and reports a magenta cast that is not there, and a sunset
reports a temperature no camera ever produced.  Restricting the estimate to
low-saturation pixels measures the *light* rather than the *subject*, and the
red/blue ratio is monotonic in colour temperature over the range that matters,
so a small calibration table beats a cubic fitted to a different problem.

``confidence`` reports how much of the frame was neutral enough to use.  A frame
with no neutral content gets a reading and a low confidence, never a silent
guess.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .frame import NEUTRAL_SATURATION, Frame
from .scales import clamp, kelvin, label, ratio, slider

#: Red/blue ratio of a neutral surface under each colour temperature, sRGB
#: primaries, normalised to 1.0 at the D65 white point. Monotonically decreasing:
#: cooler light means relatively more blue.
_TEMPERATURE_TABLE = np.array(
    [
        # (Kelvin, R/B ratio)
        (2000.0, 3.30),
        (2500.0, 2.42),
        (3000.0, 1.92),
        (3500.0, 1.60),
        (4000.0, 1.39),
        (4500.0, 1.24),
        (5000.0, 1.13),
        (5500.0, 1.05),
        (6500.0, 1.00),
        (7500.0, 0.92),
        (9000.0, 0.84),
        (11000.0, 0.77),
        (12000.0, 0.74),
    ],
    dtype=np.float64,
)

#: Fraction of the frame that must be near-neutral for a confident reading.
_CONFIDENT_COVERAGE = 0.25


def analyze_white_balance(frame: Frame) -> Dict[str, Any]:
    """Measure colour temperature (Kelvin) and green-magenta tint."""
    mask = frame.neutral_mask
    r, g, b = frame.masked_rgb_mean(mask)

    temperature = _ratio_to_kelvin(r / max(b, 1e-6))

    # Tint: green vs the red/blue average, normalised by overall level so it does
    # not scale with exposure. Positive is magenta, matching every editor's UI.
    level = max((r + g + b) / 3.0, 1e-6)
    tint = ((r + b) / 2.0 - g) / level

    coverage = frame.neutral_coverage
    return {
        "temperature": kelvin(temperature),
        "tint": slider(tint * 250.0, -100.0, 100.0),
        "neutral_coverage": ratio(coverage),
        "confidence": _confidence(frame, coverage),
    }


def _confidence(frame: Frame, coverage: float) -> float:
    """How much to trust this reading, from the size *and* purity of the neutral set.

    Coverage alone is not enough, and being wrong about that is this estimator's
    real failure mode. Under a strong global cast there may be no truly grey
    pixels at all; the mask then admits whatever is *least* saturated, which is
    still tinted, and the estimator systematically understates the cast while
    reporting plenty of coverage.

    Purity catches that: if the selected pixels average near-zero saturation
    they are real greys, and if they sit up against the selection threshold they
    are merely the least-coloured part of a coloured frame.
    """
    size = clamp(coverage / _CONFIDENT_COVERAGE, 0.0, 1.0)

    mask = frame.neutral_mask
    neutral_saturation = frame.masked_mean(frame.sat, mask)
    purity = clamp(1.0 - neutral_saturation / NEUTRAL_SATURATION, 0.0, 1.0)

    return ratio(size * (0.4 + 0.6 * purity), digits=2)


def kelvin_to_ratio(kelvin_value: float) -> float:
    """Red/blue ratio a neutral surface shows at a given colour temperature.

    The inverse of :func:`_ratio_to_kelvin`, and the reason the renderer can hit
    a temperature target instead of approximating it: the gain needed to move
    from one temperature to another is simply the ratio of their two entries in
    this table, so the correction is derived from the same curve the
    measurement came from rather than from a tuned constant.
    """
    kelvins = _TEMPERATURE_TABLE[:, 0]
    ratios = _TEMPERATURE_TABLE[:, 1]
    return float(np.interp(clamp(kelvin_value, kelvins[0], kelvins[-1]), kelvins, ratios))


def _ratio_to_kelvin(rb_ratio: float) -> float:
    """Invert the daylight-locus table: red/blue ratio -> Kelvin.

    The table is stored cool-to-warm in ratio order, so it is reversed before
    interpolating (``np.interp`` requires an increasing x).  Ratios outside the
    table clamp to its ends rather than extrapolating into nonsense.
    """
    kelvins = _TEMPERATURE_TABLE[:, 0][::-1]
    ratios = _TEMPERATURE_TABLE[:, 1][::-1]
    return float(np.interp(clamp(rb_ratio, ratios[0], ratios[-1]), ratios, kelvins))


def describe_temperature(white_balance: Dict[str, Any]) -> str:
    """Word for the frame's colour temperature."""
    return label(
        white_balance["temperature"],
        [(3500, "very warm"), (4800, "warm"), (5800, "neutral-warm"),
         (6800, "neutral"), (8000, "cool")],
        "very cool",
    )
