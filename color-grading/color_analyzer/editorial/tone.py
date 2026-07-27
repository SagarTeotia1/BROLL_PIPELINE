"""Tone stage: exposure, brightness, contrast, gamma, black and white point.

Everything here is read off the luminance distribution, using percentiles rather
than means so that a small blown highlight or a crushed corner does not move the
reading.  All five outputs map onto controls that exist in every grading tool.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .frame import Frame
from .scales import centred, label, ratio, slider, stops

#: sRGB code value of an 18% mid-grey card. Exposure is reported relative to it.
MID_GREY = 0.46

#: Percentiles defining the usable tonal range. 0.5/99.5 rather than 0/100 so a
#: handful of dead or hot pixels cannot define the black and white point.
BLACK_PERCENTILE = 0.5
WHITE_PERCENTILE = 99.5

#: p95-p5 luminance spread of a normally graded frame; the contrast slider's zero.
NEUTRAL_SPREAD = 0.55


def analyze_tone(frame: Frame) -> Dict[str, Any]:
    """Measure the frame's tonal state.

    Returns
    -------
    dict
        ``exposure`` in stops from mid-grey, ``brightness``/``contrast`` on
        -100..100, ``gamma`` as an estimated transfer exponent, and
        ``black_point``/``white_point`` as signal levels in ``[0,1]``.
    """
    black, p5, median, p95, white = frame.percentiles(
        BLACK_PERCENTILE, 5.0, 50.0, 95.0, WHITE_PERCENTILE
    )
    fit_values = frame.percentiles(*(p * 100.0 for p in _GAMMA_FIT_POINTS))
    mean = float(frame.luma_flat.mean())

    return {
        "exposure": stops(np.log2((median + 1e-4) / MID_GREY)),
        "brightness": centred(mean, neutral=0.5, span=0.5),
        "contrast": centred(p95 - p5, neutral=NEUTRAL_SPREAD, span=0.45),
        "gamma": _estimate_gamma(fit_values, p5, p95),
        "black_point": ratio(black),
        "white_point": ratio(white),
        "clipped_shadows": ratio(float((frame.luma_flat <= 0.004).mean())),
        "clipped_highlights": ratio(float((frame.luma_flat >= 0.996).mean())),
    }


#: Points across the tonal range that gamma is fitted through. Kept away from
#: the tails, where log() is steep and noise moves the percentiles most.
_GAMMA_FIT_POINTS = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)


def _estimate_gamma(values: tuple, low: float, high: float) -> float:
    """Transfer gamma, least-squares fitted across the whole tonal range.

    Under a power law the ``p``-th percentile of an evenly distributed input
    lands at ``p ** gamma`` once the range is normalised.  Taking logs makes
    that linear, ``log(q_p) = gamma * log(p)``, so gamma is the slope of a
    regression through the origin::

        gamma = sum(log(p) * log(q_p)) / sum(log(p)^2)

    **Why a fit and not a single point.**  Reading gamma off the median alone
    assumes the distribution is unimodal.  Real frames often are not — a dark
    interior with a bright window, a silhouette against sky — and on those the
    median sits in the empty valley between two modes, where it jumps as soon as
    noise nudges it across.  Eight points spread across the range means no
    single percentile can carry the answer, and the estimate stays put from
    frame to frame.

    ``low``/``high`` are the 5th and 95th percentiles, deliberately **not** the
    reported black and white points.  Those sit at 0.5/99.5 to describe clipping
    headroom, which puts them in the tails where noise moves them most.

    This describes the curve in front of you, not a recovered camera gamma: it
    assumes the ungraded distribution was roughly even, the same assumption
    every auto-levels tool makes.  Strongly clustered content (a graphic with a
    handful of flat tones) has no power law to find, and the fit will return
    something bland rather than something meaningful.

    **Known bias.** Normalising by the tonal range is what makes the reading
    independent of exposure, and it also compresses the estimate toward 1.0:
    a synthetic ramp raised to 2.2 measures about 2.1, and one raised to 0.45
    measures about 0.7.  The reading is monotonic — more gamma always reads as
    more — so it ranks and steers correctly, but do not treat it as an exact
    exponent.  The alternative, skipping normalisation, buys accuracy at the
    cost of confounding gamma with exposure, which is the worse failure for a
    control a model is going to adjust.
    """
    span = max(float(high) - float(low), 1e-3)

    log_points = np.log(np.asarray(_GAMMA_FIT_POINTS, dtype=np.float64))
    normalised = np.clip(
        (np.asarray(values, dtype=np.float64) - float(low)) / span, 0.02, 0.98
    )
    log_values = np.log(normalised)

    gamma = float((log_points * log_values).sum() / (log_points ** 2).sum())
    return round(float(np.clip(gamma, 0.3, 3.0)), 2)


def describe_brightness(tone: Dict[str, Any]) -> str:
    """Word for the frame's overall brightness."""
    return label(
        tone["brightness"],
        [(-45, "very dark"), (-15, "dark"), (15, "balanced"), (45, "bright")],
        "very bright",
    )


def describe_contrast(tone: Dict[str, Any]) -> str:
    """Word for the frame's contrast."""
    return label(
        tone["contrast"],
        [(-45, "very flat"), (-15, "flat"), (15, "normal"), (45, "punchy")],
        "very punchy",
    )
