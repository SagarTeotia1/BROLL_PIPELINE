"""HSL stage: the seven hue bands of a Lumetri / Lightroom HSL panel.

For each of red, orange, yellow, green, cyan, blue and purple the stage reports
how much of the frame occupies that band, plus that band's hue offset,
saturation and luminance — the three sliders the panel offers for each band.
Saturation and luminance are absolute 0-100 readings of those pixels; only hue
is relative, as an offset from the band centre.

Soft membership, not hard bins
------------------------------
Bands overlap by design.  A hard cut at, say, 45 degrees would flip a pixel
between "orange" and "yellow" on a one-degree shift, so the same object could
land in different bands on consecutive frames and the two bands' readings would
jitter in opposite directions.  Membership is therefore a raised cosine centred
on each band that falls to zero at the neighbouring centres, so a colour sitting
between two bands contributes partially to both and moves smoothly.

Every reading is weighted by saturation as well, because the hue of a near-grey
pixel is numerically defined but visually meaningless.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .frame import Frame
from .scales import degrees, ratio, slider

#: Band centres in degrees, in panel order.
BAND_CENTRES: Dict[str, float] = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "cyan": 180.0,
    "blue": 240.0,
    "purple": 285.0,
}

#: Half-width of each band's membership window, in degrees. Wider than the gap
#: to the nearest neighbour so adjacent bands overlap rather than leaving holes.
BAND_HALF_WIDTH: Dict[str, float] = {
    "red": 40.0,
    "orange": 30.0,
    "yellow": 40.0,
    "green": 60.0,
    "cyan": 45.0,
    "blue": 55.0,
    "purple": 50.0,
}

#: Saturation-weighted presence below which a band is reported as absent.
#: Below this the band holds a few scattered pixels, and its hue and luminance
#: are noise that rails the sliders — a frame with no yellow in it should say
#: so, not report a -60 hue offset derived from 0.4% of its pixels.
PRESENCE_FLOOR = 0.02

#: Resolution of the hue -> band-weight lookup table, in steps per degree.
#: Membership depends only on hue, so it is tabulated once at import and read
#: with a single gather per frame instead of evaluating a cosine over the whole
#: frame seven times. Half-degree steps are far finer than any grading decision.
_TABLE_STEPS_PER_DEGREE = 2


def _build_band_table() -> np.ndarray:
    """Tabulate every band's membership against hue, shape ``(steps, bands)``.

    ``0.5 * (1 + cos(pi * d / half_width))`` with the distance ratio clamped to
    1 needs no explicit windowing: the cosine is already zero at the band edge
    and stays zero beyond it.
    """
    steps = 360 * _TABLE_STEPS_PER_DEGREE
    hues = np.arange(steps, dtype=np.float32) / _TABLE_STEPS_PER_DEGREE

    columns = []
    for name, centre in BAND_CENTRES.items():
        distance = np.abs((hues - centre + 180.0) % 360.0 - 180.0)
        ratio_ = np.clip(distance / BAND_HALF_WIDTH[name], 0.0, 1.0)
        columns.append(0.5 * (1.0 + np.cos(np.pi * ratio_)))
    return np.stack(columns, axis=1).astype(np.float32)


def analyze_hsl(frame: Frame) -> Dict[str, Any]:
    """Measure all seven hue bands.

    Saturation and luminance are reported **absolutely**, on 0-100, not as a
    deviation from the frame average. Relative readings couple the bands
    together: raising orange lifts the frame mean, which drags every other
    band's number down even though nothing about those colours changed —
    measured at red falling 35 to 27 when only orange was touched. A control a
    model sets and a renderer applies has to be independent of its neighbours,
    and this is also how an HSL panel behaves.
    """

    # One gather gives every band's membership; the chroma gate is applied once
    # to the whole table rather than per band.
    indices = np.minimum(
        (frame.hue * _TABLE_STEPS_PER_DEGREE).astype(np.int32), _BAND_TABLE.shape[0] - 1
    )
    memberships = _BAND_TABLE[indices] * frame.chroma_gate[:, None]

    bands: Dict[str, Any] = {}
    for index, (name, centre) in enumerate(BAND_CENTRES.items()):
        bands[name] = _analyze_band(
            frame, memberships[:, index], centre, BAND_HALF_WIDTH[name],
        )
    return bands


def _analyze_band(frame: Frame, membership: np.ndarray, centre: float,
                  half_width: float) -> Dict[str, Any]:
    """Presence, hue offset, saturation and luminance for one band."""
    weights = membership * frame.sat

    membership_total = float(membership.sum())
    weight_total = float(weights.sum())
    presence = weight_total / max(frame.pixels, 1)

    if presence < PRESENCE_FLOOR or membership_total < 1e-6:
        return {"presence": ratio(presence), "hue": 0, "saturation": 0, "luminance": 0}

    band_hue = frame.circular_hue_mean(membership, weights=frame.sat)
    band_saturation = float(frame.sat @ membership) / membership_total
    band_luma = float(frame.luma_flat @ weights) / weight_total

    # Hue reported as a signed offset from the band centre, which is what the
    # panel's hue slider actually moves.
    hue_offset = (band_hue - centre + 180.0) % 360.0 - 180.0

    return {
        "presence": ratio(presence),
        "hue": slider(hue_offset / half_width * 100.0),
        "saturation": slider(band_saturation * 100.0, 0.0, 100.0),
        "luminance": slider(band_luma * 100.0, 0.0, 100.0),
    }


#: Built once at import; see :func:`_build_band_table`.
_BAND_TABLE = _build_band_table()
