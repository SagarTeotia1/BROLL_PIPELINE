"""Colour-wheel stage: lift, gamma and gain.

Three wheels, one per tonal zone — lift for shadows, gamma for midtones, gain
for highlights — reported the way Resolve's primaries panel presents them: a
per-channel offset plus a luminance offset, all on -100..100.

Each channel offset is that channel's mean *within the zone*, minus the zone's
own mean across all three channels.  Subtracting the zone mean is what separates
a colour cast from mere brightness: a shadow region that is simply dark has no
wheel offset, while one tinted blue reports a negative red and a positive blue
regardless of how dark it is.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .frame import Frame
from .scales import ratio, slider

#: Zone name -> wheel name, in the order editors expect them.
ZONE_WHEELS = (("shadows", "lift"), ("midtones", "gamma"), ("highlights", "gain"))

#: Channel imbalance, as a fraction of the zone's own level, that reads as full
#: deflection. At 0.5 a +100 means the channel sits 50% above the zone average —
#: a cast so strong it is unmistakable. A tighter span railed the sliders on
#: ordinary tungsten-lit footage, which throws away the difference between
#: "warm" and "extremely warm".
_CHANNEL_SPAN = 0.5

#: Weighted mean luminance of each soft zone for an evenly distributed frame,
#: and the zone's own weighted standard deviation. Both are computed from the
#: zone weighting functions rather than guessed, so a wheel's luma reads 0 on an
#: evenly exposed frame and +/-100 when the zone sits a full standard deviation
#: away from where it belongs.
_ZONE_LUMA_CENTRE = {"shadows": 0.186, "midtones": 0.500, "highlights": 0.814}
_ZONE_LUMA_SPAN = {"shadows": 0.119, "midtones": 0.124, "highlights": 0.119}


def analyze_wheels(frame: Frame) -> Dict[str, Any]:
    """Measure the lift/gamma/gain colour balance of the three tonal zones."""
    wheels: Dict[str, Any] = {}
    for zone, wheel in ZONE_WHEELS:
        wheels[wheel] = _analyze_zone(frame, zone)
    return wheels


def _analyze_zone(frame: Frame, zone: str) -> Dict[str, Any]:
    """Per-channel and luminance offsets for one tonal zone."""
    mask = frame.zone_mask(zone)
    coverage = float(mask.sum()) / max(frame.pixels, 1)

    if coverage <= 0.0:
        # An empty zone has no colour balance; report neutral rather than
        # dividing by zero and emitting noise.
        return {"red": 0, "green": 0, "blue": 0, "luma": 0, "coverage": 0.0}

    r, g, b = frame.masked_rgb_mean(mask)
    level = max((r + g + b) / 3.0, 1e-6)

    zone_luma = frame.masked_mean(frame.luma_flat, mask)
    centre = _ZONE_LUMA_CENTRE[zone]
    span = _ZONE_LUMA_SPAN[zone]

    return {
        "red": slider((r - level) / level / _CHANNEL_SPAN * 100.0),
        "green": slider((g - level) / level / _CHANNEL_SPAN * 100.0),
        "blue": slider((b - level) / level / _CHANNEL_SPAN * 100.0),
        "luma": slider((zone_luma - centre) / span * 100.0),
        "coverage": ratio(coverage),
    }
