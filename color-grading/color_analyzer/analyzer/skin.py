"""Skin-tone analysis.

Skin is detected with a robust chroma rule in YCrCb (skin clusters tightly in
``133<=Cr<=173``, ``77<=Cb<=127`` on the 8-bit scale) intersected with a plausible
hue/brightness range to reject false positives.  From the detected pixels we
describe the *grade* applied to skin:

* **skin hue / saturation / exposure** — where skin sits after grading.
* **skin consistency** — inverse of tonal spread across skin pixels; a
  consistent grade keeps skin uniform.
* **skin naturalness** — closeness of skin hue & saturation to the natural
  skin gamut (warm-orange hue, moderate saturation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import FeatureResult, ImageContext, clamp01

_CR_LO, _CR_HI = 133.0 / 255.0, 173.0 / 255.0
_CB_LO, _CB_HI = 77.0 / 255.0, 127.0 / 255.0
# Natural skin sits around a warm hue (~25 deg) with moderate saturation.
_NATURAL_HUE = 25.0
_NATURAL_SAT = 0.4


@dataclass
class SkinFeatures(FeatureResult):
    """Skin-tone analysis results."""

    skin_percentage: float = 0.0
    skin_hue: float = 0.0
    skin_saturation: float = 0.0
    skin_exposure: float = 0.0  # mean luminance of skin
    skin_consistency: float = 0.0  # [0,1], 1 = very uniform
    skin_naturalness: float = 0.0  # [0,1], 1 = natural skin tone
    detected: bool = False


class SkinAnalyzer:
    """Computes :class:`SkinFeatures` from an :class:`ImageContext`."""

    def __init__(self, min_fraction: float = 0.005) -> None:
        self.min_fraction = min_fraction

    def analyze(self, ctx: ImageContext) -> SkinFeatures:
        xp = ctx.xp
        ycrcb = ctx.ycrcb.reshape(-1, 3)
        hsv = ctx.hsv_flat
        lum = ctx.gray.reshape(-1)

        cr = ycrcb[:, 1]
        cb = ycrcb[:, 2]
        hue = hsv[:, 0]
        sat = hsv[:, 1]

        # Chroma-box rule AND plausible warm hue AND non-black/non-blown luma.
        mask = (
            (cr >= _CR_LO)
            & (cr <= _CR_HI)
            & (cb >= _CB_LO)
            & (cb <= _CB_HI)
            & (((hue <= 50.0) | (hue >= 340.0)))
            & (sat > 0.1)
            & (lum > 0.15)
            & (lum < 0.98)
        )
        count = float(mask.sum())
        fraction = count / float(mask.size)
        if fraction < self.min_fraction:
            return SkinFeatures(skin_percentage=fraction, detected=False)

        skin_hue = hue[mask]
        skin_sat = sat[mask]
        skin_lum = lum[mask]

        # Circular mean & variance of skin hue.
        theta = skin_hue * (math.pi / 180.0)
        cos_m = float(xp.cos(theta).mean())
        sin_m = float(xp.sin(theta).mean())
        mean_hue = math.degrees(math.atan2(sin_m, cos_m)) % 360.0
        hue_concentration = math.sqrt(cos_m * cos_m + sin_m * sin_m)  # [0,1]

        mean_sat = float(skin_sat.mean())
        mean_exp = float(skin_lum.mean())
        lum_std = float(skin_lum.std())

        # Consistency: high hue concentration and low luminance spread.
        consistency = clamp01(0.6 * hue_concentration + 0.4 * (1.0 - 4.0 * lum_std))

        # Naturalness: Gaussian proximity of hue & saturation to natural values.
        hue_dist = abs((mean_hue - _NATURAL_HUE + 180.0) % 360.0 - 180.0)
        hue_term = math.exp(-(hue_dist ** 2) / (2 * 20.0 ** 2))
        sat_term = math.exp(-((mean_sat - _NATURAL_SAT) ** 2) / (2 * 0.25 ** 2))
        naturalness = clamp01(0.6 * hue_term + 0.4 * sat_term)

        return SkinFeatures(
            skin_percentage=fraction,
            skin_hue=mean_hue,
            skin_saturation=mean_sat,
            skin_exposure=mean_exp,
            skin_consistency=consistency,
            skin_naturalness=naturalness,
            detected=True,
        )
