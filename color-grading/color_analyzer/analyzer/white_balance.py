"""White-balance & colour-cast estimation.

Uses the **grey-world** assumption (a well-balanced scene averages to grey) to
estimate per-channel gains and the residual cast.  Correlated colour temperature
(CCT) is derived from the mean chromaticity via McCamy's cubic approximation,
and tint (green<->magenta) from the CIE ``a``/``b`` opponent balance.

Outputs are *descriptive*: they characterise the grade's colour balance, they do
not modify the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import FeatureResult, ImageContext, clamp01, rgb_to_xyz


@dataclass
class WhiteBalanceFeatures(FeatureResult):
    """White-balance analysis results.

    ``color_temperature`` is CCT in Kelvin; ``tint`` is negative for green,
    positive for magenta.  ``*_cast`` values are non-negative magnitudes of the
    respective bias (0 when absent).  ``neutrality_score`` is 1 for a perfectly
    grey-world-neutral image and decreases with cast strength.
    """

    color_temperature: float = 6500.0
    tint: float = 0.0
    gray_world_deviation: float = 0.0
    gain_r: float = 1.0
    gain_g: float = 1.0
    gain_b: float = 1.0
    red_cast: float = 0.0
    green_cast: float = 0.0
    blue_cast: float = 0.0
    magenta_cast: float = 0.0
    yellow_cast: float = 0.0
    neutrality_score: float = 1.0


class WhiteBalanceAnalyzer:
    """Computes :class:`WhiteBalanceFeatures` from an :class:`ImageContext`."""

    def analyze(self, ctx: ImageContext) -> WhiteBalanceFeatures:
        xp = ctx.xp
        mean_rgb = ctx.rgb.reshape(-1, 3).mean(axis=0)
        r, g, b = (float(v) for v in ctx.backend.to_numpy(mean_rgb))
        gray = (r + g + b) / 3.0 + 1e-8

        # Grey-world channel gains that would neutralise the cast.
        gain_r = gray / (r + 1e-8)
        gain_g = gray / (g + 1e-8)
        gain_b = gray / (b + 1e-8)

        # Deviation from grey world: RMS of channel deviations from the mean.
        dev = math.sqrt(((r - gray) ** 2 + (g - gray) ** 2 + (b - gray) ** 2) / 3.0)

        # Directional casts (non-negative magnitudes of each bias direction).
        red_cast = max(0.0, r - (g + b) / 2.0)
        blue_cast = max(0.0, b - (r + g) / 2.0)
        green_cast = max(0.0, g - (r + b) / 2.0)
        # Yellow = red+green vs blue; magenta = red+blue vs green.
        yellow_cast = max(0.0, (r + g) / 2.0 - b)
        magenta_cast = max(0.0, (r + b) / 2.0 - g)

        cct = self._correlated_color_temperature(ctx, mean_rgb)
        tint = self._tint(ctx)

        neutrality = clamp01(1.0 - dev * 4.0)

        return WhiteBalanceFeatures(
            color_temperature=cct,
            tint=tint,
            gray_world_deviation=dev,
            gain_r=gain_r,
            gain_g=gain_g,
            gain_b=gain_b,
            red_cast=red_cast,
            green_cast=green_cast,
            blue_cast=blue_cast,
            magenta_cast=magenta_cast,
            yellow_cast=yellow_cast,
            neutrality_score=neutrality,
        )

    @staticmethod
    def _correlated_color_temperature(ctx: ImageContext, mean_rgb) -> float:
        """CCT (Kelvin) from mean chromaticity via McCamy's approximation.

        ``n = (x - 0.3320)/(0.1858 - y)`` then
        ``CCT = 449 n^3 + 3525 n^2 + 6823.3 n + 5520.33``.
        """
        xp = ctx.xp
        xyz = rgb_to_xyz(xp, mean_rgb.reshape(1, 1, 3)).reshape(3)
        X, Y, Z = (float(v) for v in ctx.backend.to_numpy(xyz))
        s = X + Y + Z + 1e-8
        x = X / s
        y = Y / s
        n = (x - 0.3320) / (0.1858 - y + 1e-8)
        cct = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
        return float(min(max(cct, 1000.0), 40000.0))

    @staticmethod
    def _tint(ctx: ImageContext) -> float:
        """Green<->magenta tint from the mean CIE a* (scaled to ~[-1,1])."""
        mean_a = float(ctx.lab[..., 1].mean())
        return max(-1.0, min(1.0, mean_a / 60.0))
