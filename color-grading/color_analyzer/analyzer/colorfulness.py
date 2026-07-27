"""Colourfulness metrics.

Implements the Hasler & Süsstrunk (2003) "Measuring colourfulness in natural
images" metric, which operates on the opponent channels

    rg = R - G,   yb = 0.5*(R + G) - B

and combines the standard deviation and mean of their magnitude::

    C = sqrt(std_rg^2 + std_yb^2) + 0.3 * sqrt(mean_rg^2 + mean_yb^2)

We also report opponent-channel variance, average CIE chroma, and a normalised
"colour richness" score in ``[0,1]`` for convenient thresholding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import FeatureResult, ImageContext, clamp01


@dataclass
class ColorfulnessFeatures(FeatureResult):
    """Colourfulness analysis results."""

    hasler_susstrunk: float = 0.0
    opponent_variance: float = 0.0
    average_chroma: float = 0.0
    color_richness: float = 0.0  # normalised [0,1]


class ColorfulnessAnalyzer:
    """Computes :class:`ColorfulnessFeatures` from an :class:`ImageContext`."""

    def analyze(self, ctx: ImageContext) -> ColorfulnessFeatures:
        xp = ctx.xp
        rgb = ctx.rgb.reshape(-1, 3) * 255.0  # metric is defined on 0-255
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]

        rg = r - g
        yb = 0.5 * (r + g) - b

        std_rg = float(rg.std())
        std_yb = float(yb.std())
        mean_rg = float(rg.mean())
        mean_yb = float(yb.mean())

        std_root = math.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_root = math.sqrt(mean_rg ** 2 + mean_yb ** 2)
        colourfulness = std_root + 0.3 * mean_root

        opponent_var = float(rg.var() + yb.var())

        # Average CIE chroma C* = sqrt(a*^2 + b*^2).
        lab = ctx.lab.reshape(-1, 3)
        chroma = float(xp.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2).mean())

        # Empirically, Hasler-Süsstrunk ~ >60 reads as "very colourful"; map to
        # a normalised richness score for downstream thresholding.
        richness = clamp01(colourfulness / 110.0)

        return ColorfulnessFeatures(
            hasler_susstrunk=colourfulness,
            opponent_variance=opponent_var,
            average_chroma=chroma,
            color_richness=richness,
        )
