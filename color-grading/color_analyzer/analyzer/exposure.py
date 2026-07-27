"""Exposure feature extraction.

Partitions the luminance range into shadows / midtones / highlights and measures
how the tonal mass is distributed, plus clipping at both ends and an overall
exposure-quality score.  ``gamma`` is estimated by matching the image's median
luminance to a mid-grey target: assuming an ``out = in^gamma`` transfer,
``gamma = log(median)/log(0.5)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import FeatureResult, ImageContext, clamp01

# Tonal-zone thresholds on normalised luminance.
_SHADOW_HI = 0.25
_HIGHLIGHT_LO = 0.75
_CLIP_LO = 0.02
_CLIP_HI = 0.98


@dataclass
class ExposureFeatures(FeatureResult):
    """Exposure analysis results (fractions in ``[0,1]``)."""

    mean_brightness: float = 0.0
    median_brightness: float = 0.0
    shadow_percentage: float = 0.0
    midtone_percentage: float = 0.0
    highlight_percentage: float = 0.0
    shadow_clipping: float = 0.0
    highlight_clipping: float = 0.0
    exposure_quality: float = 0.0  # [0,1], 1 = well exposed
    gamma_estimate: float = 1.0


class ExposureAnalyzer:
    """Computes :class:`ExposureFeatures` from an :class:`ImageContext`."""

    def analyze(self, ctx: ImageContext) -> ExposureFeatures:
        xp = ctx.xp
        lum = ctx.gray.reshape(-1)
        n = float(lum.size)

        mean_b = float(lum.mean())
        median_b = float(xp.median(lum))

        # Tonal-zone fractions via boolean-mask reductions (vectorised).
        shadow = float((lum < _SHADOW_HI).sum()) / n
        highlight = float((lum > _HIGHLIGHT_LO).sum()) / n
        midtone = max(0.0, 1.0 - shadow - highlight)

        shadow_clip = float((lum < _CLIP_LO).sum()) / n
        highlight_clip = float((lum > _CLIP_HI).sum()) / n

        # Gamma from median matching to mid-grey (0.5). Clamp to a sane range.
        med = min(max(median_b, 1e-3), 1.0 - 1e-3)
        gamma = math.log(med) / math.log(0.5)
        gamma = min(max(gamma, 0.2), 5.0)

        quality = self._quality(mean_b, shadow_clip, highlight_clip, midtone)

        return ExposureFeatures(
            mean_brightness=mean_b,
            median_brightness=median_b,
            shadow_percentage=shadow,
            midtone_percentage=midtone,
            highlight_percentage=highlight,
            shadow_clipping=shadow_clip,
            highlight_clipping=highlight_clip,
            exposure_quality=quality,
            gamma_estimate=gamma,
        )

    @staticmethod
    def _quality(mean_b: float, shadow_clip: float, highlight_clip: float, midtone: float) -> float:
        """Heuristic exposure quality in ``[0,1]``.

        Rewards a mid-range mean brightness and abundant midtones; penalises
        clipping at either end.  This is a *descriptor* of exposure health, not
        a correction target.
        """
        # Gaussian reward around mid-grey (0.45 is a common well-exposed mean).
        brightness_term = math.exp(-((mean_b - 0.45) ** 2) / (2 * 0.18 ** 2))
        clip_penalty = clamp01(1.0 - 4.0 * (shadow_clip + highlight_clip))
        midtone_term = clamp01(midtone)
        return clamp01(0.5 * brightness_term + 0.3 * midtone_term + 0.2 * clip_penalty)
