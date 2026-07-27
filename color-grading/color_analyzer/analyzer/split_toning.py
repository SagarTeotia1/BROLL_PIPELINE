"""Split-toning analysis.

Split toning applies different colour tints to shadows, midtones and highlights
(e.g. the ubiquitous teal-shadow / orange-highlight look).  The image is
partitioned by luminance into three tonal zones and each zone's tint is
characterised by its saturation-weighted mean hue, mean saturation, dominant
colour and warmth.

The overall ``split_tone_confidence`` rewards (a) meaningfully saturated tints
in shadows and highlights and (b) a large hue separation between them — the
hallmark of a deliberate split tone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .utils import FeatureResult, ImageContext, clamp01

_SHADOW_HI = 0.33
_HIGHLIGHT_LO = 0.66


@dataclass
class RegionTone(FeatureResult):
    """Tint description of a single tonal zone."""

    hue: float = 0.0  # degrees
    saturation: float = 0.0
    warmth: float = 0.0  # mean (a*+b*)/2 in Lab
    dominant_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dominant_hex: str = "#000000"
    pixel_fraction: float = 0.0
    confidence: float = 0.0  # how strongly this zone is toned


@dataclass
class SplitToningFeatures(FeatureResult):
    """Split-toning analysis results."""

    shadows: RegionTone = field(default_factory=RegionTone)
    midtones: RegionTone = field(default_factory=RegionTone)
    highlights: RegionTone = field(default_factory=RegionTone)
    shadow_highlight_hue_separation: float = 0.0  # degrees, [0,180]
    split_tone_confidence: float = 0.0


class SplitToningAnalyzer:
    """Computes :class:`SplitToningFeatures` from an :class:`ImageContext`."""

    def analyze(self, ctx: ImageContext) -> SplitToningFeatures:
        xp = ctx.xp
        lum = ctx.gray.reshape(-1)
        rgb = ctx.rgb_flat
        hsv = ctx.hsv_flat
        lab = ctx.lab_flat
        n = float(lum.size)

        # Reduce all three tonal zones with dot products against 0/1 masks.
        #
        # Each zone total is ``sum(quantity * mask)``, which is exactly a dot
        # product, so it runs through BLAS instead of a Python-level reduction.
        # The midtone zone is obtained by subtraction rather than a third dot —
        # the three zones partition the frame, so its sum is the remainder.
        #
        # Two rejected alternatives: fancy-indexing ``q[mask]`` allocates a copy
        # per zone per quantity, and ``bincount(zone, weights=q)`` forces every
        # float32 weight array up to float64 first.
        dtype = ctx.rgb.dtype
        shadow_m = (lum < _SHADOW_HI).astype(dtype)
        highlight_m = (lum > _HIGHLIGHT_LO).astype(dtype)
        cos_h, sin_h = ctx.hue_cos_sin
        sat = hsv[:, 1]

        counts = [float(shadow_m.sum()), 0.0, float(highlight_m.sum())]
        counts[1] = n - counts[0] - counts[2]

        def by_zone(q: Any) -> list:
            lo = float(q @ shadow_m)
            hi = float(q @ highlight_m)
            return [lo, float(q.sum()) - lo - hi, hi]

        sums = {
            "sat": by_zone(sat),
            "sat_cos": by_zone(sat * cos_h),
            "sat_sin": by_zone(sat * sin_h),
            "lab_a": by_zone(lab[:, 1]),
            "lab_b": by_zone(lab[:, 2]),
            "r": by_zone(rgb[:, 0]),
            "g": by_zone(rgb[:, 1]),
            "b": by_zone(rgb[:, 2]),
        }

        shadows = self._region_tone(sums, counts, 0, n)
        midtones = self._region_tone(sums, counts, 1, n)
        highlights = self._region_tone(sums, counts, 2, n)

        # Circular hue separation between shadow and highlight tints.
        sep = abs((shadows.hue - highlights.hue + 180.0) % 360.0 - 180.0)

        # Confidence: both ends toned AND well separated in hue.
        tone_strength = min(shadows.confidence, highlights.confidence)
        confidence = clamp01(tone_strength * (sep / 180.0) * 2.0)

        return SplitToningFeatures(
            shadows=shadows,
            midtones=midtones,
            highlights=highlights,
            shadow_highlight_hue_separation=sep,
            split_tone_confidence=confidence,
        )

    @staticmethod
    def _region_tone(sums: Dict[str, list], counts: list, zone: int, n: float) -> RegionTone:
        """Characterise the tint of one tonal zone from the pre-reduced sums."""
        count = float(counts[zone])
        if count < 1.0:
            return RegionTone()

        # Saturation-weighted circular mean hue (undefined for grey => ignored).
        wsum = float(sums["sat"][zone]) + 1e-8
        cos_m = float(sums["sat_cos"][zone]) / wsum
        sin_m = float(sums["sat_sin"][zone]) / wsum
        mean_hue = math.degrees(math.atan2(sin_m, cos_m)) % 360.0

        mean_sat = float(sums["sat"][zone]) / count
        warmth = float(0.5 * (float(sums["lab_a"][zone]) + float(sums["lab_b"][zone])) / count)

        r = float(sums["r"][zone]) / count
        g = float(sums["g"][zone]) / count
        b = float(sums["b"][zone]) / count
        r255, g255, b255 = int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5)

        return RegionTone(
            hue=mean_hue,
            saturation=mean_sat,
            warmth=warmth,
            dominant_rgb=(r, g, b),
            dominant_hex=f"#{r255:02x}{g255:02x}{b255:02x}",
            pixel_fraction=count / n,
            confidence=clamp01(mean_sat * 2.5),
        )
