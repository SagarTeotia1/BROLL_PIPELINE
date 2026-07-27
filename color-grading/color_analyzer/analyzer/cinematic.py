"""Cinematic / stylistic scoring.

These are *descriptive* style scores in ``[0,1]``, each a documented combination
of directly measurable quantities (hue-band mass, luminance distribution,
saturation, contrast, warmth).  They are heuristics — labels a colourist would
recognise — not ground-truth classifiers.  Every score is computed from the
shared :class:`ImageContext`, so this analyzer is self-contained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import FeatureResult, ImageContext, clamp01

# Hue bands (degrees) for the signature teal-orange look.
_TEAL_LO, _TEAL_HI = 160.0, 210.0
_ORANGE_LO, _ORANGE_HI = 15.0, 55.0
# Wider blue band used for the HSL "blue" control (background separation).
_BLUE_LO, _BLUE_HI = 200.0, 260.0


def _gauss(x: float, mu: float, sigma: float) -> float:
    """Un-normalised Gaussian membership in ``[0,1]``."""
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


@dataclass
class CinematicFeatures(FeatureResult):
    """Cinematic style scores, all in ``[0,1]``."""

    teal_dominance: float = 0.0
    orange_dominance: float = 0.0
    teal_orange_score: float = 0.0
    vintage_score: float = 0.0
    commercial_score: float = 0.0
    natural_score: float = 0.0
    hdr_score: float = 0.0
    low_key_score: float = 0.0
    high_key_score: float = 0.0
    moody_score: float = 0.0
    film_look_score: float = 0.0

    # Per-band measurements backing the HSL controls.  Each is the band's mean
    # minus the frame mean, scaled to roughly [-100, 100] — i.e. "how much more
    # saturated / brighter than the rest of the frame is this hue range".
    # Orange is where skin lives; blue is the background-separation band.
    orange_saturation: float = 0.0
    orange_luminance: float = 0.0
    blue_saturation: float = 0.0
    blue_luminance: float = 0.0


class CinematicAnalyzer:
    """Computes :class:`CinematicFeatures` from an :class:`ImageContext`."""

    def analyze(self, ctx: ImageContext) -> CinematicFeatures:
        xp = ctx.xp
        lum = ctx.gray.reshape(-1)
        hsv = ctx.hsv_flat
        lab = ctx.lab_flat
        hue = hsv[:, 0]
        sat = hsv[:, 1]

        # --- base measurements (single pass each) --------------------------
        mean_b = float(lum.mean())
        std_b = float(lum.std())
        p2, p50, p98 = ctx.luma_percentile(2.0, 50.0, 98.0)
        shadow_frac = float((lum < 0.25).sum()) / float(lum.size)
        highlight_frac = float((lum > 0.75).sum()) / float(lum.size)
        mean_sat = float(sat.mean())
        warmth = float(lab[:, 2].mean())  # Lab b*: >0 warm/yellow
        dyn_stops = math.log2((p98 + 1e-4) / (p2 + 1e-4))

        # Saturation-weighted hue-band masses (teal & orange).
        total_sat = float(sat.sum()) + 1e-8
        teal_mask = (hue >= _TEAL_LO) & (hue <= _TEAL_HI)
        orange_mask = (hue >= _ORANGE_LO) & (hue <= _ORANGE_HI)
        teal = float((sat * teal_mask).sum()) / total_sat
        orange = float((sat * orange_mask).sum()) / total_sat

        # Shadow-cool / highlight-warm evidence for the teal-orange grade.
        shadow_mask = lum < 0.33
        highlight_mask = lum > 0.66
        shadow_b = float(lab[:, 2][shadow_mask].mean()) if bool(shadow_mask.any()) else 0.0
        highlight_b = float(lab[:, 2][highlight_mask].mean()) if bool(highlight_mask.any()) else 0.0
        split_evidence = clamp01((highlight_b - shadow_b) / 25.0)

        # --- composite style scores ----------------------------------------
        teal_orange = clamp01(2.0 * math.sqrt(max(teal, 0.0) * max(orange, 0.0)) + 0.4 * split_evidence)

        # Vintage: lifted/faded blacks + muted saturation + warm/yellow cast.
        lifted = clamp01(p2 * 4.0)
        vintage = clamp01(0.4 * lifted + 0.3 * (1.0 - mean_sat) + 0.3 * _gauss(warmth, 15.0, 12.0))

        # Commercial: bright, punchy, saturated and contrasty (clean look).
        commercial = clamp01(0.35 * clamp01(mean_b * 1.4) + 0.35 * clamp01(mean_sat * 2.2) + 0.30 * clamp01(std_b * 4.0))

        # Natural: neutral warmth, moderate saturation, moderate contrast.
        natural = clamp01(0.4 * _gauss(warmth, 3.0, 10.0) + 0.3 * _gauss(mean_sat, 0.3, 0.18) + 0.3 * _gauss(std_b, 0.2, 0.12))

        # HDR: wide dynamic range with strong local contrast/detail.
        # Micro-contrast, so this asks for the small window explicitly: the
        # 15-pixel window the contrast analyzer uses measures regional contrast
        # and scores roughly twice as high on the same frame. Still cheaper than
        # before, which built the same map with a non-separable scipy convolve.
        local_contrast = float(ctx.local_std(ctx.MICRO_WINDOW).mean())
        hdr = clamp01(0.5 * clamp01((dyn_stops - 4.0) / 6.0) + 0.5 * clamp01(local_contrast * 12.0))

        # Low-key: dark, shadow-dominated.
        low_key = clamp01(0.6 * (1.0 - mean_b) + 0.4 * shadow_frac)
        # High-key: bright, highlight-dominated, gentle contrast.
        high_key = clamp01(0.5 * mean_b + 0.3 * highlight_frac + 0.2 * (1.0 - clamp01(std_b * 4.0)))

        # Moody: darkish, muted and cool.
        moody = clamp01(0.4 * (1.0 - mean_b) + 0.3 * (1.0 - mean_sat) + 0.3 * _gauss(warmth, -8.0, 12.0))

        # Film look: gentle lifted blacks + teal-orange tint + moderate colour.
        film = clamp01(0.4 * teal_orange + 0.3 * lifted + 0.3 * _gauss(mean_sat, 0.35, 0.2))

        # --- per-band saturation/luminance for the HSL controls -------------
        blue_mask = (hue >= _BLUE_LO) & (hue <= _BLUE_HI)
        o_sat, o_lum = self._band_deviation(orange_mask, sat, lum, mean_sat, mean_b)
        b_sat, b_lum = self._band_deviation(blue_mask, sat, lum, mean_sat, mean_b)

        return CinematicFeatures(
            teal_dominance=clamp01(teal * 3.0),
            orange_dominance=clamp01(orange * 3.0),
            teal_orange_score=teal_orange,
            vintage_score=vintage,
            commercial_score=commercial,
            natural_score=natural,
            hdr_score=hdr,
            low_key_score=low_key,
            high_key_score=high_key,
            moody_score=moody,
            film_look_score=film,
            orange_saturation=o_sat,
            orange_luminance=o_lum,
            blue_saturation=b_sat,
            blue_luminance=b_lum,
        )

    @staticmethod
    def _band_deviation(mask, sat, lum, mean_sat: float, mean_lum: float) -> tuple:
        """Mean saturation/luminance inside a hue band, relative to the frame.

        Returned on a ~[-100, 100] scale so it reads like an HSL slider.  Sums
        are taken with the mask as a weight rather than by fancy-indexing
        ``sat[mask]``, which would allocate a second full-length array per band.
        An empty band reports no deviation.
        """
        count = float(mask.sum())
        if count < 1.0:
            return 0.0, 0.0
        band_sat = float((sat * mask).sum()) / count
        band_lum = float((lum * mask).sum()) / count
        return (band_sat - mean_sat) * 100.0, (band_lum - mean_lum) * 100.0
