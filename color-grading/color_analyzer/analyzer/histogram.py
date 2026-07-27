"""Histogram feature extraction across colour spaces.

For each of RGB, HSV, L*a*b* and luminance we build normalised histograms and
summarise their *shape* with statistics that describe tonal distribution:

* **entropy** — information content / tonal richness.
* **skewness / kurtosis** — asymmetry and peakedness of the distribution.
* **spread** — inter-percentile width (p95 - p5), a robust dynamic-range proxy.
* **peak/valley count** — number of local maxima/minima (multi-modality).
* **smoothness** — 1 / (1 + mean |second difference|); high for smooth curves.
* **cdf** — cumulative distribution, useful for tone-curve matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .utils import (
    FeatureResult,
    ImageContext,
    entropy_from_hist,
    kurtosis,
    normalized_histogram,
    skewness,
)


@dataclass
class HistogramStats(FeatureResult):
    """Shape descriptors of a single histogram."""

    entropy: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    spread: float = 0.0
    peak_count: int = 0
    valley_count: int = 0
    smoothness: float = 0.0


@dataclass
class HistogramFeatures(FeatureResult):
    """Per-channel histogram shape stats plus stored histograms/CDFs."""

    stats: Dict[str, HistogramStats] = field(default_factory=dict)
    histograms: Dict[str, List[float]] = field(default_factory=dict)
    luminance_cdf: List[float] = field(default_factory=list)


class HistogramAnalyzer:
    """Computes :class:`HistogramFeatures` from an :class:`ImageContext`."""

    def __init__(self, bins: int = 256) -> None:
        self.bins = bins

    def analyze(self, ctx: ImageContext) -> HistogramFeatures:
        xp = ctx.xp

        # (name, channel_data, value_range) for every analysed channel.
        channels = [
            ("r", ctx.rgb[..., 0], (0.0, 1.0)),
            ("g", ctx.rgb[..., 1], (0.0, 1.0)),
            ("b", ctx.rgb[..., 2], (0.0, 1.0)),
            ("h", ctx.hsv[..., 0], (0.0, 360.0)),
            ("s", ctx.hsv[..., 1], (0.0, 1.0)),
            ("v", ctx.hsv[..., 2], (0.0, 1.0)),
            ("L", ctx.lab[..., 0], (0.0, 100.0)),
            ("a", ctx.lab[..., 1], (-128.0, 128.0)),
            ("b_lab", ctx.lab[..., 2], (-128.0, 128.0)),
            ("luma", ctx.gray, (0.0, 1.0)),
        ]

        stats: Dict[str, HistogramStats] = {}
        histograms: Dict[str, List[float]] = {}
        luminance_cdf: List[float] = []

        for name, data, rng in channels:
            prob = normalized_histogram(xp, data, self.bins, rng)
            prob_np = ctx.backend.to_numpy(prob)
            stats[name] = self._shape_stats(xp, data, prob, prob_np, rng)
            histograms[name] = [float(v) for v in prob_np.tolist()]
            if name == "luma":
                luminance_cdf = [float(v) for v in np.cumsum(prob_np).tolist()]

        return HistogramFeatures(
            stats=stats, histograms=histograms, luminance_cdf=luminance_cdf
        )

    def _shape_stats(self, xp, data, prob, prob_np, rng) -> HistogramStats:
        """Derive shape descriptors from both raw data and its histogram."""
        ent = entropy_from_hist(xp, prob)
        skew = skewness(xp, data)
        kurt = kurtosis(xp, data)

        # Robust spread: inter-percentile width normalised by the channel range.
        p5, p95 = xp.percentile(data.reshape(-1), xp.asarray([5.0, 95.0]))
        spread = float(p95 - p5) / (rng[1] - rng[0] + 1e-8)

        peaks, valleys = self._count_extrema(prob_np)

        # Smoothness from the mean magnitude of the second difference of the
        # (smoothed) histogram — smaller curvature => smoother curve.
        second_diff = np.abs(np.diff(prob_np, n=2)) if len(prob_np) > 2 else np.array([0.0])
        smoothness = 1.0 / (1.0 + float(second_diff.mean()) * len(prob_np))

        return HistogramStats(
            entropy=ent,
            skewness=skew,
            kurtosis=kurt,
            spread=spread,
            peak_count=int(peaks),
            valley_count=int(valleys),
            smoothness=smoothness,
        )

    @staticmethod
    def _count_extrema(prob_np: np.ndarray) -> tuple[int, int]:
        """Count significant local maxima/minima of a histogram.

        The histogram is lightly box-smoothed first so that quantisation noise
        does not inflate the extrema count.  Only extrema above 10% of the peak
        are counted as ``peaks`` to remain robust to tiny ripples.
        """
        if len(prob_np) < 3:
            return 0, 0
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(prob_np, kernel, mode="same")
        thresh = 0.1 * float(smooth.max()) if smooth.max() > 0 else 0.0
        peaks = 0
        valleys = 0
        for i in range(1, len(smooth) - 1):
            if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > thresh:
                peaks += 1
            elif smooth[i] < smooth[i - 1] and smooth[i] <= smooth[i + 1]:
                valleys += 1
        return peaks, valleys
