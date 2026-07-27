"""HSV analysis.

Hue requires *circular* statistics because it lives on a 360-degree ring: the
naive arithmetic mean of hues near 0 and 360 is wrong.  We therefore compute the
mean hue as the angle of the resultant vector ``sum(e^{i*theta})`` and report a
circular variance in ``[0,1]`` (0 = perfectly concentrated, 1 = uniform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from .utils import (
    ChannelStats,
    FeatureResult,
    ImageContext,
    channel_stats,
    entropy_from_hist,
    normalized_histogram,
)


@dataclass
class HSVFeatures(FeatureResult):
    """HSV analysis results (hue in degrees, saturation/value in ``[0,1]``)."""

    mean_hue: float = 0.0
    median_hue: float = 0.0
    hue_variance: float = 0.0  # circular variance in [0,1]
    hue_entropy: float = 0.0
    dominant_hue_peaks: List[float] = field(default_factory=list)  # degrees

    saturation: ChannelStats = field(default_factory=ChannelStats)
    saturation_entropy: float = 0.0

    value: ChannelStats = field(default_factory=ChannelStats)
    value_entropy: float = 0.0

    # Coarse distributions (histograms) kept for reporting/visualisation.
    hue_histogram: List[float] = field(default_factory=list)
    saturation_histogram: List[float] = field(default_factory=list)
    value_histogram: List[float] = field(default_factory=list)


class HSVAnalyzer:
    """Computes :class:`HSVFeatures` from an :class:`ImageContext`."""

    def __init__(
        self, hue_bins: int = 36, sv_bins: int = 64, n_peaks: int = 3, deep: bool = True
    ) -> None:
        self.hue_bins = hue_bins
        self.sv_bins = sv_bins
        self.n_peaks = n_peaks
        self.deep = deep

    def analyze(self, ctx: ImageContext) -> HSVFeatures:
        xp = ctx.xp
        hue = ctx.hsv[..., 0].reshape(-1)  # degrees [0,360)
        sat = ctx.hsv[..., 1]
        val = ctx.hsv[..., 2]

        # --- circular hue statistics --------------------------------------
        # Weight each hue by saturation so near-grey pixels (undefined hue)
        # do not dominate the resultant vector. cos/sin come from the shared
        # context cache (split toning and local regions want them too).
        cos_h, sin_h = ctx.hue_cos_sin
        weights = sat.reshape(-1)
        wsum = float(weights.sum()) + 1e-8
        cos_mean = float((weights * cos_h).sum()) / wsum
        sin_mean = float((weights * sin_h).sum()) / wsum
        mean_hue = (math.degrees(math.atan2(sin_mean, cos_mean))) % 360.0
        resultant = math.sqrt(cos_mean * cos_mean + sin_mean * sin_mean)
        hue_variance = 1.0 - resultant  # circular variance

        hue_prob = normalized_histogram(xp, hue, self.hue_bins, (0.0, 360.0))
        hue_entropy = entropy_from_hist(xp, hue_prob)

        # Saturation statistics are the only part the 45-parameter grade reads.
        sat_stats = channel_stats(xp, sat, self.sv_bins, (0.0, 1.0))

        if not self.deep:
            return HSVFeatures(
                mean_hue=mean_hue,
                hue_variance=hue_variance,
                hue_entropy=hue_entropy,
                saturation=sat_stats,
                saturation_entropy=sat_stats.entropy,
            )

        # --- deep-only: medians, peak picking and the stored histograms -----
        # median_hue sorts the whole frame and the three histograms are for
        # plotting; neither reaches the grade.
        median_hue = float(xp.median(hue))
        peaks = self._dominant_peaks(ctx, hue_prob)
        val_stats = channel_stats(xp, val, self.sv_bins, (0.0, 1.0))
        sat_prob = normalized_histogram(xp, sat, self.sv_bins, (0.0, 1.0))
        val_prob = normalized_histogram(xp, val, self.sv_bins, (0.0, 1.0))

        return HSVFeatures(
            mean_hue=mean_hue,
            median_hue=median_hue,
            hue_variance=hue_variance,
            hue_entropy=hue_entropy,
            dominant_hue_peaks=peaks,
            saturation=sat_stats,
            saturation_entropy=sat_stats.entropy,
            value=val_stats,
            value_entropy=val_stats.entropy,
            hue_histogram=[float(v) for v in ctx.backend.to_numpy(hue_prob).tolist()],
            saturation_histogram=[float(v) for v in ctx.backend.to_numpy(sat_prob).tolist()],
            value_histogram=[float(v) for v in ctx.backend.to_numpy(val_prob).tolist()],
        )

    def _dominant_peaks(self, ctx: ImageContext, hue_prob) -> List[float]:
        """Return the centre hues (degrees) of the ``n_peaks`` largest local
        maxima of the hue histogram (circular neighbourhood)."""
        prob = ctx.backend.to_numpy(hue_prob)
        n = len(prob)
        if n == 0:
            return []
        bin_width = 360.0 / n
        peaks: List[Tuple[float, float]] = []
        for i in range(n):
            left = prob[(i - 1) % n]
            right = prob[(i + 1) % n]
            if prob[i] >= left and prob[i] >= right and prob[i] > 0:
                centre = (i + 0.5) * bin_width
                peaks.append((float(prob[i]), centre))
        peaks.sort(reverse=True)
        return [round(c, 2) for _, c in peaks[: self.n_peaks]]
