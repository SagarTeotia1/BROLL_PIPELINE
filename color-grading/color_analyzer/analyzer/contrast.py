"""Contrast feature extraction.

Multiple, complementary contrast definitions are computed because grading styles
manipulate contrast in different ways:

* **RMS contrast** — standard deviation of luminance (global tonal separation).
* **Michelson contrast** — ``(Lmax - Lmin)/(Lmax + Lmin)`` using robust
  percentiles to reject outliers.
* **Global contrast** — p95-p5 luminance spread.
* **Local contrast** — mean local standard deviation over sliding windows
  (texture / micro-contrast), computed via box-filtered second moments.
* **Dynamic range** — log2 ratio of robust bright/dark luminance.
* **Laplacian variance** — variance of the Laplacian; a sharpness/edge-energy
  proxy that also reflects local contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .utils import FeatureResult, ImageContext


@dataclass
class ContrastFeatures(FeatureResult):
    """Contrast analysis results (luminance-domain, ``[0,1]`` unless noted)."""

    rms_contrast: float = 0.0
    michelson_contrast: float = 0.0
    global_contrast: float = 0.0
    local_contrast: float = 0.0
    histogram_spread: float = 0.0
    dynamic_range: float = 0.0  # in stops (log2)
    laplacian_variance: float = 0.0


class ContrastAnalyzer:
    """Computes :class:`ContrastFeatures` from an :class:`ImageContext`."""

    def analyze(self, ctx: ImageContext) -> ContrastFeatures:
        flat = ctx.gray.reshape(-1)  # Rec.709 luminance [0,1]

        rms = float(flat.std())

        # Percentiles come off the context's shared luminance CDF rather than
        # sorting the frame here — several analyzers want percentiles and the
        # sort was being repeated for each of them.
        p1, p5, p95, p99 = ctx.luma_percentile(1.0, 5.0, 95.0, 99.0)
        michelson = (p99 - p1) / (p99 + p1 + 1e-8)
        global_contrast = p95 - p5
        histogram_spread = float(flat.max() - flat.min())

        # Dynamic range in stops between robust dark and bright luminance.
        dyn = float(np.log2((p99 + 1e-4) / (p1 + 1e-4)))

        local_contrast = float(ctx.local_std().mean())
        lap_var = self._laplacian_variance(ctx)

        return ContrastFeatures(
            rms_contrast=rms,
            michelson_contrast=michelson,
            global_contrast=global_contrast,
            local_contrast=local_contrast,
            histogram_spread=histogram_spread,
            dynamic_range=dyn,
            laplacian_variance=lap_var,
        )

    def _laplacian_variance(self, ctx: ImageContext) -> float:
        """Variance of the Laplacian of luminance (edge energy / sharpness)."""
        gray_np = ctx.gray_np
        if cv2 is not None:
            lap = cv2.Laplacian(gray_np, cv2.CV_32F, ksize=3)
        else:  # pragma: no cover
            lap = _laplacian(gray_np)
        return float(lap.var())


def _laplacian(img: np.ndarray) -> np.ndarray:  # pragma: no cover
    """4-neighbour discrete Laplacian fallback."""
    p = np.pad(img, 1, mode="reflect")
    return (
        p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1]
    )
