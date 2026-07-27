"""Tone-curve estimation.

Without a reference "before" image, the tone curve is characterised from the
luminance *quantile function* ``Q(p)`` — the p-th percentile of luminance for
``p in [0,1]``.  If one assumes the ungraded scene had a roughly uniform tonal
distribution, ``Q(p)`` **is** the transfer curve mapping input rank to output
luminance.  From its shape we recover the classic grading controls:

* **black/white point** — endpoints ``Q(0)`` / ``Q(1)``.
* **gamma** — exponent of a best-fit power law on the normalised curve.
* **S-curve strength** — excess midtone slope relative to the mean slope
  (positive => contrasty S-curve, negative => flattened/faded).
* **lifted shadows / crushed blacks** — elevated black point / clipping at 0.
* **highlight rolloff** — compression of the slope near the top end.
* **contrast curve** — raw midtone steepness ``dQ/dp`` at ``p=0.5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .utils import FeatureResult, ImageContext, clamp01


@dataclass
class ToneCurveFeatures(FeatureResult):
    """Tone-curve analysis results."""

    black_point: float = 0.0
    white_point: float = 1.0
    gamma: float = 1.0
    s_curve_strength: float = 0.0
    lifted_shadows: float = 0.0
    crushed_blacks: float = 0.0
    highlight_rolloff: float = 0.0
    contrast_curve: float = 0.0
    curve_samples: List[float] = field(default_factory=list)  # Q(p) sampled


class ToneCurveAnalyzer:
    """Computes :class:`ToneCurveFeatures` from an :class:`ImageContext`."""

    def __init__(self, samples: int = 64) -> None:
        self.samples = samples

    def analyze(self, ctx: ImageContext) -> ToneCurveFeatures:
        xp = ctx.xp
        lum = ctx.gray.reshape(-1)

        # Quantile function Q(p) sampled on a uniform grid of p in [0,1].
        p = np.linspace(0.0, 1.0, self.samples)
        q = ctx.backend.to_numpy(xp.percentile(lum, xp.asarray(p * 100.0)))
        q = np.asarray(q, dtype=np.float64)

        black_point = float(q[0])
        white_point = float(q[-1])
        span = max(white_point - black_point, 1e-6)
        norm = (q - black_point) / span  # normalised curve in [0,1]

        gamma = self._fit_gamma(p, norm)
        slopes = np.gradient(q, p)  # raw dQ/dp
        mean_slope = float(np.mean(slopes)) + 1e-8

        def band_slope(lo: float, hi: float) -> float:
            mask = (p >= lo) & (p <= hi)
            return float(np.mean(slopes[mask])) if np.any(mask) else mean_slope

        midtone_slope = band_slope(0.4, 0.6)
        highlight_slope = band_slope(0.85, 1.0)

        s_curve = midtone_slope / mean_slope - 1.0
        highlight_rolloff = clamp01((mean_slope - highlight_slope) / mean_slope)

        # Crushed blacks: fraction of pixels pinned near zero.
        crushed = float((lum < 0.02).sum()) / float(lum.size)
        lifted = black_point  # elevated floor => faded/lifted shadows

        return ToneCurveFeatures(
            black_point=black_point,
            white_point=white_point,
            gamma=gamma,
            s_curve_strength=float(s_curve),
            lifted_shadows=lifted,
            crushed_blacks=crushed,
            highlight_rolloff=highlight_rolloff,
            contrast_curve=midtone_slope,
            curve_samples=[float(v) for v in q.tolist()],
        )

    @staticmethod
    def _fit_gamma(p: np.ndarray, norm: np.ndarray) -> float:
        """Least-squares fit of ``norm ≈ p^gamma`` in log-log space.

        Only interior points (``0 < p,norm < 1``) are used to avoid the
        singular logarithms at the endpoints.  Returns a clamped exponent.
        """
        mask = (p > 1e-3) & (p < 1.0) & (norm > 1e-3) & (norm < 1.0)
        if int(mask.sum()) < 4:
            return 1.0
        lp = np.log(p[mask])
        ln = np.log(norm[mask])
        gamma = float(np.dot(lp, ln) / (np.dot(lp, lp) + 1e-8))
        return min(max(gamma, 0.2), 5.0)
