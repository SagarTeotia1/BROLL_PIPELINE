"""Colour-harmony detection.

Classical harmony schemes are defined by the *relative* angular positions of the
dominant hues on the colour wheel.  We build a saturation-weighted hue histogram,
anchor it at the dominant hue ``h0``, and score each scheme by how well the hue
mass is explained by that scheme's target hues:

======================  =================================
Scheme                  Target hue offsets from ``h0``
======================  =================================
monochromatic           {0}
complementary           {0, 180}
analogous               {0, 30, 330}
triadic                 {0, 120, 240}
split complementary     {0, 150, 210}
tetradic                {0, 90, 180, 270}
======================  =================================

Each scheme's confidence multiplies (a) the fraction of hue mass lying within a
tolerance of its targets by (b) a *utilisation* factor requiring every target
hue to actually be populated — so a triadic scheme cannot win on a single-hue
image merely because it has spare hue slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .utils import FeatureResult, ImageContext, normalized_histogram

_TEMPLATES: Dict[str, List[float]] = {
    "monochromatic": [0.0],
    "complementary": [0.0, 180.0],
    "analogous": [0.0, 30.0, 330.0],
    "triadic": [0.0, 120.0, 240.0],
    "split_complementary": [0.0, 150.0, 210.0],
    "tetradic": [0.0, 90.0, 180.0, 270.0],
}


@dataclass
class HarmonyFeatures(FeatureResult):
    """Colour-harmony analysis results (confidences in ``[0,1]``)."""

    confidences: Dict[str, float] = field(default_factory=dict)
    best_match: str = "none"
    best_confidence: float = 0.0
    dominant_hue: float = 0.0


class HarmonyAnalyzer:
    """Computes :class:`HarmonyFeatures` from an :class:`ImageContext`."""

    def __init__(self, bins: int = 72, tolerance_deg: float = 25.0, min_frac: float = 0.05) -> None:
        self.bins = bins
        self.tolerance = tolerance_deg
        self.min_frac = min_frac

    def analyze(self, ctx: ImageContext) -> HarmonyFeatures:
        xp = ctx.xp
        hue = ctx.hsv[..., 0].reshape(-1)
        sat = ctx.hsv[..., 1].reshape(-1)

        # Saturation-weighted hue histogram (grey pixels contribute ~nothing).
        weighted = xp.histogram(
            hue, bins=self.bins, range=(0.0, 360.0), weights=sat
        )[0]
        hist = ctx.backend.to_numpy(weighted).astype(np.float64)
        total = float(hist.sum())

        centres = (np.arange(self.bins) + 0.5) * (360.0 / self.bins)

        if total < 1e-6:  # achromatic image => monochromatic by definition
            confs = {k: 0.0 for k in _TEMPLATES}
            confs["monochromatic"] = 1.0
            return HarmonyFeatures(confidences=confs, best_match="monochromatic",
                                   best_confidence=1.0, dominant_hue=0.0)

        h0 = float(centres[int(hist.argmax())])
        confs: Dict[str, float] = {}
        for name, offsets in _TEMPLATES.items():
            confs[name] = self._score(hist, centres, total, h0, offsets)

        best = max(confs, key=confs.get)
        return HarmonyFeatures(
            confidences=confs,
            best_match=best,
            best_confidence=confs[best],
            dominant_hue=h0,
        )

    def _score(
        self,
        hist: np.ndarray,
        centres: np.ndarray,
        total: float,
        h0: float,
        offsets: List[float],
    ) -> float:
        """Explained-mass * utilisation score for one harmony template."""
        targets = [(h0 + off) % 360.0 for off in offsets]
        # Circular angular distance from every bin centre to every target.
        dists = np.stack(
            [np.abs((centres - t + 180.0) % 360.0 - 180.0) for t in targets], axis=0
        )  # (n_targets, bins)
        nearest = dists.min(axis=0)
        within = nearest <= self.tolerance
        # Triangular membership weight: 1 at target hue, 0 at tolerance edge.
        weight = np.clip(1.0 - nearest / self.tolerance, 0.0, 1.0)
        explained = float((hist * weight * within).sum()) / total

        # Utilisation: each target hue must carry >= min_frac of the mass.
        used = 0
        for row in dists:
            mass = float((hist * (row <= self.tolerance)).sum()) / total
            if mass >= self.min_frac:
                used += 1
        utilisation = used / len(targets)
        return float(explained * utilisation)
