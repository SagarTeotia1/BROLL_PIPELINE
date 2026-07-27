"""Local (spatial) region analysis.

The frame is tiled into a ``grid x grid`` (default 4x4) lattice and each tile is
described independently.  This captures *spatial* grading behaviour — vignettes,
sky/ground splits, uneven white balance — that global statistics miss.  Per-tile
temperature uses the Lab ``b*`` axis (yellow<->blue) as a compact warm/cool
proxy.

Iteration is over the 16 tiles (not pixels); all per-tile reductions are
vectorised array operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .utils import FeatureResult, ImageContext


@dataclass
class GridRegion(FeatureResult):
    """Statistics of a single spatial tile."""

    row: int = 0
    col: int = 0
    avg_hue: float = 0.0
    avg_saturation: float = 0.0
    avg_brightness: float = 0.0
    contrast: float = 0.0  # luminance std within the tile
    temperature: float = 0.0  # mean Lab b* (>0 warm/yellow, <0 cool/blue)


@dataclass
class LocalRegionFeatures(FeatureResult):
    """Local-region analysis results across the whole grid."""

    grid_size: int = 4
    regions: List[GridRegion] = field(default_factory=list)
    brightness_uniformity: float = 0.0  # 1 - normalised spread of tile means
    temperature_uniformity: float = 0.0  # spatial consistency of warm/cool


class LocalRegionAnalyzer:
    """Computes :class:`LocalRegionFeatures` from an :class:`ImageContext`."""

    def __init__(self, grid: int = 4) -> None:
        self.grid = grid

    def analyze(self, ctx: ImageContext) -> LocalRegionFeatures:
        xp = ctx.xp
        h, w = ctx.height, ctx.width
        g = self.grid

        # Tile boundaries (as even as possible for non-divisible sizes).
        row_edges = [int(round(i * h / g)) for i in range(g + 1)]
        col_edges = [int(round(j * w / g)) for j in range(g + 1)]

        regions: List[GridRegion] = []
        brightness_means: List[float] = []
        temperatures: List[float] = []

        for r in range(g):
            for c in range(g):
                r0, r1 = row_edges[r], row_edges[r + 1]
                c0, c1 = col_edges[c], col_edges[c + 1]
                hsv = ctx.hsv[r0:r1, c0:c1].reshape(-1, 3)
                lum = ctx.gray[r0:r1, c0:c1].reshape(-1)
                lab_b = ctx.lab[r0:r1, c0:c1, 2].reshape(-1)
                if hsv.shape[0] == 0:
                    continue

                hue = hsv[:, 0]
                sat = hsv[:, 1]
                # Saturation-weighted circular mean hue.
                theta = hue * (math.pi / 180.0)
                wsum = float(sat.sum()) + 1e-8
                cos_m = float((sat * xp.cos(theta)).sum()) / wsum
                sin_m = float((sat * xp.sin(theta)).sum()) / wsum
                avg_hue = math.degrees(math.atan2(sin_m, cos_m)) % 360.0

                avg_sat = float(sat.mean())
                avg_bright = float(lum.mean())
                contrast = float(lum.std())
                temperature = float(lab_b.mean())

                regions.append(
                    GridRegion(
                        row=r,
                        col=c,
                        avg_hue=avg_hue,
                        avg_saturation=avg_sat,
                        avg_brightness=avg_bright,
                        contrast=contrast,
                        temperature=temperature,
                    )
                )
                brightness_means.append(avg_bright)
                temperatures.append(temperature)

        bright_uniformity = self._uniformity(brightness_means)
        temp_uniformity = self._uniformity(temperatures)

        return LocalRegionFeatures(
            grid_size=g,
            regions=regions,
            brightness_uniformity=bright_uniformity,
            temperature_uniformity=temp_uniformity,
        )

    @staticmethod
    def _uniformity(values: List[float]) -> float:
        """Return ``1/(1+std)`` of tile means — 1 for perfectly uniform grades."""
        import numpy as np

        if not values:
            return 0.0
        return float(1.0 / (1.0 + np.std(np.asarray(values, dtype=np.float64))))
