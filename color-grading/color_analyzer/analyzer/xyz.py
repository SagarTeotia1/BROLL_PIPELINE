"""CIE XYZ analysis.

XYZ is the device-independent tristimulus space underlying L*a*b*.  Beyond mean
tristimulus values we report the chromaticity coordinates ``x,y`` (the
projection onto the CIE chromaticity diagram) and a chroma proxy — the distance
of the mean chromaticity from the D65 white point, a compact scalar describing
how far the overall colour sits from neutral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import FeatureResult, ImageContext

# CIE D65 white point chromaticity coordinates.
_D65_XY = (0.3127, 0.3290)


@dataclass
class XYZFeatures(FeatureResult):
    """CIE XYZ analysis results."""

    mean_x: float = 0.0
    mean_y: float = 0.0
    mean_z: float = 0.0
    variance_x: float = 0.0
    variance_y: float = 0.0
    variance_z: float = 0.0
    chromaticity_x: float = 0.0
    chromaticity_y: float = 0.0
    chroma: float = 0.0  # distance of mean chromaticity from D65


class XYZAnalyzer:
    """Computes :class:`XYZFeatures` from an :class:`ImageContext`."""

    def analyze(self, ctx: ImageContext) -> XYZFeatures:
        xp = ctx.xp
        xyz = ctx.xyz
        mean = xyz.reshape(-1, 3).mean(axis=0)
        var = xyz.reshape(-1, 3).var(axis=0)
        mx, my, mz = float(mean[0]), float(mean[1]), float(mean[2])

        total = mx + my + mz + 1e-8
        cx = mx / total
        cy = my / total
        chroma = math.hypot(cx - _D65_XY[0], cy - _D65_XY[1])

        return XYZFeatures(
            mean_x=mx,
            mean_y=my,
            mean_z=mz,
            variance_x=float(var[0]),
            variance_y=float(var[1]),
            variance_z=float(var[2]),
            chromaticity_x=cx,
            chromaticity_y=cy,
            chroma=chroma,
        )
