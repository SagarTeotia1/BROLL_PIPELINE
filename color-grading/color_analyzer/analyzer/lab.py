"""CIE L*a*b* analysis.

L*a*b* is perceptually uniform and separates lightness (L*) from two opponent
chroma axes: a* (green<->red) and b* (blue<->yellow).  This makes it ideal for
describing colour grading:

* **Warmth** correlates with mean b* (yellow) plus mean a* (red).
* **Green-magenta balance** correlates with mean a*.
* **Average chroma** ``C* = sqrt(a*^2 + b*^2)`` measures overall colourfulness.
"""

from __future__ import annotations

from dataclasses import dataclass

from .utils import FeatureResult, ImageContext


@dataclass
class LabFeatures(FeatureResult):
    """CIE L*a*b* analysis results.

    ``warmth_score`` is the mean of a* and b* (positive => warm/red-yellow,
    negative => cool/blue-green).  ``green_magenta_score`` is mean a*
    (positive => magenta/red bias, negative => green bias).
    """

    mean_l: float = 0.0
    mean_a: float = 0.0
    mean_b: float = 0.0
    std_l: float = 0.0
    std_a: float = 0.0
    std_b: float = 0.0
    median_l: float = 0.0
    median_a: float = 0.0
    median_b: float = 0.0
    warmth_score: float = 0.0
    green_magenta_score: float = 0.0
    average_chroma: float = 0.0


class LabAnalyzer:
    """Computes :class:`LabFeatures` from an :class:`ImageContext`.

    ``deep=False`` skips the per-channel medians, which cost three full sorts of
    the frame and are not read by the 45-parameter grade.
    """

    def __init__(self, deep: bool = True) -> None:
        self.deep = deep

    def analyze(self, ctx: ImageContext) -> LabFeatures:
        xp = ctx.xp
        lab = ctx.lab.reshape(-1, 3)
        a = lab[:, 1]
        b = lab[:, 2]

        mean = lab.mean(axis=0)
        std = lab.std(axis=0)

        mean_a = float(mean[1])
        mean_b = float(mean[2])
        # Average chroma C* = sqrt(a^2 + b^2), averaged over all pixels.
        chroma = float(xp.sqrt(a * a + b * b).mean())

        features = LabFeatures(
            mean_l=float(mean[0]),
            mean_a=mean_a,
            mean_b=mean_b,
            std_l=float(std[0]),
            std_a=float(std[1]),
            std_b=float(std[2]),
            warmth_score=0.5 * (mean_a + mean_b),
            green_magenta_score=mean_a,
            average_chroma=chroma,
        )
        if self.deep:
            med = xp.median(lab, axis=0)
            features.median_l = float(med[0])
            features.median_a = float(med[1])
            features.median_b = float(med[2])
        return features
