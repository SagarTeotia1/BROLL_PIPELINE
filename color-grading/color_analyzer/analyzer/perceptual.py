"""Perceptual (CIE L*a*b*) feature extraction.

Because L*a*b* is perceptually near-uniform, Euclidean distance in it (the CIE76
``Delta E`` = ``sqrt(dL^2 + da^2 + db^2)``) approximates perceived colour
difference.  We use it for several complementary descriptors:

* **local Delta E** — mean colour difference between neighbouring pixels
  (perceptual "busyness"/texture).
* **average perceptual distance** — mean pairwise Delta E over a random sample
  (global perceptual spread of the palette).
* **lab distance** — mean Delta E of each pixel to the global mean colour
  (perceptual dispersion around the average colour).
* **perceptual contrast** — standard deviation of L* (perceived lightness
  contrast).
* **colour complexity** — normalised entropy of a coarse 3-D Lab histogram
  (how many perceptually distinct colours are present).
"""

from __future__ import annotations

from dataclasses import dataclass

from .utils import FeatureResult, ImageContext


@dataclass
class PerceptualFeatures(FeatureResult):
    """Perceptual analysis results (Delta E in CIE76 units)."""

    local_delta_e: float = 0.0
    average_perceptual_distance: float = 0.0
    lab_distance: float = 0.0
    perceptual_contrast: float = 0.0
    color_complexity: float = 0.0  # normalised [0,1]


class PerceptualAnalyzer:
    """Computes :class:`PerceptualFeatures` from an :class:`ImageContext`."""

    def __init__(self, sample_size: int = 20_000, complexity_bins: int = 8, seed: int = 0) -> None:
        self.sample_size = sample_size
        self.complexity_bins = complexity_bins
        self.seed = seed

    def analyze(self, ctx: ImageContext) -> PerceptualFeatures:
        xp = ctx.xp
        lab = ctx.lab

        # --- local Delta E: neighbour differences (right & down) -----------
        d_right = lab[:, 1:, :] - lab[:, :-1, :]
        d_down = lab[1:, :, :] - lab[:-1, :, :]
        de_right = xp.sqrt((d_right ** 2).sum(axis=2))
        de_down = xp.sqrt((d_down ** 2).sum(axis=2))
        local_de = float((float(de_right.mean()) + float(de_down.mean())) / 2.0)

        lab_flat = lab.reshape(-1, 3)
        mean_lab = lab_flat.mean(axis=0)

        # --- dispersion around the mean colour -----------------------------
        lab_distance = float(xp.sqrt(((lab_flat - mean_lab) ** 2).sum(axis=1)).mean())

        # --- average pairwise Delta E on a random sample -------------------
        avg_pairwise = self._sampled_pairwise(ctx, lab_flat)

        # --- perceptual lightness contrast ---------------------------------
        perceptual_contrast = float(lab_flat[:, 0].std())

        # --- colour complexity via 3-D Lab histogram entropy ---------------
        complexity = self._complexity(ctx, lab_flat)

        return PerceptualFeatures(
            local_delta_e=local_de,
            average_perceptual_distance=avg_pairwise,
            lab_distance=lab_distance,
            perceptual_contrast=perceptual_contrast,
            color_complexity=complexity,
        )

    def _sampled_pairwise(self, ctx: ImageContext, lab_flat) -> float:
        """Mean CIE76 Delta E between two independent random pixel samples."""
        xp = ctx.xp
        n = int(lab_flat.shape[0])
        m = min(self.sample_size, n)
        if hasattr(xp.random, "RandomState"):
            rng = xp.random.RandomState(self.seed)
            idx_a = rng.randint(0, n, size=m)
            idx_b = rng.randint(0, n, size=m)
        else:  # pragma: no cover
            idx_a = xp.random.randint(0, n, size=m)
            idx_b = xp.random.randint(0, n, size=m)
        a = lab_flat[idx_a]
        b = lab_flat[idx_b]
        return float(xp.sqrt(((a - b) ** 2).sum(axis=1)).mean())

    def _complexity(self, ctx: ImageContext, lab_flat) -> float:
        """Normalised entropy of a coarse 3-D Lab histogram (perceptual variety).

        L* is scaled from [0,100] and a*,b* from ~[-128,127] into ``bins`` cells
        per axis; the joint-histogram entropy is normalised by ``log2(bins^3)``.
        """
        import numpy as np

        lab_np = ctx.backend.to_numpy(lab_flat)
        bins = self.complexity_bins
        L = np.clip(lab_np[:, 0] / 100.0, 0, 1) * (bins - 1)
        a = np.clip((lab_np[:, 1] + 128.0) / 255.0, 0, 1) * (bins - 1)
        b = np.clip((lab_np[:, 2] + 128.0) / 255.0, 0, 1) * (bins - 1)
        codes = (L.astype(np.int64) * bins * bins + a.astype(np.int64) * bins + b.astype(np.int64))
        counts = np.bincount(codes, minlength=bins ** 3).astype(np.float64)
        p = counts / (counts.sum() + 1e-8)
        p = p[p > 0]
        if p.size <= 1:
            return 0.0
        h = -(p * np.log2(p)).sum()
        return float(h / np.log2(bins ** 3))
