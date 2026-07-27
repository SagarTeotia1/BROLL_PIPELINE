"""Dominant-colour palette extraction via a backend-agnostic K-means.

A single vectorised Lloyd's-algorithm K-means is implemented against ``xp`` so
the *same* code runs on the GPU (CuPy) or CPU (NumPy) — satisfying the "GPU
accelerated K-means with no duplicated implementation" requirement.

Distances use the expansion ``||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c`` so each
iteration is two matrix products plus an ``argmin`` — no Python pixel loops.
Pixels are (optionally) subsampled to bound cost on 4K frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

from .utils import (
    FeatureResult,
    ImageContext,
    rgb_to_hsv,
    rgb_to_lab,
)


@dataclass
class PaletteColor(FeatureResult):
    """A single dominant colour with its representations and coverage."""

    rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # [0,1]
    rgb_255: Tuple[int, int, int] = (0, 0, 0)
    hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # H[0,360], S,V[0,1]
    lab: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hex: str = "#000000"
    percentage: float = 0.0  # fraction of pixels assigned to this colour


@dataclass
class DominantColorFeatures(FeatureResult):
    """Dominant-colour analysis results."""

    colors: List[PaletteColor] = field(default_factory=list)
    palette_diversity: float = 0.0  # normalised entropy of cluster sizes
    average_palette_distance: float = 0.0  # mean pairwise LAB Delta-E


class KMeans:
    """Minimal, vectorised K-means clustering on the active backend."""

    def __init__(self, k: int = 6, max_iter: int = 12, seed: int = 0, tol: float = 1e-4) -> None:
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.tol = tol

    def fit(self, xp: Any, data: Any) -> Tuple[Any, Any]:
        """Cluster ``data`` (N,3).  Returns ``(centroids (k,3), counts (k,))``.

        On the CPU backend the memory-heavy inner loop is delegated to the
        streaming :func:`kmeans_step` kernel (Numba-accelerated when available);
        on the GPU backend the vectorised ``xp`` path is used so the whole
        iteration stays on-device.
        """
        n = int(data.shape[0])
        k = min(self.k, n)
        is_gpu = xp.__name__ != "numpy"
        rng = xp.random.RandomState(self.seed) if hasattr(xp.random, "RandomState") else None

        centroids = self._kpp_init(xp, data, k)
        prev_inertia = None
        counts = xp.zeros(k, dtype=xp.int64)

        for _ in range(self.max_iter):
            labels, sums, counts, inertia = self._assign(xp, data, centroids, is_gpu)

            # Recompute centroids with a masked divide rather than a per-cluster
            # Python loop, so the update is one vectorised step.
            counts_f = counts.astype(data.dtype).reshape(-1, 1)
            centroids = xp.where(counts_f > 0, sums / xp.maximum(counts_f, 1.0), centroids)

            # Reseed empty clusters. There are at most k of them and k is small,
            # so the scalar reads here are negligible next to the pixel passes.
            for c in range(k):
                if int(counts[c]) == 0:
                    centroids[c] = data[int(_randint(xp, rng, n))]

            if prev_inertia is not None and abs(prev_inertia - inertia) <= self.tol * (prev_inertia + 1e-8):
                break
            prev_inertia = inertia

        return centroids, counts

    @staticmethod
    def _assign(xp: Any, data: Any, centroids: Any, is_gpu: bool) -> Tuple[Any, Any, Any, float]:
        """Assign points to nearest centroid, returning labels/sums/counts/inertia."""
        if is_gpu:
            # Dense on-device path (avoids host transfers on the GPU).
            d2 = (
                (data * data).sum(axis=1, keepdims=True)
                - 2.0 * data @ centroids.T
                + (centroids * centroids).sum(axis=1)[None, :]
            )
            labels = d2.argmin(axis=1)
            n = int(data.shape[0])
            inertia = float(d2[xp.arange(n), labels].sum())
            k = int(centroids.shape[0])
            # Segment-sum by label. The previous version looped over k and
            # boolean-indexed `data[labels == c]` on every iteration, which is
            # k full masked passes per step; bincount does it in one.
            counts = xp.bincount(labels, minlength=k).astype(xp.int64)
            sums = xp.stack(
                [xp.bincount(labels, weights=data[:, ch], minlength=k) for ch in range(data.shape[1])],
                axis=1,
            ).astype(data.dtype)
            return labels, sums, counts, inertia
        # CPU: streaming kernel (Numba-accelerated when installed).
        from .numba_kernels import kmeans_step

        return kmeans_step(data, centroids)

    def _kpp_init(self, xp: Any, data: Any, k: int) -> Any:
        """k-means++ seeding for stable, well-spread initial centroids."""
        n = int(data.shape[0])
        rng = xp.random.RandomState(self.seed) if hasattr(xp.random, "RandomState") else None
        first = int(_randint(xp, rng, n))
        centroids = [data[first]]
        closest = ((data - centroids[0]) ** 2).sum(axis=1)
        for _ in range(1, k):
            probs = closest / (closest.sum() + 1e-12)
            cumprobs = xp.cumsum(probs)
            r = float(_rand(xp, rng))
            idx = int((cumprobs < r).sum())
            idx = min(idx, n - 1)
            centroids.append(data[idx])
            new_d = ((data - centroids[-1]) ** 2).sum(axis=1)
            closest = xp.minimum(closest, new_d)
        return xp.stack(centroids, axis=0)


def _randint(xp: Any, rng: Any, high: int) -> int:
    if rng is not None:
        return int(rng.randint(0, high))
    return int(xp.random.randint(0, high))


def _rand(xp: Any, rng: Any) -> float:
    if rng is not None:
        return float(rng.rand())
    return float(xp.random.rand())


class DominantColorAnalyzer:
    """Computes :class:`DominantColorFeatures` from an :class:`ImageContext`."""

    def __init__(self, k: int = 6, max_samples: int = 20_000, seed: int = 0) -> None:
        self.k = k
        self.max_samples = max_samples
        self.seed = seed

    def analyze(self, ctx: ImageContext) -> DominantColorFeatures:
        xp = ctx.xp
        data = ctx.rgb_flat.astype(xp.float32)

        # Subsample for tractable clustering on very large frames.
        n = int(data.shape[0])
        if n > self.max_samples:
            step = n // self.max_samples
            data = data[::step]

        centroids, counts = KMeans(self.k, seed=self.seed).fit(xp, data)
        total = float(counts.sum()) + 1e-8

        # Sort clusters by coverage (descending) for a stable "top-N" palette.
        counts_np = ctx.backend.to_numpy(counts)
        order = list(counts_np.argsort()[::-1])
        centroids_sorted = centroids[xp.asarray(order)]

        hsv = rgb_to_hsv(xp, centroids_sorted.reshape(1, -1, 3)).reshape(-1, 3)
        lab = rgb_to_lab(xp, centroids_sorted.reshape(1, -1, 3)).reshape(-1, 3)

        cent_np = ctx.backend.to_numpy(centroids_sorted)
        hsv_np = ctx.backend.to_numpy(hsv)
        lab_np = ctx.backend.to_numpy(lab)

        colors: List[PaletteColor] = []
        for i, idx in enumerate(order):
            r, g, b = (float(v) for v in cent_np[i])
            r255, g255, b255 = int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5)
            colors.append(
                PaletteColor(
                    rgb=(r, g, b),
                    rgb_255=(r255, g255, b255),
                    hsv=(float(hsv_np[i, 0]), float(hsv_np[i, 1]), float(hsv_np[i, 2])),
                    lab=(float(lab_np[i, 0]), float(lab_np[i, 1]), float(lab_np[i, 2])),
                    hex=f"#{r255:02x}{g255:02x}{b255:02x}",
                    percentage=float(counts_np[idx]) / total,
                )
            )

        diversity = self._diversity(counts_np)
        avg_dist = self._avg_palette_distance(lab_np)

        return DominantColorFeatures(
            colors=colors,
            palette_diversity=diversity,
            average_palette_distance=avg_dist,
        )

    @staticmethod
    def _diversity(counts_np) -> float:
        """Normalised Shannon entropy of cluster sizes in ``[0,1]``.

        1 => colours evenly distributed (diverse palette); ~0 => one colour
        dominates.
        """
        import numpy as np

        p = counts_np.astype(np.float64)
        p = p / (p.sum() + 1e-8)
        p = p[p > 0]
        if p.size <= 1:
            return 0.0
        h = -(p * np.log2(p)).sum()
        return float(h / np.log2(len(counts_np)))

    @staticmethod
    def _avg_palette_distance(lab_np) -> float:
        """Mean pairwise CIE76 Delta-E between palette colours (spread)."""
        import numpy as np

        k = lab_np.shape[0]
        if k < 2:
            return 0.0
        diff = lab_np[:, None, :] - lab_np[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        iu = np.triu_indices(k, k=1)
        return float(dist[iu].mean())
