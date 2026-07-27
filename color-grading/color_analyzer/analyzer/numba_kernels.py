"""Optional Numba-accelerated CPU kernels.

These provide a faster CPU path for the few operations that are memory-heavy
when expressed as dense array algebra.  They are *optional*: if Numba is not
installed the callers fall back to the vectorised NumPy/CuPy implementation, so
behaviour is identical either way — only performance differs.

The K-means kernel fuses assignment, inertia and per-cluster accumulation into a
single streaming pass, avoiding the ``O(N*k)`` distance matrix that the dense
path materialises.  This matters for 4K frames (millions of pixels).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # pragma: no cover - depends on host having Numba
    from numba import njit, prange

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False


if HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _kmeans_step(data: np.ndarray, centroids: np.ndarray):  # pragma: no cover
        """Single Lloyd iteration: assign points and accumulate cluster sums.

        Returns ``(labels, sums, counts, inertia)`` where ``sums[c]`` is the
        vector sum of points assigned to cluster ``c``.  A streaming inner loop
        keeps memory at ``O(N + k)`` instead of ``O(N*k)``.
        """
        n = data.shape[0]
        dim = data.shape[1]
        k = centroids.shape[0]
        labels = np.empty(n, dtype=np.int64)
        sums = np.zeros((k, dim), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        inertia = 0.0
        for i in range(n):
            best = 0
            best_d = 1e30
            for c in range(k):
                d = 0.0
                for j in range(dim):
                    diff = data[i, j] - centroids[c, j]
                    d += diff * diff
                if d < best_d:
                    best_d = d
                    best = c
            labels[i] = best
            inertia += best_d
            counts[best] += 1
            for j in range(dim):
                sums[best, j] += data[i, j]
        return labels, sums, counts, inertia


def kmeans_step_numpy(
    data: np.ndarray, centroids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Vectorised NumPy fallback with the same signature as the Numba kernel."""
    d2 = (
        (data * data).sum(axis=1, keepdims=True)
        - 2.0 * data @ centroids.T
        + (centroids * centroids).sum(axis=1)[None, :]
    )
    labels = d2.argmin(axis=1)
    inertia = float(d2[np.arange(data.shape[0]), labels].sum())
    k = centroids.shape[0]
    counts = np.bincount(labels, minlength=k)
    sums = np.zeros((k, data.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, data)
    return labels, sums, counts, inertia


def kmeans_step(data: np.ndarray, centroids: np.ndarray):
    """Dispatch to the Numba kernel when available, else the NumPy fallback."""
    if HAS_NUMBA:
        return _kmeans_step(data.astype(np.float64), centroids.astype(np.float64))
    return kmeans_step_numpy(data.astype(np.float64), centroids.astype(np.float64))
