"""Palette stage: the frame's dominant colours.

Returns a short, ordered swatch list — the thing a colourist looks at to decide
what a shot is "made of" — with each swatch's share of the frame and a role
label separating the background bulk from small saturated accents.

Stability is the whole design constraint
----------------------------------------
Clustering is normally seeded randomly, which makes the palette reproducible for
one frame but *unstable across similar frames*: two consecutive frames of the
same shot can land on different local minima and swap swatch order, and anything
downstream sees the palette churn.

So the seeding here is not random at all.  Pixels are binned into a coarse RGB
grid and the most populated cells become the initial centroids.  Two frames of
the same scene fill the same cells in the same order, so they start from the
same seeds and converge to the same palette.  ``cv2.setRNGSeed`` is set as well,
for any OpenCV routine that might consult it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from .frame import Frame
from .scales import hex_color, ratio, rgb_255, slider

#: Levels per channel in the seeding grid (6^3 = 216 cells).
GRID_LEVELS = 6

#: Pixels sampled for clustering. Taken by stride, never randomly.
MAX_SAMPLES = 20_000

#: Lloyd iterations. The grid seeding starts close, so this converges quickly.
MAX_ITERATIONS = 12

#: Centroids closer than this in RGB are merged; below it two swatches are the
#: same colour to the eye and listing both wastes a slot.
MERGE_DISTANCE = 0.06

#: Deterministic seed for any OpenCV RNG consulted downstream.
RNG_SEED = 0


def analyze_palette(frame: Frame, colors: int = 6) -> List[Dict[str, Any]]:
    """Extract up to ``colors`` dominant swatches, ordered by coverage."""
    cv2.setRNGSeed(RNG_SEED)

    samples = _sample(frame.rgb_flat)
    centroids = _grid_seed(samples, colors)
    centroids, counts = _lloyd(samples, centroids)
    centroids, counts = _merge_similar(centroids, counts)

    total = float(counts.sum()) or 1.0
    order = np.lexsort((_luma(centroids), -counts))  # coverage desc, then luma

    swatches: List[Dict[str, Any]] = []
    for index in order:
        coverage = float(counts[index]) / total
        if coverage <= 0.0:
            continue
        rgb = centroids[index]
        swatches.append({
            "hex": hex_color(rgb),
            "rgb": rgb_255(rgb),
            "coverage": ratio(coverage),
            "saturation": slider(_saturation(rgb) * 100.0, 0.0, 100.0),
            "role": _role(coverage, _saturation(rgb)),
        })
    return swatches


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def _sample(rgb_flat: np.ndarray) -> np.ndarray:
    """Subsample pixels by stride — deterministic, and spatially even."""
    count = rgb_flat.shape[0]
    if count <= MAX_SAMPLES:
        return np.ascontiguousarray(rgb_flat, dtype=np.float32)
    return np.ascontiguousarray(rgb_flat[:: count // MAX_SAMPLES], dtype=np.float32)


def _grid_seed(samples: np.ndarray, k: int) -> np.ndarray:
    """Seed centroids from the most populated cells of a coarse RGB grid.

    Cell *means* are used rather than cell centres, so the seeds already sit on
    real colours and the first Lloyd step has little to correct.
    """
    quantised = np.clip((samples * GRID_LEVELS).astype(np.int32), 0, GRID_LEVELS - 1)
    codes = (quantised[:, 0] * GRID_LEVELS + quantised[:, 1]) * GRID_LEVELS + quantised[:, 2]

    cells = GRID_LEVELS ** 3
    counts = np.bincount(codes, minlength=cells)
    sums = np.stack(
        [np.bincount(codes, weights=samples[:, c], minlength=cells) for c in range(3)],
        axis=1,
    )

    populated = np.flatnonzero(counts)
    # Ties broken by cell index so the ordering is fixed, not arbitrary.
    ranked = populated[np.lexsort((populated, -counts[populated]))]
    chosen = ranked[:k]
    return (sums[chosen] / counts[chosen, None]).astype(np.float32)


def _lloyd(samples: np.ndarray, centroids: np.ndarray) -> tuple:
    """Lloyd's algorithm to convergence or :data:`MAX_ITERATIONS`.

    Distances use ``||x-c||^2 = ||x||^2 - 2x.c + ||c||^2`` and drop the constant
    ``||x||^2``, so each step is one matrix product plus an argmin.
    """
    k = centroids.shape[0]
    counts = np.zeros(k, dtype=np.int64)
    sample_sq = None

    for _ in range(MAX_ITERATIONS):
        distances = (centroids ** 2).sum(axis=1)[None, :] - 2.0 * (samples @ centroids.T)
        labels = distances.argmin(axis=1)

        counts = np.bincount(labels, minlength=k)
        sums = np.stack(
            [np.bincount(labels, weights=samples[:, c], minlength=k) for c in range(3)],
            axis=1,
        )
        occupied = counts > 0
        updated = centroids.copy()
        updated[occupied] = (sums[occupied] / counts[occupied, None]).astype(np.float32)

        if np.allclose(updated, centroids, atol=1e-4):
            centroids = updated
            break
        centroids = updated

    return centroids, counts


def _merge_similar(centroids: np.ndarray, counts: np.ndarray) -> tuple:
    """Fold centroids within :data:`MERGE_DISTANCE` into their larger neighbour."""
    order = np.argsort(-counts)
    kept: List[int] = []
    merged_counts = counts.astype(np.int64).copy()

    for index in order:
        if counts[index] == 0:
            continue
        duplicate_of = None
        for keep in kept:
            if float(np.linalg.norm(centroids[index] - centroids[keep])) < MERGE_DISTANCE:
                duplicate_of = keep
                break
        if duplicate_of is None:
            kept.append(int(index))
        else:
            merged_counts[duplicate_of] += merged_counts[index]
            merged_counts[index] = 0

    keep_idx = np.array(sorted(kept), dtype=np.int64)
    return centroids[keep_idx], merged_counts[keep_idx]


# ---------------------------------------------------------------------------
# swatch description
# ---------------------------------------------------------------------------
def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of each centroid."""
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _saturation(rgb: np.ndarray) -> float:
    """HSV saturation of a single RGB triple."""
    high = float(rgb.max())
    low = float(rgb.min())
    return 0.0 if high < 1e-6 else (high - low) / high


def _role(coverage: float, saturation: float) -> str:
    """Editorial role of a swatch.

    An accent is the small, saturated colour that carries a shot — a neon sign,
    a costume — and is worth protecting during a grade.  Distinguishing it from
    the background bulk is the reason coverage alone is not enough.
    """
    if coverage >= 0.30:
        return "dominant"
    if saturation >= 0.45 and coverage < 0.12:
        return "accent"
    return "secondary"
