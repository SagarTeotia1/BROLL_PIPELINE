"""Shot-boundary detection.

A hard cut invalidates every assumption the tracker makes: boxes teleport, identities
swap, emotions appear to change instantly. Detecting cuts lets the pipeline

* temporarily sample **every** frame so the new shot is characterised immediately,
* reset the tracker and force re-identification instead of dragging stale track IDs
  across the cut.

The detector compares HSV histograms of consecutive frames downscaled to 64x64. That
costs ~0.2 ms per frame at 1080p, which is affordable on every decoded frame - and it
must run on every frame, because a cut that lands between two sampled frames would
otherwise be invisible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import cv2
import numpy as np

from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class SceneCutInfo:
    """Result of testing one frame for a shot boundary."""

    is_cut: bool
    score: float          # 1 - histogram correlation, in [0, 2]
    frame_index: int
    threshold: float

    def as_dict(self) -> dict:
        return {
            "is_cut": self.is_cut,
            "score": round(self.score, 4),
            "frame_index": self.frame_index,
        }


class SceneCutDetector:
    """Histogram-correlation shot detector with an adaptive floor.

    Args:
        threshold: base cut threshold on ``1 - correlation``.
        downscale: working resolution (square) for the histogram.
        min_gap: minimum number of frames between two reported cuts, which stops a
            strobing or heavily compressed sequence from firing continuously.
        adaptive: raise the effective threshold when the whole clip is high-motion.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        downscale: int = 64,
        min_gap: int = 6,
        adaptive: bool = True,
    ) -> None:
        self.base_threshold = float(threshold)
        self.downscale = int(downscale)
        self.min_gap = int(min_gap)
        self.adaptive = adaptive
        self._prev_hist: Optional[np.ndarray] = None
        self._last_cut_frame: int = -10_000
        self._recent: Deque[float] = deque(maxlen=48)
        self.cuts_detected: int = 0

    # -- api ----------------------------------------------------------------
    def update(self, image: np.ndarray, frame_index: int) -> SceneCutInfo:
        """Test one frame. Call this for **every** decoded frame."""
        hist = self._histogram(image)
        if self._prev_hist is None:
            self._prev_hist = hist
            return SceneCutInfo(False, 0.0, frame_index, self.base_threshold)

        correlation = float(
            cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
        )
        score = 1.0 - correlation
        self._prev_hist = hist
        self._recent.append(score)

        threshold = self._effective_threshold()
        is_cut = (
            score >= threshold
            and (frame_index - self._last_cut_frame) >= self.min_gap
        )
        if is_cut:
            self._last_cut_frame = frame_index
            self.cuts_detected += 1
            log.debug("Scene cut at frame %d (score %.3f >= %.3f)", frame_index, score, threshold)
        return SceneCutInfo(is_cut, score, frame_index, threshold)

    def reset(self) -> None:
        """Forget history (new video, or after a seek)."""
        self._prev_hist = None
        self._last_cut_frame = -10_000
        self._recent.clear()
        self.cuts_detected = 0

    # -- internals ----------------------------------------------------------
    def _histogram(self, image: np.ndarray) -> np.ndarray:
        # This runs on *every* decoded frame, so the resize must be cheap. Strided
        # subsampling first turns an INTER_AREA pass over a megapixel into one over a
        # few tens of kilopixels; the histogram is unaffected at this granularity.
        step = max(1, min(image.shape[0], image.shape[1]) // (self.downscale * 2))
        if step > 1:
            image = image[::step, ::step]
        small = cv2.resize(
            image, (self.downscale, self.downscale), interpolation=cv2.INTER_AREA
        )
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        # Value must be included: a cut to a differently-lit shot of the same palette
        # (or to/from black) leaves hue and saturation almost untouched.
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
        return hist

    def _effective_threshold(self) -> float:
        """Base threshold, floored by the recent motion level when adaptive."""
        if not self.adaptive or len(self._recent) < 12:
            return self.base_threshold
        median = float(np.median(self._recent))
        # A shaky handheld shot has a high baseline; require a clear jump above it.
        return max(self.base_threshold, median * 3.0 + 0.08)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self._recent)) if self._recent else 0.0


__all__ = ["SceneCutDetector", "SceneCutInfo"]
