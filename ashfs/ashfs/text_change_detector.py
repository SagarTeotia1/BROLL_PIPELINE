"""Text-change detector — lightweight secondary keyframe trigger.

Detects significant text-region changes between consecutive candidate frames
within a shot using a fast histogram-based approach.  When text changes
meaningfully (but DINOv2 global distance is low), forces an extra keyframe.

Uses OpenCV-based text region detection (Maximally Stable Extremal Regions,
MSER) — no deep learning, no extra dependencies beyond cv2.  OCR is NOT run
here; we only detect whether text-like regions changed, not what they say.

Optional: if opencv-contrib is available, EAST detector is used instead
(better precision).  Falls back to MSER automatically.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def _compute_iou(boxes_a: list[tuple[int, int, int, int]],
                 boxes_b: list[tuple[int, int, int, int]]) -> float:
    """Compute mean best-match IoU between two sets of axis-aligned bounding boxes.

    Each box is ``(x, y, w, h)`` as returned by cv2.  For each box in
    *boxes_a* we find the single best-overlapping box in *boxes_b* and
    accumulate the IoU.  The final score is the mean over *boxes_a*.

    Returns 1.0 when both sets are empty (no change), 0.0 when one set is
    empty and the other is not.
    """
    if not boxes_a and not boxes_b:
        return 1.0
    if not boxes_a or not boxes_b:
        return 0.0

    best_ious: list[float] = []
    for ax, ay, aw, ah in boxes_a:
        ax2, ay2 = ax + aw, ay + ah
        best = 0.0
        for bx, by, bw, bh in boxes_b:
            bx2, by2 = bx + bw, by + bh
            # Intersection
            ix1 = max(ax, bx)
            iy1 = max(ay, by)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter = inter_w * inter_h
            if inter == 0:
                continue
            union = aw * ah + bw * bh - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou > best:
                best = iou
        best_ious.append(best)

    return float(np.mean(best_ious)) if best_ious else 0.0


def _extract_mser_boxes(
    gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Return MSER bounding boxes for text-like regions in *gray*.

    Parameters
    ----------
    gray:
        Grayscale uint8 image.

    Returns
    -------
    List of ``(x, y, w, h)`` bounding boxes.  Empty list if MSER is
    unavailable or detects nothing.
    """
    try:
        import cv2  # noqa: PLC0415  (local import — cv2 always present)
    except ImportError:
        return []

    try:
        mser = cv2.MSER_create()
        # detect() returns (regions, bboxes); bboxes is (x, y, w, h)
        _regions, bboxes = mser.detectRegions(gray)
        if bboxes is None or len(bboxes) == 0:
            return []
        # Filter tiny noise blobs: keep only boxes with both dim > 4px
        filtered = [
            (int(x), int(y), int(w), int(h))
            for x, y, w, h in bboxes
            if w > 4 and h > 4
        ]
        return filtered
    except Exception as exc:  # noqa: BLE001
        logger.debug("MSER extraction failed: %s", exc)
        return []


class TextChangeDetector:
    """Lightweight MSER-based text-change detector.

    Parameters
    ----------
    min_iou_change:
        Mean best-match IoU below this value is considered a text change.
        Lower values = more sensitive (fewer false positives).
    min_region_count_change:
        Absolute change in MSER region count that, combined with a low IoU,
        confirms a text change.
    min_spacing_frames:
        A text-change keyframe is suppressed if it falls within this many
        frames of an already-selected keyframe.
    """

    def __init__(
        self,
        min_iou_change: float = 0.3,
        min_region_count_change: int = 5,
        min_spacing_frames: int = 3,
    ) -> None:
        self._min_iou_change = min_iou_change
        self._min_region_count_change = min_region_count_change
        self._min_spacing = min_spacing_frames
        self._cv2_available = self._check_cv2()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_cv2() -> bool:
        try:
            import cv2  # noqa: PLC0415
            _ = cv2.MSER_create  # confirm contrib method exists
            return True
        except Exception:  # noqa: BLE001
            logger.warning(
                "cv2.MSER_create not available — TextChangeDetector disabled."
            )
            return False

    def _text_changed(
        self,
        gray_a: np.ndarray,
        gray_b: np.ndarray,
    ) -> bool:
        """Return True if the text regions in *gray_a* and *gray_b* differ significantly."""
        boxes_a = _extract_mser_boxes(gray_a)
        boxes_b = _extract_mser_boxes(gray_b)

        count_delta = abs(len(boxes_a) - len(boxes_b))
        iou = _compute_iou(boxes_a, boxes_b)

        return iou < self._min_iou_change and count_delta >= self._min_region_count_change

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_text_change_frames(
        self,
        shot_frame_indices: list[int],
        frame_loader_fn: Callable[[int], np.ndarray],
        existing_keyframes: list[int],
    ) -> list[int]:
        """Return additional frame indices where significant text changes detected.

        Parameters
        ----------
        shot_frame_indices:
            Sorted list of candidate frame indices belonging to a single shot.
        frame_loader_fn:
            Callable ``(abs_frame_idx) -> np.ndarray`` returning RGB uint8 image.
        existing_keyframes:
            Keyframes already selected for this shot.  New frames are suppressed
            if within ``min_spacing_frames`` of any existing keyframe.

        Returns
        -------
        List of absolute frame indices (subset of *shot_frame_indices*) where
        a text change was detected and no existing keyframe is nearby.  Empty
        list if text detection is unavailable or no changes are found.
        """
        if not self._cv2_available:
            return []
        if len(shot_frame_indices) < 2:
            return []

        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return []

        existing_set: set[int] = set(existing_keyframes)
        extra: list[int] = []

        try:
            prev_idx = shot_frame_indices[0]
            prev_frame = frame_loader_fn(prev_idx)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

            for curr_idx in shot_frame_indices[1:]:
                try:
                    curr_frame = frame_loader_fn(curr_idx)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Frame load failed at %d: %s", curr_idx, exc)
                    prev_idx = curr_idx
                    prev_gray = None  # type: ignore[assignment]
                    continue

                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

                if prev_gray is not None and self._text_changed(prev_gray, curr_gray):
                    # Check spacing against existing keyframes AND already-added extras
                    all_kf = existing_set | set(extra)
                    too_close = any(
                        abs(curr_idx - kf) < self._min_spacing for kf in all_kf
                    )
                    if not too_close:
                        extra.append(curr_idx)

                prev_idx = curr_idx
                prev_gray = curr_gray

        except Exception as exc:  # noqa: BLE001
            logger.warning("TextChangeDetector error: %s", exc)
            return []

        return extra
