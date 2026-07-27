"""Detector interface and the :class:`Detection` record shared across the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class Detection:
    """One detected face in frame coordinates.

    Attributes:
        bbox: ``[x1, y1, x2, y2]`` in pixels of the *analysis* frame.
        score: detector confidence in ``[0, 1]``.
        landmarks: ``(5, 2)`` array (right eye, left eye, nose, right mouth, left mouth)
            or ``None`` when the model has no keypoint head.
        frame_index: source frame number, filled in by the pipeline.
        quality: face-quality score in ``[0, 1]``; ``-1`` until the gate runs.
    """

    bbox: np.ndarray
    score: float
    landmarks: Optional[np.ndarray] = None
    frame_index: int = -1
    quality: float = -1.0

    # -- geometry helpers ---------------------------------------------------
    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (
            float((self.bbox[0] + self.bbox[2]) * 0.5),
            float((self.bbox[1] + self.bbox[3]) * 0.5),
        )

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    def scaled(self, factor: float) -> "Detection":
        """Return a copy mapped back to a different resolution."""
        return Detection(
            bbox=self.bbox * factor,
            score=self.score,
            landmarks=None if self.landmarks is None else self.landmarks * factor,
            frame_index=self.frame_index,
            quality=self.quality,
        )

    def clipped(self, width: int, height: int) -> "Detection":
        """Clamp the box to the frame bounds (landmarks are left untouched)."""
        b = self.bbox.copy()
        b[0] = max(0.0, min(b[0], width - 1))
        b[1] = max(0.0, min(b[1], height - 1))
        b[2] = max(0.0, min(b[2], width - 1))
        b[3] = max(0.0, min(b[3], height - 1))
        return Detection(b, self.score, self.landmarks, self.frame_index, self.quality)

    def as_dict(self) -> dict:
        return {
            "bbox": [round(float(v), 2) for v in self.bbox],
            "score": round(float(self.score), 4),
            "quality": round(float(self.quality), 4),
            "frame_index": self.frame_index,
        }


class BaseDetector(ABC):
    """Contract every face detector implementation follows."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect faces in a single BGR frame."""

    @abstractmethod
    def detect_batch(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Detect faces in a batch of BGR frames; result index matches input index."""

    def close(self) -> None:  # pragma: no cover - optional override
        """Release GPU resources."""


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    """Greedy IoU non-maximum suppression.

    Args:
        boxes: ``(N, 4)`` array of ``[x1, y1, x2, y2]``.
        scores: ``(N,)`` confidences.
        threshold: IoU above which the lower-scoring box is dropped.

    Returns:
        Indices of the kept boxes, highest score first.
    """
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= threshold]
    return keep


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode ``(l, t, r, b)`` distances around anchor points into boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode per-keypoint offsets into ``(N, 5, 2)`` landmark coordinates."""
    n_pts = distance.shape[1] // 2
    preds = np.empty((distance.shape[0], n_pts, 2), dtype=np.float32)
    preds[:, :, 0] = points[:, 0:1] + distance[:, 0::2]
    preds[:, :, 1] = points[:, 1:2] + distance[:, 1::2]
    return preds


__all__ = ["Detection", "BaseDetector", "nms", "distance2bbox", "distance2kps"]
