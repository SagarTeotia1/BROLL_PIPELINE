"""Face quality gating.

Embedding a blurry, tiny or strongly-profile face wastes GPU time and, worse, pollutes
the identity vote with a low-information vector. This module scores each detection and
lets the pipeline drop the bad ones *before* ArcFace/HSEmotion run.

The score is a product of four sub-scores in ``[0, 1]`` (size, sharpness, pose,
exposure) so a single bad axis is enough to sink a face - which is the behaviour we
want for a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from configs.config import QualityConfig
from detector.base import Detection
from utils.image_ops import blur_score, brightness_score, yaw_ratio


@dataclass
class QualityReport:
    """Per-face quality breakdown, useful for debugging and for the GUI tooltip."""

    score: float
    size_score: float
    sharpness_score: float
    pose_score: float
    exposure_score: float
    passed: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "size": round(self.size_score, 3),
            "sharpness": round(self.sharpness_score, 3),
            "pose": round(self.pose_score, 3),
            "exposure": round(self.exposure_score, 3),
            "passed": self.passed,
            "reason": self.reason,
        }


class FaceQualityEstimator:
    """Scores and filters detections according to :class:`QualityConfig`."""

    def __init__(self, config: QualityConfig) -> None:
        self.cfg = config

    # -- scoring ------------------------------------------------------------
    def evaluate(
        self,
        detection: Detection,
        aligned_crop: Optional[np.ndarray] = None,
    ) -> QualityReport:
        """Score one detection.

        Args:
            detection: candidate face (bbox in analysis-frame pixels).
            aligned_crop: the 112x112 aligned face; when omitted the sharpness and
                exposure terms are skipped (they need pixels).
        """
        if not self.cfg.enabled:
            return QualityReport(1.0, 1.0, 1.0, 1.0, 1.0, True)

        reasons: List[str] = []

        # --- size ----------------------------------------------------------
        min_side = detection.min_side
        size_score = float(np.clip(min_side / (self.cfg.min_face_px * 2.0), 0.0, 1.0))
        if min_side < self.cfg.min_face_px:
            reasons.append(f"tiny({min_side:.0f}px)")

        # --- detector confidence -------------------------------------------
        if detection.score < self.cfg.min_det_score:
            reasons.append(f"lowconf({detection.score:.2f})")

        # --- pose ----------------------------------------------------------
        pose_score = 1.0
        if detection.landmarks is not None:
            ratio = yaw_ratio(detection.landmarks)
            # ratio 1.0 = frontal, max_yaw_ratio = gate
            pose_score = float(np.clip(1.0 - (ratio - 1.0) / (self.cfg.max_yaw_ratio - 1.0), 0.0, 1.0))
            if ratio > self.cfg.max_yaw_ratio:
                reasons.append(f"profile({ratio:.1f})")

        # --- sharpness / exposure ------------------------------------------
        sharpness_score = 1.0
        exposure_score = 1.0
        if aligned_crop is not None and aligned_crop.size:
            blur = blur_score(aligned_crop)
            sharpness_score = float(np.clip(blur / (self.cfg.min_blur_var * 4.0), 0.0, 1.0))
            if blur < self.cfg.min_blur_var:
                reasons.append(f"blurry({blur:.0f})")
            bright = brightness_score(aligned_crop)
            # Penalise crushed blacks / blown highlights symmetrically around 0.5.
            exposure_score = float(np.clip(1.0 - abs(bright - 0.5) * 2.4, 0.0, 1.0))
            if bright < 0.08 or bright > 0.94:
                reasons.append(f"exposure({bright:.2f})")

        score = float(size_score * sharpness_score * pose_score * max(exposure_score, 0.15))
        passed = not reasons
        return QualityReport(
            score=score,
            size_score=size_score,
            sharpness_score=sharpness_score,
            pose_score=pose_score,
            exposure_score=exposure_score,
            passed=passed,
            reason=",".join(reasons),
        )

    # -- batch helpers ------------------------------------------------------
    def filter(
        self,
        detections: Sequence[Detection],
        crops: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> Tuple[List[Detection], List[QualityReport]]:
        """Return only the detections that pass, plus every report (aligned by index).

        The detections that pass have their ``quality`` field populated.
        """
        kept: List[Detection] = []
        reports: List[QualityReport] = []
        for i, det in enumerate(detections):
            crop = crops[i] if crops is not None and i < len(crops) else None
            report = self.evaluate(det, crop)
            det.quality = report.score
            reports.append(report)
            if report.passed:
                kept.append(det)
        return kept, reports

    def rank(self, detections: Sequence[Detection]) -> List[Detection]:
        """Sort detections best-first (quality if scored, otherwise area x score)."""
        def key(d: Detection) -> float:
            if d.quality >= 0:
                return d.quality
            return d.area * d.score
        return sorted(detections, key=key, reverse=True)


__all__ = ["QualityReport", "FaceQualityEstimator"]
