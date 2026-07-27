"""Face detection: SCRFD plus the face-quality gate."""

from detector.base import BaseDetector, Detection  # noqa: F401
from detector.face_quality import FaceQualityEstimator, QualityReport  # noqa: F401
from detector.scrfd import SCRFDDetector  # noqa: F401

__all__ = [
    "BaseDetector",
    "Detection",
    "SCRFDDetector",
    "FaceQualityEstimator",
    "QualityReport",
]
