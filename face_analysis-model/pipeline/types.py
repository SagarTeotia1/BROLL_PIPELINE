"""Message types exchanged between pipeline stages.

Everything that travels through a queue is a frozen-ish dataclass rather than a tuple,
so a stage can be modified without breaking the unpacking in five other files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from detector.base import Detection


@dataclass
class RawFrame:
    """A decoded frame as it leaves the decoder.

    ``scale`` maps frame coordinates back to *source* pixels. It is ``1.0`` for a
    native-resolution decode and ``> 1`` when the decoder downscaled during colour
    conversion (see :class:`pipeline.frame_source.VideoSource`), which is much cheaper
    than converting at full size and resizing afterwards.
    """

    index: int                 # frame number in the source
    timestamp: float           # presentation time in seconds
    image: np.ndarray          # BGR
    is_keyframe: bool = False
    scale: float = 1.0         # frame coords * scale -> source coords

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]


@dataclass
class SampledFrame:
    """A frame selected for analysis, already scaled to the analysis resolution."""

    index: int
    timestamp: float
    image: np.ndarray          # BGR at analysis resolution
    scale: float               # analysis -> source coordinate factor
    scene_cut: bool = False
    stride_used: int = 1


@dataclass
class FaceObservation:
    """One face in one analysed frame, after tracking / recognition / emotion."""

    track_id: int
    bbox: np.ndarray
    det_score: float
    quality: float = 0.0
    actor_id: int = -1
    actor_name: str = "Unknown"
    similarity: float = 0.0
    locked: bool = False
    emotion: str = "Neutral"
    emotion_confidence: float = 0.0
    raw_emotion: str = ""
    raw_emotion_confidence: float = 0.0
    emotion_changed: bool = False
    thumbnail: Optional[np.ndarray] = None   # small BGR crop for the GUI panel

    def as_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "actor": self.actor_name,
            "similarity": round(self.similarity, 4),
            "emotion": self.emotion,
            "confidence": round(self.emotion_confidence, 4),
            "quality": round(self.quality, 3),
        }


@dataclass
class FrameResult:
    """Everything the pipeline learned about one analysed frame."""

    index: int
    timestamp: float
    faces: List[FaceObservation] = field(default_factory=list)
    scene_cut: bool = False
    stride_used: int = 1
    latency_ms: float = 0.0
    frame_size: tuple[int, int] = (0, 0)     # (height, width) at analysis resolution

    @property
    def num_faces(self) -> int:
        return len(self.faces)


@dataclass
class ProgressUpdate:
    """Periodic pipeline telemetry for the GUI status bar."""

    frames_decoded: int = 0
    frames_analysed: int = 0
    total_frames: int = 0
    timestamp: float = 0.0
    duration: float = 0.0
    decode_fps: float = 0.0
    analysis_fps: float = 0.0
    detect_fps: float = 0.0
    recognition_fps: float = 0.0
    emotion_fps: float = 0.0
    inference_fps: float = 0.0
    realtime_factor: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    queue_depths: Dict[str, int] = field(default_factory=dict)
    events: int = 0
    expression_changes: int = 0
    cast_faces: int = 0            # faces matched to a registered actor so far
    elapsed: float = 0.0           # wall-clock seconds since analysis started
    finished: bool = False

    @property
    def percent(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(100.0, 100.0 * self.frames_decoded / self.total_frames)


@dataclass
class RecognitionRequest:
    """A face queued for ArcFace embedding."""

    track_id: int
    frame_index: int
    timestamp: float
    crop: np.ndarray           # aligned 112x112 BGR


@dataclass
class EmotionRequest:
    """A face queued for HSEmotion classification."""

    track_id: int
    frame_index: int
    timestamp: float
    crop: np.ndarray           # loose crop at the classifier's input size


__all__ = [
    "RawFrame",
    "SampledFrame",
    "FaceObservation",
    "FrameResult",
    "ProgressUpdate",
    "RecognitionRequest",
    "EmotionRequest",
    "Detection",
]
