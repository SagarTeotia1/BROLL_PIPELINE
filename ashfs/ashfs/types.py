"""Shared types for ASH-FS pipeline."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class DenseSignals:
    """Per-frame emotion and identity signals at candidate-frame density.

    Stored as a dense time-series independent of keyframe selection so that
    fast-changing signals (emotion flickers, speaker changes) are not forced
    onto the sparse VLM keyframe grid.
    """
    frame_indices: list          # absolute frame indices (same as candidate frames)
    emotions: list               # list[dict | None] — {'dominant_emotion': str, 'scores': dict}
    identities: list             # list[str | None] — 'speaker_N' cluster label or None
    extraction_fps: float        # fps at which signals were extracted
    extractor_available: bool    # False if deepface not installed


@dataclass
class Shot:
    start_frame: int
    end_frame: int
    duration_s: float
    fps: float = 0.0

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class ShotEmbeddings:
    shot: Shot
    frame_indices: list  # absolute frame indices in video
    embeddings: np.ndarray  # shape: (N, 768) DINOv2


@dataclass
class ScoredShot:
    shot: Shot
    frame_indices: list
    embeddings: np.ndarray
    variance_score: float
    motion_score: float
    complexity: float

    @property
    def frame_count(self) -> int:
        """Number of candidate frames available in this shot."""
        return len(self.frame_indices)


@dataclass
class BudgetedShot:
    scored_shot: ScoredShot
    budget: int  # number of keyframes to select


@dataclass
class SelectedKeyframes:
    shot: Shot
    frame_indices: list           # absolute frame indices selected
    embeddings: np.ndarray        # DINOv2 embeddings of selected frames
    siglip_embeddings: Optional[np.ndarray] = None  # stored for future retrieval
    frame_roles: list = field(default_factory=list)  # 'medoid','common','unique','extra'


@dataclass
class TreeNode:
    level: str  # 'shot' | 'chunk' | 'scene' | 'sequence' | 'video'
    node_id: str
    children: list
    keyframes: list             # SelectedKeyframes from shots
    aggregate_embedding: Optional[np.ndarray] = None
    expand: bool = True         # True=expand children, False=summarize
    diversity_score: float = 0.0
