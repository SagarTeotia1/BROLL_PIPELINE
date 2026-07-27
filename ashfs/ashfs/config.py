"""Configuration loader for ASH-FS pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import yaml


# ---------------------------------------------------------------------------
# Nested config dataclasses — mirror config.yaml exactly
# ---------------------------------------------------------------------------

@dataclass
class ShotSegmentationConfig:
    window_size_range: list[int]       # [min, max]
    min_shot_duration_s: float
    max_shot_duration_s: float
    glrt_threshold: float              # cosine distance threshold for soft boundary detection
    candidate_fps: float               # fps to sample candidate frames from video
    flow_boundary_enabled: bool        # opt-in: optical-flow secondary shot boundary
    flow_threshold: float              # mean optical flow magnitude to trigger (default 3.0)
    flow_min_gap_frames: int           # ignore if within N frames of a hard boundary (default 10)
    gpu_shot_threshold: float          # frame-diff mean abs pixel delta (0–1) to declare hard cut


@dataclass
class TextChangeConfig:
    enabled: bool                      # opt-in — costs ~10ms/frame on CPU
    min_iou_change: float              # IoU drop below this → text changed (default 0.3)
    min_region_count_change: int       # abs change in MSER region count (default 5)


@dataclass
class ComplexityConfig:
    variance_weight: float
    motion_weight: float
    complexity_threshold_general: float
    complexity_threshold_talking_head: float
    complexity_threshold_fast_cut: float
    content_vertical: Literal["general", "talking_head", "fast_cut"]


@dataclass
class BudgetConfig:
    base_frames_per_shot: int
    max_frames_per_shot: int
    duration_bonus_interval_s: float   # +1 frame per this many seconds beyond threshold
    duration_bonus_start_s: float      # start adding bonus frames after this duration
    max_total_frames_per_video: Optional[int]  # None = no cap, fully adaptive
    duration_floor_interval_s: float   # min 1 keyframe per N seconds (floor, not bonus)
    duration_floor_enabled: bool       # toggle the duration floor on/off


@dataclass
class DualKeyframeConfig:
    lambda_common: float               # weight toward typicality for common frame
    alpha_unique: float                # weight toward atypicality for unique frame
    neighborhood_k: int               # temporal neighborhood size for volatility
    min_spacing_frames: int           # minimum gap between selected frames


@dataclass
class HierarchyConfig:
    chunk_size_shots: int              # shots per chunk node
    expansion_threshold_chunk: float
    expansion_threshold_scene: float
    expansion_threshold_sequence: float
    # Per-vertical chunk expansion thresholds — override expansion_threshold_chunk
    expansion_threshold_talking_head: float
    expansion_threshold_fast_cut: float
    expansion_threshold_general: float


@dataclass
class EmbeddingsConfig:
    dinov2_model: str                  # torch.hub model name
    dinov2_batch_size: int
    dinov2_device: Literal["cuda", "cpu"]
    siglip2_enabled: bool              # stored for future retrieval use
    siglip2_batch_size: int


@dataclass
class DenseSignalsConfig:
    enabled: bool                # opt-in — requires deepface
    batch_size: int              # frames to process per batch
    face_detector: str           # 'opencv' | 'retinaface' — opencv fastest for dense sampling


@dataclass
class FallbackConfig:
    uniform_fps: float
    max_fallback_frames: int


@dataclass
class Config:
    shot_segmentation: ShotSegmentationConfig
    complexity: ComplexityConfig
    budget: BudgetConfig
    dual_keyframe: DualKeyframeConfig
    hierarchy: HierarchyConfig
    embeddings: EmbeddingsConfig
    fallback: FallbackConfig
    text_change: TextChangeConfig
    dense_signals: DenseSignalsConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> Config:
    """Load config.yaml from *path* and return a validated Config dataclass."""
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    shot_seg = raw["shot_segmentation"]
    shot_segmentation = ShotSegmentationConfig(
        window_size_range=list(shot_seg["window_size_range"]),
        min_shot_duration_s=float(shot_seg["min_shot_duration_s"]),
        max_shot_duration_s=float(shot_seg["max_shot_duration_s"]),
        glrt_threshold=float(shot_seg["glrt_threshold"]),
        candidate_fps=float(shot_seg["candidate_fps"]),
        flow_boundary_enabled=bool(shot_seg.get("flow_boundary_enabled", False)),
        flow_threshold=float(shot_seg.get("flow_threshold", 3.0)),
        flow_min_gap_frames=int(shot_seg.get("flow_min_gap_frames", 10)),
        gpu_shot_threshold=float(shot_seg.get("gpu_shot_threshold", 0.15)),
    )

    tc = raw.get("text_change", {})
    text_change = TextChangeConfig(
        enabled=bool(tc.get("enabled", False)),
        min_iou_change=float(tc.get("min_iou_change", 0.3)),
        min_region_count_change=int(tc.get("min_region_count_change", 5)),
    )

    cplx = raw["complexity"]
    complexity = ComplexityConfig(
        variance_weight=float(cplx["variance_weight"]),
        motion_weight=float(cplx["motion_weight"]),
        complexity_threshold_general=float(cplx["complexity_threshold_general"]),
        complexity_threshold_talking_head=float(cplx["complexity_threshold_talking_head"]),
        complexity_threshold_fast_cut=float(cplx["complexity_threshold_fast_cut"]),
        content_vertical=cplx["content_vertical"],
    )

    bgt = raw["budget"]
    budget = BudgetConfig(
        base_frames_per_shot=int(bgt["base_frames_per_shot"]),
        max_frames_per_shot=int(bgt["max_frames_per_shot"]),
        duration_bonus_interval_s=float(bgt["duration_bonus_interval_s"]),
        duration_bonus_start_s=float(bgt["duration_bonus_start_s"]),
        max_total_frames_per_video=(
            int(bgt["max_total_frames_per_video"])
            if bgt.get("max_total_frames_per_video") is not None
            else None
        ),
        duration_floor_interval_s=float(bgt.get("duration_floor_interval_s", 5.0)),
        duration_floor_enabled=bool(bgt.get("duration_floor_enabled", True)),
    )

    dk = raw["dual_keyframe"]
    dual_keyframe = DualKeyframeConfig(
        lambda_common=float(dk["lambda_common"]),
        alpha_unique=float(dk["alpha_unique"]),
        neighborhood_k=int(dk["neighborhood_k"]),
        min_spacing_frames=int(dk["min_spacing_frames"]),
    )

    hier = raw["hierarchy"]
    hierarchy = HierarchyConfig(
        chunk_size_shots=int(hier["chunk_size_shots"]),
        expansion_threshold_chunk=float(hier["expansion_threshold_chunk"]),
        expansion_threshold_scene=float(hier["expansion_threshold_scene"]),
        expansion_threshold_sequence=float(hier["expansion_threshold_sequence"]),
        expansion_threshold_talking_head=float(hier.get("expansion_threshold_talking_head", 0.08)),
        expansion_threshold_fast_cut=float(hier.get("expansion_threshold_fast_cut", 0.30)),
        expansion_threshold_general=float(hier.get("expansion_threshold_general", 0.20)),
    )

    emb = raw["embeddings"]
    embeddings = EmbeddingsConfig(
        dinov2_model=str(emb["dinov2_model"]),
        dinov2_batch_size=int(emb["dinov2_batch_size"]),
        dinov2_device=str(emb["dinov2_device"]),
        siglip2_enabled=bool(emb["siglip2_enabled"]),
        siglip2_batch_size=int(emb["siglip2_batch_size"]),
    )

    fb = raw["fallback"]
    fallback = FallbackConfig(
        uniform_fps=float(fb["uniform_fps"]),
        max_fallback_frames=int(fb["max_fallback_frames"]),
    )

    ds = raw.get("dense_signals", {})
    dense_signals = DenseSignalsConfig(
        enabled=bool(ds.get("enabled", False)),
        batch_size=int(ds.get("batch_size", 32)),
        face_detector=str(ds.get("face_detector", "opencv")),
    )

    return Config(
        shot_segmentation=shot_segmentation,
        complexity=complexity,
        budget=budget,
        dual_keyframe=dual_keyframe,
        hierarchy=hierarchy,
        embeddings=embeddings,
        fallback=fallback,
        text_change=text_change,
        dense_signals=dense_signals,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VERTICAL_THRESHOLD_MAP: dict[str, str] = {
    "general": "complexity_threshold_general",
    "talking_head": "complexity_threshold_talking_head",
    "fast_cut": "complexity_threshold_fast_cut",
}


def get_complexity_threshold(cfg: Config) -> float:
    """Return the complexity threshold for the configured content vertical.

    Uses *cfg.complexity.content_vertical* to select among the three
    thresholds stored in *cfg.complexity*.  Raises *ValueError* for unknown
    vertical values so misconfiguration surfaces immediately at call-time.
    """
    vertical: str = cfg.complexity.content_vertical
    field_name = _VERTICAL_THRESHOLD_MAP.get(vertical)
    if field_name is None:
        raise ValueError(
            f"Unknown content_vertical {vertical!r}. "
            f"Expected one of: {sorted(_VERTICAL_THRESHOLD_MAP)}"
        )
    return float(getattr(cfg.complexity, field_name))
