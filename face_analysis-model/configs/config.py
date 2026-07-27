"""Central configuration.

Every tunable parameter of the pipeline lives here as a typed dataclass field.
The whole tree can be round-tripped to YAML so that a run is reproducible:

    cfg = AppConfig.load("configs/default.yaml")
    cfg.sampling.frame_stride = 2
    cfg.save("outputs/run_2024/config.yaml")

Nothing else in the codebase reads environment variables or hardcodes a path;
if you need to change behaviour, change it here.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

# Repository root = parent of the ``configs`` package.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    """Filesystem layout. Relative paths are resolved against the repo root."""

    weights_dir: str = "models/weights"
    cache_dir: str = "cache"
    output_dir: str = "outputs"
    assets_dir: str = "assets"
    cast_db: str = "cache/cast_database.db"
    cast_pickle: str = "cache/cast_database.pkl"
    log_dir: str = "outputs/logs"

    def resolve(self, value: str) -> Path:
        """Turn one of the fields into an absolute :class:`Path`."""
        p = Path(value)
        return p if p.is_absolute() else (ROOT_DIR / p)

    def ensure_dirs(self) -> None:
        """Create every directory this config points at."""
        for name in ("weights_dir", "cache_dir", "output_dir", "assets_dir", "log_dir"):
            self.resolve(getattr(self, name)).mkdir(parents=True, exist_ok=True)
        for name in ("cast_db", "cast_pickle"):
            self.resolve(getattr(self, name)).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class GPUConfig:
    """Device selection and low-level execution knobs."""

    device: str = "cuda"           # "cuda" | "cpu"
    device_id: int = 0
    # FP16 is requested per model and then *validated* against the FP32 reference on
    # the target provider (models/precision.py); a graph that breaks in half precision
    # silently keeps FP32 rather than emitting NaN.
    fp16: bool = True
    # Frames are staged and uploaded as NHWC uint8 (the decoder's native layout, one
    # byte per channel). ONNX Runtime's CUDA EP consumes NCHW, so the transpose happens
    # once on the GPU inside GpuNormalizer. Set False only for A/B measurements.
    channels_last: bool = True
    use_cuda_streams: bool = True  # overlap H2D copies with compute
    pinned_memory: bool = True     # page-locked staging buffers
    torch_compile: bool = False    # compile the fused preprocessing kernel
    use_tensorrt: bool = False     # enable TensorrtExecutionProvider when available
    trt_cache_dir: str = "cache/trt"
    # ORT arena: 6 GB card -> keep the arena from ballooning.
    gpu_mem_limit_mb: int = 3072
    cudnn_conv_algo_search: str = "HEURISTIC"  # EXHAUSTIVE | HEURISTIC | DEFAULT
    # Number of intra-op threads for any CPU fallback kernels.
    intra_op_threads: int = 4


@dataclass
class VideoConfig:
    """Decoding behaviour."""

    backend: str = "auto"          # "auto" | "pyav" | "opencv"
    hwaccel: bool = True           # try NVDEC (h264_cuvid / hevc_cuvid) first
    decode_threads: int = 4        # PyAV thread_count for the SW fallback
    queue_size: int = 64           # raw frame queue depth
    # Frames are analysed at this longest-side resolution; playback keeps native res.
    analysis_long_side: int = 1280


@dataclass
class SamplingConfig:
    """Frame sampling / adaptive stride policy."""

    frame_stride: int = 4          # analyse 1 out of every N frames (30fps -> 7.5fps)
    adaptive: bool = True
    min_stride: int = 1
    max_stride: int = 8
    # Scene cuts temporarily force dense sampling.
    scene_cut_enabled: bool = True
    scene_cut_threshold: float = 0.35   # 1 - histogram correlation
    scene_cut_boost_frames: int = 12    # sampled frames analysed at min_stride after a cut
    # Adaptive relaxation: after this many "calm" sampled frames the stride grows.
    calm_window: int = 30
    stride_relax_step: int = 1


@dataclass
class DetectorConfig:
    """SCRFD face detector."""

    model: str = "scrfd_10g_bnkps"
    input_size: int = 640          # square letterbox input
    score_threshold: float = 0.45
    nms_threshold: float = 0.4
    max_faces: int = 12
    batch_size: int = 4            # frames per detector batch


@dataclass
class QualityConfig:
    """Face-quality gates applied before any embedding work."""

    enabled: bool = True
    min_face_px: int = 48          # shortest bbox side, in analysis-resolution pixels
    min_blur_var: float = 22.0     # variance of Laplacian on the aligned crop
    # Nose position along the eye axis, expressed as max/min of the two halves:
    # 1.0 = frontal, 3.0 ~= 40 degrees of yaw, >6 is a hard profile. Roll-invariant.
    max_yaw_ratio: float = 3.0
    min_det_score: float = 0.50


@dataclass
class TrackerConfig:
    """ByteTrack association parameters.

    The ``match_*`` values are **maximum association costs**, where the cost is
    ``1 - IoU`` (score-fused in the high-confidence passes). Larger = more permissive:
    0.8 accepts an overlap as low as ~0.2 IoU, which is what lets a fast-moving face
    stay on the same track between sampled frames.
    """

    high_threshold: float = 0.55        # detections entering the first pass
    low_threshold: float = 0.15         # detections entering the recovery pass
    match_threshold: float = 0.80       # max cost, pass 1 (high-score detections)
    match_threshold_low: float = 0.50   # max cost, pass 2 (recovery)
    match_threshold_new: float = 0.70   # max cost when confirming a newborn track
    new_track_threshold: float = 0.60
    max_time_lost: int = 30             # sampled frames a track survives unmatched
    min_hits: int = 2                   # frames before a track is reported as confirmed


@dataclass
class RecognitionConfig:
    """ArcFace embedding + gallery matching."""

    model: str = "arcface_r50"
    input_size: int = 112
    batch_size: int = 16
    similarity_threshold: float = 0.38   # cosine, below -> "Unknown"
    margin_threshold: float = 0.06       # best-vs-second-best margin for a confident lock
    # Cast-only mode: emotion is computed *only* for faces that matched a registered
    # actor. Unrecognised people (extras, crowd, background) are still tracked - the
    # tracker needs them to keep boxes apart - but never reach HSEmotion and never
    # appear in the UI or the timeline. This is both the product behaviour that is
    # wanted and a large compute saving on busy shots.
    cast_only: bool = True
    # Re-identify a locked track every N sampled frames (drift / occlusion recovery).
    reid_interval: int = 45
    # A track is considered identified after this many agreeing votes.
    vote_window: int = 5
    min_votes: int = 3


@dataclass
class EmotionConfig:
    """HSEmotion classifier + temporal smoothing."""

    model: str = "hsemotion_enet_b0_8"
    input_size: int = 224
    batch_size: int = 16
    confidence_threshold: float = 0.40
    smoothing: str = "hybrid"      # "majority" | "ema" | "hybrid"
    window: int = 7                # sliding window length (sampled frames)
    min_agree: int = 4             # votes required inside the window to switch
    ema_alpha: float = 0.45        # EMA factor on the probability vector
    switch_confidence: float = 0.55  # a single high-confidence frame may switch early


@dataclass
class TimelineConfig:
    """Event assembly rules."""

    min_event_duration: float = 0.30   # events shorter than this are merged away
    merge_gap: float = 0.60            # same emotion resuming within this gap -> one event
    close_track_after: float = 1.50    # seconds without observation closes an open event
    include_unknown: bool = False      # emit events for unrecognised faces


@dataclass
class PipelineConfig:
    """Queue depths and batching latency."""

    detect_queue: int = 16
    track_queue: int = 32
    infer_queue: int = 32
    result_queue: int = 64
    # A partially filled batch is flushed after this long so latency stays bounded.
    max_batch_latency_ms: float = 25.0


@dataclass
class GUIConfig:
    """Player + panels."""

    window_title: str = "Face Recognition & Emotion Timeline"
    playback_fps_cap: int = 60
    # The player is a monitor, not the product: it is capped to a narrow column so the
    # analysis panels own the window.
    preview_width: int = 480
    video_panel_width: int = 420
    face_thumb_size: int = 72
    max_face_cards: int = 6
    stats_interval_ms: int = 500
    overlay_boxes: bool = True
    show_unknown_faces: bool = False   # draw/list faces that matched no registered actor
    max_event_rows: int = 500          # live event table cap


@dataclass
class LoggingConfig:
    level: str = "INFO"
    to_file: bool = True
    profile: bool = True           # per-stage FPS / queue telemetry
    profile_interval_s: float = 2.0


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    """Root of the configuration tree."""

    paths: PathConfig = field(default_factory=PathConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | os.PathLike[str]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: Optional[str | os.PathLike[str]] = None) -> "AppConfig":
        """Load YAML config; missing file or missing keys fall back to defaults."""
        if path is None:
            path = ROOT_DIR / "configs" / "default.yaml"
        p = Path(path)
        if not p.exists():
            cfg = cls()
            cfg.paths.ensure_dirs()
            return cfg
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(data)
        cfg.paths.ensure_dirs()
        return cfg

    # -- convenience --------------------------------------------------------
    def override(self, dotted: Dict[str, Any]) -> "AppConfig":
        """Apply ``{"sampling.frame_stride": 2}`` style overrides in place."""
        for key, value in dotted.items():
            node: Any = self
            parts = key.split(".")
            for part in parts[:-1]:
                node = getattr(node, part)
            leaf = parts[-1]
            if not hasattr(node, leaf):
                raise KeyError(f"Unknown config key: {key}")
            current = getattr(node, leaf)
            setattr(node, leaf, type(current)(value) if current is not None else value)
        return self

    def describe(self) -> str:
        """Compact human-readable summary used in logs."""
        return (
            f"device={self.gpu.device}:{self.gpu.device_id} fp16={self.gpu.fp16} "
            f"stride={self.sampling.frame_stride} det_bs={self.detector.batch_size} "
            f"rec_bs={self.recognition.batch_size} emo_bs={self.emotion.batch_size} "
            f"sim_thr={self.recognition.similarity_threshold}"
        )


# ---------------------------------------------------------------------------
# Recursive dataclass construction (tolerant of unknown / missing keys)
# ---------------------------------------------------------------------------
def _build(cls: Type[T], data: Dict[str, Any]) -> T:
    """Instantiate ``cls`` from ``data``, recursing into nested dataclasses.

    ``from __future__ import annotations`` turns ``field.type`` into a string, so
    the nested type is recovered from ``default_factory`` (every nested field in
    this module declares one) rather than from the annotation.
    """
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        factory = f.default_factory  # type: ignore[misc]
        if (
            isinstance(value, dict)
            and factory is not dataclasses.MISSING
            and isinstance(factory, type)
            and dataclasses.is_dataclass(factory)
        ):
            kwargs[f.name] = _build(factory, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "ROOT_DIR",
    "AppConfig",
    "PathConfig",
    "GPUConfig",
    "VideoConfig",
    "SamplingConfig",
    "DetectorConfig",
    "QualityConfig",
    "TrackerConfig",
    "RecognitionConfig",
    "EmotionConfig",
    "TimelineConfig",
    "PipelineConfig",
    "GUIConfig",
    "LoggingConfig",
]
