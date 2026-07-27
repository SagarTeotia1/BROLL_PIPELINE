"""Streaming analysis pipeline: decoding, sampling, batching and orchestration."""

from pipeline.batching import BatchCollector  # noqa: F401
from pipeline.frame_source import DecodeThread, VideoMetadata, VideoSource, probe  # noqa: F401
from pipeline.sampler import AdaptiveSampler  # noqa: F401
from pipeline.scene_detect import SceneCutDetector  # noqa: F401
from pipeline.types import (  # noqa: F401
    FaceObservation,
    FrameResult,
    ProgressUpdate,
    RawFrame,
    SampledFrame,
)
from pipeline.worker import AnalysisPipeline, PipelineCallbacks  # noqa: F401

__all__ = [
    "AnalysisPipeline",
    "PipelineCallbacks",
    "VideoSource",
    "VideoMetadata",
    "DecodeThread",
    "probe",
    "AdaptiveSampler",
    "SceneCutDetector",
    "BatchCollector",
    "RawFrame",
    "SampledFrame",
    "FrameResult",
    "FaceObservation",
    "ProgressUpdate",
]
