"""Frame sampling policy: fixed stride, scene-cut boost and adaptive relaxation.

Default behaviour is "analyse 1 frame in 4" (30 fps source -> 7.5 analysed fps), which
is the single biggest throughput lever in the system. On top of that:

* **Scene-cut boost** - right after a shot boundary the stride drops to
  ``sampling.min_stride`` for ``scene_cut_boost_frames`` analysed frames so the new shot
  is characterised immediately, then decays back.
* **Adaptive relaxation** - if nothing interesting happened for ``calm_window`` analysed
  frames (no new track, no identity change, no emotion change), the stride grows by
  ``stride_relax_step`` up to ``max_stride``. A static interview shot ends up analysed
  at ~3.75 fps; a busy scene stays at 7.5 fps or better.

The sampler also owns the analysis-resolution downscale: detection does not need 4K, and
capping the long side at ``video.analysis_long_side`` cuts detector cost dramatically
while the GUI still plays the native-resolution video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


from configs.config import SamplingConfig, VideoConfig
from pipeline.scene_detect import SceneCutDetector
from pipeline.types import RawFrame, SampledFrame
from utils.image_ops import resize_long_side
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class SamplerStats:
    """Counters exposed for logging and the benchmark report."""

    frames_seen: int = 0
    frames_sampled: int = 0
    scene_cuts: int = 0
    boosted_frames: int = 0
    current_stride: int = 4

    @property
    def sample_ratio(self) -> float:
        return self.frames_sampled / self.frames_seen if self.frames_seen else 0.0

    def as_dict(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "frames_sampled": self.frames_sampled,
            "sample_ratio": round(self.sample_ratio, 4),
            "scene_cuts": self.scene_cuts,
            "boosted_frames": self.boosted_frames,
            "final_stride": self.current_stride,
        }


class AdaptiveSampler:
    """Decides which decoded frames become analysis frames."""

    def __init__(self, sampling: SamplingConfig, video: VideoConfig) -> None:
        self.cfg = sampling
        self.vcfg = video
        self.base_stride = max(1, int(sampling.frame_stride))
        self.stride = self.base_stride
        self.scene = (
            SceneCutDetector(
                threshold=sampling.scene_cut_threshold,
                min_gap=max(2, self.base_stride),
            )
            if sampling.scene_cut_enabled
            else None
        )
        self._boost_remaining = 0
        self._calm_counter = 0
        self._next_index = 0
        self.stats = SamplerStats(current_stride=self.stride)

    # -- main entry point ---------------------------------------------------
    def consider(self, frame: RawFrame) -> Optional[SampledFrame]:
        """Decide whether ``frame`` should be analysed.

        Returns a :class:`SampledFrame` (already downscaled) or ``None``.
        """
        self.stats.frames_seen += 1
        is_cut = False
        if self.scene is not None:
            info = self.scene.update(frame.image, frame.index)
            is_cut = info.is_cut
            if is_cut:
                self.stats.scene_cuts += 1
                self._boost_remaining = self.cfg.scene_cut_boost_frames
                self.stride = max(self.cfg.min_stride, 1)
                self._calm_counter = 0
                self._next_index = frame.index  # analyse this frame now

        if frame.index < self._next_index:
            return None

        stride_used = self.stride
        self._next_index = frame.index + max(1, self.stride)
        self.stats.frames_sampled += 1
        self.stats.current_stride = self.stride

        if self._boost_remaining > 0:
            self._boost_remaining -= 1
            self.stats.boosted_frames += 1
            if self._boost_remaining == 0:
                self.stride = self.base_stride

        # Usually a no-op: the decoder already emits analysis resolution. It still runs
        # for OpenCV/pass-through decodes and for sources smaller than the cap.
        image, scale = resize_long_side(frame.image, self.vcfg.analysis_long_side)
        return SampledFrame(
            index=frame.index,
            timestamp=frame.timestamp,
            image=image,
            # Compose the decoder's downscale with any extra resize applied here so the
            # factor always maps analysis pixels back to *source* pixels.
            scale=frame.scale * (1.0 / scale if scale else 1.0),
            scene_cut=is_cut,
            stride_used=stride_used,
        )

    # -- adaptive feedback --------------------------------------------------
    def report_activity(self, active: bool) -> None:
        """Tell the sampler whether the last analysed frame was 'interesting'.

        ``active`` should be True when a track was born, an identity changed or an
        emotion switched. Sustained inactivity relaxes the stride; any activity snaps
        it back to the configured base.
        """
        if not self.cfg.adaptive:
            return
        if active:
            self._calm_counter = 0
            if self.stride > self.base_stride:
                self.stride = self.base_stride
                log.debug("Sampler: activity detected, stride back to %d", self.stride)
            return

        self._calm_counter += 1
        if (
            self._boost_remaining == 0
            and self._calm_counter >= self.cfg.calm_window
            and self.stride < self.cfg.max_stride
        ):
            self.stride = min(self.cfg.max_stride, self.stride + self.cfg.stride_relax_step)
            self._calm_counter = 0
            log.debug("Sampler: calm scene, stride relaxed to %d", self.stride)

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        self.stride = self.base_stride
        self._boost_remaining = 0
        self._calm_counter = 0
        self._next_index = 0
        self.stats = SamplerStats(current_stride=self.stride)
        if self.scene is not None:
            self.scene.reset()

    def expected_processed_fps(self, source_fps: float) -> float:
        """Analysed FPS implied by the *current* stride."""
        return source_fps / max(1, self.stride)

    def actual_processed_fps(self, source_fps: float) -> float:
        """Analysed FPS actually achieved, from the sampling ratio."""
        return source_fps * self.stats.sample_ratio


__all__ = ["AdaptiveSampler", "SamplerStats"]
