"""GPU-accelerated shot boundary detection via PyTorch CUDA frame-difference.

Primary path: PyTorch CUDA
  - Reads video frames in large batches directly to GPU tensor
  - Computes mean absolute pixel diff between consecutive frames
  - Marks frames where diff > threshold as shot boundaries
  - No TensorFlow / TransNetV2 dependency
  - ~20-50x faster than PySceneDetect on L4 GPU

Fallback: PySceneDetect (CPU)
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GPUShotDetector:
    """Shot boundary detector using PyTorch CUDA frame-diff (GPU) with PySceneDetect fallback.

    Parameters
    ----------
    threshold:
        Mean absolute pixel difference (0–1 normalised) to declare a boundary.
        0.15 catches hard cuts cleanly for talking-head / podcast content.
        Raise to 0.25 for fast-cut / action content.
    pyscenedetect_threshold:
        ContentDetector threshold used in the CPU fallback.
    min_shot_duration_s:
        Retained for API compat with shot_segmenter — not applied here;
        enforcement happens in ShotSegmenter._enforce_duration_constraints.
    batch_size:
        Frames per GPU batch.  2048 is safe for L4 (24 GB) at 320×180.
        Reduce to 512 if running on smaller VRAM (e.g. T4 16 GB).
    resize_wh:
        (width, height) to resize frames before diff computation.
        Smaller = faster; 320×180 is more than enough for cut detection.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        pyscenedetect_threshold: float = 27.0,
        min_shot_duration_s: float = 0.5,
        batch_size: int = 2048,
        resize_wh: tuple[int, int] = (320, 180),
    ) -> None:
        self._threshold = threshold
        self._psd_threshold = pyscenedetect_threshold
        self._min_shot_s = min_shot_duration_s
        self._batch_size = batch_size
        self._resize_wh = resize_wh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_boundaries(self, video_path: str) -> list[int]:
        """Return sorted list of shot-boundary frame indices (excluding frame 0).

        Tries PyTorch CUDA first; falls back to PySceneDetect on any error.
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(
                    "Using PyTorch CUDA shot detection on %s (batch=%d, resize=%s).",
                    torch.cuda.get_device_name(0), self._batch_size, self._resize_wh,
                )
                return self._detect_pytorch_cuda(video_path)
            else:
                logger.info("CUDA unavailable — using PySceneDetect (CPU).")
        except Exception as exc:
            logger.warning(
                "PyTorch GPU shot detection failed (%s) — falling back to PySceneDetect.", exc,
                exc_info=True,
            )
        return self._detect_pyscenedetect(video_path)

    # ------------------------------------------------------------------
    # PyTorch CUDA path
    # ------------------------------------------------------------------

    def _detect_pytorch_cuda(self, video_path: str) -> list[int]:
        import torch

        device = torch.device("cuda")
        W, H = self._resize_wh

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path!r}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Adaptive decode stride: sample at ~10fps for shot detection.
        # Hard cuts are instant (detectable even at 10fps).
        # Slow zooms/pans change over seconds — 10fps is more than enough.
        # This cuts decode time by 3x on 30fps footage, 6x on 60fps.
        target_sample_fps = 10.0
        stride = max(1, round(video_fps / target_sample_fps))
        effective_fps = video_fps / stride

        logger.info(
            "PyTorch CUDA shot detection: %d frames, stride=%d (%.1f→%.1ffps), resize=%dx%d.",
            total_frames, stride, video_fps, effective_fps, W, H,
        )

        all_diffs: list[float] = []
        sampled_frame_indices: list[int] = []  # actual frame index for each diff sample
        raw_batch: list[np.ndarray] = []
        batch_start_indices: list[int] = []
        prev_last: Optional[np.ndarray] = None
        frame_idx = 0

        def _flush(frames: list[np.ndarray], prev: Optional[np.ndarray]) -> list[float]:
            if prev is not None:
                frames = [prev] + frames
            arr = np.stack(frames, axis=0).astype(np.float32)
            t = torch.from_numpy(arr).div_(255.0).to(device)
            diffs = (t[1:] - t[:-1]).abs().mean(dim=(1, 2, 3))
            return diffs.cpu().numpy().tolist()

        while True:
            if stride > 1:
                # grab() skips without decoding — very fast
                for _ in range(stride - 1):
                    grabbed = cap.grab()
                    if not grabbed:
                        break
                    frame_idx += 1

            ok, frame = cap.read()
            if not ok:
                if raw_batch:
                    all_diffs.extend(_flush(raw_batch, prev_last))
                break

            small = cv2.resize(frame, (W, H))
            raw_batch.append(small)
            sampled_frame_indices.append(frame_idx)
            frame_idx += 1

            if len(raw_batch) >= self._batch_size:
                all_diffs.extend(_flush(raw_batch, prev_last))
                prev_last = raw_batch[-1]
                raw_batch = []

        cap.release()

        # Map diff index back to actual frame index (boundary at the second frame of the pair).
        # sampled_frame_indices[i+1] is the frame where the cut happened.
        boundaries: list[int] = []
        for i, d in enumerate(all_diffs):
            if d > self._threshold and i + 1 < len(sampled_frame_indices):
                boundaries.append(sampled_frame_indices[i + 1])

        logger.info(
            "PyTorch CUDA detected %d shot boundaries (threshold=%.3f, stride=%d).",
            len(boundaries), self._threshold, stride,
        )
        return sorted(boundaries)

    # ------------------------------------------------------------------
    # PySceneDetect fallback (CPU)
    # ------------------------------------------------------------------

    def _detect_pyscenedetect(self, video_path: str) -> list[int]:
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError as exc:
            raise ImportError(
                "PySceneDetect not installed. pip install scenedetect"
            ) from exc

        logger.info("Running PySceneDetect CPU fallback.")
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self._psd_threshold))
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()

        boundaries = [
            start_tc.get_frames()
            for i, (start_tc, _) in enumerate(scene_list)
            if i > 0
        ]
        logger.info("PySceneDetect found %d boundaries.", len(boundaries))
        return sorted(boundaries)
