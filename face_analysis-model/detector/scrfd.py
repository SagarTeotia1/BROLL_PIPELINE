"""SCRFD face detector on ONNX Runtime with batched GPU preprocessing.

SCRFD (Sample and Computation Redistribution for Face Detection) is an anchor-based
single-stage detector; the ``*_bnkps`` variants also regress five landmarks, which is
exactly what ArcFace alignment needs, so detection and alignment share one forward pass.

Performance notes
-----------------
* Only **uint8** pixels cross PCIe. Letterboxing happens on the CPU (cheap, SIMD in
  OpenCV), the batch is staged in **pinned** memory, uploaded ``non_blocking``, and
  normalised on the GPU by :class:`utils.image_ops.GpuNormalizer`.
* Anchor centres are generated once per input size and cached; decoding a 640x640
  batch is pure vectorised numpy.
* The graph is run through :class:`models.onnx_engine.OnnxEngine`, which binds the
  input tensor by pointer (no host round-trip).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from configs.config import AppConfig
from detector.base import BaseDetector, Detection, distance2bbox, distance2kps, nms
from models.model_zoo import ensure_model, get_spec
from models.onnx_engine import OnnxEngine
from utils.gpu import PinnedRing
from utils.image_ops import GpuNormalizer, letterbox
from utils.logging_utils import get_logger

log = get_logger(__name__)

# Output-count -> (fmc, strides, num_anchors, use_kps). Matches InsightFace exports.
_HEAD_LAYOUTS: Dict[int, Tuple[int, Tuple[int, ...], int, bool]] = {
    6: (3, (8, 16, 32), 2, False),
    9: (3, (8, 16, 32), 2, True),
    10: (5, (8, 16, 32, 64, 128), 1, False),
    15: (5, (8, 16, 32, 64, 128), 1, True),
}


class SCRFDDetector(BaseDetector):
    """Batched SCRFD detector.

    Args:
        config: full application config (reads ``detector`` and ``gpu`` sections).
        device: torch device the engine runs on.
        stream_handle: optional CUDA stream pointer shared with ORT.
    """

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        stream_handle: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self.dcfg = config.detector
        self.device = device
        spec = get_spec(self.dcfg.model)
        weights_dir = config.paths.resolve(config.paths.weights_dir)
        model_path = ensure_model(self.dcfg.model, weights_dir)

        self.engine = OnnxEngine(
            model_path,
            device=device,
            fp16=config.gpu.fp16,
            fp16_cache_dir=weights_dir / "fp16",
            gpu_mem_limit_mb=config.gpu.gpu_mem_limit_mb,
            cudnn_conv_algo_search=config.gpu.cudnn_conv_algo_search,
            intra_op_threads=config.gpu.intra_op_threads,
            use_tensorrt=config.gpu.use_tensorrt,
            trt_cache_dir=config.paths.resolve(config.gpu.trt_cache_dir),
            user_compute_stream=stream_handle,
            name="scrfd",
            probe_spatial=160,   # multiple of 32; keeps the batch probe cheap
        )

        n_out = len(self.engine.output_names)
        if n_out not in _HEAD_LAYOUTS:
            raise RuntimeError(
                f"unexpected SCRFD head: {n_out} outputs (supported: {sorted(_HEAD_LAYOUTS)})"
            )
        self.fmc, self.strides, self.num_anchors, self.use_kps = _HEAD_LAYOUTS[n_out]

        self.input_size = int(self.dcfg.input_size)
        # The published SCRFD export is batch-1 only; ``effective_batch`` reflects the
        # validated verdict from models.batch_patch instead of assuming.
        self.batch_size = self.engine.effective_batch(int(self.dcfg.batch_size))
        if self.batch_size < int(self.dcfg.batch_size):
            log.info(
                "Detector batch clamped from %d to %d (graph limit); frames are "
                "processed sequentially and the per-face models still batch across frames",
                self.dcfg.batch_size, self.batch_size,
            )
        self.normalizer = GpuNormalizer(
            mean=spec.mean, std=spec.std, device=device,
            dtype=self.engine.input_dtype,
            bgr_to_rgb=spec.bgr_to_rgb,
            compile_kernel=config.gpu.torch_compile,
        )
        self._pinned = PinnedRing(
            slots=3, max_batch=self.batch_size,
            elem_shape=(self.input_size, self.input_size, 3),
            dtype=torch.uint8, enabled=config.gpu.pinned_memory,
        )
        self._anchor_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._input_name = self.engine.input_names[0]
        log.info(
            "SCRFD ready: %s, %d outputs, strides=%s, anchors=%d, kps=%s, batch=%d",
            model_path.name, n_out, self.strides, self.num_anchors, self.use_kps,
            self.batch_size,
        )

    # -- public API ---------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect faces in one BGR frame."""
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Detect faces across a list of BGR frames.

        Frames are letterboxed to a common square input, so they may have different
        native resolutions. Batches larger than ``detector.batch_size`` are split.
        """
        if not images:
            return []
        results: List[List[Detection]] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            results.extend(self._detect_chunk(chunk))
        return results

    # -- internals ----------------------------------------------------------
    def _detect_chunk(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        n = len(images)
        size = self.input_size
        host = self._pinned.acquire(n)  # (n, size, size, 3) uint8 pinned
        scales: List[float] = []
        host_np = host.numpy()
        for i, img in enumerate(images):
            padded, scale, _ = letterbox(img, size)
            host_np[i] = padded
            scales.append(scale)

        device_u8 = host.to(self.device, non_blocking=True)
        blob = self.normalizer(device_u8)
        outputs = self.engine.run({self._input_name: blob})
        return self._decode_batch(outputs, n, scales)

    def _decode_batch(
        self, outputs: List[np.ndarray], batch: int, scales: Sequence[float]
    ) -> List[List[Detection]]:
        threshold = self.dcfg.score_threshold
        results: List[List[Detection]] = []

        # Per-level slices: outputs are flattened batch-major by the exported graph.
        level_scores = [np.asarray(outputs[i], dtype=np.float32) for i in range(self.fmc)]
        level_bboxes = [
            np.asarray(outputs[i + self.fmc], dtype=np.float32) for i in range(self.fmc)
        ]
        level_kps = (
            [np.asarray(outputs[i + 2 * self.fmc], dtype=np.float32) for i in range(self.fmc)]
            if self.use_kps
            else []
        )

        for b in range(batch):
            boxes_all: List[np.ndarray] = []
            scores_all: List[np.ndarray] = []
            kps_all: List[np.ndarray] = []
            for level, stride in enumerate(self.strides):
                per_img = level_scores[level].shape[0] // batch
                sl = slice(b * per_img, (b + 1) * per_img)
                scores = level_scores[level][sl].reshape(-1)
                keep = np.nonzero(scores >= threshold)[0]
                if keep.size == 0:
                    continue
                anchors = self._anchors(stride)
                bbox_pred = level_bboxes[level][sl].reshape(per_img, -1)[keep] * stride
                boxes = distance2bbox(anchors[keep], bbox_pred)
                boxes_all.append(boxes)
                scores_all.append(scores[keep])
                if self.use_kps:
                    kps_pred = level_kps[level][sl].reshape(per_img, -1)[keep] * stride
                    kps_all.append(distance2kps(anchors[keep], kps_pred))

            if not boxes_all:
                results.append([])
                continue

            boxes = np.concatenate(boxes_all, axis=0)
            scores = np.concatenate(scores_all, axis=0)
            kps = np.concatenate(kps_all, axis=0) if kps_all else None

            order = scores.argsort()[::-1]
            boxes, scores = boxes[order], scores[order]
            if kps is not None:
                kps = kps[order]
            keep_idx = nms(boxes, scores, self.dcfg.nms_threshold)[: self.dcfg.max_faces]

            inv = 1.0 / scales[b]
            dets: List[Detection] = []
            for i in keep_idx:
                dets.append(
                    Detection(
                        bbox=boxes[i] * inv,
                        score=float(scores[i]),
                        landmarks=None if kps is None else (kps[i] * inv).astype(np.float32),
                    )
                )
            results.append(dets)
        return results

    def _anchors(self, stride: int) -> np.ndarray:
        """Anchor centres for one FPN level at the current input size (cached)."""
        key = (self.input_size, self.input_size, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        h = self.input_size // stride
        w = self.input_size // stride
        ys, xs = np.mgrid[:h, :w]
        centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
        centers = centers.reshape(-1, 2)
        if self.num_anchors > 1:
            centers = np.repeat(centers, self.num_anchors, axis=0)
        self._anchor_cache[key] = centers
        return centers

    def warmup(self) -> None:
        """Run one dummy batch so cuDNN picks its algorithms off the hot path."""
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.detect_batch([dummy] * self.batch_size)
        log.debug("SCRFD warmed up")

    def close(self) -> None:
        self.engine.close()


__all__ = ["SCRFDDetector"]
