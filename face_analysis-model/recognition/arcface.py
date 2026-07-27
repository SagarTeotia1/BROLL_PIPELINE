"""ArcFace embedding extraction.

Takes canonical 112x112 aligned crops (produced by :func:`utils.image_ops.align_face`)
and returns L2-normalised 512-d identity vectors. Cosine similarity between two such
vectors is a plain dot product, which is what :mod:`recognition.matcher` exploits.

Batching is mandatory here: a single 112x112 forward pass leaves an RTX 3050 almost
idle, while a batch of 16 costs barely more wall time than a batch of 1.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from configs.config import AppConfig
from models.model_zoo import ensure_model, get_spec
from models.onnx_engine import OnnxEngine
from utils.gpu import PinnedRing
from utils.image_ops import GpuNormalizer, align_face
from utils.logging_utils import get_logger

log = get_logger(__name__)


class ArcFaceEmbedder:
    """Batched ArcFace inference.

    Args:
        config: application config (reads ``recognition`` and ``gpu``).
        device: torch device.
        stream_handle: optional CUDA stream pointer shared with ONNX Runtime.
    """

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        stream_handle: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self.rcfg = config.recognition
        self.device = device
        spec = get_spec(self.rcfg.model)
        weights_dir = config.paths.resolve(config.paths.weights_dir)
        model_path = ensure_model(self.rcfg.model, weights_dir)

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
            name="arcface",
        )
        self.input_size = int(self.rcfg.input_size)
        self.batch_size = self.engine.effective_batch(int(self.rcfg.batch_size))
        self.normalizer = GpuNormalizer(
            mean=spec.mean, std=spec.std, device=device,
            dtype=self.engine.input_dtype, bgr_to_rgb=spec.bgr_to_rgb,
            compile_kernel=config.gpu.torch_compile,
        )
        self._pinned = PinnedRing(
            slots=3, max_batch=self.batch_size,
            elem_shape=(self.input_size, self.input_size, 3),
            dtype=torch.uint8, enabled=config.gpu.pinned_memory,
        )
        self._input_name = self.engine.input_names[0]
        self.dimension = self._probe_dimension()
        log.info(
            "ArcFace ready: %s, dim=%d, batch=%d", model_path.name, self.dimension,
            self.batch_size,
        )

    def _probe_dimension(self) -> int:
        shape = self.engine.outputs[0].shape
        for dim in reversed(shape):
            if isinstance(dim, int) and dim > 1:
                return dim
        return 512

    # -- public API ---------------------------------------------------------
    def embed_aligned(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Embed pre-aligned 112x112 BGR crops.

        Returns:
            ``(N, dim)`` float32 array of L2-normalised embeddings.
        """
        if not crops:
            return np.zeros((0, self.dimension), dtype=np.float32)
        out = np.empty((len(crops), self.dimension), dtype=np.float32)
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start : start + self.batch_size]
            out[start : start + len(chunk)] = self._embed_chunk(chunk)
        return out

    def embed_faces(
        self,
        image: np.ndarray,
        landmarks_list: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Align then embed several faces from one frame."""
        crops = [align_face(image, lm, self.input_size) for lm in landmarks_list]
        return self.embed_aligned(crops)

    # -- internals ----------------------------------------------------------
    def _embed_chunk(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        n = len(crops)
        host = self._pinned.acquire(n)
        host_np = host.numpy()
        for i, crop in enumerate(crops):
            if crop.shape[0] != self.input_size or crop.shape[1] != self.input_size:
                raise ValueError(
                    f"ArcFace expects {self.input_size}x{self.input_size} crops, got {crop.shape}"
                )
            host_np[i] = crop
        device_u8 = host.to(self.device, non_blocking=True)
        blob = self.normalizer(device_u8)
        raw = self.engine.run({self._input_name: blob})[0]
        emb = np.asarray(raw, dtype=np.float32).reshape(n, -1)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        np.maximum(norms, 1e-9, out=norms)
        return emb / norms

    def warmup(self) -> None:
        """Prime cuDNN kernels with a full-size dummy batch."""
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.embed_aligned([dummy] * self.batch_size)
        log.debug("ArcFace warmed up")

    def close(self) -> None:
        self.engine.close()


def average_embeddings(
    embeddings: np.ndarray, weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Fuse several embeddings of the same identity into one centroid.

    Each vector is already unit length, so a weighted mean followed by a re-normalise
    gives the spherical centroid - the standard InsightFace gallery construction.

    Args:
        embeddings: ``(N, dim)`` L2-normalised vectors.
        weights: optional per-vector weights (e.g. face quality scores).
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("average_embeddings needs a non-empty (N, dim) array")
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        if w.shape[0] != embeddings.shape[0]:
            raise ValueError("weights length must match embeddings")
        w = np.maximum(w, 1e-3)
        centroid = (embeddings * w).sum(axis=0) / w.sum()
    else:
        centroid = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return (centroid / max(norm, 1e-9)).astype(np.float32)


__all__ = ["ArcFaceEmbedder", "average_embeddings"]
