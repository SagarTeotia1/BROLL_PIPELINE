"""HSEmotion facial-expression recognition (8 AffectNet classes).

HSEmotion is an EfficientNet trained on AffectNet + VGAF; it takes a *loose* face crop
(not the tight ArcFace alignment) resized to the network resolution with ImageNet
normalisation, and returns logits over eight expressions.

Class order of the released ONNX models is alphabetical AffectNet::

    Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise

which is remapped here to the product-facing vocabulary (Angry / Happy / Sad / ...).

As with ArcFace, inference is always batched across all faces of all frames currently
in flight - see :class:`pipeline.batching.BatchCollector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from configs.config import AppConfig
from models.model_zoo import ensure_model, get_spec
from models.onnx_engine import OnnxEngine
from utils.gpu import PinnedRing
from utils.image_ops import GpuNormalizer, crop_bbox
from utils.logging_utils import get_logger

log = get_logger(__name__)

# Native model order (AffectNet-8, alphabetical).
RAW_LABELS: Tuple[str, ...] = (
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
)

# Product-facing vocabulary.
EMOTION_LABELS: Tuple[str, ...] = (
    "Angry",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
)

# 7-class variants drop Contempt.
RAW_LABELS_7: Tuple[str, ...] = (
    "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
)
EMOTION_LABELS_7: Tuple[str, ...] = (
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise",
)

# Colours used by the GUI and the timeline PNG (BGR for OpenCV, RGB hex for Qt/matplotlib).
EMOTION_COLORS: Dict[str, str] = {
    "Angry": "#e5484d",
    "Contempt": "#a35bd6",
    "Disgust": "#7c9a3b",
    "Fear": "#8e6fd6",
    "Happy": "#f5b73d",
    "Neutral": "#6b7684",
    "Sad": "#3d7ff5",
    "Surprise": "#22b8a6",
    "Unknown": "#454b54",
}


@dataclass
class EmotionResult:
    """One emotion prediction."""

    label: str
    confidence: float
    probabilities: np.ndarray

    def top_k(self, k: int = 3, labels: Sequence[str] = EMOTION_LABELS) -> List[Tuple[str, float]]:
        """The ``k`` most likely expressions, highest first."""
        idx = np.argsort(-self.probabilities)[:k]
        return [(labels[i], float(self.probabilities[i])) for i in idx]

    def as_dict(self) -> dict:
        return {"emotion": self.label, "confidence": round(self.confidence, 4)}


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    return x / np.maximum(x.sum(axis=axis, keepdims=True), 1e-9)


class HSEmotionClassifier:
    """Batched HSEmotion inference.

    Args:
        config: application config (reads ``emotion`` and ``gpu``).
        device: torch device.
        stream_handle: optional shared CUDA stream pointer.
    """

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        stream_handle: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self.ecfg = config.emotion
        self.device = device
        spec = get_spec(self.ecfg.model)
        weights_dir = config.paths.resolve(config.paths.weights_dir)
        model_path = ensure_model(self.ecfg.model, weights_dir)

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
            name="hsemotion",
        )
        # The spec's input size is authoritative (B0 -> 224, B2 -> 260), but honour a
        # static shape baked into the graph if there is one.
        graph_size = self._graph_spatial()
        self.input_size = graph_size or int(self.ecfg.input_size or spec.input_size)
        self.batch_size = self.engine.effective_batch(int(self.ecfg.batch_size))
        self.num_classes = self._graph_classes()
        self.labels = EMOTION_LABELS if self.num_classes == 8 else EMOTION_LABELS_7

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
        log.info(
            "HSEmotion ready: %s, %d classes, input=%d, batch=%d",
            model_path.name, self.num_classes, self.input_size, self.batch_size,
        )

    def _graph_spatial(self) -> Optional[int]:
        shape = self.engine.inputs[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 1:
            return int(shape[2])
        return None

    def _graph_classes(self) -> int:
        shape = self.engine.outputs[0].shape
        for dim in reversed(shape):
            if isinstance(dim, int) and dim > 1:
                return int(dim)
        return 8

    # -- public API ---------------------------------------------------------
    def predict_crops(self, crops: Sequence[np.ndarray]) -> List[EmotionResult]:
        """Classify pre-cropped BGR faces (any size; resized internally)."""
        if not crops:
            return []
        results: List[EmotionResult] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start : start + self.batch_size]
            results.extend(self._predict_chunk(chunk))
        return results

    def predict_faces(
        self,
        image: np.ndarray,
        bboxes: Sequence[Sequence[float]],
        scale: float = 1.15,
    ) -> List[EmotionResult]:
        """Crop faces out of a frame and classify them in one batch."""
        crops = [crop_bbox(image, box, scale=scale, size=self.input_size) for box in bboxes]
        return self.predict_crops(crops)

    def make_crop(self, image: np.ndarray, bbox: Sequence[float], scale: float = 1.15) -> np.ndarray:
        """Produce the network-sized crop for one face (used by the batch collector)."""
        return crop_bbox(image, bbox, scale=scale, size=self.input_size)

    # -- internals ----------------------------------------------------------
    def _predict_chunk(self, crops: Sequence[np.ndarray]) -> List[EmotionResult]:
        n = len(crops)
        host = self._pinned.acquire(n)
        host_np = host.numpy()
        size = self.input_size
        for i, crop in enumerate(crops):
            if crop.shape[0] != size or crop.shape[1] != size:
                interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
                crop = cv2.resize(crop, (size, size), interpolation=interp)
            host_np[i] = crop
        device_u8 = host.to(self.device, non_blocking=True)
        blob = self.normalizer(device_u8)
        logits = np.asarray(self.engine.run({self._input_name: blob})[0], dtype=np.float32)
        logits = logits.reshape(n, -1)
        probs = softmax(logits, axis=1)

        out: List[EmotionResult] = []
        for i in range(n):
            j = int(np.argmax(probs[i]))
            out.append(
                EmotionResult(
                    label=self.labels[j] if j < len(self.labels) else "Unknown",
                    confidence=float(probs[i, j]),
                    probabilities=probs[i].copy(),
                )
            )
        return out

    def warmup(self) -> None:
        """Prime the graph with a full dummy batch."""
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.predict_crops([dummy] * self.batch_size)
        log.debug("HSEmotion warmed up")

    def close(self) -> None:
        self.engine.close()


__all__ = [
    "HSEmotionClassifier",
    "EmotionResult",
    "EMOTION_LABELS",
    "EMOTION_LABELS_7",
    "EMOTION_COLORS",
    "RAW_LABELS",
    "softmax",
]
