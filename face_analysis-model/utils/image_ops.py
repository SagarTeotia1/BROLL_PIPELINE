"""Image geometry, face alignment and GPU-side normalisation.

Contains the pieces every model shares:

* :func:`umeyama` / :func:`estimate_norm` / :func:`align_face` - the standard ArcFace
  5-point similarity warp to a 112x112 canonical crop.
* :func:`letterbox` - aspect-preserving pad used by the SCRFD detector.
* :class:`GpuNormalizer` - fused uint8 NHWC -> float NCHW (optionally FP16 and
  ``channels_last``) normalisation that runs on the GPU, optionally ``torch.compile``d.
* Quality metrics (blur, pose, brightness) used by the quality gate.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from utils.logging_utils import get_logger

log = get_logger(__name__)

# Canonical 5-point template used by every ArcFace/InsightFace model (112x112).
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],   # right eye
        [73.5318, 51.5014],   # left eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # right mouth corner
        [70.7299, 92.2041],   # left mouth corner
    ],
    dtype=np.float32,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Similarity transform (Umeyama, 1991)
# ---------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> np.ndarray:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Args:
        src: ``(N, 2)`` source points.
        dst: ``(N, 2)`` destination points.
        estimate_scale: include isotropic scale in the estimate.

    Returns:
        ``(3, 3)`` homogeneous transform matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    if estimate_scale:
        var_src = src_demean.var(axis=0).sum()
        scale = 1.0 if var_src < 1e-12 else (S @ d) / var_src
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T


def estimate_norm(landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Return the ``(2, 3)`` affine matrix aligning 5 landmarks to the template."""
    assert landmarks.shape == (5, 2), f"expected (5,2) landmarks, got {landmarks.shape}"
    dst = ARCFACE_TEMPLATE.copy()
    if image_size != 112:
        dst = dst * (image_size / 112.0)
    matrix = umeyama(landmarks.astype(np.float32), dst)
    return matrix[:2, :].astype(np.float32)


def align_face(
    image: np.ndarray, landmarks: np.ndarray, image_size: int = 112
) -> np.ndarray:
    """Warp a face to the canonical ArcFace crop.

    Args:
        image: full BGR frame.
        landmarks: ``(5, 2)`` points in frame coordinates.
        image_size: output square size.

    Returns:
        ``(image_size, image_size, 3)`` BGR crop.
    """
    M = estimate_norm(landmarks, image_size)
    return cv2.warpAffine(
        image, M, (image_size, image_size), borderValue=0.0, flags=cv2.INTER_LINEAR
    )


def crop_bbox(
    image: np.ndarray, bbox: Sequence[float], scale: float = 1.0, size: Optional[int] = None
) -> np.ndarray:
    """Square, padded crop around a bbox - used for emotion input and thumbnails."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1) * scale
    half = side * 0.5
    left, top = int(round(cx - half)), int(round(cy - half))
    right, bottom = int(round(cx + half)), int(round(cy + half))

    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - w), max(0, bottom - h)
    left, top = max(0, left), max(0, top)
    right, bottom = min(w, right), min(h, bottom)
    if right <= left or bottom <= top:
        out = np.zeros((max(1, int(side)), max(1, int(side)), 3), dtype=image.dtype)
    else:
        out = image[top:bottom, left:right]
        if pad_l or pad_t or pad_r or pad_b:
            out = cv2.copyMakeBorder(
                out, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
            )
    if size is not None and out.shape[0] != size:
        interp = cv2.INTER_AREA if out.shape[0] > size else cv2.INTER_LINEAR
        out = cv2.resize(out, (size, size), interpolation=interp)
    return out


# ---------------------------------------------------------------------------
# Letterbox for the detector
# ---------------------------------------------------------------------------
def letterbox(
    image: np.ndarray, size: int, color: int = 0
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize keeping aspect ratio and pad to ``size x size`` (top-left anchored).

    Returns ``(padded_image, scale, (pad_x, pad_y))``. SCRFD's decoding assumes the
    image is anchored at the origin, so padding is added right/bottom only and
    ``pad_x``/``pad_y`` are always 0 - they are returned for API symmetry with
    centre-padded variants.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), color, dtype=image.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas, scale, (0, 0)


def resize_long_side(image: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    """Downscale so the longest side equals ``long_side`` (never upscales)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= long_side:
        return image, 1.0
    scale = long_side / longest
    return (
        cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA),
        scale,
    )


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def blur_score(image: np.ndarray) -> float:
    """Variance of the Laplacian - low values mean out-of-focus / motion blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def brightness_score(image: np.ndarray) -> float:
    """Mean luma in ``[0, 1]``; extreme values indicate crushed or blown crops."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(gray.mean()) / 255.0


def yaw_ratio(landmarks: np.ndarray) -> float:
    """Profile-ness estimate from 5 landmarks, invariant to in-plane rotation.

    The nose is projected onto the **eye axis** and its position expressed as a fraction
    ``t`` of the inter-ocular distance: ``t == 0.5`` is frontal, ``t -> 0`` or ``1`` is a
    full profile. The returned value is ``max(t, 1-t) / min(t, 1-t)``, so it is always
    ``>= 1`` and one threshold covers both turn directions.

    Measuring along the eye axis rather than the image's x axis matters: a tilted head
    (roll) collapses the horizontal distance to one eye and would otherwise be reported
    as an extreme profile.
    """
    eye_a, eye_b, nose = (
        landmarks[0].astype(np.float64),
        landmarks[1].astype(np.float64),
        landmarks[2].astype(np.float64),
    )
    axis = eye_b - eye_a
    span = float(np.hypot(axis[0], axis[1]))
    if span < 1e-6:
        return 99.0
    t = float(np.dot(nose - eye_a, axis) / (span * span))
    t = min(max(t, 1e-3), 1.0 - 1e-3)
    lo, hi = min(t, 1.0 - t), max(t, 1.0 - t)
    return float(hi / lo)


def sharpness_normalised(image: np.ndarray) -> float:
    """Blur score squashed into ``[0, 1]`` for use as a quality weight."""
    return float(np.clip(blur_score(image) / 300.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# GPU normalisation
# ---------------------------------------------------------------------------
class GpuNormalizer:
    """Fused uint8 NHWC -> normalised float NCHW conversion on the GPU.

    The three networks want different scalings, so the mean/std are parameters:

    ==============  ==================  ================
    model           mean                std
    ==============  ==================  ================
    SCRFD           127.5               128.0
    ArcFace         127.5               127.5
    HSEmotion       ImageNet * 255      ImageNet * 255
    ==============  ==================  ================

    ``torch.compile`` fuses the subtract/divide/permute into a single kernel; it is
    opt-in because the ~20 s compile cost only pays off on long videos.

    Layout note: the *input* is NHWC uint8 - the natural layout of a decoded frame, so
    nothing is transposed on the CPU and only one byte per channel crosses PCIe. The
    *output* is always NCHW-contiguous because that is what ONNX Runtime's CUDA
    execution provider requires; handing it a ``channels_last`` tensor would either be
    copied back internally or, when bound by raw pointer, silently reinterpreted.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        bgr_to_rgb: bool = True,
        compile_kernel: bool = False,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.bgr_to_rgb = bgr_to_rgb
        self._mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._fn = self._forward
        if compile_kernel and device.type == "cuda":
            try:
                self._fn = torch.compile(self._forward, dynamic=True, mode="reduce-overhead")
                log.info("GpuNormalizer: torch.compile enabled")
            except Exception as exc:  # pragma: no cover - depends on triton availability
                log.warning("torch.compile unavailable (%s); using eager normaliser", exc)
                self._fn = self._forward

    def _forward(self, batch_u8: torch.Tensor) -> torch.Tensor:
        # batch_u8: (N, H, W, 3) uint8 on device
        x = batch_u8.permute(0, 3, 1, 2).float()
        if self.bgr_to_rgb:
            x = x.flip(1)
        x = (x - self._mean) / self._std
        return x

    def __call__(self, batch_u8: torch.Tensor) -> torch.Tensor:
        """Normalise a uint8 NHWC device tensor into the model's NCHW input tensor."""
        return self._fn(batch_u8).to(self.dtype).contiguous()


def hsemotion_stats() -> Tuple[Sequence[float], Sequence[float]]:
    """ImageNet mean/std expressed in 0-255 units (HSEmotion preprocessing)."""
    return tuple(IMAGENET_MEAN * 255.0), tuple(IMAGENET_STD * 255.0)


def draw_face_box(
    frame: np.ndarray,
    bbox: Sequence[float],
    label: str,
    color: Tuple[int, int, int] = (0, 220, 120),
    thickness: int = 2,
) -> None:
    """Draw a labelled detection box in place (used by the CLI preview writer)."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    if not label:
        return
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(0, y1 - th - base - 4)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA,
    )


__all__ = [
    "ARCFACE_TEMPLATE",
    "umeyama",
    "estimate_norm",
    "align_face",
    "crop_bbox",
    "letterbox",
    "resize_long_side",
    "blur_score",
    "brightness_score",
    "yaw_ratio",
    "sharpness_normalised",
    "GpuNormalizer",
    "hsemotion_stats",
    "draw_face_box",
]
