"""CUDA-aware execution backend built on OpenCV only.

The expensive per-pixel work in this engine is three operations — resize,
colour-space conversion and box blur.  :class:`Backend` routes those through
``cv2.cuda`` when the installed OpenCV was built with CUDA *and* a device is
present, and through the ordinary CPU functions otherwise.  Callers never
branch; they hand in a NumPy array and get one back.

Why the capability probe is this careful
----------------------------------------
The stock ``opencv-python`` wheel from PyPI exposes the ``cv2.cuda`` *namespace*
but ships no CUDA kernels: ``hasattr(cv2, "cuda")`` is ``True`` while
``cv2.cuda.resize`` does not exist and ``getCudaEnabledDeviceCount()`` returns 0.
Probing only the namespace, or only the device count, produces a backend that
claims GPU support and then raises on first use.  Each entry point is therefore
checked for the specific function it needs, and any failure downgrades that one
operation rather than the whole backend.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np


def _cuda_device_count() -> int:
    """Number of usable CUDA devices, or 0 when OpenCV has no CUDA support."""
    cuda = getattr(cv2, "cuda", None)
    if cuda is None:
        return 0
    try:
        return int(cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def _has_cuda_fn(name: str) -> bool:
    """True when ``cv2.cuda.<name>`` exists in this build."""
    cuda = getattr(cv2, "cuda", None)
    return cuda is not None and hasattr(cuda, name)


class Backend:
    """Dispatches image operations to CUDA or CPU.

    Parameters
    ----------
    prefer_gpu:
        Set ``False`` to force the CPU path even on a CUDA-capable build.
    device_id:
        CUDA device to bind when more than one is present.
    """

    #: Operations this engine can offload, and the ``cv2.cuda`` function each needs.
    _OPS = {"resize": "resize", "cvt_color": "cvtColor", "blur": "blur"}

    def __init__(self, prefer_gpu: bool = True, device_id: int = 0) -> None:
        self.device_count = _cuda_device_count()
        self._gpu_ops = {}
        self.use_gpu = False

        if prefer_gpu and self.device_count > 0:
            try:
                cv2.cuda.setDevice(int(device_id))
                self._gpu_ops = {op: _has_cuda_fn(fn) for op, fn in self._OPS.items()}
                self.use_gpu = any(self._gpu_ops.values())
            except Exception:
                self._gpu_ops = {}
                self.use_gpu = False

        self.device_id = int(device_id) if self.use_gpu else -1

    # -- introspection ------------------------------------------------------
    def describe(self) -> dict:
        """Backend state, embedded in the analysis output for reproducibility."""
        return {
            "device": "cuda" if self.use_gpu else "cpu",
            "cuda_devices": self.device_count,
            "accelerated_ops": sorted(op for op, ok in self._gpu_ops.items() if ok),
        }

    def _offloads(self, op: str) -> bool:
        return bool(self._gpu_ops.get(op))

    # -- operations ---------------------------------------------------------
    def resize(self, image: np.ndarray, size: Tuple[int, int], interpolation: int) -> np.ndarray:
        """Resize to ``(width, height)``.

        Note the CUDA path is only taken for downscales with ``INTER_AREA``
        substituted: ``cv2.cuda.resize`` does not implement area interpolation,
        and silently changing the filter would change the measured statistics
        between backends. Correctness beats the offload here.
        """
        if interpolation == cv2.INTER_AREA or not self._offloads("resize"):
            return cv2.resize(image, size, interpolation=interpolation)
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(image)
            out = cv2.cuda.resize(gpu, size, interpolation=interpolation)
            return out.download()
        except Exception:
            return cv2.resize(image, size, interpolation=interpolation)

    def cvt_color(self, image: np.ndarray, code: int) -> np.ndarray:
        """Colour-space conversion."""
        if not self._offloads("cvt_color"):
            return cv2.cvtColor(image, code)
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(image)
            return cv2.cuda.cvtColor(gpu, code).download()
        except Exception:
            return cv2.cvtColor(image, code)

    def blur(self, image: np.ndarray, ksize: int) -> np.ndarray:
        """Normalised box blur with a ``ksize x ksize`` kernel."""
        if not self._offloads("blur"):
            return cv2.blur(image, (ksize, ksize))
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(image)
            flt = cv2.cuda.createBoxFilter(gpu.type(), -1, (ksize, ksize))
            return flt.apply(gpu).download()
        except Exception:
            return cv2.blur(image, (ksize, ksize))


#: Shared default; callers may construct their own.
DEFAULT_BACKEND: Optional[Backend] = None


def default_backend() -> Backend:
    """Process-wide backend, built on first use."""
    global DEFAULT_BACKEND
    if DEFAULT_BACKEND is None:
        DEFAULT_BACKEND = Backend()
    return DEFAULT_BACKEND
