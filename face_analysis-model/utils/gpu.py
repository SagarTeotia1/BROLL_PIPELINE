"""GPU plumbing: DLL discovery, device selection, pinned buffers, CUDA streams, NVML.

Import order matters on Windows: ``onnxruntime-gpu`` looks for ``cudart64_12.dll`` /
``cudnn64_9.dll`` on the DLL search path. A pip-installed PyTorch ships those libraries
inside ``torch/lib`` (and ``nvidia/*/bin``), so :func:`prepare_cuda_dll_path` registers
those directories *before* onnxruntime is imported. ``models.onnx_engine`` calls it at
module import time; nothing else needs to care.
"""

from __future__ import annotations

import functools
import os
import platform
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.logging_utils import get_logger

log = get_logger(__name__)

_DLL_READY = False
_DLL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Windows DLL discovery
# ---------------------------------------------------------------------------
def prepare_cuda_dll_path() -> List[str]:
    """Add PyTorch's bundled CUDA/cuDNN directories to the DLL search path.

    Returns the list of directories that were registered (empty on non-Windows,
    where the loader uses RPATH instead).
    """
    global _DLL_READY
    with _DLL_LOCK:
        if _DLL_READY or platform.system() != "Windows":
            _DLL_READY = True
            return []

        candidates: List[str] = []
        try:
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib):
                candidates.append(torch_lib)
        except Exception:  # pragma: no cover - torch always present in practice
            pass

        # pip wheels: site-packages/nvidia/<component>/bin
        for base in sys.path:
            nvidia_root = os.path.join(base, "nvidia")
            if not os.path.isdir(nvidia_root):
                continue
            for component in sorted(os.listdir(nvidia_root)):
                bin_dir = os.path.join(nvidia_root, component, "bin")
                if os.path.isdir(bin_dir):
                    candidates.append(bin_dir)

        # Classic CUDA toolkit installation.
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and os.path.isdir(os.path.join(cuda_path, "bin")):
            candidates.append(os.path.join(cuda_path, "bin"))

        registered: List[str] = []
        for directory in dict.fromkeys(candidates):  # dedupe, keep order
            try:
                os.add_dll_directory(directory)
                registered.append(directory)
            except (OSError, AttributeError):
                continue
        if registered:
            os.environ["PATH"] = os.pathsep.join(registered + [os.environ.get("PATH", "")])
        _DLL_READY = True
        log.debug("CUDA DLL directories registered: %s", registered)
        return registered


# ---------------------------------------------------------------------------
# Device information
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceInfo:
    """Static description of the compute device in use."""

    name: str
    index: int
    available: bool
    total_memory_mb: float
    capability: Tuple[int, int]
    supports_fp16: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if not self.available:
            return "CPU (no CUDA device)"
        return (
            f"{self.name} (cuda:{self.index}, {self.total_memory_mb:.0f} MB, "
            f"sm_{self.capability[0]}{self.capability[1]})"
        )


@functools.lru_cache(maxsize=4)
def get_device_info(device_id: int = 0) -> DeviceInfo:
    """Query CUDA device properties (cached)."""
    if not torch.cuda.is_available():
        return DeviceInfo("cpu", -1, False, 0.0, (0, 0), False)
    props = torch.cuda.get_device_properties(device_id)
    cap = (props.major, props.minor)
    return DeviceInfo(
        name=props.name,
        index=device_id,
        available=True,
        total_memory_mb=props.total_memory / (1024 ** 2),
        capability=cap,
        # Every CUDA arch we target (sm_53+) has native FP16 math.
        supports_fp16=cap >= (5, 3),
    )


def resolve_device(preferred: str = "cuda", device_id: int = 0) -> torch.device:
    """Return a usable torch device, falling back to CPU with a warning."""
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    if preferred.startswith("cuda"):
        log.warning("CUDA requested but unavailable - falling back to CPU (expect slow runs)")
    return torch.device("cpu")


def apply_torch_performance_flags() -> None:
    """Enable the TF32 / cuDNN autotuning knobs that are safe for inference."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# NVML telemetry
# ---------------------------------------------------------------------------
@dataclass
class GpuTelemetry:
    """A single GPU usage sample."""

    utilization: float = 0.0        # percent, SM busy
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_c: float = 0.0
    torch_allocated_mb: float = 0.0
    torch_reserved_mb: float = 0.0

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return 100.0 * self.memory_used_mb / self.memory_total_mb


class GpuMonitor:
    """Thread-safe NVML sampler with a pure-torch fallback.

    NVML is opened lazily; if ``nvidia-ml-py`` is missing or the driver refuses the
    handle, utilisation is reported as 0 and only torch allocator stats are filled in.
    """

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id
        self._handle = None
        self._nvml = None
        self._lock = threading.Lock()
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            log.debug("NVML initialised for device %d", self.device_id)
        except Exception as exc:  # pragma: no cover - depends on host driver
            log.debug("NVML unavailable (%s); GPU utilisation will read 0", exc)
            self._nvml = None
            self._handle = None

    def sample(self) -> GpuTelemetry:
        """Take one telemetry sample. Cheap enough for a 2 Hz GUI timer."""
        tel = GpuTelemetry()
        if torch.cuda.is_available():
            tel.torch_allocated_mb = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
            tel.torch_reserved_mb = torch.cuda.memory_reserved(self.device_id) / (1024 ** 2)
        with self._lock:
            if self._nvml is None or self._handle is None:
                info = get_device_info(self.device_id)
                tel.memory_total_mb = info.total_memory_mb
                tel.memory_used_mb = tel.torch_reserved_mb
                return tel
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                tel.utilization = float(util.gpu)
                tel.memory_used_mb = mem.used / (1024 ** 2)
                tel.memory_total_mb = mem.total / (1024 ** 2)
                try:
                    tel.temperature_c = float(
                        self._nvml.nvmlDeviceGetTemperature(self._handle, 0)
                    )
                except Exception:
                    tel.temperature_c = 0.0
            except Exception:  # pragma: no cover
                pass
        return tel

    def close(self) -> None:
        with self._lock:
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception:  # pragma: no cover
                    pass
                self._nvml = None
                self._handle = None


# ---------------------------------------------------------------------------
# Pinned host staging buffers
# ---------------------------------------------------------------------------
class PinnedRing:
    """Ring of page-locked host buffers reused for H2D transfers.

    Allocating pinned memory is expensive (it locks pages with the driver), so the
    pipeline allocates a small ring once and cycles through it. Each slot is large
    enough for ``max_batch * elem_shape``; a caller asks for a view sized to the
    actual batch.

    The ring is *not* thread safe by design - give each stage its own instance so
    there is no lock on the hot path.
    """

    def __init__(
        self,
        slots: int,
        max_batch: int,
        elem_shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
        enabled: bool = True,
    ) -> None:
        self.slots = max(1, slots)
        self.max_batch = max_batch
        self.elem_shape = tuple(elem_shape)
        self.dtype = dtype
        self.enabled = enabled and torch.cuda.is_available()
        shape = (max_batch, *self.elem_shape)
        self._buffers: List[torch.Tensor] = [
            torch.empty(shape, dtype=dtype, pin_memory=self.enabled) for _ in range(self.slots)
        ]
        self._cursor = 0

    def acquire(self, batch: int) -> torch.Tensor:
        """Return a ``[batch, *elem_shape]`` view of the next pinned slot."""
        if batch > self.max_batch:
            # Oversized request: fall back to a one-off buffer rather than crash.
            return torch.empty(
                (batch, *self.elem_shape), dtype=self.dtype, pin_memory=self.enabled
            )
        buf = self._buffers[self._cursor]
        self._cursor = (self._cursor + 1) % self.slots
        return buf[:batch]

    def fill_from_numpy(self, array: np.ndarray) -> torch.Tensor:
        """Copy a numpy batch into the next pinned slot and return the view."""
        view = self.acquire(array.shape[0])
        view.copy_(torch.from_numpy(np.ascontiguousarray(array)))
        return view

    @property
    def nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self._buffers)


# ---------------------------------------------------------------------------
# CUDA streams
# ---------------------------------------------------------------------------
class StreamContext:
    """Owns an optional non-default CUDA stream for a pipeline stage.

    When streams are disabled (CPU device or config off) every method degrades to a
    no-op so call sites stay branch-free.
    """

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        self.device = device
        self.enabled = bool(enabled and device.type == "cuda" and torch.cuda.is_available())
        self.stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if self.enabled else None
        )

    def __enter__(self) -> "StreamContext":
        if self.stream is not None:
            self._ctx = torch.cuda.stream(self.stream)
            self._ctx.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self.stream is not None:
            self._ctx.__exit__(*exc)

    def synchronize(self) -> None:
        """Block until every op queued on this stream has completed."""
        if self.stream is not None:
            self.stream.synchronize()

    @property
    def cuda_stream_handle(self) -> int:
        """Raw ``cudaStream_t`` value, for handing to ONNX Runtime."""
        return int(self.stream.cuda_stream) if self.stream is not None else 0


def to_device(
    tensor: torch.Tensor,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Move + cast in one shot with the requested memory format."""
    out = tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
    if channels_last and out.dim() == 4:
        out = out.contiguous(memory_format=torch.channels_last)
    return out


def empty_cache() -> None:
    """Release cached blocks back to the driver (call between videos, not per frame)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def memory_summary(device_id: int = 0) -> Dict[str, float]:
    """Allocator snapshot in MB, handy for leak hunting in the logs."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    scale = 1024 ** 2
    return {
        "allocated_mb": torch.cuda.memory_allocated(device_id) / scale,
        "reserved_mb": torch.cuda.memory_reserved(device_id) / scale,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device_id) / scale,
    }


__all__ = [
    "prepare_cuda_dll_path",
    "DeviceInfo",
    "get_device_info",
    "resolve_device",
    "apply_torch_performance_flags",
    "GpuTelemetry",
    "GpuMonitor",
    "PinnedRing",
    "StreamContext",
    "to_device",
    "empty_cache",
    "memory_summary",
]
