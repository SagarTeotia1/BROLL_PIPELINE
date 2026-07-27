"""ONNX Runtime execution engine with torch-interop zero-copy input binding.

This is the single place in the codebase that talks to ``onnxruntime``. It provides:

* **Provider setup** - CUDA (optionally TensorRT) with a bounded arena so a 6 GB card
  does not get eaten by the allocator, plus a CPU fallback that still works.
* **FP16 caching** - a model is converted once with ``onnxconverter-common`` and stored
  under ``weights/fp16``; conversion failures degrade to FP32 instead of crashing.
* **IOBinding** - inputs are bound straight to CUDA ``torch.Tensor.data_ptr()``, so the
  preprocessed batch never round-trips through host memory. Outputs are allocated on
  the device by ORT and pulled back only when the consumer is CPU-side.
* **User compute stream** - when enabled, ORT runs on the *same* CUDA stream as the
  torch preprocessing, removing an inter-stream sync per batch.

Import side effect: :func:`utils.gpu.prepare_cuda_dll_path` runs before ``onnxruntime``
is imported so the CUDA/cuDNN DLLs shipped with PyTorch are discoverable on Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from models.batch_patch import ensure_batchable
from models.precision import validate_fp16
from utils.gpu import prepare_cuda_dll_path
from utils.logging_utils import get_logger

prepare_cuda_dll_path()  # MUST precede the onnxruntime import on Windows.

import onnxruntime as ort  # noqa: E402  (deliberate late import)

log = get_logger(__name__)

_TORCH_TO_NUMPY: Dict[torch.dtype, type] = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.uint8: np.uint8,
}
_ORT_TO_TORCH: Dict[str, torch.dtype] = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(int64)": torch.int64,
    "tensor(int32)": torch.int32,
    "tensor(uint8)": torch.uint8,
}
_ORT_TO_NUMPY: Dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
}


@dataclass
class EngineIO:
    """Description of one model input/output tensor."""

    name: str
    shape: Tuple[Union[int, str], ...]
    dtype_str: str

    @property
    def torch_dtype(self) -> torch.dtype:
        return _ORT_TO_TORCH.get(self.dtype_str, torch.float32)

    @property
    def numpy_dtype(self) -> type:
        return _ORT_TO_NUMPY.get(self.dtype_str, np.float32)

    @property
    def is_dynamic_batch(self) -> bool:
        return len(self.shape) > 0 and not isinstance(self.shape[0], int)


def available_providers() -> List[str]:
    """Providers compiled into the installed onnxruntime build."""
    return list(ort.get_available_providers())


def cuda_is_available() -> bool:
    """True when ORT can actually place work on a GPU."""
    return "CUDAExecutionProvider" in available_providers() and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# FP16 conversion cache
# ---------------------------------------------------------------------------
def convert_to_fp16(model_path: Path, cache_dir: Path) -> Path:
    """Return an FP16 copy of ``model_path`` (cached), or the original on failure.

    Uses ``onnxconverter-common``. ``keep_io_types=False`` makes the graph accept
    half-precision inputs directly, which halves host->device bandwidth.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (model_path.stem + ".fp16.onnx")
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        import onnx
        from onnxconverter_common import float16

        log.info("Converting %s to FP16 (one-off, cached)", model_path.name)
        model = onnx.load(str(model_path))
        fp16_model = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            disable_shape_infer=True,
            # These ops are numerically fragile in half precision.
            op_block_list=["Resize", "TopK", "NonMaxSuppression", "RoiAlign"],
        )
        onnx.save(fp16_model, str(target))
        log.info("FP16 model cached at %s", target.name)
        return target
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        log.warning("FP16 conversion failed for %s (%s); staying in FP32", model_path.name, exc)
        return model_path


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class OnnxEngine:
    """A single ONNX graph bound to a device, callable with torch or numpy batches.

    Example:
        engine = OnnxEngine("weights/w600k_r50.onnx", device=torch.device("cuda"), fp16=True)
        embeddings = engine.run({engine.input_names[0]: batch_tensor})[0]
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: torch.device,
        fp16: bool = True,
        fp16_cache_dir: Optional[Path] = None,
        gpu_mem_limit_mb: int = 3072,
        cudnn_conv_algo_search: str = "HEURISTIC",
        intra_op_threads: int = 4,
        use_tensorrt: bool = False,
        trt_cache_dir: Optional[Path] = None,
        user_compute_stream: Optional[int] = None,
        name: Optional[str] = None,
        allow_batch_patch: bool = True,
        probe_spatial: Optional[int] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.name = name or self.model_path.stem
        self.device = device
        self.device_id = device.index or 0
        # Tentative: the provider list only says the build *supports* CUDA, not that
        # the DLLs load. The real answer comes from the session below.
        self.on_cuda = device.type == "cuda" and cuda_is_available()
        self.fp16 = bool(fp16 and self.on_cuda)

        cache = Path(fp16_cache_dir or self.model_path.parent / "fp16")

        # Decide (and cache) whether this graph may be run with batch > 1 before any
        # precision conversion, so the verdict is about the original weights.
        active_path, self.max_batch = ensure_batchable(
            self.model_path, cache.parent / "batch",
            probe_spatial=probe_spatial, enabled=allow_batch_patch,
        )
        self._fp32_path = active_path      # batch-decided graph, still float32
        if self.fp16:
            converted = convert_to_fp16(active_path, cache)
            if converted != active_path and validate_fp16(
                active_path, converted, cache,
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.on_cuda else ["CPUExecutionProvider"]
                ),
                probe_spatial=probe_spatial,
            ):
                active_path = converted
            else:
                # Conversion produced NaN/garbage on the target provider: stay FP32.
                self.fp16 = False
        self.active_path = active_path

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.inter_op_num_threads = 1
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = not self.on_cuda
        sess_options.log_severity_level = 3

        providers, provider_options = self._build_providers(
            gpu_mem_limit_mb=gpu_mem_limit_mb,
            cudnn_conv_algo_search=cudnn_conv_algo_search,
            use_tensorrt=use_tensorrt,
            trt_cache_dir=trt_cache_dir,
            user_compute_stream=user_compute_stream,
        )

        try:
            self.session = ort.InferenceSession(
                str(active_path), sess_options=sess_options,
                providers=providers, provider_options=provider_options,
            )
        except Exception as exc:
            log.warning(
                "Session creation with %s failed (%s); retrying with default options",
                providers, exc,
            )
            fallback = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.on_cuda \
                else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                str(active_path), sess_options=sess_options, providers=fallback
            )

        # The CUDA EP silently degrades to CPU when its DLLs cannot be loaded (wrong
        # CUDA major version, missing cuDNN). Trust the *session*, not the build flags:
        # IOBinding to device pointers would raise on a CPU-only session.
        active_providers = list(self.session.get_providers())
        gpu_active = any(
            p in active_providers
            for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        )
        if self.on_cuda and not gpu_active:
            log.error(
                "onnxruntime could not activate a GPU provider for '%s' (got %s). "
                "Check that onnxruntime-gpu matches your CUDA major version - torch "
                "reports %s. Falling back to CPU execution.",
                self.name, active_providers,
                getattr(torch.version, "cuda", "unknown"),
            )
            self.on_cuda = False
            if self.fp16:
                # FP16 kernels are not supported by the CPU EP for these graphs.
                log.warning("Reloading '%s' in FP32 for CPU execution", self.name)
                self.session = ort.InferenceSession(
                    str(self._fp32_path), sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                self.active_path = self._fp32_path
                self.fp16 = False

        self.inputs: List[EngineIO] = [
            EngineIO(i.name, tuple(i.shape), i.type) for i in self.session.get_inputs()
        ]
        self.outputs: List[EngineIO] = [
            EngineIO(o.name, tuple(o.shape), o.type) for o in self.session.get_outputs()
        ]
        self.input_names = [i.name for i in self.inputs]
        self.output_names = [o.name for o in self.outputs]
        self._binding = self.session.io_binding() if self.on_cuda else None
        self._run_options = ort.RunOptions()

        log.info(
            "Engine '%s' ready | providers=%s | fp16=%s | max_batch=%s | in=%s | out=%d tensors",
            self.name, self.session.get_providers(), self.fp16,
            self.max_batch or "dynamic",
            [(i.name, i.shape) for i in self.inputs], len(self.outputs),
        )

    # -- provider construction ---------------------------------------------
    def _build_providers(
        self,
        gpu_mem_limit_mb: int,
        cudnn_conv_algo_search: str,
        use_tensorrt: bool,
        trt_cache_dir: Optional[Path],
        user_compute_stream: Optional[int],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if not self.on_cuda:
            return ["CPUExecutionProvider"], [{}]

        cuda_opts: Dict[str, Any] = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": int(gpu_mem_limit_mb) * 1024 * 1024,
            "cudnn_conv_algo_search": cudnn_conv_algo_search,
            "do_copy_in_default_stream": "1",
            "cudnn_conv_use_max_workspace": "1",
        }
        if user_compute_stream:
            # Share torch's stream: ORT then enqueues onto the same stream that
            # produced the input tensors, so no cross-stream event is needed.
            cuda_opts["has_user_compute_stream"] = "1"
            cuda_opts["user_compute_stream"] = str(user_compute_stream)
            cuda_opts["do_copy_in_default_stream"] = "0"

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        options: List[Dict[str, Any]] = [cuda_opts, {}]

        if use_tensorrt and "TensorrtExecutionProvider" in available_providers():
            cache = Path(trt_cache_dir or self.model_path.parent / "trt")
            cache.mkdir(parents=True, exist_ok=True)
            trt_opts = {
                "device_id": self.device_id,
                "trt_fp16_enable": self.fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
                "trt_max_workspace_size": int(gpu_mem_limit_mb) * 1024 * 1024,
                "trt_timing_cache_enable": True,
            }
            providers.insert(0, "TensorrtExecutionProvider")
            options.insert(0, trt_opts)
        return providers, options

    # -- properties ---------------------------------------------------------
    @property
    def input_dtype(self) -> torch.dtype:
        return self.inputs[0].torch_dtype

    @property
    def providers(self) -> List[str]:
        return list(self.session.get_providers())

    def static_input_shape(self) -> Tuple[Union[int, str], ...]:
        return self.inputs[0].shape

    # -- execution ----------------------------------------------------------
    def run(
        self,
        feeds: Dict[str, Union[torch.Tensor, np.ndarray]],
        output_names: Optional[Sequence[str]] = None,
    ) -> List[np.ndarray]:
        """Run the graph and return outputs as host numpy arrays.

        CUDA torch tensors are bound by pointer (no host round-trip); everything else
        goes through the standard ``session.run`` path.
        """
        names = list(output_names) if output_names else self.output_names
        if self._binding is not None and self._all_cuda_tensors(feeds):
            return self._run_iobinding(feeds, names)
        np_feeds = {
            k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in feeds.items()
        }
        return self.session.run(names, np_feeds, self._run_options)

    def run_device(
        self,
        feeds: Dict[str, torch.Tensor],
        output_names: Optional[Sequence[str]] = None,
    ) -> List[torch.Tensor]:
        """Run the graph keeping outputs on the GPU as torch tensors when possible.

        Falls back to a host copy (then re-upload) if the runtime cannot expose the
        output buffers via DLPack, so callers always get device tensors.
        """
        names = list(output_names) if output_names else self.output_names
        if self._binding is None or not self._all_cuda_tensors(feeds):
            arrays = self.run(feeds, names)
            return [torch.from_numpy(a).to(self.device, non_blocking=True) for a in arrays]

        self._bind(feeds, names, device_outputs=True)
        self.session.run_with_iobinding(self._binding, self._run_options)
        tensors: List[torch.Tensor] = []
        for ov in self._binding.get_outputs():
            tensors.append(self._ortvalue_to_torch(ov))
        self._binding.clear_binding_inputs()
        self._binding.clear_binding_outputs()
        return tensors

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _all_cuda_tensors(feeds: Dict[str, Any]) -> bool:
        return bool(feeds) and all(
            isinstance(v, torch.Tensor) and v.is_cuda for v in feeds.values()
        )

    def _bind(
        self, feeds: Dict[str, torch.Tensor], names: Sequence[str], device_outputs: bool
    ) -> None:
        binding = self._binding
        assert binding is not None
        for spec in self.inputs:
            tensor = feeds.get(spec.name)
            if tensor is None:
                raise KeyError(f"missing input '{spec.name}' for engine {self.name}")
            want = spec.torch_dtype
            if tensor.dtype != want:
                tensor = tensor.to(want)
            tensor = tensor.contiguous()
            feeds[spec.name] = tensor  # keep a reference alive until run() returns
            binding.bind_input(
                name=spec.name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=_TORCH_TO_NUMPY[tensor.dtype],
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )
        for name in names:
            if device_outputs:
                binding.bind_output(name, device_type="cuda", device_id=self.device_id)
            else:
                binding.bind_output(name)

    def _run_iobinding(
        self, feeds: Dict[str, torch.Tensor], names: Sequence[str]
    ) -> List[np.ndarray]:
        # Outputs land in pinned host memory; ORT performs the D2H copy on its own
        # stream while we already hold the next batch in flight.
        self._bind(feeds, names, device_outputs=False)
        # The preprocessing kernels must have retired before ORT reads the pointers.
        torch.cuda.current_stream(self.device).synchronize()
        self.session.run_with_iobinding(self._binding, self._run_options)
        arrays = self._binding.copy_outputs_to_cpu()
        self._binding.clear_binding_inputs()
        self._binding.clear_binding_outputs()
        return arrays

    def _ortvalue_to_torch(self, ortvalue: Any) -> torch.Tensor:
        """Wrap an ORT device tensor as torch, via DLPack when the build allows it."""
        try:
            from torch.utils.dlpack import from_dlpack

            return from_dlpack(ortvalue)
        except Exception:
            return torch.from_numpy(ortvalue.numpy()).to(self.device, non_blocking=True)

    # -- lifecycle ----------------------------------------------------------
    def effective_batch(self, requested: int) -> int:
        """Clamp a requested batch size to what the graph actually supports."""
        if self.max_batch <= 0:
            return max(1, requested)
        return max(1, min(requested, self.max_batch))

    def warmup(self, batch_sizes: Sequence[int] = (1,), spatial: Optional[int] = None) -> None:
        """Force kernel selection / TRT engine build before the first real frame."""
        spec = self.inputs[0]
        shape = list(spec.shape)
        for i, dim in enumerate(shape):
            if not isinstance(dim, int) or dim <= 0:
                shape[i] = 3 if i == 1 else (spatial or 112)
        for bs in batch_sizes:
            shape[0] = bs
            dummy = torch.zeros(
                shape, dtype=spec.torch_dtype,
                device=self.device if self.on_cuda else torch.device("cpu"),
            )
            try:
                self.run({spec.name: dummy})
            except Exception as exc:  # pragma: no cover - shape guessing may miss
                log.debug("Warmup batch %d for %s skipped: %s", bs, self.name, exc)
                return
        log.debug("Engine '%s' warmed up for batches %s", self.name, list(batch_sizes))

    def close(self) -> None:
        """Drop the session and its device allocations."""
        self._binding = None
        self.session = None  # type: ignore[assignment]

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "OnnxEngine",
    "EngineIO",
    "available_providers",
    "cuda_is_available",
    "convert_to_fp16",
]
