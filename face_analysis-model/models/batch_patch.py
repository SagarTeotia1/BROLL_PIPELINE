"""Batch-capability probing for released ONNX graphs.

Not every published model can be run with a batch larger than one. The InsightFace
SCRFD export (``det_10g.onnx``) declares a **static** batch of 1, and simply rewriting
the input dimension to a symbolic one produces *silently wrong* results: feeding the
same image twice yields two different outputs, because a reshape inside the detection
head folds the batch axis into the anchor axis.

Guessing here would be a correctness bug that only shows up as mysteriously missed
faces, so this module decides empirically:

1. if the graph already declares a dynamic batch dimension -> batching is allowed,
2. otherwise, patch dimension 0 to a symbolic name and **validate**: run the patched
   graph with the same tensor duplicated and require both halves to match a single-image
   reference within tolerance,
3. if validation fails -> the model is pinned to batch 1 and the caller loops.

The verdict is cached next to the weights (``<stem>.batch.json``) so the probe runs once
per machine, not once per process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from utils.logging_utils import get_logger

log = get_logger(__name__)

_TOLERANCE = 1e-3


def declared_batch(shape: Sequence) -> int:
    """Return the declared batch size: ``0`` when dynamic, else the fixed value."""
    if not shape:
        return 0
    dim = shape[0]
    if isinstance(dim, int) and dim > 0:
        return dim
    return 0


def ensure_batchable(
    model_path: Path,
    cache_dir: Path,
    probe_spatial: Optional[int] = None,
    enabled: bool = True,
) -> Tuple[Path, int]:
    """Return ``(model_to_use, max_batch)`` for a graph.

    Args:
        model_path: the original ONNX file.
        cache_dir: where patched graphs and verdicts are stored.
        probe_spatial: square input size used for validation when the graph's spatial
            dimensions are dynamic (SCRFD needs a multiple of 32).
        enabled: set ``False`` to skip patching entirely and trust the declaration.

    Returns:
        ``max_batch`` is ``0`` for "unlimited / dynamic" and ``N`` for a hard cap.
    """
    import onnxruntime as ort

    model_path = Path(model_path)
    cache_dir = Path(cache_dir)
    so = ort.SessionOptions()
    so.log_severity_level = 3

    session = ort.InferenceSession(
        str(model_path), so, providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    if len(inputs) != 1:
        # Multi-input graphs are out of scope for the probe; trust the declaration.
        return model_path, declared_batch(inputs[0].shape) if inputs else 1

    static_batch = declared_batch(inputs[0].shape)
    if static_batch == 0:
        log.debug("%s declares a dynamic batch dimension", model_path.name)
        del session
        return model_path, 0
    if static_batch > 1 or not enabled:
        del session
        return model_path, static_batch

    verdict_file = cache_dir / f"{model_path.stem}.batch.json"
    patched_path = cache_dir / f"{model_path.stem}.dynbatch.onnx"
    if verdict_file.exists():
        try:
            verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
            if verdict.get("dynamic") and patched_path.exists():
                del session
                return patched_path, 0
            if not verdict.get("dynamic"):
                del session
                return model_path, 1
        except (OSError, ValueError):
            pass  # corrupt cache -> re-probe

    ok = _patch_and_validate(model_path, patched_path, session, probe_spatial)
    cache_dir.mkdir(parents=True, exist_ok=True)
    verdict_file.write_text(
        json.dumps({"dynamic": ok, "model": model_path.name}, indent=2), encoding="utf-8"
    )
    del session
    if ok:
        log.info("%s validated for dynamic batching", model_path.name)
        return patched_path, 0
    log.info(
        "%s is batch-1 only (dynamic-batch validation failed); the wrapper will loop",
        model_path.name,
    )
    patched_path.unlink(missing_ok=True)
    return model_path, 1


def _patch_and_validate(
    model_path: Path,
    patched_path: Path,
    reference_session,
    probe_spatial: Optional[int],
) -> bool:
    """Rewrite dim 0 to a symbol and check that batching preserves per-item results."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnx is a hard requirement in practice
        return False

    try:
        model = onnx.load(str(model_path))
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            if len(dims):
                dims[0].dim_param = "batch_out"
        patched_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(patched_path))

        shape = list(reference_session.get_inputs()[0].shape)
        spatial = probe_spatial or 160
        resolved = [
            d if isinstance(d, int) and d > 0 else (spatial if i >= 2 else 1)
            for i, d in enumerate(shape)
        ]
        resolved[0] = 1
        rng = np.random.default_rng(1234)
        sample = rng.standard_normal(resolved).astype(np.float32)

        name = reference_session.get_inputs()[0].name
        single = reference_session.run(None, {name: sample})

        so = ort.SessionOptions()
        so.log_severity_level = 3
        patched_session = ort.InferenceSession(
            str(patched_path), so, providers=["CPUExecutionProvider"]
        )
        doubled = patched_session.run(
            None, {name: np.concatenate([sample, sample], axis=0)}
        )
    except Exception as exc:
        log.debug("Dynamic-batch patch failed for %s: %s", model_path.name, exc)
        return False

    if len(doubled) != len(single):
        return False
    for ref, batched in zip(single, doubled):
        if batched.shape[0] != ref.shape[0] * 2:
            return False
        half = batched.shape[0] // 2
        scale = max(1.0, float(np.abs(ref).max()))
        if np.abs(batched[:half] - ref).max() / scale > _TOLERANCE:
            return False
        if np.abs(batched[half:] - ref).max() / scale > _TOLERANCE:
            return False
    return True


__all__ = ["ensure_batchable", "declared_batch"]
