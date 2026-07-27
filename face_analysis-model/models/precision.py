"""FP16 safety validation.

Half precision is close to free performance on Ampere - when the graph tolerates it.
It does not always. The HSEmotion EfficientNet-B0 export, for example, converts cleanly
and runs correctly on the CPU provider, but produces **NaN** on the CUDA provider: some
fused half-precision kernel in that topology overflows.

Shipping that would silently turn every emotion into garbage, so precision is validated
the same way batching is: convert, run FP32 and FP16 side by side on the *target*
providers with the same input, and keep FP16 only if the outputs are finite and match.

The verdict is cached (``<stem>.precision.json``) so the probe costs a couple of seconds
once per machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from utils.logging_utils import get_logger

log = get_logger(__name__)

# Half precision has ~3 decimal digits; allow a generous relative band before we
# declare the conversion broken. We are guarding against NaN/garbage, not rounding.
_REL_TOLERANCE = 0.02


def validate_fp16(
    fp32_path: Path,
    fp16_path: Path,
    cache_dir: Path,
    providers: Sequence[str],
    probe_spatial: Optional[int] = None,
    batch: int = 2,
) -> bool:
    """Return ``True`` when ``fp16_path`` reproduces ``fp32_path`` on ``providers``.

    Args:
        fp32_path: reference graph.
        fp16_path: converted graph.
        cache_dir: where the verdict is cached.
        providers: the execution providers the engine will actually use.
        probe_spatial: square size for graphs with dynamic spatial dimensions.
        batch: probe batch size (clamped to 1 for batch-1 graphs automatically).
    """
    import onnxruntime as ort

    cache_dir = Path(cache_dir)
    verdict_file = cache_dir / f"{Path(fp32_path).stem}.precision.json"
    provider_key = "+".join(providers)
    if verdict_file.exists():
        try:
            verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
            if verdict.get("providers") == provider_key:
                return bool(verdict.get("fp16_ok"))
        except (OSError, ValueError):
            pass

    ok = False
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        ref = ort.InferenceSession(str(fp32_path), so, providers=list(providers))
        half = ort.InferenceSession(str(fp16_path), so, providers=list(providers))

        spec = ref.get_inputs()[0]
        shape = [
            d if isinstance(d, int) and d > 0 else (probe_spatial or 224)
            for d in spec.shape
        ]
        shape[0] = batch if not (isinstance(spec.shape[0], int) and spec.shape[0] > 0) \
            else int(spec.shape[0])
        if len(shape) == 4:
            shape[1] = 3

        rng = np.random.default_rng(7)
        # Values in the range a normalised image actually produces.
        sample = (rng.random(shape).astype(np.float32) * 4.0 - 2.0)

        out32 = ref.run(None, {spec.name: np.ascontiguousarray(sample)})
        out16 = half.run(
            None, {half.get_inputs()[0].name: np.ascontiguousarray(sample.astype(np.float16))}
        )
        ok = _outputs_match(out32, out16)
    except Exception as exc:
        log.warning("FP16 validation failed for %s (%s); staying in FP32", Path(fp32_path).name, exc)
        ok = False

    cache_dir.mkdir(parents=True, exist_ok=True)
    verdict_file.write_text(
        json.dumps({"fp16_ok": ok, "providers": provider_key}, indent=2), encoding="utf-8"
    )
    if ok:
        log.info("FP16 validated for %s on %s", Path(fp32_path).name, provider_key)
    else:
        log.warning(
            "FP16 rejected for %s on %s (non-finite or divergent outputs) - using FP32",
            Path(fp32_path).name, provider_key,
        )
    return ok


def _outputs_match(reference: List[np.ndarray], candidate: List[np.ndarray]) -> bool:
    if len(reference) != len(candidate):
        return False
    for ref, cand in zip(reference, candidate):
        cand32 = np.asarray(cand, dtype=np.float32)
        ref32 = np.asarray(ref, dtype=np.float32)
        if cand32.shape != ref32.shape:
            return False
        if not np.isfinite(cand32).all():
            return False
        scale = max(1.0, float(np.abs(ref32).max()))
        if float(np.abs(cand32 - ref32).max()) / scale > _REL_TOLERANCE:
            return False
    return True


__all__ = ["validate_fp16"]
