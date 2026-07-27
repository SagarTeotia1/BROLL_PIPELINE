from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from models.qwen_vllm import QwenVLLM
from pipeline.level3.context_builder import FrameContext
from shared.types import KeyframeRecord, QwenFrameOutput

logger = logging.getLogger(__name__)


async def run_qwen_analysis(
    contexts: list[FrameContext],
    batch_size: int = 16,  # kept for API compat — vLLM schedules internally
) -> list[tuple[KeyframeRecord, QwenFrameOutput | None]]:
    """Run Qwen VL inference over all frame contexts in a single vLLM submission.

    Submits all frames at once. vLLM's internal scheduler handles concurrent
    execution up to max_num_seqs, maintaining prefix-cache reuse across frames.
    This eliminates the sequential batch-wait-batch overhead from the prior
    implementation.

    Args:
        contexts:   FrameContext objects, one per keyframe.
        batch_size: Unused — kept for call-site compatibility.

    Returns:
        List of (KeyframeRecord, QwenFrameOutput | None) in ascending timestamp order.
        None entries indicate frames where inference or parsing failed.
    """
    sorted_contexts = sorted(contexts, key=lambda c: c.keyframe.timestamp_s)

    qwen = QwenVLLM.get()

    all_input = [
        {"request_id": ctx.request_id, "messages": ctx.messages}
        for ctx in sorted_contexts
    ]

    logger.info(
        "[L3] Submitting %d frames to Qwen (single vLLM submission)", len(all_input)
    )

    raw_outputs = await qwen.analyse_batch(all_input)

    results: list[tuple[KeyframeRecord, QwenFrameOutput | None]] = []
    n_ok = 0

    for ctx, raw in zip(sorted_contexts, raw_outputs):
        kf = ctx.keyframe

        if not raw:
            logger.error(
                "Empty Qwen output for frame %s (t=%.2fs)", kf.id, kf.timestamp_s
            )
            results.append((kf, None))
            continue

        try:
            parsed_dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON parse error for frame %s (t=%.2fs): %s | raw=%r",
                kf.id,
                kf.timestamp_s,
                exc,
                raw[:200],
            )
            results.append((kf, None))
            continue

        try:
            output = QwenFrameOutput.model_validate(parsed_dict)
        except ValidationError as exc:
            logger.error(
                "Pydantic validation error for frame %s (t=%.2fs): %s",
                kf.id,
                kf.timestamp_s,
                exc,
            )
            results.append((kf, None))
            continue
        except Exception as exc:
            logger.error(
                "Unexpected validation error for frame %s (t=%.2fs): %s",
                kf.id,
                kf.timestamp_s,
                exc,
                exc_info=True,
            )
            results.append((kf, None))
            continue

        results.append((kf, output))
        n_ok += 1

    logger.info(
        "[L3] Qwen analysis complete: %d/%d succeeded, %d failed.",
        n_ok,
        len(sorted_contexts),
        len(sorted_contexts) - n_ok,
    )
    return results
