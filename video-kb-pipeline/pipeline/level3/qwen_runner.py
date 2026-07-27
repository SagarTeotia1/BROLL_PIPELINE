from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from models.qwen_vllm import QwenVLLM
from pipeline.level3.context_builder import FrameContext
from shared.types import KeyframeRecord, QwenFrameOutput
from shared.utils import ProgressTracker

logger = logging.getLogger(__name__)


def _compress_captions(analyses: list[QwenFrameOutput]) -> str:
    """Compress the last few frame captions into a short rolling summary string.

    Pulls caption, beat_type, likely_consequence, and scene_mood from each
    analysis and formats them as a compact multi-line context block.

    Args:
        analyses: Recent QwenFrameOutput objects (last 3 used).

    Returns:
        A compact string suitable for use as prev_scene_summary.
    """
    recent = analyses[-3:] if len(analyses) > 3 else analyses
    if not recent:
        return ""

    parts: list[str] = []
    for i, analysis in enumerate(recent):
        frame_parts: list[str] = []

        if analysis.caption:
            frame_parts.append(f"Caption: {analysis.caption}")

        beat_type = analysis.story_beat.beat_type if analysis.story_beat else ""
        act_position = analysis.story_beat.act_position if analysis.story_beat else ""
        if beat_type:
            beat_str = f"{beat_type} ({act_position})" if act_position else beat_type
            frame_parts.append(f"Beat: {beat_str}")

        likely_consequence = (
            analysis.causality.likely_consequence if analysis.causality else ""
        )
        if likely_consequence:
            frame_parts.append(f"Consequence implied: {likely_consequence}")

        pids = [p.pid for p in analysis.people] if analysis.people else []
        if pids:
            frame_parts.append(f"People: {', '.join(pids)}")

        frame_parts.append(f"t={analysis.t:.1f}s")

        if frame_parts:
            parts.append(f"[Frame {i + 1}] " + " | ".join(frame_parts))

    return "\n".join(parts)


async def run_qwen_analysis(
    contexts: list[FrameContext],
    batch_size: int = 8,
) -> list[tuple[KeyframeRecord, QwenFrameOutput | None]]:
    """Run Qwen VL inference over all frame contexts in batches.

    Processes frames in ascending timestamp order. After every shot boundary
    (scene_change=True in the model output), carries the last 3 captions forward
    as a rolling summary for use in subsequent batches. Note: the rolling summary
    is tracked here for logging/debugging purposes; the actual summary injected
    into each frame's prompt is built by context_builder before this function
    is called, so updating it here does not affect the prompts in this run.
    It is available for the caller to use when calling build_all_frame_contexts
    again for the next group of shots.

    The QwenVLLM singleton must already have load() called before invoking this
    function.

    Args:
        contexts: FrameContext objects, one per keyframe to analyse.
        batch_size: Number of frames to submit to vLLM concurrently per batch.

    Returns:
        List of (KeyframeRecord, QwenFrameOutput | None) in ascending timestamp order.
        Frames where inference failed or the output could not be parsed have None.
    """
    # Sort by timestamp to ensure sequential processing
    sorted_contexts = sorted(contexts, key=lambda c: c.keyframe.timestamp_s)

    qwen = QwenVLLM.get()
    results: list[tuple[KeyframeRecord, QwenFrameOutput | None]] = []

    # Running buffer of successful outputs for rolling summary tracking
    recent_successful: list[QwenFrameOutput] = []

    n_batches = -(-len(sorted_contexts) // batch_size)  # ceil division
    progress = ProgressTracker(
        "Qwen L3 inference",
        total=len(sorted_contexts),
        logger=logger,
        log_every=5,  # log every 5% progress
    )

    for batch_start in range(0, len(sorted_contexts), batch_size):
        batch_contexts = sorted_contexts[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        batch_input = [
            {
                "request_id": ctx.request_id,
                "messages": ctx.messages,
            }
            for ctx in batch_contexts
        ]

        logger.info(
            "[L3] Qwen batch %d/%d (frames %d-%d of %d)",
            batch_num,
            n_batches,
            batch_start + 1,
            batch_start + len(batch_contexts),
            len(sorted_contexts),
        )

        raw_outputs = await qwen.analyse_batch(batch_input)
        progress.update(len(batch_contexts))

        for ctx, raw in zip(batch_contexts, raw_outputs):
            kf = ctx.keyframe

            if not raw:
                logger.error(
                    "Empty output for frame %s (t=%.2fs) — marking as None",
                    kf.id,
                    kf.timestamp_s,
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
                    "Unexpected error validating frame %s (t=%.2fs): %s",
                    kf.id,
                    kf.timestamp_s,
                    exc,
                    exc_info=True,
                )
                results.append((kf, None))
                continue

            results.append((kf, output))
            recent_successful.append(output)

            # Track shot boundaries for rolling summary
            if output.scene_change:
                rolling_summary = _compress_captions(recent_successful)
                logger.debug(
                    "Scene change detected at frame %s (t=%.2fs). "
                    "Rolling summary updated (%d chars).",
                    kf.id,
                    kf.timestamp_s,
                    len(rolling_summary),
                )
                # Keep only the last 3 for memory efficiency
                recent_successful = recent_successful[-3:]

    progress.done()
    logger.info(
        "[L3] Qwen analysis complete: %d frames processed, %d succeeded, %d failed.",
        len(sorted_contexts),
        sum(1 for _, o in results if o is not None),
        sum(1 for _, o in results if o is None),
    )
    return results
