"""Level-6 Caption/Text Overlay Agent.

Implements CLAUDE.md "LEVEL 6 — SPECIALIZED ACTION AGENTS" ->
"Caption/Text Overlay Agent": for every `TEXT_OVERLAY_REQUEST` op in a
finalized `edit_plan.operations`, decide caption STYLE (the one LLM-worthy
judgment call at L6 — word-by-word karaoke timing for reels vs. a static
lower-third block for a full cut) via Groq, then build the actual FFmpeg
filter string per caption. The caption *text* itself is never generated or
paraphrased — it is read from real transcript data
(`transcript_segments.text` for the op's time window, falling back to
`frame_analyses.qwen_output.dialogue_subtitle` when no transcript segment
overlaps that window) and only echoed back by the LLM; this module
independently verifies that echo before trusting it (rule 18 — L6 never
re-interprets intent, which includes "improving" the wording).

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
interface — same `_get_groq_client`/`_call_groq_tool` pattern as
`pipeline/level4/grounding_runner.py` and
`pipeline/level5/planner_runner.py` (forced tool_choice,
`json.loads()` on `tool_calls[0].function.arguments`, pydantic validation
before use).

Filter format choice — plain chained `drawtext`, not `ass`/`subtitles`:
Both `static_block` and `karaoke_word_by_word` are built as one or more
`drawtext` filters, each gated by `enable='between(t,start,end)'`, chained
with commas into a single filter string. A `.ass` subtitle file would give
richer native karaoke fill/highlight effects, but it needs a *file path*
written to disk that survives to render time — `render_direct`'s
`extra_filters: dict[cut_list_item_id -> ffmpeg filter STRING]` extension
point (see `pipeline/level6/editing_director.py`) only carries a filter
string per clip, no side-channel for attaching a generated asset file. A
chained-`drawtext` string is fully self-contained in that one string, so
it fits the locked contract without inventing an unsupported file-passing
mechanism; word-by-word emphasis is achieved by giving emphasis words a
larger `fontsize`/different `fontcolor` in their own `drawtext` clause.
Documented limitation: this doesn't get ASS's automatic line-wrapping or
built-in karaoke fill-sweep animation — acceptable for a v1 style pass,
flagged here for anyone revisiting captions once `render_direct` grows a
side-asset channel.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import logging

import asyncpg

from knowledge_base.postgres.queries import (
    get_client_style_profile,
    get_edit_plan,
    get_frame_analyses_with_timestamps_for_video,
    get_transcript_segments_for_video,
    get_video,
)
from shared.config import settings
from shared.types import CaptionStyleItem, ChooseCaptionStyleOutput, TranscriptSegment
from prompts.l6_caption_style import (
    CHOOSE_CAPTION_STYLE_TOOL,
    L6_CAPTION_STYLE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# CLAUDE.md rule 23 — batch cap, same reasoning as L4's Grounding Agent /
# L5's Pass A: cap per-call list size so per-item quality doesn't degrade
# on an edit plan with many TEXT_OVERLAY_REQUEST ops.
_CAPTION_BATCH_SIZE = 20
_CAPTION_LLM_MAX_TOKENS = 4096

# Small, bounded retry for a transient Groq API failure — same pattern as
# L4/L5's `_LLM_CALL_ATTEMPTS`.
_LLM_CALL_ATTEMPTS = 2

# How far outside a caption's own [start_time, end_time] to still count a
# transcript_segment as "for this window" — mirrors the small slack used
# elsewhere for float boundary comparisons (e.g. L5's _TIME_BOUND_EPSILON,
# though wider here since transcript segment boundaries are Whisper-native,
# not scene-snapped).
_WINDOW_SLACK_S = 0.25

# Text-drift guard: below this similarity ratio (difflib SequenceMatcher,
# normalized), the LLM's echoed `text` is considered to have diverged from
# the real transcript source and is discarded in favor of the original
# (rule 18 — never trust the LLM's rewrite of the words themselves).
_TEXT_MATCH_MIN_RATIO = 0.90


# ---------------------------------------------------------------------------
# Groq client + generic structured tool-call helper — same pattern as
# pipeline/level4/grounding_runner.py::_call_tool /
# pipeline/level5/planner_runner.py::_call_groq_tool.
# ---------------------------------------------------------------------------


def _get_groq_client():
    """Name kept for minimal diff — actually returns an OpenRouter client
    now (see shared/llm_client.py)."""
    from shared.llm_client import get_llm_client

    return get_llm_client()


async def _call_groq_tool(
    client,
    model: str,
    system_prompt: str,
    tool_schema: dict,
    payload: dict,
    max_tokens: int,
    *,
    pool=None,
    video_id: str | None = None,
    stage: str | None = None,
) -> dict | None:
    """Call Groq's OpenAI-compatible `chat.completions.create` with a forced
    tool call, return the tool's structured arguments as a dict, or None if
    every attempt failed. See `pipeline/level5/planner_runner.py::
    _call_groq_tool` for the full explanation of why Groq's shape differs
    from Anthropic's (`tool_calls[0].function.arguments` is a JSON STRING
    here, not a pre-parsed dict)."""
    import time as _time

    fn_name = tool_schema["function"]["name"]
    last_exc: Exception | None = None
    for attempt in range(1, _LLM_CALL_ATTEMPTS + 1):
        _t0 = _time.monotonic()
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": fn_name}},
                # See pipeline/level4/grounding_runner.py::_call_tool for the
                # real-video finding behind this.
                extra_body={"provider": {"require_parameters": True}},
            )
            usage = getattr(response, "usage", None)
            _latency_ms = int((_time.monotonic() - _t0) * 1000)
            if usage is not None:
                logger.info(
                    "L6 caption LLM call model=%s tool=%s prompt_tokens=%s completion_tokens=%s",
                    model, fn_name,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )
            # L7 7c — durable cost/latency log, non-fatal.
            from shared.llm_client import log_llm_call as _log_llm_call
            await _log_llm_call(
                pool,
                video_id=video_id,
                level=6,
                stage=stage or "l6_caption",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
                latency_ms=_latency_ms,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                logger.error(
                    "L6 caption LLM call returned no tool_calls (model=%s, tool=%s)",
                    model, fn_name,
                )
                return None
            arguments = tool_calls[0].function.arguments  # JSON string
            return json.loads(arguments)
        except Exception as exc:  # noqa: BLE001 — log & retry/return, never raise into caller loop
            last_exc = exc
            logger.warning(
                "L6 caption LLM call attempt %d/%d failed (model=%s, tool=%s): %s",
                attempt, _LLM_CALL_ATTEMPTS, model, fn_name, exc,
            )
    logger.error(
        "L6 caption LLM call exhausted %d attempts (model=%s, tool=%s): %s",
        _LLM_CALL_ATTEMPTS, model, fn_name, last_exc,
    )
    return None


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Real-transcript text sourcing (never invented — CLAUDE.md grounding rule)
# ---------------------------------------------------------------------------


def _transcript_text_for_window(
    segments: list[TranscriptSegment], start: float, end: float
) -> tuple[str, list[dict]]:
    """Join `text` from every transcript_segment overlapping
    [start - slack, end + slack], in time order, plus the flattened
    word-level list (for karaoke timing) across those segments.

    Returns ("", []) if nothing overlaps — caller falls back to L3's
    `dialogue_subtitle`.
    """
    lo, hi = start - _WINDOW_SLACK_S, end + _WINDOW_SLACK_S
    overlapping = [
        seg for seg in segments
        if seg.start_time is not None and seg.end_time is not None
        and seg.start_time <= hi and seg.end_time >= lo
    ]
    overlapping.sort(key=lambda s: s.start_time)
    text = " ".join(seg.text.strip() for seg in overlapping if seg.text and seg.text.strip())
    words: list[dict] = []
    for seg in overlapping:
        for w in seg.words or []:
            try:
                words.append(
                    {"word": str(w["word"]).strip(), "start": float(w["start"]), "end": float(w["end"])}
                )
            except (KeyError, TypeError, ValueError):
                continue
    return text, words


def _dialogue_subtitle_fallback(
    frame_rows: list[dict], start: float, end: float
) -> str:
    """Fall back to L3's `frame_analyses.qwen_output.dialogue_subtitle` for
    the frame nearest the window midpoint, when no transcript_segment
    overlaps the caption's time window (e.g. a caption over a beat with no
    speech but on-screen dialogue text L3 already grounded in the
    transcript at frame-analysis time)."""
    midpoint = (start + end) / 2.0
    candidates = [r for r in frame_rows if r.get("timestamp_s") is not None]
    if not candidates:
        return ""
    nearest = min(candidates, key=lambda r: abs(r["timestamp_s"] - midpoint))
    subtitle = (nearest.get("qwen_output") or {}).get("dialogue_subtitle")
    return (subtitle or "").strip()


# ---------------------------------------------------------------------------
# Op graph helpers — TEXT_OVERLAY_REQUEST ops attach to a SELECT_CLIP op via
# that clip's `downstream_ops: ["TEXT_OVERLAY_REQUEST:<op_id>"]` list (see
# prompts/l5_sequencing.py). This builds the reverse map: overlay op_id ->
# the SELECT_CLIP op_id it applies to.
# ---------------------------------------------------------------------------


def _build_target_map(operations: list[dict]) -> dict[str, str]:
    target_by_overlay_op_id: dict[str, str] = {}
    for op in operations:
        if op.get("type") != "SELECT_CLIP":
            continue
        for ref in op.get("downstream_ops") or []:
            if not isinstance(ref, str) or ":" not in ref:
                continue
            ref_type, _, ref_op_id = ref.partition(":")
            if ref_type == "TEXT_OVERLAY_REQUEST" and ref_op_id:
                target_by_overlay_op_id[ref_op_id] = op["op_id"]
    return target_by_overlay_op_id


# ---------------------------------------------------------------------------
# FFmpeg drawtext filter building
# ---------------------------------------------------------------------------

_POSITION_Y = {
    "top": "h*0.08",
    "center": "(h-text_h)/2",
    "lower_third": "h*0.78",
    "bottom": "h-text_h-h*0.05",
}
_FONT_SIZE = {
    "small": 28,
    "medium": 40,
    "large": 56,
}
_EMPHASIS_FONT_BUMP = 12
_EMPHASIS_COLOR = "yellow"
_BASE_COLOR = "white"


def _escape_drawtext(text: str) -> str:
    """Escape a caption string for FFmpeg's `drawtext` `text=` value.

    `drawtext` needs backslash, single-quote, and colon escaped (colon is
    the filter option separator, single-quote closes the `text='...'`
    literal). Order matters: backslash first, so later-inserted escape
    backslashes aren't themselves re-escaped.
    """
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")


def _drawtext_clause(
    text: str, start: float, end: float, y_expr: str, fontsize: int, color: str
) -> str:
    escaped = _escape_drawtext(text)
    return (
        f"drawtext=text='{escaped}':fontsize={fontsize}:fontcolor={color}:"
        f"x=(w-text_w)/2:y={y_expr}:box=1:boxcolor=black@0.5:boxborderw=8:"
        f"enable='between(t,{start:.3f},{end:.3f})'"
    )


def build_caption_filter(style: CaptionStyleItem, start: float, end: float, words: list[dict]) -> str:
    """Build the FFmpeg filter string for one caption, chaining `drawtext`
    clauses (see module docstring for why `drawtext` over `ass`).

    `static_block`: one `drawtext` clause spanning [start, end].
    `karaoke_word_by_word`: one `drawtext` clause per word (clipped to
    [start, end], since word timestamps come from the underlying
    transcript_segments and should already fall inside that range — clipped
    defensively in case of upstream slack), each visible only for its own
    [word.start, word.end] window; words in `style.emphasis_words` render
    larger and in `_EMPHASIS_COLOR`.
    """
    y_expr = _POSITION_Y.get(style.position, _POSITION_Y["lower_third"])
    base_size = _FONT_SIZE.get(style.font_size_tier, _FONT_SIZE["medium"])

    if style.timing_mode == "karaoke_word_by_word" and words:
        emphasis_set = {w.strip().lower() for w in style.emphasis_words if w.strip()}
        clauses = []
        for w in words:
            w_start = max(start, w["start"])
            w_end = min(end, w["end"])
            if w_end <= w_start:
                continue
            is_emphasis = w["word"].strip().lower().strip(".,!?") in emphasis_set
            size = base_size + _EMPHASIS_FONT_BUMP if is_emphasis else base_size
            color = _EMPHASIS_COLOR if is_emphasis else _BASE_COLOR
            clauses.append(_drawtext_clause(w["word"], w_start, w_end, y_expr, size, color))
        if clauses:
            return ",".join(clauses)
        # No usable word timestamps in range — fall through to static_block.
        logger.warning(
            "build_caption_filter: karaoke_word_by_word requested but no usable "
            "word-level timestamps in [%.3f, %.3f] — falling back to static_block",
            start, end,
        )

    return _drawtext_clause(style.text, start, end, y_expr, base_size, _BASE_COLOR)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def run_caption_overlay(pool: asyncpg.Pool, edit_plan_id: str) -> list[dict]:
    """For every `TEXT_OVERLAY_REQUEST` op in `edit_plan_id`'s finalized
    operations, source real transcript text for its time window, call Groq
    for STYLE only, validate the echoed text against the source, and build
    an FFmpeg filter string.

    Returns a list of
    `{"op_id": <TEXT_OVERLAY_REQUEST op_id>, "target_op_id": <SELECT_CLIP
    op_id this caption attaches to, or None if undeterminable>, "filter":
    <ffmpeg filter string>, "style": {...}, "text": <real source text used>}`
    — same per-op-keyed shape as Engineer B's `build_ffmpeg_color_filters`
    output, ready for `pipeline/level6/updater.py` to remap `target_op_id`
    to the matching `cut_list_item.id` (via `cut_list_items.op_id`) and
    merge into Engineer A's `render_direct(..., extra_filters=...)` dict —
    the reconciliation across engineers' different keying schemes is
    `updater.py`'s integration job, not this module's.

    Ops this function skips (logged, never guessed per rule 18):
      - a TEXT_OVERLAY_REQUEST with no determinable time window (neither
        its own start_time/end_time nor a SELECT_CLIP referencing it via
        downstream_ops) — the op's rationale/parameters are underspecified
        in the EditPlan; that's a signal to fix L5's output shape, not for
        L6 to invent a window.
      - a TEXT_OVERLAY_REQUEST whose time window has no real transcript
        text available at all (neither transcript_segments nor L3's
        dialogue_subtitle) — nothing grounded to caption with.

    If this video's `client_id` resolves to a `client_style_profiles` row
    with a non-empty `caption_style` (CLAUDE.md "PIPELINE ADDENDUM 2" ->
    "3. Client Style Profiles"), it is passed to the LLM as an OPTIONAL
    styling prior alongside the existing platform-target context — a SOFT
    PRIOR only (rule 26), never a replacement for the per-line style
    judgment this agent already makes, and never license to alter `text`.
    No client_id, or a client_id with no profile / empty caption_style,
    leaves that field null and reproduces exactly today's behavior.
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        logger.error("run_caption_overlay: no edit_plan found for id=%s", edit_plan_id)
        return []

    operations = plan.operations
    overlay_ops = [op for op in operations if op.get("type") == "TEXT_OVERLAY_REQUEST"]
    if not overlay_ops:
        logger.info("run_caption_overlay: edit_plan_id=%s has no TEXT_OVERLAY_REQUEST ops", edit_plan_id)
        return []

    target_by_overlay_op_id = _build_target_map(operations)
    select_ops_by_id = {op["op_id"]: op for op in operations if op.get("type") == "SELECT_CLIP"}

    segments = await get_transcript_segments_for_video(pool, plan.video_id)
    frame_rows = await get_frame_analyses_with_timestamps_for_video(pool, plan.video_id)

    # ------------------------------------------------------------------
    # Resolve each overlay op's time window + real source text.
    # ------------------------------------------------------------------
    prepared: dict[str, dict] = {}  # op_id -> {start, end, text, words, target_op_id}
    for op in overlay_ops:
        op_id = op["op_id"]
        target_op_id = target_by_overlay_op_id.get(op_id)
        start, end = op.get("start_time"), op.get("end_time")
        if start is None or end is None:
            target_op = select_ops_by_id.get(target_op_id) if target_op_id else None
            if target_op is not None:
                start, end = target_op.get("start_time"), target_op.get("end_time")

        if start is None or end is None or start >= end:
            logger.warning(
                "run_caption_overlay: TEXT_OVERLAY_REQUEST op_id=%s has no determinable "
                "time window (own start/end null and no matching SELECT_CLIP downstream_ops "
                "reference) — L5 output underspecified, skipping (rule 18: L6 never guesses).",
                op_id,
            )
            continue

        text, words = _transcript_text_for_window(segments, start, end)
        if not text:
            text = _dialogue_subtitle_fallback(frame_rows, start, end)
            words = []  # dialogue_subtitle has no word-level timestamps — static only

        if not text:
            logger.warning(
                "run_caption_overlay: TEXT_OVERLAY_REQUEST op_id=%s window [%.3f, %.3f] has "
                "no real transcript text (no transcript_segments overlap, no "
                "dialogue_subtitle) — skipping, nothing grounded to caption with.",
                op_id, start, end,
            )
            continue

        prepared[op_id] = {
            "start": start, "end": end, "text": text, "words": words, "target_op_id": target_op_id,
        }

    if not prepared:
        logger.info("run_caption_overlay: edit_plan_id=%s no captionable TEXT_OVERLAY_REQUEST ops", edit_plan_id)
        return []

    # CLAUDE.md "PIPELINE ADDENDUM 2" -> "3. Client Style Profiles":
    # additive, soft-prior-only lookup. No client_id / no profile row /
    # empty caption_style all leave client_caption_style as None, which
    # reproduces exactly today's payload shape (see rule 26).
    client_caption_style: dict | None = None
    video = await get_video(pool, plan.video_id)
    if video is not None and video.client_id:
        profile = await get_client_style_profile(pool, video.client_id)
        if profile is not None and profile.caption_style:
            client_caption_style = profile.caption_style

    # ------------------------------------------------------------------
    # Batched Groq calls for style only.
    # ------------------------------------------------------------------
    client = _get_groq_client()
    styles_by_op_id: dict[str, CaptionStyleItem] = {}
    items = list(prepared.items())
    for batch in _chunk(items, _CAPTION_BATCH_SIZE):
        payload = {
            "platform": plan.platform,
            "client_style_prior": client_caption_style,
            "captions": [
                {"op_id": op_id, "transcript_text": info["text"], "start_time": info["start"], "end_time": info["end"]}
                for op_id, info in batch
            ],
        }
        raw = await _call_groq_tool(
            client, settings.L6_CAPTION_MODEL, L6_CAPTION_STYLE_SYSTEM_PROMPT,
            CHOOSE_CAPTION_STYLE_TOOL, payload, _CAPTION_LLM_MAX_TOKENS,
            pool=pool, video_id=plan.video_id, stage="l6_caption",
        )
        if raw is None:
            logger.error(
                "run_caption_overlay: edit_plan_id=%s a batch of %d caption(s) failed its "
                "LLM call — those ops will not be captioned this run",
                edit_plan_id, len(batch),
            )
            continue
        try:
            parsed = ChooseCaptionStyleOutput.model_validate(raw)
        except Exception as exc:
            logger.error("run_caption_overlay: failed to validate style batch response: %s", exc)
            continue
        for item in parsed.styles:
            if item.op_id not in prepared:
                logger.warning(
                    "run_caption_overlay: LLM returned op_id=%r not in this batch — "
                    "dropping (closed-set discipline)",
                    item.op_id,
                )
                continue
            styles_by_op_id[item.op_id] = item

    # ------------------------------------------------------------------
    # Text-drift guard + filter building.
    # ------------------------------------------------------------------
    results: list[dict] = []
    for op_id, info in prepared.items():
        style = styles_by_op_id.get(op_id)
        if style is None:
            # LLM call/validation failed for this op — skip rather than
            # render with an invented default style (still real text, but
            # no style judgment was actually made for it this run).
            continue

        source_text = info["text"]
        ratio = difflib.SequenceMatcher(None, style.text.strip().lower(), source_text.strip().lower()).ratio()
        if ratio < _TEXT_MATCH_MIN_RATIO:
            logger.warning(
                "run_caption_overlay: op_id=%s LLM echoed text diverged from source "
                "transcript (similarity=%.2f < %.2f) — rule 18 violation, using original "
                "transcript text instead of the LLM's version. llm_text=%r source_text=%r",
                op_id, ratio, _TEXT_MATCH_MIN_RATIO, style.text, source_text,
            )
            style = style.model_copy(update={"text": source_text})

        filter_str = build_caption_filter(style, info["start"], info["end"], info["words"])
        results.append(
            {
                "op_id": op_id,
                "target_op_id": info["target_op_id"],
                "filter": filter_str,
                "style": style.model_dump(),
                "text": style.text,
            }
        )

    logger.info(
        "run_caption_overlay: edit_plan_id=%s built %d/%d caption filter(s)",
        edit_plan_id, len(results), len(overlay_ops),
    )
    return results
