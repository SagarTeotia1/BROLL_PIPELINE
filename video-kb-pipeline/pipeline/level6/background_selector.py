"""Level-6 Compositing Agent — background/B-roll selection (A3).

Implements CLAUDE.md "PIPELINE ADDENDUM — Motion Graphics/Compositing +
Hardening" -> A3 ("Background/B-roll selection (the judgment call)"): same
tier logic as Color Grading / Caption agents — retrieval is cheap and
deterministic, the *pick* is the one LLM-worthy step.

Scene-level, not cut-order-level: `background_assignments.scene_id` FKs to
the canonical `scenes` table (L4's Story Architect output), not to any
`edit_plan_id`/`cut_list_item_id` — a background pick is a property of the
scene itself, reusable across any edit plan later built from it. This
differs from `run_color_grading`/`run_caption_overlay`, both of which key
off `cut_list_items` for a specific `edit_plan_id`.

Drift note (CLAUDE.md task brief: "check which table is real vs planned"):
CLAUDE.md A3's one-line description says "embed scene transcript + `scene_
mood`/`tags` (already in `frame_analyses`)". That's half-drifted from the
actual schema — `frame_analyses` (per-FRAME) genuinely does carry `scene_
mood`/`tags`/`beat_type`, but there is no scene-level table with those
columns directly; the canonical per-SCENE table is `scenes` (from L4's
`005_l4_reasoning.sql`), which carries `summary`/`emotional_arc`/
`participants`/time range but not `scene_mood`/`tags` itself. This module
resolves the drift by building each scene's query text from BOTH: `scenes`
(the real FK target, for summary/emotional_arc/time range) plus an
aggregation of the `frame_analyses` rows whose timestamp falls inside that
scene's time range (the real source of scene_mood/tags/beat_type signal;
see `get_frame_analyses_fields_for_video`), plus the real transcript text
overlapping that window. Nothing here is invented — every input to the
embedding query is a real upstream field, just assembled across two tables
instead of one.

Grounding discipline (rule 12 / CLAUDE.md's "never invent an asset"): the
LLM pick is constrained to the `stock_assets` rows actually returned by
`search_stock_assets` for that scene — `select_background`'s tool schema
and this module's post-validation both enforce that `asset_id` is one of
the candidates offered, never invented, never borrowed from another
scene's candidate list.

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
interface — same `_get_groq_client`/`_call_groq_tool` pattern as
`pipeline/level6/color_grading_runner.py` / `caption_overlay.py` (forced
tool_choice, `json.loads()` on `tool_calls[0].function.arguments`, pydantic
validation before any write).
"""
from __future__ import annotations

import asyncio
import json
import logging

import asyncpg
from pydantic import ValidationError

from knowledge_base.postgres.queries import (
    bulk_upsert_background_assignments,
    get_frame_analyses_fields_for_video,
    get_scenes_for_video,
    get_transcript_segments_for_video,
    search_stock_assets,
)
from shared.config import settings
from shared.types import (
    BackgroundAssignmentRecord,
    SceneRecord,
    SelectBackgroundOutput,
    TranscriptSegment,
)
from shared.utils import gen_id
from prompts.background_selection import (
    L6_BACKGROUND_SELECTION_SYSTEM_PROMPT,
    SELECT_BACKGROUND_TOOL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# CLAUDE.md rule 23 — batch cap applies here too. Each scene's payload also
# carries up to _TOP_K candidates with description/tags text, so kept
# smaller than L4/L5's ~35 to compensate — same reasoning
# color_grading_runner.py used for its own smaller _COLOR_BATCH_SIZE.
_SCENE_BATCH_SIZE = 8
_BACKGROUND_MAX_TOKENS = 8192

# Top-k stock_assets candidates retrieved per scene before the LLM pick.
_TOP_K = 6

# Small, bounded retry for a transient Groq API failure — same pattern as
# L4/L5/L6-color/L6-caption's `_LLM_CALL_ATTEMPTS`, not a retry loop.
_LLM_CALL_ATTEMPTS = 2

# How many tags to keep (union, in first-seen order) when aggregating
# frame_analyses.tags across a scene's window — keeps the embedding query
# text bounded on a long/dense scene instead of growing unbounded.
_MAX_AGGREGATED_TAGS = 12

# How many transcript characters to keep from a scene's window — same
# "bounded, not the full history" discipline used for L4's rolling summary.
_MAX_TRANSCRIPT_CHARS = 600


# ---------------------------------------------------------------------------
# Groq client + generic structured tool-call helper (mirrors L4/L5/L6-color/
# L6-caption exactly).
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
    tool call and return the tool's structured arguments as a dict, or None
    if every attempt failed. See `pipeline/level6/color_grading_runner.py::
    _call_groq_tool` for the full rationale — identical pattern here."""
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
                    "L6 compositing (background) LLM call model=%s tool=%s prompt_tokens=%s completion_tokens=%s",
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
                stage=stage or "l6_background_selection",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
                latency_ms=_latency_ms,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                logger.error(
                    "L6 compositing (background) LLM call returned no tool_calls (model=%s, tool=%s)",
                    model, fn_name,
                )
                return None
            arguments = tool_calls[0].function.arguments  # JSON string
            return json.loads(arguments)
        except Exception as exc:  # noqa: BLE001 — log & retry/return, never raise into caller loop
            last_exc = exc
            logger.warning(
                "L6 compositing (background) LLM call attempt %d/%d failed (model=%s, tool=%s): %s",
                attempt, _LLM_CALL_ATTEMPTS, model, fn_name, exc,
            )
    logger.error(
        "L6 compositing (background) LLM call exhausted %d attempts (model=%s, tool=%s): %s",
        _LLM_CALL_ATTEMPTS, model, fn_name, last_exc,
    )
    return None


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Per-scene query text assembly — real upstream data only, see module
# docstring's "Drift note" for why this joins scenes + frame_analyses +
# transcript_segments instead of one single table.
# ---------------------------------------------------------------------------


def _aggregate_frame_signal(
    scene: SceneRecord, frame_rows: list[dict]
) -> tuple[str | None, list[str]]:
    """Return (majority scene_mood, aggregated tags) for frame_analyses rows
    whose timestamp falls inside [scene.start_time, scene.end_time]."""
    in_window = [
        r for r in frame_rows
        if r["timestamp_s"] is not None
        and scene.start_time <= r["timestamp_s"] <= scene.end_time
    ]
    mood_counts: dict[str, int] = {}
    tags: list[str] = []
    for r in in_window:
        mood = r.get("scene_mood")
        if mood:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        for t in r.get("tags") or []:
            if t not in tags:
                tags.append(t)
    majority_mood = max(mood_counts, key=mood_counts.get) if mood_counts else None
    return majority_mood, tags[:_MAX_AGGREGATED_TAGS]


def _transcript_text_for_scene(
    segments: list[TranscriptSegment], scene: SceneRecord
) -> str:
    overlapping = [
        seg for seg in segments
        if seg.start_time is not None and seg.end_time is not None
        and seg.start_time <= scene.end_time and seg.end_time >= scene.start_time
    ]
    overlapping.sort(key=lambda s: s.start_time)
    text = " ".join(seg.text.strip() for seg in overlapping if seg.text and seg.text.strip())
    return text[:_MAX_TRANSCRIPT_CHARS]


def _build_query_text(
    scene: SceneRecord, mood: str | None, tags: list[str], transcript_text: str
) -> str:
    """Real-data-only text handed to the embedder — every component is
    either a `scenes` column or an aggregation of real `frame_analyses`/
    `transcript_segments` rows, never invented."""
    parts: list[str] = []
    if scene.summary:
        parts.append(scene.summary)
    if scene.emotional_arc:
        parts.append(scene.emotional_arc)
    if mood:
        parts.append(f"mood: {mood}")
    if tags:
        parts.append(" ".join(tags))
    if transcript_text:
        parts.append(transcript_text)
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_background_selection(
    pool: asyncpg.Pool,
    video_id: str,
    license_type: str | None = None,
) -> list[BackgroundAssignmentRecord]:
    """Compute + write `background_assignments` for every `scenes` row of
    *video_id* that has at least one usable `stock_assets` candidate.

    Scene-level (see module docstring) — not tied to any particular
    `edit_plan_id`; re-running for the same video is a safe UPSERT keyed on
    `scene_id` (rule 9/15, `bulk_upsert_background_assignments`).

    *license_type*, when given, restricts `search_stock_assets` to that
    license tier only (see CLAUDE.md "Before building A3: stock
    licensing" — filtering is available now even though enforcement is not
    mandatory day one).

    Returns only the assignments this call actually decided on and wrote —
    scenes with zero retrievable candidates, or whose batch's LLM call
    failed outright, or whose LLM pick was `asset_id: null` (a legitimate
    "nothing fits" answer, not an error), get no row this run.
    """
    scenes = await get_scenes_for_video(pool, video_id)
    if not scenes:
        logger.warning(
            "run_background_selection: video_id=%s has no scenes — has L4 "
            "(Story Architect) run yet?",
            video_id,
        )
        return []

    frame_rows = await get_frame_analyses_fields_for_video(pool, video_id)
    segments = await get_transcript_segments_for_video(pool, video_id)

    # ------------------------------------------------------------------
    # Build each scene's grounded query text, embed, and retrieve top-k
    # stock_assets candidates.
    # ------------------------------------------------------------------
    from models.embed_model import LocalEmbedder

    embedder = LocalEmbedder.get()
    embedder.load()  # idempotent — no-op if already loaded (see embed_model.py docstring)

    query_texts: list[str] = []
    scene_by_index: list[SceneRecord] = []
    for scene in scenes:
        mood, tags = _aggregate_frame_signal(scene, frame_rows)
        transcript_text = _transcript_text_for_scene(segments, scene)
        query_text = _build_query_text(scene, mood, tags, transcript_text)
        if not query_text:
            logger.warning(
                "run_background_selection: scene=%s (video_id=%s) has no "
                "usable grounding text (no summary/emotional_arc/mood/tags/"
                "transcript in its window) — skipping, nothing to embed.",
                scene.id, video_id,
            )
            continue
        query_texts.append(query_text)
        scene_by_index.append(scene)

    if not query_texts:
        logger.warning(
            "run_background_selection: video_id=%s — no scenes had usable "
            "grounding text, nothing to do",
            video_id,
        )
        return []

    embeddings = await asyncio.to_thread(embedder.embed, query_texts)

    scene_payloads: dict[str, dict] = {}  # scene_id -> {query_text, candidates}
    for scene, query_text, embedding in zip(scene_by_index, query_texts, embeddings):
        candidates = await search_stock_assets(
            pool, embedding, limit=_TOP_K, license_type=license_type
        )
        if not candidates:
            logger.warning(
                "run_background_selection: scene=%s (video_id=%s) — "
                "search_stock_assets returned zero candidates (license_type=%r) "
                "— skipping, nothing to pick from.",
                scene.id, video_id, license_type,
            )
            continue
        scene_payloads[scene.id] = {
            "query_text": query_text,
            "candidates": [
                {
                    "asset_id": c.id,
                    "description": c.description,
                    "tags": c.tags,
                    "license_type": c.license_type,
                }
                for c in candidates
            ],
        }

    if not scene_payloads:
        logger.warning(
            "run_background_selection: video_id=%s — no scene had any "
            "retrievable stock_assets candidate, nothing to do",
            video_id,
        )
        return []

    # ------------------------------------------------------------------
    # Batch to Groq, validate, collect.
    # ------------------------------------------------------------------
    client = _get_groq_client()
    records: list[BackgroundAssignmentRecord] = []
    scene_ids = list(scene_payloads.keys())

    for batch_ids in _chunk(scene_ids, _SCENE_BATCH_SIZE):
        payload = {
            "scenes": [
                {"scene_id": sid, **scene_payloads[sid]}
                for sid in batch_ids
            ]
        }
        raw = await _call_groq_tool(
            client, settings.L6_COMPOSITING_MODEL, L6_BACKGROUND_SELECTION_SYSTEM_PROMPT,
            SELECT_BACKGROUND_TOOL, payload, _BACKGROUND_MAX_TOKENS,
            pool=pool, video_id=video_id, stage="l6_background_selection",
        )
        if raw is None:
            logger.error(
                "run_background_selection: video_id=%s a batch of %d scene(s) "
                "failed its LLM call — those scenes get no background_assignments "
                "this run",
                video_id, len(batch_ids),
            )
            continue
        try:
            parsed = SelectBackgroundOutput.model_validate(raw)
        except ValidationError as exc:
            logger.error(
                "run_background_selection: video_id=%s failed to validate "
                "response for a batch of %d scene(s): %s",
                video_id, len(batch_ids), exc,
            )
            continue

        candidate_ids_by_scene = {
            sid: {c["asset_id"] for c in scene_payloads[sid]["candidates"]}
            for sid in batch_ids
        }
        for pick in parsed.picks:
            if pick.scene_id not in candidate_ids_by_scene:
                logger.warning(
                    "run_background_selection: LLM returned scene_id=%r not in "
                    "this batch — dropping (closed-set discipline)",
                    pick.scene_id,
                )
                continue
            if pick.asset_id is None:
                logger.info(
                    "run_background_selection: scene=%s — LLM chose no "
                    "candidate ('null' pick, %s)",
                    pick.scene_id, pick.rationale or "no rationale given",
                )
                continue
            if pick.asset_id not in candidate_ids_by_scene[pick.scene_id]:
                logger.warning(
                    "run_background_selection: scene=%s — LLM returned "
                    "asset_id=%r not among this scene's own candidates — "
                    "dropping (rule 12 closed-set discipline, never invent "
                    "an asset).",
                    pick.scene_id, pick.asset_id,
                )
                continue
            records.append(
                BackgroundAssignmentRecord(
                    id=gen_id(),
                    scene_id=pick.scene_id,
                    asset_id=pick.asset_id,
                    start_offset=pick.start_offset,
                    loop=pick.loop,
                    rationale=pick.rationale or None,
                )
            )

    if records:
        await bulk_upsert_background_assignments(pool, records)

    logger.info(
        "run_background_selection: video_id=%s wrote %d/%d background_assignments "
        "(%d scenes had no usable grounding text or candidates)",
        video_id, len(records), len(scenes), len(scenes) - len(scene_payloads),
    )
    return records
