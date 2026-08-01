"""Level-6 Compositing Agent — emphasis DECISION (A5, decision half only).

Implements CLAUDE.md "PIPELINE ADDENDUM — Motion Graphics/Compositing +
Hardening" -> A5 ("Zoom in/out and highlight/emphasis effects") — decision
half ONLY: "reuses existing signal, no new data collection: `beat_type`/
`tension_level`/`emotion` from `frame_analyses`, face bboxes from `face_
appearances`. Rule-first (e.g. "zoom into speaker face when tension_level
spikes"), LLM only if a rule can't decide." The FFmpeg mechanics
(`zoompan`/`crop` Ken-Burns motion, overlay-shape callouts) are a SEPARATE,
later phase in `pipeline/level6/editing_director.py` — this module never
touches that file and never emits mechanics parameters (start_rect/
end_rect/easing/duration); `emphasis_effects.parameters` here is
decision-level only (which person/region, why) for that later phase to
consume.

Cut-order-level, not scene-level (unlike `background_selector.py`):
`emphasis_effects.cut_list_item_id` FKs to `cut_list_items` — "what to
emphasize" is inherently about a specific clip in the finalized cut, same
keying as `sequence_color_adjustments`. Requires the Editing Director to
have already run (`cut_list_items` must exist) — same precondition as
`run_color_grading`.

Deterministic rule pass (no LLM call at all for the common case):
  R1 — `tension_level` in {"high", "critical"} AND a single face clearly
       dominates the clip's screen time -> zoom on that face.
  R2 — `beat_type` in {"climax", "inciting_incident"} AND a single face
       clearly dominates -> zoom on that face.
"Dominant" = exactly one face detected, OR the top face's frame_count is
more than double the runner-up's (a clear, not marginal, majority — avoids
the rule firing on a roughly 50/50 two-shot where the deterministic pass
has no real basis to pick one person over the other).

Everything else — no dominant face, or dominant face exists but neither
rule's signal condition is met — is genuinely ambiguous and gets a single
batched Groq fallback call (never one call per clip; rule 23's batch-cap
discipline). Clips with zero detected faces in their window skip entirely
(rule-based AND LLM path) — there is nothing real to target, and per rule
12/18 this module never invents a target.

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
interface — same `_get_groq_client`/`_call_groq_tool` pattern as
`pipeline/level6/color_grading_runner.py` / `caption_overlay.py` /
`background_selector.py` (forced tool_choice, `json.loads()` on
`tool_calls[0].function.arguments`, pydantic validation before any write).
"""
from __future__ import annotations

import asyncio
import json
import logging

import asyncpg
from pydantic import ValidationError

from knowledge_base.postgres.queries import (
    bulk_insert_emphasis_effects,
    get_cut_list_items_for_edit_plan,
    get_edit_plan,
    get_face_appearances_for_video_sampled,
    get_frame_analyses_fields_for_video,
    get_persons_for_video,
    get_transcript_segments_for_video,
)
from shared.config import settings
from shared.types import (
    CutListItemRecord,
    DecideEmphasisOutput,
    EmphasisEffectRecord,
    FaceAppearanceRecord,
    TranscriptSegment,
)
from shared.utils import gen_id
from prompts.emphasis_selection import (
    DECIDE_EMPHASIS_TOOL,
    L6_EMPHASIS_DECISION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# CLAUDE.md rule 23 — batch cap, same reasoning as L4 Grounding / L5 Pass A /
# L6 caption/color agents.
_EMPHASIS_LLM_BATCH_SIZE = 30
_EMPHASIS_MAX_TOKENS = 8192

# Small, bounded retry for a transient Groq API failure.
_LLM_CALL_ATTEMPTS = 2

# A face "clearly dominates" a clip when its frame_count exceeds the
# runner-up's by more than this factor — a marginal majority (e.g. 3 vs 2
# frames) is NOT treated as a confident deterministic signal; that case
# falls through to the LLM fallback instead.
_DOMINANCE_RATIO = 2.0

# tension_level / beat_type values a rule treats as a "spike" — closed set,
# matches prompts/qwen_v2.py's documented enums (see shared/types.py's
# QwenEmotionalTone / QwenStoryBeat docstrings for the same enums).
_TENSION_SPIKE_LEVELS = {"high", "critical"}
_BEAT_SPIKE_TYPES = {"climax", "inciting_incident"}

_WINDOW_SLACK_S = 0.25
_MAX_TRANSCRIPT_CHARS = 300


# ---------------------------------------------------------------------------
# Groq client + generic structured tool-call helper (mirrors L4/L5/L6-color/
# L6-caption/background_selector.py exactly).
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
    """See `pipeline/level6/color_grading_runner.py::_call_groq_tool` for
    the full rationale — identical pattern here."""
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
                    "L6 compositing (emphasis) LLM call model=%s tool=%s prompt_tokens=%s completion_tokens=%s",
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
                stage=stage or "l6_emphasis_selection",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
                latency_ms=_latency_ms,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                logger.error(
                    "L6 compositing (emphasis) LLM call returned no tool_calls (model=%s, tool=%s)",
                    model, fn_name,
                )
                return None
            arguments = tool_calls[0].function.arguments  # JSON string
            return json.loads(arguments)
        except Exception as exc:  # noqa: BLE001 — log & retry/return, never raise into caller loop
            last_exc = exc
            logger.warning(
                "L6 compositing (emphasis) LLM call attempt %d/%d failed (model=%s, tool=%s): %s",
                attempt, _LLM_CALL_ATTEMPTS, model, fn_name, exc,
            )
    logger.error(
        "L6 compositing (emphasis) LLM call exhausted %d attempts (model=%s, tool=%s): %s",
        _LLM_CALL_ATTEMPTS, model, fn_name, last_exc,
    )
    return None


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Per-clip window aggregation — real upstream data only.
# ---------------------------------------------------------------------------


def _nearest_frame(frame_rows: list[dict], midpoint: float) -> dict | None:
    candidates = [r for r in frame_rows if r.get("timestamp_s") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r["timestamp_s"] - midpoint))


def _frame_signal_for_window(frame_rows: list[dict], start: float, end: float) -> dict:
    """Return {beat_type, tension_level, scene_mood, caption} for a clip's
    window — majority vote across every frame_analyses row whose timestamp
    falls inside [start, end]; falls back to the single nearest frame
    (by window midpoint) when none fall strictly inside, same fallback
    shape `caption_overlay.py::_dialogue_subtitle_fallback` uses."""
    in_window = [
        r for r in frame_rows
        if r.get("timestamp_s") is not None and start <= r["timestamp_s"] <= end
    ]
    if not in_window:
        nearest = _nearest_frame(frame_rows, (start + end) / 2.0)
        in_window = [nearest] if nearest else []

    def majority(key: str) -> str | None:
        counts: dict[str, int] = {}
        for r in in_window:
            v = r.get(key)
            if v:
                counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get) if counts else None

    caption = in_window[0].get("caption") if in_window else None
    return {
        "beat_type": majority("beat_type"),
        "tension_level": majority("tension_level"),
        "scene_mood": majority("scene_mood"),
        "caption": caption,
    }


def _transcript_excerpt_for_window(
    segments: list[TranscriptSegment], start: float, end: float
) -> str:
    lo, hi = start - _WINDOW_SLACK_S, end + _WINDOW_SLACK_S
    overlapping = [
        seg for seg in segments
        if seg.start_time is not None and seg.end_time is not None
        and seg.start_time <= hi and seg.end_time >= lo
    ]
    overlapping.sort(key=lambda s: s.start_time)
    text = " ".join(seg.text.strip() for seg in overlapping if seg.text and seg.text.strip())
    return text[:_MAX_TRANSCRIPT_CHARS]


def _avg_bbox(bboxes: list[dict]) -> dict | None:
    valid = [b for b in bboxes if isinstance(b, dict) and all(k in b for k in ("x", "y", "w", "h"))]
    if not valid:
        return None
    n = len(valid)
    return {
        "x": sum(float(b["x"]) for b in valid) / n,
        "y": sum(float(b["y"]) for b in valid) / n,
        "w": sum(float(b["w"]) for b in valid) / n,
        "h": sum(float(b["h"]) for b in valid) / n,
    }


def _faces_for_window(
    faces: list[FaceAppearanceRecord], start: float, end: float, pid_by_person_id: dict[str, str]
) -> list[dict]:
    """Group face_appearances within [start, end] by resolved pid
    (person_id -> persons.pid). Faces with no resolved person_id are
    excluded — a target must be a real, nameable pid (never an anonymous
    track_id) so the FFmpeg mechanics phase and any human review can
    identify who is being emphasized; this is the same "closed-set,
    real-identity-only" discipline used everywhere else in the pipeline.

    Returns candidates sorted by frame_count descending:
      [{"pid": ..., "frame_count": ..., "dominant_emotion": ...,
        "avg_bbox": {...} or None}, ...]
    """
    in_window = [
        f for f in faces
        if f.timestamp_s is not None and start <= f.timestamp_s <= end and f.person_id
    ]
    grouped: dict[str, dict] = {}
    for f in in_window:
        pid = pid_by_person_id.get(f.person_id)
        if pid is None:
            continue
        g = grouped.setdefault(pid, {"frame_count": 0, "bboxes": [], "emotions": {}})
        g["frame_count"] += 1
        if f.bbox:
            g["bboxes"].append(f.bbox)
        if f.emotion:
            g["emotions"][f.emotion] = g["emotions"].get(f.emotion, 0) + 1

    candidates = []
    for pid, g in grouped.items():
        dominant_emotion = max(g["emotions"], key=g["emotions"].get) if g["emotions"] else None
        candidates.append(
            {
                "pid": pid,
                "frame_count": g["frame_count"],
                "dominant_emotion": dominant_emotion,
                "avg_bbox": _avg_bbox(g["bboxes"]),
            }
        )
    candidates.sort(key=lambda c: c["frame_count"], reverse=True)
    return candidates


def _dominant_face(candidates: list[dict]) -> dict | None:
    """A face 'clearly dominates' when it's the only candidate, or its
    frame_count beats the runner-up's by more than `_DOMINANCE_RATIO` —
    see module docstring for why a marginal majority is deliberately NOT
    treated as a confident deterministic signal."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    top, runner_up = candidates[0], candidates[1]
    if runner_up["frame_count"] <= 0:
        return top
    if top["frame_count"] / runner_up["frame_count"] > _DOMINANCE_RATIO:
        return top
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_emphasis_selection(
    pool: asyncpg.Pool, edit_plan_id: str
) -> list[EmphasisEffectRecord]:
    """Decide zoom/highlight emphasis (WHAT/WHOM only — see module
    docstring) for every resolvable `cut_list_items` row under
    *edit_plan_id*, rule-first with a batched LLM fallback for the
    genuinely ambiguous subset, and write `emphasis_effects` rows.

    Returns only the effects this call actually decided on and wrote —
    clips with no detected faces in their window, or whose fallback
    batch's LLM call/validation failed, or for which the LLM's own honest
    answer was "none", get no row (idempotent-safe rerun: this function
    generates fresh ids per call, same documented caveat as
    `bulk_insert_cut_list_items`/`bulk_insert_sequence_color_adjustments` —
    a caller wanting a clean re-decide should clear stale rows first).
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        logger.warning("run_emphasis_selection: edit_plan_id=%s not found", edit_plan_id)
        return []

    cut_items = await get_cut_list_items_for_edit_plan(pool, edit_plan_id)
    if not cut_items:
        logger.warning(
            "run_emphasis_selection: edit_plan_id=%s has no cut_list_items — "
            "has the Editing Director run yet?",
            edit_plan_id,
        )
        return []

    frame_rows = await get_frame_analyses_fields_for_video(pool, plan.video_id)
    faces = await get_face_appearances_for_video_sampled(pool, plan.video_id)
    persons = await get_persons_for_video(pool, plan.video_id)
    pid_by_person_id = {p.id: p.pid for p in persons}
    segments = await get_transcript_segments_for_video(pool, plan.video_id)

    # ------------------------------------------------------------------
    # Rule pass — decide what can be decided deterministically; collect
    # the rest for the batched LLM fallback.
    # ------------------------------------------------------------------
    records: list[EmphasisEffectRecord] = []
    ambiguous: list[tuple[CutListItemRecord, dict, list[dict]]] = []  # (item, signal, candidates)
    n_no_faces = 0

    for item in cut_items:
        signal = _frame_signal_for_window(frame_rows, item.source_start, item.source_end)
        candidates = _faces_for_window(faces, item.source_start, item.source_end, pid_by_person_id)
        if not candidates:
            n_no_faces += 1
            continue  # nothing real to target — rule 12/18: never invent one

        dominant = _dominant_face(candidates)
        rule_hit: str | None = None
        if dominant is not None:
            if signal["tension_level"] in _TENSION_SPIKE_LEVELS:
                rule_hit = f"tension_level={signal['tension_level']!r} spike with dominant face"
            elif signal["beat_type"] in _BEAT_SPIKE_TYPES:
                rule_hit = f"beat_type={signal['beat_type']!r} spike with dominant face"

        if dominant is not None and rule_hit is not None:
            records.append(
                EmphasisEffectRecord(
                    id=gen_id(),
                    cut_list_item_id=item.id,
                    effect_type="zoom",
                    parameters={
                        "target_pid": dominant["pid"],
                        "target_bbox": dominant["avg_bbox"],
                        "frame_count": dominant["frame_count"],
                        "dominant_emotion": dominant["dominant_emotion"],
                        "decided_by": "rule",
                        "trigger": rule_hit,
                    },
                    rationale=f"Deterministic rule: {rule_hit} (frame_count={dominant['frame_count']}).",
                )
            )
        else:
            ambiguous.append((item, signal, candidates))

    # ------------------------------------------------------------------
    # LLM fallback, batched — only the genuinely ambiguous subset.
    # ------------------------------------------------------------------
    if ambiguous:
        client = _get_groq_client()
        candidates_by_item_id: dict[str, list[dict]] = {}
        prepared_items: list[dict] = []
        for item, signal, candidates in ambiguous:
            transcript_excerpt = _transcript_excerpt_for_window(
                segments, item.source_start, item.source_end
            )
            candidates_by_item_id[item.id] = candidates
            prepared_items.append(
                {
                    "cut_list_item_id": item.id,
                    "beat_type": signal["beat_type"],
                    "tension_level": signal["tension_level"],
                    "scene_mood": signal["scene_mood"],
                    "caption": signal["caption"],
                    "transcript_excerpt": transcript_excerpt,
                    "candidate_faces": [
                        {
                            "pid": c["pid"],
                            "frame_count": c["frame_count"],
                            "dominant_emotion": c["dominant_emotion"],
                        }
                        for c in candidates
                    ],
                }
            )

        for batch in _chunk(prepared_items, _EMPHASIS_LLM_BATCH_SIZE):
            payload = {"clips": batch}
            raw = await _call_groq_tool(
                client, settings.L6_COMPOSITING_MODEL, L6_EMPHASIS_DECISION_SYSTEM_PROMPT,
                DECIDE_EMPHASIS_TOOL, payload, _EMPHASIS_MAX_TOKENS,
                pool=pool, video_id=plan.video_id, stage="l6_emphasis_selection",
            )
            if raw is None:
                logger.error(
                    "run_emphasis_selection: edit_plan_id=%s a batch of %d "
                    "ambiguous clip(s) failed its LLM call — those clips get "
                    "no emphasis_effects this run",
                    edit_plan_id, len(batch),
                )
                continue
            try:
                parsed = DecideEmphasisOutput.model_validate(raw)
            except ValidationError as exc:
                logger.error(
                    "run_emphasis_selection: edit_plan_id=%s failed to validate "
                    "response for a batch of %d clip(s): %s",
                    edit_plan_id, len(batch), exc,
                )
                continue

            batch_ids = {b["cut_list_item_id"] for b in batch}
            for decision in parsed.decisions:
                if decision.cut_list_item_id not in batch_ids:
                    logger.warning(
                        "run_emphasis_selection: LLM returned cut_list_item_id=%r "
                        "not in this batch — dropping (closed-set discipline)",
                        decision.cut_list_item_id,
                    )
                    continue
                if decision.effect_type not in ("zoom", "highlight"):
                    continue  # "none" — legitimate, expected, no row written
                item_candidates = {
                    c["pid"]: c for c in candidates_by_item_id[decision.cut_list_item_id]
                }
                if decision.target_pid not in item_candidates:
                    logger.warning(
                        "run_emphasis_selection: cut_list_item=%s — LLM returned "
                        "target_pid=%r not among this clip's own candidate_faces "
                        "— dropping (rule 12 closed-set discipline).",
                        decision.cut_list_item_id, decision.target_pid,
                    )
                    continue
                target = item_candidates[decision.target_pid]
                records.append(
                    EmphasisEffectRecord(
                        id=gen_id(),
                        cut_list_item_id=decision.cut_list_item_id,
                        effect_type=decision.effect_type,
                        parameters={
                            "target_pid": target["pid"],
                            "target_bbox": target["avg_bbox"],
                            "frame_count": target["frame_count"],
                            "dominant_emotion": target["dominant_emotion"],
                            "decided_by": "llm",
                            "trigger": "llm_judgment",
                        },
                        rationale=decision.rationale or None,
                    )
                )

    if records:
        await bulk_insert_emphasis_effects(pool, records)

    logger.info(
        "run_emphasis_selection: edit_plan_id=%s wrote %d/%d emphasis_effects "
        "(%d rule-decided, %d clips had no detected faces, %d sent to LLM fallback)",
        edit_plan_id, len(records), len(cut_items),
        len(records) - sum(1 for r in records if r.parameters.get("decided_by") == "llm"),
        n_no_faces, len(ambiguous),
    )
    return records
