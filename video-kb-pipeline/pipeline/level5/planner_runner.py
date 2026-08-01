"""Level-5 Planning — Selection & Scoring (Pass A) + Sequencing & Pacing
(Pass B), hard-duration enforcement, plan validation, and diff-based
revisions.

Implements CLAUDE.md "LEVEL 5 — PLANNING (Editing Director's Plan)" end to
end for one video:

  run_selection_pass       — Pass A: batched, closed-set relevance scoring
                              of every `scenes` row against user intent +
                              hard constraints, grounded in
                              `storylines(status='final')` + `scenes`.
  run_sequencing_pass       — Pass B: ONE call turning Pass A's ranked,
                              pruned candidates into an ordered
                              EditOperation[] array.
  enforce_duration          — hard duration constraint, enforced
                              programmatically: sum SELECT_CLIP durations,
                              bounded LLM trim/extend retries, then a pure
                              -code fallback trim (never an unbounded loop).
  validate_plan             — reject-with-reason gate before a plan is
                              usable: scene_id existence, timestamp bounds,
                              contiguous sequence_index.
  run_level5_planning       — top-level orchestrator; never writes a plan
                              it knows is broken (same discipline as L4's
                              finalizer).
  apply_revision            — ONE Groq call producing a DIFF against an
                              existing plan (never a full re-plan), applied
                              to produce a new plan version.

Read contract (CLAUDE.md "Read contract (rule 16, restated as a hard
boundary)"): this module reads `storylines(status='final')` and `scenes`
only, via `get_final_storyline_for_video` / `get_scenes_for_video`. It
never touches `frame_analyses`, `kg_edges`, `speaker_turns`, or raw
`searchable_facts` — if L5 needs something not already in `storylines`/
`scenes`, that is a signal to extend L4's output shape, not to reach
around this module's queries.

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
interface. `qwen/qwen3.6-27b` for both passes (`settings.L5_SELECTION_MODEL`
/ `settings.L5_SEQUENCING_MODEL` — Groq's Qwen lineup is single-tier, same
situation as L4). Tool-call arguments come back as a JSON STRING
(`response.choices[0].message.tool_calls[0].function.arguments`) — this
module `json.loads()`s it before pydantic validation, unlike Anthropic's
pre-parsed `tool_use` block `input` dict used in the (stale, being
migrated) L4 runners.
"""
from __future__ import annotations

import asyncio
import json
import logging

import asyncpg
from pydantic import ValidationError

from knowledge_base.postgres.queries import (
    get_client_style_profile,
    get_edit_plan,
    get_final_storyline_for_video,
    get_latest_edit_plan_for_video,
    get_reward_signal,
    get_scenes_for_video,
    get_top_scoring_scenes_for_client,
    get_video,
    insert_edit_plan,
    insert_edit_plan_revision,
)
from pipeline.feedback.correction_logger import log_edit_plan_revision_correction
from shared.config import settings
from shared.types import (
    ApplyPlanRevisionOutput,
    BuildEditPlanOutput,
    EditOperationItem,
    EditPlanRecord,
    EditPlanRevisionRecord,
    ScoreCandidateScenesOutput,
)
from shared.utils import gen_id
from prompts.l5_selection import L5_SELECTION_SYSTEM_PROMPT, SCORE_CANDIDATE_SCENES_TOOL
from prompts.l5_sequencing import (
    APPLY_PLAN_REVISION_TOOL,
    BUILD_EDIT_PLAN_TOOL,
    L5_DURATION_CORRECTION_SYSTEM_PROMPT,
    L5_REVISION_SYSTEM_PROMPT,
    L5_SEQUENCING_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# CLAUDE.md rule 23 / "Batch cap, same reasoning as the L4 Grounding Agent" —
# cap Pass A at ~30-40 scenes/call, sub-batch, merge scores.
_SELECTION_BATCH_SIZE = 35
_SELECTION_MAX_TOKENS = 8192
_SEQUENCING_MAX_TOKENS = 8192

# CLAUDE.md rule 22 — duration-fit retries cap at 2-3 attempts, then a
# programmatic fallback (never an unbounded LLM retry loop).
_DURATION_CORRECTION_MAX_ATTEMPTS = 3

# PIPELINE ADDENDUM 3, LEVEL 9, item 9b (reward -> few-shot injection).
# Both numbers given literally in CLAUDE.md 9b's text.
_CLIENT_STYLE_SAMPLE_FLOOR = 5
_CLIENT_STYLE_EXAMPLE_LIMIT = 5
_DEFAULT_DURATION_TOLERANCE_PCT = 10.0

# Small, bounded retry for a transient Groq API failure (network blip,
# 5xx) — not a retry loop. Same pattern as L4's `_LLM_CALL_ATTEMPTS`.
_LLM_CALL_ATTEMPTS = 2

# Float tolerance for the "inside scene bounds" timestamp check — scene
# boundaries and LLM-returned floats can differ by sub-millisecond rounding.
_TIME_BOUND_EPSILON = 0.05

# EditOperation.type closed set — CLAUDE.md "EditPlan schema": L5's own
# decisions (SELECT_CLIP/TRIM/REORDER/CUT_TO) plus L6 dispatch requests.
_ALL_OP_TYPES = {
    "SELECT_CLIP",
    "TRIM",
    "REORDER",
    "CUT_TO",
    "COLOR_MATCH_REQUEST",
    "TEXT_OVERLAY_REQUEST",
    "AUDIO_DUCK_REQUEST",
    "B_ROLL_INSERT_REQUEST",
}
_SELECT_CLIP = "SELECT_CLIP"


# ---------------------------------------------------------------------------
# Groq client + generic structured tool-call helper
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
    pool: asyncpg.Pool | None = None,
    video_id: str | None = None,
    stage: str | None = None,
) -> dict | None:
    """Call Groq's OpenAI-compatible `chat.completions.create` with a forced
    tool call and return the tool's structured arguments as a dict, or None
    if every attempt failed.

    `tool_choice={"type": "function", "function": {"name": ...}}` forces
    the model to return the structured tool call — no free-text JSON
    parsing. IMPORTANT: unlike Anthropic's Messages API (pre-parsed
    `tool_use.input` dict), Groq's OpenAI-compatible shape returns
    `message.tool_calls[0].function.arguments` as a JSON STRING that must
    be `json.loads()`-ed before pydantic validation.
    """
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
                # real-video finding behind this — OpenRouter routes to
                # multiple backend providers with inconsistent tool-calling
                # reliability; this restricts routing to providers that
                # declare full support for every param in this request.
                extra_body={"provider": {"require_parameters": True}},
            )
            usage = getattr(response, "usage", None)
            _latency_ms = int((_time.monotonic() - _t0) * 1000)
            if usage is not None:
                logger.info(
                    "L5 planning LLM call model=%s tool=%s prompt_tokens=%s completion_tokens=%s",
                    model,
                    fn_name,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )
            # L7 7c — durable cost/latency log, non-fatal.
            from shared.llm_client import log_llm_call as _log_llm_call
            await _log_llm_call(
                pool,
                video_id=video_id,
                level=5,
                stage=stage or "l5_planning",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
                latency_ms=_latency_ms,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                logger.error(
                    "L5 planning LLM call returned no tool_calls (model=%s, tool=%s)",
                    model, fn_name,
                )
                return None
            arguments = tool_calls[0].function.arguments  # JSON string
            return json.loads(arguments)
        except Exception as exc:  # noqa: BLE001 — log & retry/return, never raise into caller loop
            last_exc = exc
            logger.warning(
                "L5 planning LLM call attempt %d/%d failed (model=%s, tool=%s): %s",
                attempt, _LLM_CALL_ATTEMPTS, model, fn_name, exc,
            )
    logger.error(
        "L5 planning LLM call exhausted %d attempts (model=%s, tool=%s): %s",
        _LLM_CALL_ATTEMPTS, model, fn_name, last_exc,
    )
    return None


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Pass A — Selection & Scoring
# ---------------------------------------------------------------------------


async def run_selection_pass(
    pool: asyncpg.Pool,
    video_id: str,
    user_prompt: str,
    target_duration_s: float | None = None,
    platform: str | None = None,
    must_include: list[str] | None = None,
    must_exclude: list[str] | None = None,
) -> list[dict]:
    """Pass A: score every `scenes` row for *video_id* against *user_prompt*
    + hard constraints, batched at `_SELECTION_BATCH_SIZE`/call, merged into
    one ranked (descending relevance_score) candidate list.

    Reads `storylines(status='final')` only to confirm L4 has finalized
    this video (L5 read-contract gate) — the actual scoring input is
    `scenes` (canonical_scene_id/summary/participants/emotional_arc/
    causal_link_to_next/usability_score), per the read contract box in
    CLAUDE.md. Returns [] if there is no final storyline, no scenes, or
    every batch's LLM call failed.
    """
    storyline = await get_final_storyline_for_video(pool, video_id)
    if storyline is None:
        logger.warning(
            "run_selection_pass: video_id=%s has no storylines(status='final') row — "
            "L5 read contract requires a finalized storyline before planning (has L4 run yet?)",
            video_id,
        )
        return []

    scenes = await get_scenes_for_video(pool, video_id)
    if not scenes:
        logger.warning("run_selection_pass: video_id=%s has no scenes rows — nothing to score", video_id)
        return []

    must_include = must_include or []
    must_exclude = must_exclude or []

    scenes_by_id = {s.id: s for s in scenes}
    known_scene_ids = set(scenes_by_id)
    candidate_payload = [
        {
            "scene_id": s.id,
            "canonical_scene_id": s.canonical_scene_id,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "participants": s.participants,
            "summary": s.summary,
            "emotional_arc": s.emotional_arc,
            "causal_link_to_next": s.causal_link_to_next,
            "usability_score": s.usability_score,
        }
        for s in scenes
    ]

    client = _get_groq_client()
    scored: dict[str, dict] = {}
    batches = _chunk(candidate_payload, _SELECTION_BATCH_SIZE)
    for batch_idx, batch in enumerate(batches):
        payload = {
            "user_prompt": user_prompt,
            "target_duration_s": target_duration_s,
            "platform": platform,
            "must_include": must_include,
            "must_exclude": must_exclude,
            "candidate_scenes": batch,
        }
        raw = await _call_groq_tool(
            client, settings.L5_SELECTION_MODEL, L5_SELECTION_SYSTEM_PROMPT,
            SCORE_CANDIDATE_SCENES_TOOL, payload, _SELECTION_MAX_TOKENS,
            pool=pool, video_id=video_id, stage="l5_selection",
        )
        if raw is None:
            logger.error(
                "run_selection_pass: video_id=%s batch %d/%d LLM call failed — "
                "those scenes will not be ranked this run",
                video_id, batch_idx + 1, len(batches),
            )
            continue
        try:
            parsed = ScoreCandidateScenesOutput.model_validate(raw)
        except ValidationError as exc:
            logger.error(
                "run_selection_pass: video_id=%s batch %d/%d failed to validate response: %s",
                video_id, batch_idx + 1, len(batches), exc,
            )
            continue
        for item in parsed.candidates:
            if item.scene_id not in known_scene_ids:
                logger.warning(
                    "run_selection_pass: video_id=%s LLM returned scene_id=%r not in "
                    "candidate_scenes — dropping (closed-set discipline)",
                    video_id, item.scene_id,
                )
                continue
            scored[item.scene_id] = {
                "relevance_score": item.relevance_score,
                "rationale": item.rationale,
            }

    ranked: list[dict] = []
    for scene_id, score in scored.items():
        s = scenes_by_id[scene_id]
        ranked.append(
            {
                "scene_id": s.id,
                "canonical_scene_id": s.canonical_scene_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "participants": s.participants,
                "summary": s.summary,
                "emotional_arc": s.emotional_arc,
                "causal_link_to_next": s.causal_link_to_next,
                "usability_score": s.usability_score,
                "relevance_score": score["relevance_score"],
                "rationale": score["rationale"],
            }
        )
    ranked.sort(key=lambda r: r["relevance_score"], reverse=True)
    logger.info(
        "run_selection_pass: video_id=%s scored %d/%d scenes across %d batch(es)",
        video_id, len(ranked), len(scenes), len(batches),
    )
    return ranked


# ---------------------------------------------------------------------------
# Pass B — Sequencing & Pacing
# ---------------------------------------------------------------------------


async def run_sequencing_pass(
    pool: asyncpg.Pool,
    video_id: str,
    ranked_candidates: list[dict],
    target_duration_s: float | None = None,
    platform: str | None = None,
    pacing_preference: str | None = None,
    client_pacing_preference: str | None = None,
    user_prompt: str | None = None,
    client_style_examples: list[dict] | None = None,
) -> list[dict]:
    """Pass B: ONE Groq call turning Pass A's ranked/pruned candidates into
    the ordered `EditOperation[]` array. `pool`/`video_id` are accepted for
    a consistent signature with `run_selection_pass` (and for future use —
    e.g. re-validating candidates against live `scenes` state) but this
    pass makes no additional DB reads; all context comes from
    *ranked_candidates*, already bounded by Pass A regardless of source
    video length (CLAUDE.md: "context bounded regardless of source video
    length").

    *client_pacing_preference* is the OPTIONAL soft prior from CLAUDE.md
    "PIPELINE ADDENDUM 2" -> "3. Client Style Profiles" —
    `client_style_profiles.pacing_preference` for this video's `client_id`,
    when a profile row exists (looked up by the caller,
    `run_level5_planning`, via `get_video` + `get_client_style_profile`).
    Distinct from *pacing_preference* (the user's explicit per-edit
    request): the two are never conflated, and the prompt itself (rule 26)
    tells the model the explicit user request always wins on conflict.
    Leaving this None (no profile, or no client_id on the video) reproduces
    exactly today's prompt/behavior — additive, not required.

    *client_style_examples* — CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 9 —
    REWARD & PUNISHMENT", item 9b: an OPTIONAL, bounded (<= 5, rule 23) list
    of this client's best-reward-scored past scene summaries, looked up by
    the caller (`run_level5_planning`) via `get_reward_signal(scope_type=
    'client')` + `get_top_scoring_scenes_for_client` ONLY when that
    reward_signals row has `sample_count >= 5`. Soft style prior only (rule
    26) — never a source of grounded content, and an empty/None list (the
    common case today — see 9b's live-data caveat) reproduces exactly
    today's prompt/behavior.

    *user_prompt* — real-video finding, first live L5 run: the user's raw
    editing intent previously reached ONLY Pass A (relevance scoring), never
    this pass, so an explicit structural instruction in the prompt itself
    ("start with X, then Y, end with Z") had no channel to influence
    ordering — Pass B ordered purely by causal_link_to_next + relevance,
    producing a sequence that ignored the user's literal requested structure
    even though Pass A had correctly scored the right scenes as relevant.
    Passed through now so Pass B can honor explicit ordering language when
    present; still subordinate to grounded data (rule 18 — L6/L5 don't
    invent content, they sequence what Pass A already selected).
    """
    if not ranked_candidates:
        logger.warning(
            "run_sequencing_pass: video_id=%s got no ranked_candidates — nothing to sequence",
            video_id,
        )
        return []

    client = _get_groq_client()
    payload = {
        "ranked_candidates": ranked_candidates,
        "target_duration_s": target_duration_s,
        "platform": platform,
        "pacing_preference": pacing_preference,
        "client_style_pacing_preference": client_pacing_preference,
        "user_prompt": user_prompt,
        # PIPELINE ADDENDUM 3, LEVEL 9, item 9b — soft style prior only
        # (rule 26), see this function's docstring addition below and
        # prompts/l5_sequencing.py. Empty list is the common case and must
        # change nothing about how Pass B sequences.
        "client_style_examples": client_style_examples or [],
    }
    raw = await _call_groq_tool(
        client, settings.L5_SEQUENCING_MODEL, L5_SEQUENCING_SYSTEM_PROMPT,
        BUILD_EDIT_PLAN_TOOL, payload, _SEQUENCING_MAX_TOKENS,
        pool=pool, video_id=video_id, stage="l5_sequencing",
    )
    if raw is None:
        logger.error("run_sequencing_pass: video_id=%s LLM call failed", video_id)
        return []

    # Per-item, not per-batch (same fix as L4's grounding_runner malformed-
    # cluster handling): a single malformed op must not discard every other
    # op in an otherwise-valid plan. Real-video finding: `BuildEditPlanOutput.
    # model_validate(raw)` as one all-or-nothing call meant 3 dispatch ops
    # missing sequence_index (correctly, per the prompt's own instruction —
    # see EditOperationItem.sequence_index docstring) discarded an entire
    # valid 18-op plan.
    raw_ops = raw.get("operations", []) if isinstance(raw, dict) else []
    if not isinstance(raw_ops, list):
        logger.error(
            "run_sequencing_pass: video_id=%s 'operations' missing or not a list in tool output",
            video_id,
        )
        return []
    parsed_ops: list[EditOperationItem] = []
    for i, item_raw in enumerate(raw_ops):
        try:
            parsed_ops.append(EditOperationItem.model_validate(item_raw))
        except ValidationError as exc:
            logger.error(
                "run_sequencing_pass: video_id=%s dropping malformed op at index %d: %s | raw=%r",
                video_id, i, exc, item_raw,
            )

    known_scene_ids = {c["scene_id"] for c in ranked_candidates}
    candidates_by_scene_id = {c["scene_id"]: c for c in ranked_candidates}
    operations: list[dict] = []
    n_dropped = 0
    n_backfilled = 0
    for op in parsed_ops:
        if op.type not in _ALL_OP_TYPES:
            logger.warning(
                "run_sequencing_pass: video_id=%s dropping op_id=%s with unknown type=%r",
                video_id, op.op_id, op.type,
            )
            n_dropped += 1
            continue
        if op.scene_id is not None and op.scene_id not in known_scene_ids:
            logger.warning(
                "run_sequencing_pass: video_id=%s op_id=%s references scene_id=%r not in "
                "ranked_candidates — dropping (closed-set discipline)",
                video_id, op.op_id, op.scene_id,
            )
            n_dropped += 1
            continue
        op_dict = op.model_dump()
        if op.type == _SELECT_CLIP:
            # Real-video finding, first live L5 run: the tool schema marks
            # scene_id/start_time/end_time nullable (so the model can return
            # a dispatch-only op), but the model sometimes omits them on a
            # genuine SELECT_CLIP too, and validate_plan hard-rejects the
            # WHOLE plan on the first such op — same failure class as
            # several L4 real-video findings (don't trust the model to
            # re-emit ground truth it was already given). Fix, same
            # principle as story_architect_runner's "never trust the LLM's
            # own start/end, use real data instead": a null scene_id is
            # unrecoverable (nothing to look up) — drop the op. A present
            # scene_id with null start_time/end_time IS recoverable — Pass
            # A's ranked_candidates already carries that scene's real
            # bounds, so backfill the full-clip span instead of discarding
            # a perfectly valid selection over one missing field pair.
            if op.scene_id is None:
                logger.warning(
                    "run_sequencing_pass: video_id=%s op_id=%s is SELECT_CLIP with no "
                    "scene_id — unrecoverable, dropping",
                    video_id, op.op_id,
                )
                n_dropped += 1
                continue
            if op.start_time is None or op.end_time is None:
                candidate = candidates_by_scene_id[op.scene_id]
                op_dict["start_time"] = candidate["start_time"]
                op_dict["end_time"] = candidate["end_time"]
                n_backfilled += 1
                logger.info(
                    "run_sequencing_pass: video_id=%s op_id=%s missing start_time/end_time — "
                    "backfilled full scene span [%.2f, %.2f) from ranked_candidates",
                    video_id, op.op_id, candidate["start_time"], candidate["end_time"],
                )
        operations.append(op_dict)

    logger.info(
        "run_sequencing_pass: video_id=%s produced %d operations (%d dropped, %d backfilled)",
        video_id, len(operations), n_dropped, n_backfilled,
    )
    return operations


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


def _select_clip_duration(operations: list[dict]) -> float:
    total = 0.0
    for op in operations:
        if op.get("type") != _SELECT_CLIP:
            continue
        start, end = op.get("start_time"), op.get("end_time")
        if start is not None and end is not None:
            total += max(0.0, end - start)
    return total


def _reorder_for_sequencing(operations: list[dict]) -> list[dict]:
    """Group SELECT_CLIP ops (sorted by their own sequence_index) first,
    followed by every other op type, preserving their relative order.
    Used before `_resequence` so contiguous renumbering has a stable,
    predictable input order."""
    select_ops = sorted(
        (op for op in operations if op.get("type") == _SELECT_CLIP),
        key=lambda op: op.get("sequence_index", 0),
    )
    other_ops = [op for op in operations if op.get("type") != _SELECT_CLIP]
    return select_ops + other_ops


def _resequence(operations: list[dict]) -> list[dict]:
    """Reassign `sequence_index` contiguously from 0 across SELECT_CLIP ops
    only, in the given list order (CLAUDE.md validation rule — "sequence_
    index is contiguous, no gaps/dupes"). Non-SELECT_CLIP ops are returned
    unchanged."""
    result: list[dict] = []
    idx = 0
    for op in operations:
        if op.get("type") == _SELECT_CLIP:
            op = dict(op)
            op["sequence_index"] = idx
            idx += 1
        result.append(op)
    return result


def _normalize_operations(operations: list[dict]) -> list[dict]:
    return _resequence(_reorder_for_sequencing(operations))


def _programmatic_trim(
    operations: list[dict], target_duration_s: float, relevance_by_scene_id: dict[str, float]
) -> list[dict]:
    """Pure code, no LLM call (CLAUDE.md rule 22 fallback): drop the
    lowest-relevance SELECT_CLIP op(s), by `relevance_score` from Pass A
    joined back in via *relevance_by_scene_id*, until the remaining
    SELECT_CLIP duration fits *target_duration_s*.

    Only trims — does not attempt to programmatically *extend* a
    too-short plan (picking new content to add back in is a judgment call,
    not something safe to do without LLM/human review); a still-short
    plan after this fallback is left as-is rather than silently padded.

    Known limitation: does not prune dispatch-request ops (COLOR_MATCH_
    REQUEST etc.) left orphaned by a dropped SELECT_CLIP's `downstream_ops`
    reference. `validate_plan` does not check for orphaned downstream
    references, so this is non-fatal — flagged here for a future pass.
    """
    current = _select_clip_duration(operations)
    if current <= target_duration_s:
        return operations

    select_ops = [op for op in operations if op.get("type") == _SELECT_CLIP]
    drop_order = sorted(select_ops, key=lambda op: relevance_by_scene_id.get(op.get("scene_id"), 0.0))

    dropped_ids: set[str] = set()
    remaining = current
    for op in drop_order:
        if remaining <= target_duration_s:
            break
        dropped_ids.add(op["op_id"])
        remaining -= max(0.0, (op.get("end_time") or 0.0) - (op.get("start_time") or 0.0))

    result = [op for op in operations if op.get("op_id") not in dropped_ids]
    logger.info(
        "_programmatic_trim: dropped %d lowest-relevance SELECT_CLIP op(s) to fit "
        "target_duration_s=%.1f (was %.1fs)",
        len(dropped_ids), target_duration_s, current,
    )
    return _normalize_operations(result)


async def enforce_duration(
    operations: list[dict],
    target_duration_s: float | None,
    ranked_candidates: list[dict] | None = None,
    tolerance_pct: float = _DEFAULT_DURATION_TOLERANCE_PCT,
    *,
    pool: asyncpg.Pool | None = None,
    video_id: str | None = None,
) -> list[dict]:
    """Hard duration constraint, enforced programmatically (CLAUDE.md "Hard
    vs. soft constraints — don't trust the LLM with arithmetic"): sum
    `(end_time - start_time)` across SELECT_CLIP ops, compare to
    *target_duration_s*. Within *tolerance_pct*, returns *operations*
    unchanged. If not, feeds the specific overage/shortfall back to Groq
    asking it to trim/extend from the EXISTING selected set (never
    re-planning from scratch), bounded at `_DURATION_CORRECTION_MAX_
    ATTEMPTS` (2-3, per rule 22). If still out of tolerance after that,
    falls back to `_programmatic_trim` — pure code, no further LLM call.

    Deviation from the literal 3-arg stub named in the task description:
    this adds an optional *ranked_candidates* parameter. It is needed for
    two things the spec requires but the 3-arg signature can't carry: (1)
    giving the correction LLM call scene-bound context so it doesn't
    invent a timestamp outside a scene's range, and (2) relevance scores
    for the final programmatic-trim fallback ("drop the lowest-relevance
    clip(s)" — relevance_score only exists on Pass A's output, not on the
    operations themselves). If omitted, the LLM correction call still runs
    (with an empty candidates list) but the fallback trim degrades to
    treating every clip as equal relevance (drops in list order).

    If *target_duration_s* is None or <= 0 (no hard constraint given),
    returns *operations* unchanged — there is nothing to enforce.
    """
    if target_duration_s is None or target_duration_s <= 0:
        return operations

    ranked_candidates = ranked_candidates or []
    relevance_by_scene_id = {c["scene_id"]: c.get("relevance_score", 0.0) for c in ranked_candidates}
    known_scene_ids = {c["scene_id"] for c in ranked_candidates} if ranked_candidates else None

    tolerance = target_duration_s * (tolerance_pct / 100.0)
    current = _select_clip_duration(operations)
    if abs(current - target_duration_s) <= tolerance:
        return operations

    client = _get_groq_client()
    corrected = operations
    for attempt in range(1, _DURATION_CORRECTION_MAX_ATTEMPTS + 1):
        current = _select_clip_duration(corrected)
        if abs(current - target_duration_s) <= tolerance:
            return corrected

        overage = current - target_duration_s
        payload = {
            "operations": corrected,
            "ranked_candidates": ranked_candidates,
            "target_duration_s": target_duration_s,
            "current_duration_s": current,
            "overage_s": overage,
        }
        raw = await _call_groq_tool(
            client, settings.L5_SEQUENCING_MODEL, L5_DURATION_CORRECTION_SYSTEM_PROMPT,
            BUILD_EDIT_PLAN_TOOL, payload, _SEQUENCING_MAX_TOKENS,
            pool=pool, video_id=video_id, stage="l5_duration_correction",
        )
        if raw is None:
            logger.warning(
                "enforce_duration: correction attempt %d/%d LLM call failed",
                attempt, _DURATION_CORRECTION_MAX_ATTEMPTS,
            )
            break
        try:
            parsed = BuildEditPlanOutput.model_validate(raw)
        except ValidationError as exc:
            logger.warning(
                "enforce_duration: correction attempt %d/%d failed to validate response: %s",
                attempt, _DURATION_CORRECTION_MAX_ATTEMPTS, exc,
            )
            break

        new_ops: list[dict] = []
        for op in parsed.operations:
            if op.type not in _ALL_OP_TYPES:
                continue
            if known_scene_ids is not None and op.scene_id is not None and op.scene_id not in known_scene_ids:
                continue
            new_ops.append(op.model_dump())
        if new_ops:
            corrected = new_ops
        else:
            logger.warning(
                "enforce_duration: correction attempt %d/%d returned no usable operations, keeping prior result",
                attempt, _DURATION_CORRECTION_MAX_ATTEMPTS,
            )
            break

    # Bounded LLM retries exhausted (or broke early) — programmatic
    # fallback, no LLM call (rule 22).
    current = _select_clip_duration(corrected)
    if abs(current - target_duration_s) <= tolerance:
        return corrected
    logger.warning(
        "enforce_duration: LLM correction attempts exhausted (current=%.1fs target=%.1fs) — "
        "falling back to programmatic trim",
        current, target_duration_s,
    )
    return _programmatic_trim(corrected, target_duration_s, relevance_by_scene_id)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


async def validate_plan(pool: asyncpg.Pool, video_id: str, operations: list[dict]) -> str | None:
    """Returns a failure reason string, or None if *operations* is a usable
    plan for *video_id*. Never silently coerces — every check either
    passes or the whole plan is rejected with a specific reason (CLAUDE.md
    "Validation before a plan is usable").

    Checks:
      - operations is non-empty and contains at least one SELECT_CLIP op
      - every op `type` is in the closed EditOperation-type set
      - every op with a non-null `scene_id` references a scene that
        actually exists in `scenes` for this video_id
      - every op with a non-null `scene_id` + start_time/end_time falls
        inside that scene's own [start_time, end_time] bounds
      - SELECT_CLIP ops all carry a scene_id + start_time + end_time
      - `sequence_index` is contiguous starting at 0, no gaps/dupes,
        across SELECT_CLIP ops only
    """
    if not operations:
        return "operations list is empty"

    for op in operations:
        op_type = op.get("type")
        if op_type not in _ALL_OP_TYPES:
            return f"operation {op.get('op_id')!r} has unknown type {op_type!r}"

    select_ops = [op for op in operations if op.get("type") == _SELECT_CLIP]
    if not select_ops:
        return "plan contains no SELECT_CLIP operations"

    for op in select_ops:
        if not op.get("scene_id"):
            return f"SELECT_CLIP op {op.get('op_id')!r} is missing scene_id"
        if op.get("start_time") is None or op.get("end_time") is None:
            return f"SELECT_CLIP op {op.get('op_id')!r} is missing start_time/end_time"
        if op.get("sequence_index") is None:
            # sequence_index is optional at the schema level (dispatch ops
            # legitimately omit it — see EditOperationItem docstring), but a
            # SELECT_CLIP op needs a real int for the contiguity check below
            # to even run; sorted() would TypeError on a mix of int/None.
            return f"SELECT_CLIP op {op.get('op_id')!r} is missing sequence_index"

    scenes = await get_scenes_for_video(pool, video_id)
    scenes_by_id = {s.id: s for s in scenes}

    for op in operations:
        scene_id = op.get("scene_id")
        if scene_id is None:
            continue
        if scene_id not in scenes_by_id:
            return (
                f"operation {op.get('op_id')!r} references scene_id={scene_id!r} "
                f"not found in scenes for video_id={video_id}"
            )
        start, end = op.get("start_time"), op.get("end_time")
        if start is None or end is None:
            continue
        scene = scenes_by_id[scene_id]
        if (
            start < scene.start_time - _TIME_BOUND_EPSILON
            or end > scene.end_time + _TIME_BOUND_EPSILON
            or start >= end
        ):
            return (
                f"operation {op.get('op_id')!r} time range [{start}, {end}] falls outside "
                f"scene {scene_id!r} bounds [{scene.start_time}, {scene.end_time}] — "
                "no hallucinated timestamps"
            )

    indices = sorted(op.get("sequence_index") for op in select_ops)
    expected = list(range(len(select_ops)))
    if indices != expected:
        return (
            f"SELECT_CLIP sequence_index values {indices} are not contiguous from 0 "
            f"(expected {expected})"
        )

    return None


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def run_level5_planning(
    pool: asyncpg.Pool,
    video_id: str,
    user_prompt: str,
    target_duration_s: float | None = None,
    platform: str | None = None,
    must_include: list[str] | None = None,
    must_exclude: list[str] | None = None,
    pacing_preference: str | None = None,
) -> dict:
    """Run Level-5 planning for *video_id* end to end:
    run_selection_pass -> run_sequencing_pass -> enforce_duration ->
    validate_plan -> write `edit_plans` (only if validation passes).

    Versioning semantics: `edit_plans.version` is scoped to `video_id`
    only (matching the schema's `UNIQUE (video_id, version)` constraint,
    which has no per-prompt grouping key). Each successful planning run
    for a video — regardless of `user_prompt` — takes the next version
    number after `get_latest_edit_plan_for_video`'s current max, the same
    "insert a new version, never overwrite" discipline already used for
    `storylines` (rule 15). A different user_prompt on the same video is
    NOT a fresh version-1 sequence; it is simply the next version of that
    video's plan history. This keeps "get the latest plan for a video"
    O(1) and unambiguous, at the cost of version numbers not being
    prompt-scoped — acceptable since `edit_plans.user_prompt` is stored on
    each row for full audit/lineage anyway.

    Returns `{"ok": True, "edit_plan_id": ..., "version": ..., "achieved_duration_s": ...}`
    on success, or `{"ok": False, "reason": "..."}` without writing
    anything if any stage fails — same "never write a plan you know is
    broken" discipline as L4's finalizer.
    """
    storyline = await get_final_storyline_for_video(pool, video_id)
    if storyline is None:
        return {
            "ok": False,
            "reason": (
                f"no storylines(status='final') row for video_id={video_id} — "
                "L4 must finalize a storyline before L5 can plan"
            ),
        }

    ranked = await run_selection_pass(
        pool, video_id, user_prompt, target_duration_s, platform, must_include, must_exclude
    )
    if not ranked:
        return {
            "ok": False,
            "reason": "Pass A (Selection & Scoring) produced no ranked candidate scenes",
        }

    # CLAUDE.md "PIPELINE ADDENDUM 2" -> "3. Client Style Profiles": additive,
    # soft-prior-only lookup. A video with no client_id, or a client_id with
    # no profile row, leaves client_pacing_preference as None — Pass B's
    # prompt/behavior is then byte-for-byte what it was before this feature.
    client_pacing_preference: str | None = None
    video = await get_video(pool, video_id)
    if video is not None and video.client_id:
        profile = await get_client_style_profile(pool, video.client_id)
        if profile is not None:
            client_pacing_preference = profile.pacing_preference

    # PIPELINE ADDENDUM 3, LEVEL 9, item 9b — reward -> few-shot injection.
    # Additive, soft-prior-only (rule 26). A video with no client_id, or a
    # client_id whose reward_signals(scope_type='client') row doesn't exist
    # yet or has sample_count below the floor, leaves client_style_examples
    # as an empty list — byte-for-byte the same prompt/behavior as before
    # this feature. On real data today, no video has client_id set (see
    # migration 016_client_style_profiles.sql's videos.client_id, still
    # unpopulated) so this path is correct-and-wired but not yet reachable —
    # honestly reflects the same state as client_pacing_preference above.
    client_style_examples: list[dict] = []
    if video is not None and video.client_id:
        reward_signal = await get_reward_signal(pool, "client", video.client_id)
        if reward_signal is not None and reward_signal.sample_count >= _CLIENT_STYLE_SAMPLE_FLOOR:
            top_scenes = await get_top_scoring_scenes_for_client(
                pool, video.client_id, limit=_CLIENT_STYLE_EXAMPLE_LIMIT
            )
            client_style_examples = [
                {
                    "canonical_scene_id": s["canonical_scene_id"],
                    "summary": s["summary"],
                    "emotional_arc": s["emotional_arc"],
                }
                for s in top_scenes
            ]

    operations = await run_sequencing_pass(
        pool, video_id, ranked, target_duration_s, platform, pacing_preference,
        client_pacing_preference, user_prompt, client_style_examples,
    )
    if not operations:
        return {
            "ok": False,
            "reason": "Pass B (Sequencing & Pacing) produced no usable operations",
        }

    operations = await enforce_duration(
        operations, target_duration_s, ranked_candidates=ranked, pool=pool, video_id=video_id,
    )

    reason = await validate_plan(pool, video_id, operations)
    if reason is not None:
        return {"ok": False, "reason": reason}

    achieved_duration_s = _select_clip_duration(operations)

    latest = await get_latest_edit_plan_for_video(pool, video_id)
    version = (latest.version + 1) if latest is not None else 1

    plan = EditPlanRecord(
        id=gen_id(),
        video_id=video_id,
        storyline_id=storyline.id,
        user_prompt=user_prompt,
        target_duration_s=target_duration_s,
        platform=platform,
        status="draft",
        version=version,
        operations=operations,
        achieved_duration_s=achieved_duration_s,
    )
    edit_plan_id = await insert_edit_plan(pool, plan)
    logger.info(
        "run_level5_planning: video_id=%s wrote edit_plans id=%s version=%d achieved_duration_s=%.1f",
        video_id, edit_plan_id, version, achieved_duration_s,
    )
    return {
        "ok": True,
        "edit_plan_id": edit_plan_id,
        "version": version,
        "achieved_duration_s": achieved_duration_s,
    }


# ---------------------------------------------------------------------------
# Revisions — diffs, not regenerates (CLAUDE.md rule 21)
# ---------------------------------------------------------------------------


def _apply_diff(operations: list[dict], diff_ops: list[dict]) -> list[dict]:
    """Apply an `add`/`modify`/`remove`-tagged diff onto *operations*,
    preserving original relative order and appending new ops at the end
    (final ordering is normalized by `_normalize_operations` afterward)."""
    by_id: dict[str, dict] = {op["op_id"]: dict(op) for op in operations}
    order: list[str] = [op["op_id"] for op in operations]

    for diff in diff_ops:
        action = diff.get("action", "modify")
        op_id = diff.get("op_id")
        if not op_id:
            continue
        clean = {k: v for k, v in diff.items() if k != "action"}
        if action == "remove":
            by_id.pop(op_id, None)
            if op_id in order:
                order.remove(op_id)
        else:  # "add" or "modify" — both are an upsert by op_id
            by_id[op_id] = clean
            if op_id not in order:
                order.append(op_id)

    return [by_id[op_id] for op_id in order if op_id in by_id]


async def apply_revision(pool: asyncpg.Pool, edit_plan_id: str, user_feedback: str) -> dict:
    """Apply one round of user feedback to an existing edit plan as a DIFF
    (CLAUDE.md rule 21), not a regenerate: ONE Groq call asking specifically
    what should change, an `edit_plan_revisions` row recording the
    `diff_operations` (only the changed ops), then the diff applied to
    produce a new `edit_plans` row (new version, same video_id) —
    validated the same way as `validate_plan` before writing.

    Returns `{"ok": True, "edit_plan_id": ..., "version": ..., "achieved_duration_s": ...}`
    or `{"ok": False, "reason": "..."}`. The `edit_plan_revisions` row is
    written as soon as the LLM call succeeds and validates (it records
    what was attempted, which is useful audit trail even if the resulting
    plan then fails `validate_plan` and no new `edit_plans` row is
    written) — only the new plan version write is gated on validation.
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        return {"ok": False, "reason": f"edit_plan_id={edit_plan_id} not found"}

    client = _get_groq_client()
    payload = {"operations": plan.operations, "user_feedback": user_feedback}
    raw = await _call_groq_tool(
        client, settings.L5_SEQUENCING_MODEL, L5_REVISION_SYSTEM_PROMPT,
        APPLY_PLAN_REVISION_TOOL, payload, _SEQUENCING_MAX_TOKENS,
        pool=pool, video_id=plan.video_id, stage="l5_revision",
    )
    if raw is None:
        return {"ok": False, "reason": "revision LLM call failed"}

    try:
        parsed = ApplyPlanRevisionOutput.model_validate(raw)
    except ValidationError as exc:
        return {"ok": False, "reason": f"revision response failed validation: {exc}"}

    diff_ops = [d.model_dump() for d in parsed.diff_operations]
    if not diff_ops:
        return {"ok": False, "reason": "revision LLM call returned an empty diff"}

    revision = EditPlanRevisionRecord(
        id=gen_id(), edit_plan_id=edit_plan_id, user_feedback=user_feedback, diff_operations=diff_ops
    )
    await insert_edit_plan_revision(pool, revision)
    await log_edit_plan_revision_correction(
        pool,
        video_id=plan.video_id,
        edit_plan_id=edit_plan_id,
        original_operations=plan.operations,
        diff_operations=diff_ops,
        correction_source="client",
        reason=user_feedback,
    )

    new_operations = _apply_diff(plan.operations, diff_ops)
    new_operations = _normalize_operations(new_operations)

    reason = await validate_plan(pool, plan.video_id, new_operations)
    if reason is not None:
        return {"ok": False, "reason": reason}

    achieved_duration_s = _select_clip_duration(new_operations)
    latest = await get_latest_edit_plan_for_video(pool, plan.video_id)
    next_version = (latest.version + 1) if latest is not None else plan.version + 1

    new_plan = EditPlanRecord(
        id=gen_id(),
        video_id=plan.video_id,
        storyline_id=plan.storyline_id,
        user_prompt=plan.user_prompt,
        target_duration_s=plan.target_duration_s,
        platform=plan.platform,
        status="draft",
        version=next_version,
        operations=new_operations,
        achieved_duration_s=achieved_duration_s,
    )
    new_edit_plan_id = await insert_edit_plan(pool, new_plan)
    logger.info(
        "apply_revision: edit_plan_id=%s -> new edit_plan_id=%s version=%d achieved_duration_s=%.1f",
        edit_plan_id, new_edit_plan_id, next_version, achieved_duration_s,
    )
    return {
        "ok": True,
        "edit_plan_id": new_edit_plan_id,
        "version": next_version,
        "achieved_duration_s": achieved_duration_s,
    }
