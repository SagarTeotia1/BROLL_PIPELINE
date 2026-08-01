"""Level-6 Color Grading Agent — sequence-aware color continuity.

Implements CLAUDE.md "LEVEL 6 — SPECIALIZED ACTION AGENTS" -> "Color Grading
Agent". The hard part (45-parameter per-shot analysis) is already built at
L2 (`color_grades`, `color-grading/color_analyzer/analyzer/schema.py`). This
module does NOT re-analyze color — it solves continuity across a sequence
assembled from non-adjacent source shots, which the original per-shot
analysis never had to consider (CLAUDE.md rule 19: sequence-aware agents key
off CUT order, not source order).

Two entry points:

  run_color_grading(pool, edit_plan_id, mood_target=None)
      For every `cut_list_items` row (finalized cut order), resolves the
      underlying shot's `color_grades.parameters`, builds cut-order neighbor
      context (1-2 clips before/after in `sequence_index` order), batches
      clips to Groq (`settings.L6_COLOR_MODEL`), validates + clamps every
      returned delta against schema.py's documented per-parameter bounds,
      and writes `sequence_color_adjustments` rows.

  build_ffmpeg_color_filters(base_parameters, sequence_delta)
      Pure function, no DB, no LLM call: applies `sequence_delta` on top of
      `base_parameters` and returns a ready-to-embed FFmpeg filter string
      per CLAUDE.md's "no LUT baking for v1 — direct FFmpeg parametric
      filter mapping" decision. See the docstring on that function for the
      exact (judgment-call) unit conversions used.

Shot resolution (CLAUDE.md "join each to its shot's color_grades.parameters
via the op's scene_id"): a `cut_list_items` row has no scene_id of its own —
it only carries an `op_id`, which is resolved against the parent
`edit_plans.operations` JSON to find that operation's `scene_id`, which in
turn is resolved against `scenes` to get a time range. If a scene covers
multiple shots, the shot whose midpoint is closest to the CLIP's own
midpoint (`cut_list_items.source_start`/`source_end`, i.e. the actual
snapped cut points — not the scene's full range) is used, on the theory that
what actually ends up on screen for this specific clip is a better proxy for
"which shot's grade applies" than the scene's nominal boundaries. If no
scene is found for an op_id (unset scene_id, or an op_id/scene_id that no
longer resolves), the same nearest-midpoint search falls back to the full
shot list for the video rather than dropping the clip outright.

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
interface, forced tool call — same `_call_groq_tool` pattern as
`pipeline/level4/grounding_runner.py` / `pipeline/level5/planner_runner.py`.
Tool-call arguments come back as a JSON STRING
(`response.choices[0].message.tool_calls[0].function.arguments`) that must
be `json.loads()`-ed before pydantic validation — NOT a pre-parsed dict.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import asyncpg
from pydantic import ValidationError

from knowledge_base.postgres.queries import (
    bulk_insert_sequence_color_adjustments,
    get_client_style_profile,
    get_color_grades_for_video,
    get_cut_list_items_for_edit_plan,
    get_edit_plan,
    get_scenes_for_video,
    get_shots_for_video,
    get_video,
)
from shared.config import settings
from shared.types import (
    ColorGradeRecord,
    ComputeSequenceDeltaOutput,
    CutListItemRecord,
    SceneRecord,
    SequenceColorAdjustmentRecord,
    ShotRecord,
)
from shared.utils import gen_id
from prompts.l6_color_grading import (
    COMPUTE_SEQUENCE_DELTAS_TOOL,
    L6_COLOR_GRADING_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# CLAUDE.md rule 23 — batch cap applies here too. Each clip's payload also
# carries up to 4 neighbor payloads, so the effective per-call item count is
# larger than a flat list of the same length would be; kept smaller than
# L4/L5's ~35 to compensate (never "put the whole sequence in one call
# regardless of video length").
_COLOR_BATCH_SIZE = 6
_COLOR_MAX_TOKENS = 8192

# How many clips before/after in CUT order (sequence_index, not source
# order — CLAUDE.md rule 19) to include as neighbor context per clip.
_NEIGHBOR_RADIUS = 2

# Small, bounded retry for a transient Groq API failure — same pattern as
# L4/L5's `_LLM_CALL_ATTEMPTS`, not a retry loop.
_LLM_CALL_ATTEMPTS = 2

# Float tolerance for "midpoint falls inside scene bounds".
_TIME_BOUND_EPSILON = 0.05


# ---------------------------------------------------------------------------
# color-grading schema access (mirrors pipeline/level2/color_runner.py's
# sys.path pattern — avoids a top-level import of a sibling monorepo module)
# ---------------------------------------------------------------------------

# Root of the monorepo — four levels up from this file:
#   video-kb-pipeline/pipeline/level6/color_grading_runner.py
#   -> video-kb-pipeline/pipeline/level6/
#   -> video-kb-pipeline/pipeline/
#   -> video-kb-pipeline/
#   -> <monorepo root>/
_MONOREPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_COLOR_GRADING_ROOT = (
    Path("/app_modules/color_grading")
    if Path("/app_modules/color_grading").exists()
    else _MONOREPO_ROOT / "color-grading"
)

_PARAM_BY_NAME: dict[str, Any] | None = None
_ADJUSTABLE_NAMES: tuple[str, ...] | None = None


def _ensure_color_grading_on_path() -> None:
    root = str(_COLOR_GRADING_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_schema() -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return (PARAM_BY_NAME, adjustable_param_names) from the real
    color-grading schema module — the single source of truth for every
    parameter's documented [lo, hi] bounds. Never hardcode bounds here."""
    global _PARAM_BY_NAME, _ADJUSTABLE_NAMES
    if _PARAM_BY_NAME is None:
        _ensure_color_grading_on_path()
        from color_analyzer.analyzer.schema import ADJUSTABLE, PARAM_BY_NAME  # type: ignore[import]

        _PARAM_BY_NAME = PARAM_BY_NAME
        _ADJUSTABLE_NAMES = tuple(p.name for p in ADJUSTABLE if p.kind in ("float", "int"))
    return _PARAM_BY_NAME, _ADJUSTABLE_NAMES  # type: ignore[return-value]


def _clamp(value: float, lo: float | None, hi: float | None) -> float:
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


# ---------------------------------------------------------------------------
# Groq client + generic structured tool-call helper (mirrors L4/L5 exactly)
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
    if every attempt failed. See `pipeline/level5/planner_runner.py`'s
    `_call_groq_tool` for the full rationale — this is the same pattern:
    `tool_choice` forces the structured call, and
    `message.tool_calls[0].function.arguments` is a JSON STRING that must be
    `json.loads()`-ed (unlike Anthropic's pre-parsed `tool_use.input`)."""
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
                    "L6 color grading LLM call model=%s tool=%s prompt_tokens=%s completion_tokens=%s",
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
                level=6,
                stage=stage or "l6_color_grading",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
                latency_ms=_latency_ms,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                logger.error(
                    "L6 color grading LLM call returned no tool_calls (model=%s, tool=%s)",
                    model, fn_name,
                )
                return None
            arguments = tool_calls[0].function.arguments  # JSON string
            return json.loads(arguments)
        except Exception as exc:  # noqa: BLE001 — log & retry/return, never raise into caller loop
            last_exc = exc
            logger.warning(
                "L6 color grading LLM call attempt %d/%d failed (model=%s, tool=%s): %s",
                attempt, _LLM_CALL_ATTEMPTS, model, fn_name, exc,
            )
    logger.error(
        "L6 color grading LLM call exhausted %d attempts (model=%s, tool=%s): %s",
        _LLM_CALL_ATTEMPTS, model, fn_name, last_exc,
    )
    return None


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Shot resolution: cut_list_item -> op_id -> scene_id -> shot -> color_grade
# ---------------------------------------------------------------------------


def _shot_midpoint(shot: ShotRecord) -> float | None:
    if shot.start_time is None or shot.end_time is None:
        return None
    return (shot.start_time + shot.end_time) / 2.0


def _select_shot_for_clip(
    item: CutListItemRecord,
    scene: SceneRecord | None,
    shots: list[ShotRecord],
) -> ShotRecord | None:
    """Pick the shot whose midpoint is closest to *item*'s own midpoint.

    Restricts the candidate pool to shots inside *scene*'s time range when a
    scene is known and at least one shot's midpoint actually falls inside it
    (documented choice — see module docstring); otherwise (no scene, or no
    shot midpoint lands inside the scene's bounds) falls back to searching
    every shot in the video, so a clip is never silently dropped just
    because the scene/shot boundaries don't line up exactly.
    """
    clip_mid = (item.source_start + item.source_end) / 2.0

    candidates = shots
    if scene is not None:
        in_scene = [
            s for s in shots
            if (mid := _shot_midpoint(s)) is not None
            and scene.start_time - _TIME_BOUND_EPSILON <= mid <= scene.end_time + _TIME_BOUND_EPSILON
        ]
        if in_scene:
            candidates = in_scene

    best: ShotRecord | None = None
    best_dist = float("inf")
    for s in candidates:
        mid = _shot_midpoint(s)
        if mid is None:
            continue
        dist = abs(mid - clip_mid)
        if dist < best_dist:
            best_dist, best = dist, s
    return best


def _flatten_base_parameters(
    parameters: dict[str, Any], adjustable_names: tuple[str, ...]
) -> dict[str, float]:
    """Flatten `color_grades.parameters` (per-param
    {"current", "recommended", "delta"}) to {param_name: value} for the
    adjustable numeric params only — this is the value the sequence-delta
    LLM call reasons about. Prefers "recommended" (the shot's own
    already-committed grade); falls back to "current" if a recommended
    value is missing for some reason."""
    flat: dict[str, float] = {}
    for name in adjustable_names:
        entry = parameters.get(name)
        if not isinstance(entry, dict):
            continue
        val = entry.get("recommended")
        if val is None:
            val = entry.get("current")
        if isinstance(val, (int, float)):
            flat[name] = float(val)
    return flat


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_color_grading(
    pool: asyncpg.Pool,
    edit_plan_id: str,
    mood_target: str | None = None,
) -> list[SequenceColorAdjustmentRecord]:
    """Compute + write `sequence_color_adjustments` for every resolvable
    `cut_list_items` row under *edit_plan_id*.

    *mood_target* is an optional explicit mood/style request (e.g.
    "warmer", "cinematic"); when omitted, only cut-order-neighbor
    continuity drives the correction (per CLAUDE.md, `scenes.emotional_arc`
    is the other source of mood_target — callers wanting that behavior
    should look it up per-clip and pass it in; this function itself makes
    no scene-mood policy decision, matching L6's "narrow and mostly
    deterministic" design principle — mood sourcing is a caller concern).

    If this video's `client_id` resolves to a `client_style_profiles` row
    with a non-empty `brand_colors` (CLAUDE.md "PIPELINE ADDENDUM 2" -> "3.
    Client Style Profiles"), it is looked up here and passed to the LLM as
    an OPTIONAL `target_brand_bias` field alongside the existing
    neighbor-delta context — a SOFT PRIOR only (rule 26), never a
    replacement for the neighbor-continuity reasoning this agent already
    does. No client_id, or a client_id with no profile / empty
    brand_colors, leaves `target_brand_bias` null and reproduces exactly
    today's behavior.

    Returns only the adjustments this call actually decided on and wrote —
    clips that could not be resolved to a shot/color_grade, or whose batch's
    LLM call failed outright, are skipped (logged) and simply have no
    `sequence_color_adjustments` row; a rerun is safe (idempotent inserts
    keyed by a fresh id — see `bulk_insert_sequence_color_adjustments`).
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        logger.warning("run_color_grading: edit_plan_id=%s not found", edit_plan_id)
        return []

    items = await get_cut_list_items_for_edit_plan(pool, edit_plan_id)
    if not items:
        logger.warning(
            "run_color_grading: edit_plan_id=%s has no cut_list_items — "
            "has the Editing Director run yet?",
            edit_plan_id,
        )
        return []

    # CLAUDE.md "PIPELINE ADDENDUM 2" -> "3. Client Style Profiles":
    # additive, soft-prior-only lookup. No client_id / no profile row /
    # empty brand_colors all leave target_brand_bias as None, which
    # reproduces exactly today's payload shape (see rule 26).
    target_brand_bias: dict | None = None
    video = await get_video(pool, plan.video_id)
    if video is not None and video.client_id:
        profile = await get_client_style_profile(pool, video.client_id)
        if profile is not None and profile.brand_colors:
            target_brand_bias = profile.brand_colors

    op_scene_by_op_id: dict[str, str | None] = {
        op.get("op_id"): op.get("scene_id") for op in plan.operations
    }
    scenes = await get_scenes_for_video(pool, plan.video_id)
    scenes_by_id: dict[str, SceneRecord] = {s.id: s for s in scenes}
    shots = await get_shots_for_video(pool, plan.video_id)
    grades = await get_color_grades_for_video(pool, plan.video_id)

    grades_by_shot_id: dict[str, ColorGradeRecord] = {}
    for g in grades:
        if g.shot_id and g.shot_id not in grades_by_shot_id:
            grades_by_shot_id[g.shot_id] = g

    _, adjustable_names = _get_schema()

    # ------------------------------------------------------------------
    # Resolve each cut_list_item to its shot's color_grades row, in
    # sequence_index order (items already come sorted from the query).
    # ------------------------------------------------------------------
    resolved: dict[str, dict[str, Any]] = {}  # cut_list_item_id -> context
    for item in items:
        scene_id = op_scene_by_op_id.get(item.op_id)
        scene = scenes_by_id.get(scene_id) if scene_id else None
        if scene_id and scene is None:
            logger.warning(
                "run_color_grading: cut_list_item=%s op_id=%s scene_id=%r not found in "
                "scenes for video_id=%s — falling back to nearest-shot search over the "
                "full shot list",
                item.id, item.op_id, scene_id, plan.video_id,
            )

        shot = _select_shot_for_clip(item, scene, shots)
        if shot is None:
            logger.warning(
                "run_color_grading: cut_list_item=%s — no shot with a usable midpoint "
                "found for video_id=%s, skipping",
                item.id, plan.video_id,
            )
            continue

        grade = grades_by_shot_id.get(shot.id)
        if grade is None:
            logger.warning(
                "run_color_grading: cut_list_item=%s resolved to shot=%s but no "
                "color_grades row exists for that shot — skipping (has L2 color "
                "grading run for this video?)",
                item.id, shot.id,
            )
            continue

        flat = _flatten_base_parameters(grade.parameters, adjustable_names)
        if not flat:
            logger.warning(
                "run_color_grading: cut_list_item=%s shot=%s color_grades row has no "
                "usable adjustable parameters — skipping",
                item.id, shot.id,
            )
            continue

        resolved[item.id] = {
            "item": item,
            "shot": shot,
            "base_parameters": grade.parameters,  # unchanged, full nested dict
            "flat": flat,
        }

    if not resolved:
        logger.warning(
            "run_color_grading: edit_plan_id=%s — no cut_list_items resolved to a "
            "usable color_grades row, nothing to do",
            edit_plan_id,
        )
        return []

    resolved_order = [item.id for item in items if item.id in resolved]

    # ------------------------------------------------------------------
    # Build per-clip payloads with cut-order neighbor context.
    # ------------------------------------------------------------------
    clip_payloads: list[dict[str, Any]] = []
    for idx, item_id in enumerate(resolved_order):
        ctx = resolved[item_id]
        lo = max(0, idx - _NEIGHBOR_RADIUS)
        hi = min(len(resolved_order), idx + _NEIGHBOR_RADIUS + 1)
        neighbor_ids = [resolved_order[i] for i in range(lo, hi) if i != idx]
        neighbors = [
            {
                "sequence_index": resolved[nid]["item"].sequence_index,
                "base_parameters": resolved[nid]["flat"],
            }
            for nid in neighbor_ids
        ]
        clip_payloads.append(
            {
                "cut_list_item_id": item_id,
                "sequence_index": ctx["item"].sequence_index,
                "base_parameters": ctx["flat"],
                "neighbors": neighbors,
            }
        )

    # ------------------------------------------------------------------
    # Batch to Groq, validate, clamp, collect.
    # ------------------------------------------------------------------
    client = _get_groq_client()
    param_by_name, _ = _get_schema()
    records: list[SequenceColorAdjustmentRecord] = []
    n_dropped_keys = 0

    batches = _chunk(clip_payloads, _COLOR_BATCH_SIZE)
    for batch_idx, batch in enumerate(batches):
        payload = {"clips": batch, "mood_target": mood_target, "target_brand_bias": target_brand_bias}
        raw = await _call_groq_tool(
            client, settings.L6_COLOR_MODEL, L6_COLOR_GRADING_SYSTEM_PROMPT,
            COMPUTE_SEQUENCE_DELTAS_TOOL, payload, _COLOR_MAX_TOKENS,
            pool=pool, video_id=plan.video_id, stage="l6_color_grading",
        )
        if raw is None:
            logger.error(
                "run_color_grading: edit_plan_id=%s batch %d/%d LLM call failed — "
                "those clips get no sequence_color_adjustments this run",
                edit_plan_id, batch_idx + 1, len(batches),
            )
            continue
        try:
            parsed = ComputeSequenceDeltaOutput.model_validate(raw)
        except ValidationError as exc:
            logger.error(
                "run_color_grading: edit_plan_id=%s batch %d/%d failed to validate response: %s",
                edit_plan_id, batch_idx + 1, len(batches), exc,
            )
            continue

        for result in parsed.adjustments:
            if result.cut_list_item_id not in resolved:
                logger.warning(
                    "run_color_grading: LLM returned cut_list_item_id=%r not in this "
                    "batch — dropping (closed-set discipline)",
                    result.cut_list_item_id,
                )
                continue

            ctx = resolved[result.cut_list_item_id]
            clip_flat = ctx["flat"]
            validated_delta: dict[str, float] = {}
            for key, delta in result.sequence_delta.items():
                if key not in clip_flat:
                    # Either not an adjustable param at all, or not present
                    # on THIS clip's own base_parameters (rule: never emit a
                    # delta for a param absent from the clip's own base) —
                    # never trust the LLM's arithmetic/keys, drop silently
                    # to the log, same discipline as L5 duration enforcement.
                    n_dropped_keys += 1
                    continue
                param = param_by_name.get(key)
                if param is None or param.kind not in ("float", "int"):
                    n_dropped_keys += 1
                    continue
                try:
                    delta_f = float(delta)
                except (TypeError, ValueError):
                    n_dropped_keys += 1
                    continue
                base_val = clip_flat[key]
                final_val = _clamp(base_val + delta_f, param.lo, param.hi)
                clamped_delta = final_val - base_val
                if clamped_delta == 0 and delta_f != 0:
                    logger.info(
                        "run_color_grading: clamped %s delta for cut_list_item=%s "
                        "(requested final=%.4f, bounds=[%s,%s])",
                        key, result.cut_list_item_id, base_val + delta_f, param.lo, param.hi,
                    )
                validated_delta[key] = clamped_delta

            records.append(
                SequenceColorAdjustmentRecord(
                    id=gen_id(),
                    edit_plan_id=edit_plan_id,
                    cut_list_item_id=result.cut_list_item_id,
                    base_parameters=ctx["base_parameters"],
                    sequence_delta=validated_delta,
                    rationale=result.rationale or None,
                )
            )

    if records:
        await bulk_insert_sequence_color_adjustments(pool, records)

    logger.info(
        "run_color_grading: edit_plan_id=%s wrote %d/%d sequence_color_adjustments "
        "(%d cut_list_items unresolved to a shot/grade, %d out-of-schema delta keys dropped)",
        edit_plan_id, len(records), len(items), len(items) - len(resolved), n_dropped_keys,
    )
    return records


# ---------------------------------------------------------------------------
# build_ffmpeg_color_filters — pure function, no DB, no LLM
# ---------------------------------------------------------------------------


def _final_value(
    base_parameters: dict[str, Any],
    sequence_delta: dict[str, float],
    name: str,
    default: float,
) -> float:
    """Resolve the final value for one param: base_parameters[name]'s
    "recommended" (falling back to "current", then *default* if the param
    is entirely absent) plus sequence_delta.get(name, 0)."""
    entry = base_parameters.get(name)
    base: float | None = None
    if isinstance(entry, dict):
        base = entry.get("recommended")
        if base is None:
            base = entry.get("current")
    if base is None:
        base = default
    delta = sequence_delta.get(name) or 0.0
    return float(base) + float(delta)


def build_ffmpeg_color_filters(
    base_parameters: dict[str, Any], sequence_delta: dict[str, float]
) -> str:
    """Apply *sequence_delta* on top of *base_parameters* and return an
    FFmpeg filter-chain string (comma-joined, ready to drop into a
    `filter_complex` graph for one clip) — CLAUDE.md's "no LUT baking for
    v1 — direct FFmpeg parametric filter mapping" decision.

    Pure function: no DB read, no LLM call, no I/O beyond importing the
    color-grading schema module for per-parameter bounds (same
    `_ensure_color_grading_on_path` mechanism `run_color_grading` uses).

    Mapping (CLAUDE.md "Color: no LUT baking needed for v1" list) and the
    UNIT CONVERSIONS used for each — every conversion below is this
    module's own reasonable judgment call, NOT verified against any
    color-science reference; spot-check against a real graded frame before
    trusting it visually (flagged again in the task report):

      white_balance.temperature [2000K, 12000K]
        -> colortemperature=temperature=<K>
        Direct passthrough — ffmpeg's `colortemperature` filter accepts
        1000-40000K, so the schema's range needs no rescaling, only a
        defensive clamp.

      primary.exposure [-3, 3] stops, primary.contrast [-100, 100],
      primary.gamma [0.5, 2.0], presence.saturation [-100, 100]
        -> one eq=contrast=:brightness=:saturation=:gamma= filter
        - eq_brightness = exposure / 3.0             (stops -> eq's
          roughly [-1, 1] additive brightness range; NOT a physically
          exact stops conversion — a stop is a multiplicative doubling,
          eq's brightness is additive, so this is a linear approximation
          scaled so the schema's max exposure maps to eq's max brightness)
        - eq_contrast   = 1 + contrast / 100          (schema's percent-ish
          [-100,100] -> eq's multiplicative [0, 2], where -100 -> 0 (flat
          gray) and +100 -> 2 (double contrast), 0 -> 1 (no change))
        - eq_gamma      = primary.gamma, passthrough  (both scales already
          use "1.0 = no change" gamma semantics, so no conversion beyond
          clamping into eq's own [0.1, 10] range)
        - eq_saturation = 1 + saturation / 100        (same percent-ish
          mapping as contrast, onto eq's [0, 3] multiplicative range)
        tone_curve.shadow_lift [0,1] / tone_curve.contrast_strength [0,1]
        are folded into eq_brightness/eq_contrast as small additive nudges
        (+0.15 * shadow_lift to brightness, +0.2 * contrast_strength to
        contrast) rather than mapped to a literal `curves` filter — a v1
        simplification per the task's explicit "fold into eq if a literal
        curves mapping is too complex" allowance. This is the crudest
        conversion in this function and the first one to replace with a
        real `curves` control-point mapping if v1's visual output isn't
        good enough.

      color_wheels.{lift,gamma,gain}_{temp,strength}
        -> colorbalance=rs=:gs=:bs=:rm=:gm=:bm=:rh=:gh=:bh=
        Schema's temp is a signed direction [-100 cool/teal, 100
        warm/orange] and strength is intensity [0,1] — NOT direct R/B
        offsets. Conversion: signed = (temp / 100) * strength, clamped to
        colorbalance's own [-1, 1] range; positive signed pushes red up
        and blue down by the same magnitude (warm = more red, less blue),
        green is left at 0 (the schema carries no green-axis signal for
        these three wheels). lift -> shadows (rs/gs/bs), gamma ->
        midtones (rm/gm/bm), gain -> highlights (rh/gh/bh) — a direct,
        undisputed 1:1 match to colorbalance's own three-band structure.
        The `colorbalance` filter clause is omitted entirely when all six
        offsets round to ~0, to avoid cluttering the filter chain with a
        no-op.

    Returns the filter chain as a single comma-joined string, e.g.
    "colortemperature=temperature=6100,eq=contrast=1.05:brightness=0.02:
    saturation=1.08:gamma=1.32,colorbalance=rs=0.12:gs=0:bs=-0.12:...".
    Meant to plug into Engineer A's render pipeline (`render_direct`'s
    `extra_filters` extension point) keyed by `cut_list_item.id` — that
    hookup did not exist in `pipeline/level6/editing_director.py` at the
    time this function was written; wire it there once available.
    """
    param_by_name, _ = _get_schema()

    def bounds(name: str) -> tuple[float | None, float | None]:
        p = param_by_name.get(name)
        return (p.lo, p.hi) if p is not None else (None, None)

    filters: list[str] = []

    # --- white balance -> colortemperature ---
    temp_lo, temp_hi = bounds("white_balance.temperature")
    temperature = _clamp(
        _final_value(base_parameters, sequence_delta, "white_balance.temperature", 6500.0),
        temp_lo, temp_hi,
    )
    filters.append(f"colortemperature=temperature={temperature:.0f}")

    # --- primary + presence + tone_curve (folded) -> eq ---
    exposure = _clamp(_final_value(base_parameters, sequence_delta, "primary.exposure", 0.0), -3.0, 3.0)
    contrast = _clamp(_final_value(base_parameters, sequence_delta, "primary.contrast", 0.0), -100.0, 100.0)
    gamma = _clamp(_final_value(base_parameters, sequence_delta, "primary.gamma", 1.0), 0.5, 2.0)
    saturation = _clamp(_final_value(base_parameters, sequence_delta, "presence.saturation", 0.0), -100.0, 100.0)
    shadow_lift = _clamp(_final_value(base_parameters, sequence_delta, "tone_curve.shadow_lift", 0.0), 0.0, 1.0)
    contrast_strength = _clamp(
        _final_value(base_parameters, sequence_delta, "tone_curve.contrast_strength", 0.0), 0.0, 1.0
    )

    eq_brightness = _clamp(exposure / 3.0 + shadow_lift * 0.15, -1.0, 1.0)
    eq_contrast = _clamp(1.0 + contrast / 100.0 + contrast_strength * 0.2, 0.0, 3.0)
    eq_gamma = _clamp(gamma, 0.1, 10.0)
    eq_saturation = _clamp(1.0 + saturation / 100.0, 0.0, 3.0)

    filters.append(
        f"eq=contrast={eq_contrast:.4f}:brightness={eq_brightness:.4f}:"
        f"saturation={eq_saturation:.4f}:gamma={eq_gamma:.4f}"
    )

    # --- color wheels -> colorbalance ---
    def wheel_offset(temp_key: str, strength_key: str) -> float:
        temp = _clamp(_final_value(base_parameters, sequence_delta, temp_key, 0.0), -100.0, 100.0)
        strength = _clamp(_final_value(base_parameters, sequence_delta, strength_key, 0.0), 0.0, 1.0)
        return _clamp((temp / 100.0) * strength, -1.0, 1.0)

    lift_r = wheel_offset("color_wheels.lift_temp", "color_wheels.lift_strength")
    gamma_r = wheel_offset("color_wheels.gamma_temp", "color_wheels.gamma_strength")
    gain_r = wheel_offset("color_wheels.gain_temp", "color_wheels.gain_strength")
    lift_b, gamma_b, gain_b = -lift_r, -gamma_r, -gain_r

    if any(abs(v) > 1e-4 for v in (lift_r, gamma_r, gain_r)):
        filters.append(
            "colorbalance="
            f"rs={lift_r:.4f}:gs=0:bs={lift_b:.4f}:"
            f"rm={gamma_r:.4f}:gm=0:bm={gamma_b:.4f}:"
            f"rh={gain_r:.4f}:gh=0:bh={gain_b:.4f}"
        )

    return ",".join(filters)
