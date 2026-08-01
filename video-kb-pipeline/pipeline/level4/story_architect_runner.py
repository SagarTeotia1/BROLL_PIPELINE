"""Level-4 Story Architect Agent — canonical scene timeline + storyline
synthesis for one video.

Implements CLAUDE.md "LEVEL 4 — REASONING" -> "Agent 2 — Story Architect
Agent": windowed batches of 3-5 consecutive scenes (not one-call-per-scene),
a rolling summary synced once per batch, and a PRUNED per-frame
representation (caption/causality/continuity/people[pid+story_role+action]/
beat_type/scene_mood/tension_level) — NEVER the full raw `qwen_output` JSON,
per the explicit ~3x token-cost warning in CLAUDE.md.

Scope note: this module writes Postgres only (`scenes`, `storylines`, both
`draft`). Neo4j edge-type correction, the `scenes`/`storylines` pgvector
embedding writes (B7 dropped Pinecone), and flipping `storylines.status` to
`final` are the responsibility of `pipeline/level4/updater.py` and
`pipeline/level4/finalizer.py` respectively (out of scope here — see
CLAUDE.md "Neo4j + Pinecone propagation" and "Finalization &
source-of-truth contract").

Entry point: `run_story_architect(pool, video_id)` — the exact name/signature
other L4 modules (finalizer, modal_app.py wiring) call into.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter

import asyncpg
from pydantic import ValidationError

from knowledge_base.postgres.queries import (
    bulk_upsert_scenes,
    delete_scenes_for_video,
    get_color_grades_for_video,
    get_frame_analyses_with_timestamps_for_video,
    get_latest_storyline,
    get_persons_for_video,
    get_reward_signal,
    get_shots_for_video,
    get_speaker_turns_for_video,
    get_top_scoring_scenes_for_client,
    get_transcript_segments_for_video,
    get_video,
    upsert_storyline_draft,
)
from shared.config import settings
from shared.types import (
    ColorGradeRecord,
    PersonRecord,
    QwenFrameOutput,
    SceneBeatItem,
    SceneRecord,
    SpeakerTurnRecord,
    StorylineRecord,
    StorylineSummaryOutput,
    TranscriptSegment,
    WriteSceneBeatsOutput,
)
from shared.utils import gen_id
from prompts.story_architect import (
    STORY_ARCHITECT_SYSTEM_PROMPT,
    WRITE_SCENE_BEATS_TOOL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# CLAUDE.md: "batch 3-5 consecutive scenes per call". 4 is the balanced
# target size _make_windows() partitions toward.
_WINDOW_TARGET_SIZE = 4
_STORY_LLM_MAX_TOKENS = 8192
_SUMMARY_LLM_MAX_TOKENS = 1024

# Rolling summary bound (CLAUDE.md rule 14) — truncate defensively if a call
# ignores the "3 sentences max" instruction.
_ROLLING_SUMMARY_MAX_SENTENCES = 3

# PIPELINE ADDENDUM 3, LEVEL 9, item 9b (reward -> few-shot injection).
# Both numbers are given literally in CLAUDE.md 9b's own text, not invented
# here: only inject examples when the client's reward_signals row has
# sample_count >= 5, and cap injected examples at 5 (rule 23 — batch caps
# apply everywhere an LLM sees a list that scales with data volume).
_CLIENT_STYLE_SAMPLE_FLOOR = 5
_CLIENT_STYLE_EXAMPLE_LIMIT = 5

# Small, bounded retry for a transient API failure — not a retry loop.
# Bumped 2->3: malformed-JSON from cheaper OpenRouter-routed models (no
# grammar-constrained tool-calling) is a confirmed real failure mode.
_LLM_CALL_ATTEMPTS = 3

# usability_score formula weights — see _compute_usability_score() docstring
# for the full formula. First pass; documented for others to refine (per
# CLAUDE.md "L4 addition needed to support L5").
_USABILITY_TECHNICAL_WEIGHT = 0.6
_USABILITY_TRANSCRIPT_WEIGHT = 0.4

# ---------------------------------------------------------------------------
# Groq client + generic structured tool-call helper
#
# Deliberately self-contained rather than importing the private helpers in
# `pipeline/level4/grounding_runner.py` — that module is owned by a
# different, concurrently-developed L4 agent, and coupling to its private
# (underscore-prefixed) internals would be fragile. The shape mirrors it
# closely so the two agents read consistently.
#
# Groq's `chat.completions.create` is OpenAI-compatible: the system prompt
# is a `role: "system"` entry in `messages` (not a separate top-level
# `system=` kwarg like Anthropic), and a forced tool call comes back as
# `response.choices[0].message.tool_calls[0].function.arguments` — a JSON
# STRING that must be `json.loads()`-ed, not a pre-parsed dict like
# Anthropic's `tool_use` block `.input`.
# ---------------------------------------------------------------------------


def _get_groq_client():
    """Name kept for minimal diff — actually returns an OpenRouter client
    now (see shared/llm_client.py)."""
    from shared.llm_client import get_llm_client

    return get_llm_client()


async def _call_tool(
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
    """Call the Groq chat.completions API with a forced tool call and return
    the tool call's structured arguments as a dict, or None if every attempt
    failed.

    Uses `tool_choice={"type": "function", "function": {"name": ...}}` —
    structured output, never free-text JSON parsing (per CLAUDE.md's
    structured-output rule for both L4 agents). `tool_schema` is the
    OpenAI-compatible `{"type": "function", "function": {"name",
    "description", "parameters"}}` shape — NOT Anthropic's top-level
    `input_schema`.
    """
    # openai's public exception hierarchy (openai.APIError / RateLimitError /
    # APIStatusError / APIConnectionError / ...) is Stainless-generated,
    # same pattern as the groq/anthropic SDKs it replaces — imported here
    # only so an ImportError surfaces clearly if the installed openai
    # version ever drops these names.
    import openai  # noqa: F401
    import time as _time

    tool_name = tool_schema["function"]["name"]
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
                tool_choice={"type": "function", "function": {"name": tool_name}},
                # See grounding_runner.py::_call_tool for the real-video
                # finding behind this — OpenRouter routes to multiple
                # backend providers with inconsistent tool-calling
                # reliability; this restricts routing to providers that
                # declare full support for every param in this request.
                # reasoning.enabled=True: unlike grounding's closed-set
                # classification, this stage does narrative synthesis (scene
                # merging, causal links) — worth the extra latency/cost at
                # this stage's low call volume (~50/video).
                extra_body={
                    "provider": {"require_parameters": True},
                    "reasoning": {"enabled": True},
                },
            )
            usage = getattr(response, "usage", None)
            _latency_ms = int((_time.monotonic() - _t0) * 1000)
            if usage is not None:
                logger.info(
                    "L4 story_architect LLM call model=%s tool=%s prompt_tokens=%s completion_tokens=%s",
                    model,
                    tool_name,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )
            # L7 7c — durable cost/latency log, non-fatal.
            from shared.llm_client import log_llm_call as _log_llm_call
            await _log_llm_call(
                pool,
                video_id=video_id,
                level=4,
                stage=stage or "story_architect",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
                latency_ms=_latency_ms,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            found_tool_call = False
            for call in tool_calls:
                if getattr(call.function, "name", None) == tool_name:
                    found_tool_call = True
                    try:
                        return json.loads(call.function.arguments)
                    except json.JSONDecodeError as exc:
                        # Same fix as grounding_runner.py's _call_tool: this
                        # used to return None immediately on the first
                        # malformed-JSON attempt, bypassing _LLM_CALL_ATTEMPTS
                        # entirely — fall through to the next attempt instead.
                        last_exc = exc
                        logger.warning(
                            "L4 story_architect LLM call attempt %d/%d returned malformed "
                            "JSON in tool arguments (model=%s, tool=%s): %s",
                            attempt, _LLM_CALL_ATTEMPTS, model, tool_name, exc,
                        )
                        break
            if found_tool_call:
                continue
            logger.warning(
                "L4 story_architect LLM call attempt %d/%d returned no matching tool_call "
                "(model=%s, tool=%s)",
                attempt, _LLM_CALL_ATTEMPTS, model, tool_name,
            )
        except Exception as exc:  # noqa: BLE001 — log & retry/return, never raise into caller loop
            last_exc = exc
            logger.warning(
                "L4 story_architect LLM call attempt %d/%d failed (model=%s, tool=%s): %s",
                attempt, _LLM_CALL_ATTEMPTS, model, tool_name, exc,
            )
    logger.error(
        "L4 story_architect LLM call exhausted %d attempts (model=%s, tool=%s): %s",
        _LLM_CALL_ATTEMPTS, model, tool_name, last_exc,
    )
    return None


def _truncate_rolling_summary(text: str, max_sentences: int = _ROLLING_SUMMARY_MAX_SENTENCES) -> str:
    """Defensively bound the rolling summary to *max_sentences* (rule 14).

    Splits on '.', '!', '?' followed by whitespace/end-of-string — good
    enough for English narrative prose; not a full sentence tokenizer.
    """
    text = (text or "").strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    bounded = " ".join(s.strip() for s in sentences[:max_sentences] if s.strip())
    return bounded


# ---------------------------------------------------------------------------
# Input assembly: group frame_analyses into candidate scenes
# ---------------------------------------------------------------------------


class _SceneGroup:
    """A chronological run of frames belonging to one physical scene, as
    determined by Qwen's own `scene_change` boundary flag — NOT by exact
    `scene_id` string match, since that string is exactly what's
    inconsistent per-frame (see CLAUDE.md "Why L4 exists"). The frames in
    one group may carry several different scene_id string "aliases" that
    the LLM reconciles into one canonical id.
    """

    __slots__ = ("frames", "keyframe_ids", "start_time", "end_time")

    def __init__(self) -> None:
        self.frames: list[QwenFrameOutput] = []
        self.keyframe_ids: list[str] = []
        # Real coverage bounds for this scene's `scenes` row. Default to the
        # raw keyframe-timestamp span; _snap_group_boundaries() widens these
        # to close gaps against shot boundaries once all groups are built
        # (see that function's docstring for why raw keyframe span alone is
        # insufficient — keyframes are a sparse subset of a shot's frames).
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    @property
    def raw_start_time(self) -> float:
        return min(f.t for f in self.frames)

    @property
    def raw_end_time(self) -> float:
        return max(f.t for f in self.frames)

    @property
    def majority_scene_id(self) -> str:
        counts = Counter(f.scene_id for f in self.frames if f.scene_id)
        if not counts:
            return f"scene_at_{self.raw_start_time:.1f}s"
        return counts.most_common(1)[0][0]

    @property
    def alias_scene_ids(self) -> list[str]:
        majority = self.majority_scene_id
        seen: list[str] = []
        for f in self.frames:
            if f.scene_id and f.scene_id != majority and f.scene_id not in seen:
                seen.append(f.scene_id)
        return seen


def _prune_frame(frame: QwenFrameOutput) -> dict:
    """The PRUNED per-frame representation CLAUDE.md requires for Agent 2's
    input: caption, causality, continuity, people[pid+story_role+action
    only], beat_type, scene_mood, tension_level. NEVER the full raw
    qwen_output JSON (~3x token cost at scene-batch scale)."""
    return {
        "caption": frame.caption,
        "causality": frame.causality.model_dump() if frame.causality else None,
        "continuity": frame.continuity.model_dump() if frame.continuity else None,
        "people": [
            {"pid": p.pid, "story_role": p.story_role, "action": p.action}
            for p in frame.people
        ],
        "beat_type": frame.story_beat.beat_type if frame.story_beat else None,
        "scene_mood": frame.emotional_tone.scene_mood if frame.emotional_tone else None,
        "tension_level": frame.emotional_tone.tension_level if frame.emotional_tone else None,
    }


async def _build_scene_groups(pool: asyncpg.Pool, video_id: str) -> list[_SceneGroup]:
    rows = await get_frame_analyses_with_timestamps_for_video(pool, video_id)
    groups: list[_SceneGroup] = []
    current: _SceneGroup | None = None
    n_skipped = 0
    for row in rows:
        try:
            parsed = QwenFrameOutput.model_validate(row["qwen_output"])
        except ValidationError as exc:
            n_skipped += 1
            logger.warning(
                "story_architect: skipping unparseable qwen_output for keyframe_id=%s: %s",
                row.get("keyframe_id"), exc,
            )
            continue
        if current is None or parsed.scene_change:
            current = _SceneGroup()
            groups.append(current)
        current.frames.append(parsed)
        current.keyframe_ids.append(row["keyframe_id"])
    if n_skipped:
        logger.warning(
            "story_architect: video_id=%s skipped %d/%d frames with unparseable qwen_output",
            video_id, n_skipped, len(rows),
        )
    return groups


def _snap_group_boundaries(groups: list[_SceneGroup], shots: list) -> None:
    """Widen each group's start_time/end_time so the written `scenes` rows
    are gap-free against shot coverage (mutates groups in place).

    Real-video finding (first live L4 run): a group's raw keyframe span
    (min/max timestamp of its own Qwen-analyzed frames) is narrower than
    the shot(s) it actually belongs to, because keyframes are a sparse
    ASHFS-selected subset of a shot's frames, not every frame. Writing
    scenes with raw keyframe bounds left real timestamp ranges — the edges
    of shots between the last selected keyframe and the shot boundary —
    uncovered by any `scenes` row, tripping the finalizer's "every
    chunk/shot time range covered by >=1 scenes row" completeness check
    (CLAUDE.md finalization contract #1).

    Fix: make groups' covered ranges contiguous and touching across the
    whole video span, not just the fuzzy set of frame timestamps within
    each. Each group's boundary with its neighbor is the midpoint between
    their raw (frame-timestamp) endpoints; the very first/last group is
    additionally snapped out to the overall min/max shot start/end. This
    is a pure widening — it can only grow coverage, never shrink it below
    the raw keyframe span, so it cannot invalidate anything already
    correct (e.g. speaker-turn overlap matching, which uses these same
    bounds and only gains from a wider window).
    """
    if not groups:
        return
    for g in groups:
        g.start_time = g.raw_start_time
        g.end_time = g.raw_end_time

    for i in range(len(groups) - 1):
        boundary = (groups[i].raw_end_time + groups[i + 1].raw_start_time) / 2.0
        groups[i].end_time = boundary
        groups[i + 1].start_time = boundary

    shot_starts = [s.start_time for s in shots if s.start_time is not None]
    shot_ends = [s.end_time for s in shots if s.end_time is not None]
    if shot_starts:
        groups[0].start_time = min(groups[0].start_time, min(shot_starts))
    if shot_ends:
        groups[-1].end_time = max(groups[-1].end_time, max(shot_ends))


def _resolve_participant_pid(raw: str, cast_payload: dict[str, str]) -> str | None:
    """Map one LLM-returned participant string back to a real pid, or None
    if it can't be resolved to anything in the closed cast set.

    Real-video finding: the model sometimes returns "P1 (Samay Raina)"
    instead of the bare pid "P1" it was given in `cast_payload` keys — an
    exact-match check against `cast_payload` drops every participant in
    that scene, even though the identity is unambiguous. Tries, in order:
    exact pid match, the text before a "(" (handles "P1 (Name)"), then a
    case-insensitive match against the cast's display names (handles the
    model returning the name alone). Still closed-set discipline (CLAUDE.md
    rule 12) — never invents a pid outside `cast_payload`.
    """
    raw = raw.strip()
    if raw in cast_payload:
        return raw
    head = raw.split("(", 1)[0].strip()
    if head in cast_payload:
        return head
    for pid, name in cast_payload.items():
        if name and name.lower() in (raw.lower(), head.lower()):
            return pid
    return None


def _fallback_summary(group: _SceneGroup) -> str:
    """Build a deterministic, non-empty summary from the group's own
    already-pruned frame captions (no LLM call — same data _prune_frame
    already extracted for the window this group was in).

    Real-video finding: the original fallback wrote a static placeholder
    string ("(no beat returned...)") for any scene whose window either
    failed outright or under-returned beats. That guarantees time coverage
    (the documented reason this function exists) but produces a summary
    with zero semantic content — and the scene most likely to hit this path
    is exactly the largest/most complex one in a window (hardest to
    summarize within the window's token budget), which is often the single
    most important scene in the video. Observed live: a 127s scene
    containing the entire main skit fell back this way, then L5 Pass A
    correctly-but-wrongly scored it 0.0 relevance against a genuinely empty
    "summary" — a real scene got treated as irrelevant not because it was,
    but because its only textual signal was a placeholder. Concatenating
    real captions here costs nothing (pure string join, data already in
    hand) and gives every downstream consumer (L5 selection, dashboard,
    search) something grounded to work with even when narrative synthesis
    genuinely failed.
    """
    seen: list[str] = []
    for f in group.frames:
        cap = (f.caption or "").strip()
        if cap and cap not in seen:
            seen.append(cap)
    if not seen:
        return "(no beat returned by story architect for this scene — no frame captions available either)"
    joined = " ".join(seen)
    return joined[:800] + ("..." if len(joined) > 800 else "")


def _fallback_participants(group: _SceneGroup) -> list[str]:
    """Dedup pids across the group's frames — same no-LLM, data-already-in-hand
    principle as `_fallback_summary`."""
    seen: list[str] = []
    for f in group.frames:
        for p in f.people:
            if p.pid and p.pid not in seen:
                seen.append(p.pid)
    return seen


def _fallback_scene_record(
    video_id: str,
    group: _SceneGroup,
    color_grades: list[ColorGradeRecord],
    segments: list[TranscriptSegment],
) -> SceneRecord:
    """Minimal SceneRecord built from deterministic group data alone (no LLM
    narrative synthesis) — used whenever the Story Architect Agent didn't
    return usable output for a scene, whether because the whole window's
    call failed/didn't validate or because this specific scene_index was
    missing from an otherwise-valid response. Always has real start_time/
    end_time (post-_snap_group_boundaries), so a scene using this fallback
    still counts toward the finalizer's time-coverage check — the whole
    point of this helper existing is that "the model didn't answer" must
    never mean "this time range has no scenes row at all." Summary/
    participants are backfilled from real frame data (see
    `_fallback_summary`/`_fallback_participants`) rather than left empty, so
    downstream consumers (L5 relevance scoring especially) don't mistake
    "narrative synthesis failed" for "this scene has no content."
    """
    return SceneRecord(
        id=gen_id(),
        video_id=video_id,
        canonical_scene_id=group.majority_scene_id,
        discarded_aliases=group.alias_scene_ids,
        start_time=group.start_time,
        end_time=group.end_time,
        participants=_fallback_participants(group),
        summary=_fallback_summary(group),
        emotional_arc="unknown",
        causal_link_to_next=None,
        usability_score=_compute_usability_score(group, color_grades, segments),
    )


def _disambiguate_canonical_scene_ids(records: list[SceneRecord], video_id: str) -> None:
    """Mutate *records* in place so no two rows share a `canonical_scene_id`.

    Real-video finding: `bulk_upsert_scenes` UPSERTs on
    `(video_id, canonical_scene_id)`. Qwen reuses scene_id labels across
    genuinely different, non-adjacent physical scenes (the whole reason
    canonical reconciliation exists — see CLAUDE.md "Why L4 exists"); when
    the Story Architect Agent's chosen canonical id for two different
    scenes in the same video collides, the UPSERT silently overwrites one
    scene's row with the other's, dropping its time range from the
    `scenes` table entirely — a real coverage gap that survives even
    though `_snap_group_boundaries` already made the input groups
    gap-free. First occurrence of a label keeps it unchanged (so a rerun
    with identical LLM output stays idempotent per rule 15); every later
    occurrence gets a time-based suffix, which is always unique since
    distinct scenes never share both a start_time and an end_time.
    """
    seen: set[str] = set()
    for rec in records:
        cid = rec.canonical_scene_id
        if cid not in seen:
            seen.add(cid)
            continue
        new_id = f"{cid}__at_{rec.start_time:.1f}s"
        while new_id in seen:
            new_id = f"{new_id}_"
        logger.warning(
            "run_story_architect: video_id=%s canonical_scene_id=%r reused by scene at "
            "[%.1fs, %.1fs) — Qwen assigned the same scene label to two different physical "
            "scenes; disambiguating to %r so the UPSERT doesn't overwrite one scene's row "
            "with the other's, silently dropping its time coverage",
            video_id, cid, rec.start_time, rec.end_time, new_id,
        )
        rec.canonical_scene_id = new_id
        seen.add(new_id)


def _make_windows(groups: list[_SceneGroup], target: int = _WINDOW_TARGET_SIZE) -> list[list[_SceneGroup]]:
    """Partition *groups* into windows of 3-5 consecutive scenes.

    Balanced partition (not simple fixed-size chunking with a ragged tail) —
    picks a window count via round(n/target), then spreads the remainder
    across windows so every window lands in [3, 5] whenever n >= 3.
    """
    n = len(groups)
    if n == 0:
        return []
    if n <= 5:
        return [groups]
    num_windows = max(1, round(n / target))
    base, remainder = divmod(n, num_windows)
    windows: list[list[_SceneGroup]] = []
    idx = 0
    for w in range(num_windows):
        size = base + (1 if w < remainder else 0)
        windows.append(groups[idx : idx + size])
        idx += size
    return windows


# ---------------------------------------------------------------------------
# Speaker turns / transcript join (per-scene "resolved dialogue")
# ---------------------------------------------------------------------------


def _overlapping_transcript_text(segments: list[TranscriptSegment], start: float, end: float) -> str:
    matching = [
        seg.text for seg in segments
        if seg.start_time < end and seg.end_time > start
    ]
    return " ".join(matching)


def _scene_speaker_turns(
    group: _SceneGroup,
    turns: list[SpeakerTurnRecord],
    segments: list[TranscriptSegment],
    pid_by_person_db_id: dict[str, str],
) -> list[dict]:
    """Resolved dialogue for this scene's time range — only turns with a
    non-null person_id (i.e. actually resolved by L2 fusion or the
    Grounding Agent). Unresolved turns are not invented a speaker."""
    result: list[dict] = []
    for turn in turns:
        if turn.person_id is None:
            continue
        if not (turn.start_time < group.end_time and turn.end_time > group.start_time):
            continue
        pid = pid_by_person_db_id.get(turn.person_id)
        if not pid:
            continue
        result.append(
            {
                "person_id": pid,
                "start_time": turn.start_time,
                "end_time": turn.end_time,
                "transcript_text": _overlapping_transcript_text(segments, turn.start_time, turn.end_time),
            }
        )
    return result


# ---------------------------------------------------------------------------
# usability_score — deterministic, NO LLM call (CLAUDE.md "L4 addition
# needed to support L5")
# ---------------------------------------------------------------------------


def _color_grade_quality(grade: ColorGradeRecord) -> float | None:
    """Roll up one color_grades row's quality-adjacent params into [0, 1].

    Uses the four `quality.*` fields the color-grading engine emits
    (observed on real data — see the "current" sub-key of each):
      - quality.dynamic_range   (float stops, higher is better; ~0-8 typical
                                  range observed, clamped and normalized)
      - quality.noise_risk      ("low" | "medium" | "high")
      - quality.highlight_clipping (bool — True is bad)
      - quality.shadow_crush    (bool — True is bad)

    Each sub-score is mapped to [0, 1] and averaged unweighted. Returns None
    if none of the four keys are present (grade row has an unexpected shape).
    """
    params = grade.parameters or {}
    scores: list[float] = []

    dr = params.get("quality.dynamic_range")
    if isinstance(dr, dict) and isinstance(dr.get("current"), (int, float)):
        scores.append(max(0.0, min(1.0, dr["current"] / 8.0)))

    noise = params.get("quality.noise_risk")
    if isinstance(noise, dict):
        scores.append({"low": 1.0, "medium": 0.6, "high": 0.2}.get(noise.get("current"), 0.5))

    clipping = params.get("quality.highlight_clipping")
    if isinstance(clipping, dict) and isinstance(clipping.get("current"), bool):
        scores.append(0.0 if clipping["current"] else 1.0)

    crush = params.get("quality.shadow_crush")
    if isinstance(crush, dict) and isinstance(crush.get("current"), bool):
        scores.append(0.0 if crush["current"] else 1.0)

    if not scores:
        return None
    return sum(scores) / len(scores)


def _compute_usability_score(
    group: _SceneGroup,
    color_grades: list[ColorGradeRecord],
    segments: list[TranscriptSegment],
) -> float:
    """Deterministic usability_score in [0, 1] for one scene — NO LLM call.

    Formula (first pass — documented for others to refine, per CLAUDE.md):
      usability_score = 0.6 * technical_quality + 0.4 * transcript_confidence

      technical_quality: mean of per-frame `_color_grade_quality()` over
      every color_grades row whose timestamp_s falls inside
      [scene.start_time, scene.end_time]. Defaults to 0.5 (neutral/unknown)
      when no color_grades rows fall in range.

      transcript_confidence: mean of `transcript_segments.confidence` (the
      Whisper word/segment-level confidence already stored by L1) over every
      segment overlapping the scene's time range. Defaults to 0.5 (neutral —
      a silent, dialogue-free scene is not inherently low quality) when no
      segments overlap.

    Weights (0.6 technical / 0.4 transcript) favor visual technical quality
    slightly, since color-grading signal is denser (per-shot) than
    transcript confidence (only present where there's dialogue).
    """
    in_range_grades = [
        g for g in color_grades
        if g.timestamp_s is not None and group.start_time <= g.timestamp_s <= group.end_time
    ]
    quality_scores = [q for q in (_color_grade_quality(g) for g in in_range_grades) if q is not None]
    technical_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

    overlapping_segs = [
        seg for seg in segments
        if seg.start_time < group.end_time and seg.end_time > group.start_time and seg.confidence is not None
    ]
    transcript_confidence = (
        sum(seg.confidence for seg in overlapping_segs) / len(overlapping_segs)
        if overlapping_segs else 0.5
    )

    score = _USABILITY_TECHNICAL_WEIGHT * technical_quality + _USABILITY_TRANSCRIPT_WEIGHT * transcript_confidence
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Title/synopsis generation (see module docstring / final report for the
# design-choice note: this is ONE additional short LLM call, not a
# CLAUDE.md-specified verbatim prompt — deterministic concatenation of scene
# summaries was judged too flat for L5's actual narrative-synthesis need).
# ---------------------------------------------------------------------------

_STORYLINE_SUMMARY_SYSTEM_PROMPT = """\
You write the title and synopsis for a video's storyline document — the \
single summary a downstream planning agent will read instead of the raw \
scene timeline. Be concrete and grounded ONLY in the scene beats given to \
you; never invent a plot point, person, or event absent from the input.

Input: an ordered list of scene beats (canonical_scene_id, summary, \
emotional_arc, causal_link_to_next, participants) covering the whole video, \
plus the cast (pid -> display_name).

Write:
  title: a short (<=10 word) descriptive title for the whole video.
  synopsis: 2-4 sentences summarizing the throughline across all scenes, \
grounded in the causal_link_to_next chain where present.

Return via the write_storyline_summary tool call.
"""

_WRITE_STORYLINE_SUMMARY_TOOL = {
    "type": "function",
    "function": {
        "name": "write_storyline_summary",
        "description": "Return a title and synopsis for the whole video, grounded only in the given scene beats.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "synopsis": {"type": "string"},
            },
            "required": ["title", "synopsis"],
        },
    },
}


async def _generate_title_synopsis(
    client, scene_records: list[SceneRecord], cast_payload: dict[str, str],
    *, pool: asyncpg.Pool | None = None, video_id: str | None = None,
) -> tuple[str | None, str | None]:
    if not scene_records:
        return None, None
    beats_payload = [
        {
            "canonical_scene_id": s.canonical_scene_id,
            "summary": s.summary,
            "emotional_arc": s.emotional_arc,
            "causal_link_to_next": s.causal_link_to_next,
            "participants": s.participants,
        }
        for s in scene_records
    ]
    raw = await _call_tool(
        client, settings.L4_STORY_ARCHITECT_MODEL, _STORYLINE_SUMMARY_SYSTEM_PROMPT,
        _WRITE_STORYLINE_SUMMARY_TOOL, {"scene_beats": beats_payload, "cast": cast_payload},
        _SUMMARY_LLM_MAX_TOKENS,
        pool=pool, video_id=video_id, stage="story_architect_summary",
    )
    if raw is None:
        return None, None
    try:
        parsed = StorylineSummaryOutput.model_validate(raw)
    except ValidationError as exc:
        logger.error("story_architect: failed to validate title/synopsis response: %s", exc)
        return None, None
    return parsed.title, parsed.synopsis


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_story_architect(pool: asyncpg.Pool, video_id: str) -> StorylineRecord | None:
    """Run the Story Architect Agent for *video_id* end to end.

    - Groups frame_analyses into candidate scenes on Qwen's own
      scene_change boundary.
    - Batches scenes into windows of 3-5, processing chronologically with a
      rolling summary carried forward once per batch.
    - Writes canonical `scenes` rows (with a deterministic usability_score,
      no LLM call) and one `storylines` row in `draft` status.

    Returns the written StorylineRecord, or None if there was nothing to
    process (no frame_analyses for this video yet — L3 hasn't run) or every
    window's LLM call failed outright.
    """
    groups = await _build_scene_groups(pool, video_id)
    if not groups:
        logger.warning(
            "run_story_architect: video_id=%s has no parseable frame_analyses — "
            "nothing to do (has L3 run yet?)",
            video_id,
        )
        return None

    shots = await get_shots_for_video(pool, video_id)
    _snap_group_boundaries(groups, shots)

    persons: list[PersonRecord] = await get_persons_for_video(pool, video_id)
    pid_by_person_db_id = {p.id: p.pid for p in persons}
    cast_payload: dict[str, str] = {p.pid: (p.display_name or p.pid) for p in persons}

    turns = await get_speaker_turns_for_video(pool, video_id)
    segments = await get_transcript_segments_for_video(pool, video_id)
    color_grades = await get_color_grades_for_video(pool, video_id)

    windows = _make_windows(groups)
    logger.info(
        "run_story_architect: video_id=%s grouped %d frames into %d candidate scenes, %d windows",
        video_id, sum(len(g.frames) for g in groups), len(groups), len(windows),
    )

    # PIPELINE ADDENDUM 3, LEVEL 9, item 9b -- reward -> few-shot injection.
    # Additive, soft-prior-only (rule 26): looked up ONCE per run, not per
    # window. A video with no client_id, or a client_id whose
    # reward_signals(scope_type='client') row doesn't exist yet or has
    # sample_count below the floor, leaves client_style_examples as an
    # empty list -- the payload key is always present (never conditionally
    # omitted) so downstream logging/debugging can see "checked, found
    # nothing" vs "never checked", but an empty list changes nothing about
    # model behavior versus the field not existing at all.
    client_style_examples: list[dict] = []
    video_meta = await get_video(pool, video_id)
    if video_meta is not None and video_meta.client_id:
        reward_signal = await get_reward_signal(pool, "client", video_meta.client_id)
        if reward_signal is not None and reward_signal.sample_count >= _CLIENT_STYLE_SAMPLE_FLOOR:
            top_scenes = await get_top_scoring_scenes_for_client(
                pool, video_meta.client_id, limit=_CLIENT_STYLE_EXAMPLE_LIMIT
            )
            client_style_examples = [
                {
                    "canonical_scene_id": s["canonical_scene_id"],
                    "summary": s["summary"],
                    "emotional_arc": s["emotional_arc"],
                }
                for s in top_scenes
            ]
            logger.info(
                "run_story_architect: video_id=%s client_id=%s injecting %d client_style_examples "
                "(reward_signals sample_count=%d)",
                video_id, video_meta.client_id, len(client_style_examples), reward_signal.sample_count,
            )

    client = _get_groq_client()
    rolling_summary = ""
    scene_records: list[SceneRecord] = []

    for window_idx, window in enumerate(windows):
        scenes_payload = []
        for scene_idx, group in enumerate(window):
            scenes_payload.append(
                {
                    "scene_index": scene_idx,
                    "scene_frames": [_prune_frame(f) for f in group.frames],
                    "speaker_turns": _scene_speaker_turns(group, turns, segments, pid_by_person_db_id),
                }
            )
        payload = {
            "rolling_summary": rolling_summary,
            "scenes": scenes_payload,
            "cast": cast_payload,
            "client_style_examples": client_style_examples,
        }

        raw = await _call_tool(
            client, settings.L4_STORY_ARCHITECT_MODEL, STORY_ARCHITECT_SYSTEM_PROMPT,
            WRITE_SCENE_BEATS_TOOL, payload, _STORY_LLM_MAX_TOKENS,
            pool=pool, video_id=video_id, stage="story_architect_scene_beats",
        )
        if raw is None:
            logger.error(
                "run_story_architect: video_id=%s window %d/%d LLM call failed — "
                "writing fallback records for this window's %d scenes (no narrative "
                "synthesis, but time coverage is preserved; safe to retry on next run)",
                video_id, window_idx + 1, len(windows), len(window),
            )
            for group in window:
                scene_records.append(_fallback_scene_record(video_id, group, color_grades, segments))
            continue

        try:
            parsed = WriteSceneBeatsOutput.model_validate(raw)
        except ValidationError as exc:
            logger.error(
                "run_story_architect: video_id=%s window %d/%d failed to validate response: %s — "
                "writing fallback records for this window's %d scenes",
                video_id, window_idx + 1, len(windows), exc, len(window),
            )
            for group in window:
                scene_records.append(_fallback_scene_record(video_id, group, color_grades, segments))
            continue

        # Match beats back to input scenes by scene_index, never by position
        # (zip() truncated to the shorter list and silently dropped whichever
        # scene the model omitted — a real gap in scene time coverage,
        # violating the "every chunk/shot time range covered by >=1 scenes
        # row" finalizer rule). Beats with an out-of-range or duplicate
        # scene_index are logged and ignored rather than trusted.
        beats_by_index: dict[int, SceneBeatItem] = {}
        for beat in parsed.scene_beats:
            if not (0 <= beat.scene_index < len(window)):
                logger.warning(
                    "run_story_architect: video_id=%s window %d/%d beat has "
                    "out-of-range scene_index=%d for %d input scenes — ignored",
                    video_id, window_idx + 1, len(windows), beat.scene_index, len(window),
                )
                continue
            if beat.scene_index in beats_by_index:
                logger.warning(
                    "run_story_architect: video_id=%s window %d/%d duplicate "
                    "scene_index=%d — keeping first, ignoring later one",
                    video_id, window_idx + 1, len(windows), beat.scene_index,
                )
                continue
            beats_by_index[beat.scene_index] = beat

        if len(beats_by_index) != len(window):
            logger.warning(
                "run_story_architect: video_id=%s window %d/%d returned %d usable beats for "
                "%d input scenes — missing scenes get a fallback record so time coverage "
                "is never gapped",
                video_id, window_idx + 1, len(windows), len(beats_by_index), len(window),
            )

        for scene_idx, group in enumerate(window):
            beat = beats_by_index.get(scene_idx)
            if beat is None:
                # Fallback: synthesize a minimal record from deterministic
                # group data alone, so this scene's time range is still
                # covered even though the model didn't return a beat for it.
                logger.warning(
                    "run_story_architect: video_id=%s scene_index=%d (scene=%s) had no "
                    "matching beat — writing fallback record with no narrative synthesis",
                    video_id, scene_idx, group.majority_scene_id,
                )
                scene_records.append(_fallback_scene_record(video_id, group, color_grades, segments))
                continue

            canonical_id = beat.canonical_scene_id.strip() or group.majority_scene_id
            resolved = [(p, _resolve_participant_pid(p, cast_payload)) for p in beat.participants]
            valid_pids = {pid for _, pid in resolved if pid is not None}
            unresolved_raw = [raw for raw, pid in resolved if pid is None]
            if unresolved_raw:
                logger.warning(
                    "run_story_architect: video_id=%s scene=%s dropped participants not in "
                    "cast: %s", video_id, canonical_id, sorted(unresolved_raw),
                )
            record = SceneRecord(
                id=gen_id(),
                video_id=video_id,
                canonical_scene_id=canonical_id,
                discarded_aliases=list(beat.discarded_aliases) or group.alias_scene_ids,
                # Ground-truth times come from the actual frame timestamps,
                # never trusted from the LLM's own arithmetic (same
                # precedent as L5's achieved_duration_s).
                start_time=group.start_time,
                end_time=group.end_time,
                participants=sorted(valid_pids),
                summary=beat.summary,
                emotional_arc=beat.emotional_arc,
                causal_link_to_next=beat.causal_link_to_next,
                usability_score=_compute_usability_score(group, color_grades, segments),
            )
            scene_records.append(record)

        rolling_summary = _truncate_rolling_summary(parsed.updated_rolling_summary)

    if not scene_records:
        logger.error(
            "run_story_architect: video_id=%s — every window failed, nothing written", video_id
        )
        return None

    _disambiguate_canonical_scene_ids(scene_records, video_id)
    # Delete-then-insert, not rely on UPSERT alone: canonical_scene_id is
    # derived from non-deterministic LLM text, so a re-run's UPSERT key
    # rarely matches a prior run's — see delete_scenes_for_video()'s
    # docstring for the real-video duplication this caused (video_id
    # 97199656 accumulated ~25 near-duplicate overlapping scene rows across
    # reruns before this fix). This makes reruns actually idempotent, which
    # is what rule 15 already claimed.
    deleted = await delete_scenes_for_video(pool, video_id)
    if deleted:
        logger.info(
            "run_story_architect: video_id=%s cleared %d scene(s) from a prior run before rewriting",
            video_id, deleted,
        )
    await bulk_upsert_scenes(pool, scene_records)
    logger.info(
        "run_story_architect: video_id=%s wrote %d canonical scenes", video_id, len(scene_records)
    )

    title, synopsis = await _generate_title_synopsis(
        client, scene_records, cast_payload, pool=pool, video_id=video_id,
    )

    existing = await get_latest_storyline(pool, video_id)
    if existing is None:
        version = 1
        storyline_id = gen_id()
    elif existing.status == "draft":
        # Idempotent rerun — overwrite our own prior draft in place.
        version = existing.version
        storyline_id = existing.id
    else:  # existing.status == "final" — never overwrite, insert a new version (rule 15)
        version = existing.version + 1
        storyline_id = gen_id()

    beats = [
        {
            "canonical_scene_id": s.canonical_scene_id,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "participants": s.participants,
            "summary": s.summary,
            "emotional_arc": s.emotional_arc,
            "causal_link_to_next": s.causal_link_to_next,
        }
        for s in scene_records
    ]

    storyline = StorylineRecord(
        id=storyline_id,
        video_id=video_id,
        version=version,
        status="draft",
        title=title,
        synopsis=synopsis,
        cast_members=cast_payload,
        beats=beats,
    )
    await upsert_storyline_draft(pool, storyline)
    logger.info(
        "run_story_architect: video_id=%s wrote storylines draft version=%d (title=%r)",
        video_id, version, title,
    )
    return storyline
