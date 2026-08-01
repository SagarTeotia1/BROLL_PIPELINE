"""Level-4 Grounding Agent — speaker turn resolution, relation
canonicalization, and fact dedup for one video.

Implements the three sub-tasks described in CLAUDE.md "LEVEL 4 — REASONING"
-> "Agent 1 — Grounding Agent":

  1a. run_speaker_resolution     — batched LLM tiebreak for ambiguous
                                    speaker_turns rows, with confidence-gated
                                    escalation to a stronger model tier.
  1b. run_relation_canonicalization — cluster kg_edges.relation strings by
                                    local embedding similarity, then ONE
                                    batched LLM call maps every cluster onto
                                    the versioned ontology_relations table.
  1c. run_fact_dedup             — pure cosine-similarity pass over
                                    searchable_facts embeddings, no LLM call.

Scope note: this module writes Postgres only (speaker_turns, kg_edges, and
searchable_facts). Neo4j edge-type correction and the `scenes`/`storylines`
pgvector embedding writes (B7 dropped Pinecone) are the responsibility of
`pipeline/level4/updater.py` (out of scope for this module — see CLAUDE.md
"Neo4j + Pinecone propagation").
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter

import asyncpg

from knowledge_base.postgres.queries import (
    bulk_update_fact_superseded_by,
    bulk_update_kg_edges_canonical_relation,
    bulk_update_speaker_turn_resolutions,
    get_distinct_kg_edge_relations,
    get_face_appearances_for_video_sampled,
    get_frame_analyses_with_timestamps_for_video,
    get_ontology_relations,
    get_persons_for_video,
    get_searchable_facts_for_video,
    get_transcript_segments_for_video,
    get_unresolved_speaker_turns,
    insert_pipeline_alert,
)
from shared.config import settings
from shared.types import (
    FaceAppearanceRecord,
    RelationClusterResult,
    SpeakerTurnRecord,
    TranscriptSegment,
)
from shared.utils import get_transcript_snippet
from prompts.grounding_relation import (
    CANONICALIZE_RELATIONS_TOOL,
    GROUNDING_RELATION_SYSTEM_PROMPT,
)
from prompts.grounding_speaker import (
    GROUNDING_SPEAKER_SYSTEM_PROMPT,
    RESOLVE_SPEAKER_TURNS_TOOL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# CLAUDE.md rule 23 / L5 "batch cap, same reasoning" precedent — cap batch
# size so per-item accuracy doesn't silently degrade on long lists.
_SPEAKER_BATCH_SIZE = 35
_RELATION_LLM_MAX_TOKENS = 4096
_SPEAKER_LLM_MAX_TOKENS = 8192

# Real-video finding: 1b sent every cluster in ONE uncapped call — with
# CANONICALIZE_RELATIONS_TOOL requiring a `reasoning` string per cluster and
# ~150+ distinct relation clusters typical for a video (CLAUDE.md's own
# "200+ distinct relation strings observed" note), output routinely exceeded
# _RELATION_LLM_MAX_TOKENS and got cut off mid-JSON ("Expecting ',' delimiter"
# deep in the response) — a real truncation bug, not the "weak backend"
# malformed-JSON case the retry logic was built for. This also explains the
# climbing OTHER-bucket rate observed across reruns of the same video:
# whichever clusters got truncated out varied run to run and silently
# defaulted to OTHER. Sub-batch, same pattern as 1a's _SPEAKER_BATCH_SIZE —
# closes the rule-23 gap (1b previously only capped raw_relations shown per
# cluster, never the cluster count itself).
_RELATION_BATCH_SIZE = 25

# Confidence-gated escalation threshold (finalization contract #2).
_LOW_CONFIDENCE_THRESHOLD = 0.5

# CLAUDE.md B4 observability thresholds — not spec'd numerically there
# (only the OTHER-bucket ~5% figure is), so these are conservative
# defaults: a video tripping either signals it's worth a human look, same
# "alert, don't silently decay" spirit as the OTHER-bucket check below.
_UNRESOLVED_FINAL_RATE_THRESHOLD = 0.20   # >20% of turns permanently ambiguous
_ESCALATION_TRIGGER_RATE_THRESHOLD = 0.30  # >30% of first-pass answers needed a 2nd (stronger) pass
_OTHER_BUCKET_RATE_THRESHOLD = 0.05        # already spec'd in CLAUDE.md 1b

# Embedding-similarity thresholds. Fact dedup's 0.95 is spec'd directly in
# CLAUDE.md ("threshold ~0.95 cosine"); the relation-clustering threshold is
# not spec'd numerically — 0.86 is a conservative default for grouping
# near-duplicate short predicate phrases with bge-large-en-v1.5 (embeddings
# are L2-normalised, so cosine similarity == dot product).
_RELATION_CLUSTER_SIM_THRESHOLD = 0.86
_FACT_DEDUP_SIM_THRESHOLD = 0.95
# Real-video cost finding: clusters can have dozens of near-duplicate raw
# relation strings; showing all of them bloated input tokens with no
# decision-quality benefit. Cap the sample shown to the LLM — the full list
# is still used for write-back (see run_relation_canonicalization).
_MAX_RELATION_EXAMPLES_PER_CLUSTER = 8

# Retries for a transient Groq API failure (network blip, 5xx) —
# small and bounded, not a retry loop.
_LLM_CALL_ATTEMPTS = 3  # bumped 2->3: malformed-JSON from cheaper OpenRouter-routed
# models (no grammar-constrained tool-calling) is a confirmed real, somewhat
# frequent failure mode on real video content — give it one more shot before
# giving up on a batch.


# ---------------------------------------------------------------------------
# Groq client + generic structured tool-call helper
#
# Groq's `chat.completions.create` is OpenAI-compatible: the system prompt
# is a `role: "system"` entry in `messages` (not a separate top-level
# `system=` kwarg like Anthropic), and a forced tool call comes back as
# `response.choices[0].message.tool_calls[0].function.arguments` — a JSON
# STRING that must be `json.loads()`-ed, not a pre-parsed dict like
# Anthropic's `tool_use` block `.input`.
# ---------------------------------------------------------------------------


def _get_groq_client():
    """Name kept for minimal diff across this module's call sites — actually
    returns an OpenRouter client now (see shared/llm_client.py). Provider
    moved from Groq to OpenRouter for the real cheap/strong two-tier split;
    Groq remains available as an explicit fallback via get_llm_client(prefer_groq=True)."""
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

    Uses `tool_choice={"type": "function", "function": {"name": ...}}` so
    the model MUST return the structured tool call — no free-text JSON
    parsing, no parse-failure retries of that kind (per CLAUDE.md 1a design
    rationale). `tool_schema` is the OpenAI-compatible
    `{"type": "function", "function": {"name", "description", "parameters"}}`
    shape (see prompts/grounding_speaker.py etc.) — NOT Anthropic's
    top-level `input_schema`.
    """
    # openai's public exception hierarchy (openai.APIError / RateLimitError /
    # APIStatusError / APIConnectionError / ...) is Stainless-generated,
    # same pattern as the groq/anthropic SDKs it replaces — imported here
    # only so an ImportError surfaces clearly if the installed openai
    # version ever drops these names.
    import openai  # noqa: F401
    import json as _json
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
                    {"role": "user", "content": _json.dumps(payload)},
                ],
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                # Real-video finding: OpenRouter routes this model across
                # multiple backend providers (Alibaba Cloud Int., SiliconFlow,
                # Nebius observed) with inconsistent tool-calling reliability
                # — some returned syntactically malformed JSON in forced tool
                # calls. require_parameters restricts routing to providers
                # that actually declare full support for every parameter in
                # this request (tools/tool_choice included), filtering out
                # weaker backends before they get a chance to misbehave.
                # reasoning.enabled=False: grounding is closed-set
                # classification, not synthesis — no benefit from a
                # reasoning pass, and it only adds latency/cost across the
                # high call volume this stage runs at (unlike Story
                # Architect, which turns it on — see story_architect_runner.py).
                extra_body={
                    "provider": {"require_parameters": True},
                    "reasoning": {"enabled": False},
                },
            )
            usage = getattr(response, "usage", None)
            _latency_ms = int((_time.monotonic() - _t0) * 1000)
            if usage is not None:
                logger.info(
                    "L4 grounding LLM call model=%s tool=%s prompt_tokens=%s completion_tokens=%s",
                    model,
                    tool_name,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )
            # L7 7c — durable cost/latency log, non-fatal (see
            # shared/llm_client.py::log_llm_call docstring).
            from shared.llm_client import log_llm_call as _log_llm_call
            await _log_llm_call(
                pool,
                video_id=video_id,
                level=4,
                stage=stage or "grounding",
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
                        return _json.loads(call.function.arguments)
                    except _json.JSONDecodeError as exc:
                        # Real-video finding: this used to `return None` here
                        # immediately, on the FIRST attempt — bypassing
                        # _LLM_CALL_ATTEMPTS entirely, even though malformed
                        # JSON syntax from a cheaper OpenRouter-routed model
                        # (no grammar-constrained tool-calling, unlike
                        # vLLM's local guided decoding) is exactly the kind
                        # of transient failure a retry is for. Fall through
                        # to the next attempt instead of returning.
                        last_exc = exc
                        logger.warning(
                            "L4 grounding LLM call attempt %d/%d returned malformed JSON "
                            "in tool arguments (model=%s, tool=%s): %s",
                            attempt, _LLM_CALL_ATTEMPTS, model, tool_name, exc,
                        )
                        break
            if found_tool_call:
                continue
            logger.warning(
                "L4 grounding LLM call attempt %d/%d returned no matching tool_call "
                "(model=%s, tool=%s)",
                attempt, _LLM_CALL_ATTEMPTS, model, tool_name,
            )
        except Exception as exc:  # noqa: BLE001 — log & retry/return, never raise into caller loop
            last_exc = exc
            logger.warning(
                "L4 grounding LLM call attempt %d/%d failed (model=%s, tool=%s): %s",
                attempt,
                _LLM_CALL_ATTEMPTS,
                model,
                tool_name,
                exc,
            )
    logger.error(
        "L4 grounding LLM call exhausted %d attempts (model=%s, tool=%s): %s",
        _LLM_CALL_ATTEMPTS,
        model,
        tool_name,
        last_exc,
    )
    return None


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_resolutions_lenient(raw: dict) -> list["SpeakerResolutionItem"]:
    """Validate each item in raw["resolutions"] individually instead of the
    whole batch at once via SpeakerResolutionBatch.model_validate().

    Real-video finding: a batch of 35 turns had one malformed item mid-list
    (a cheaper OpenRouter-routed model deviated from the tool schema partway
    through a long structured output) — all-or-nothing batch validation
    discarded every OTHER item in that batch too, including ones that were
    genuinely valid. That's real data loss for turns the model actually
    resolved correctly. Per-item validation keeps the good ones and only
    drops the specific malformed entries, logged individually (still "reject
    and log on schema mismatch," per rule 12 — just scoped to the item that
    actually violated the schema, not everything sharing its batch).
    """
    from shared.types import SpeakerResolutionItem

    items_raw = raw.get("resolutions")
    if not isinstance(items_raw, list):
        logger.error(
            "run_speaker_resolution: 'resolutions' missing or not a list in tool output — "
            "dropping entire response: %r", raw,
        )
        return []

    valid: list[SpeakerResolutionItem] = []
    for i, item_raw in enumerate(items_raw):
        try:
            valid.append(SpeakerResolutionItem.model_validate(item_raw))
        except Exception as exc:
            logger.error(
                "run_speaker_resolution: dropping malformed resolution item at index %d "
                "(kept %d valid items from the rest of this batch): %s | raw=%r",
                i, len(valid), exc, item_raw,
            )
    return valid


# ---------------------------------------------------------------------------
# 1a. Speaker turn resolution
# ---------------------------------------------------------------------------


def _normalize_pid(raw_pid: str | None, known_pids: set[str]) -> str | None:
    """Best-effort case normalization: Qwen frame output has been observed
    to emit lowercase pids ("p1") while `persons.pid` is canonically
    uppercase ("P1"). Map to the canonical cast pid when possible so the
    LLM sees consistent identifiers; otherwise pass the raw value through
    unchanged (still useful signal, just won't match `cast`)."""
    if not raw_pid:
        return raw_pid
    upper = raw_pid.strip().upper()
    return upper if upper in known_pids else raw_pid


def _build_turn_context(
    turn: SpeakerTurnRecord,
    known_pids: set[str],
    segments: list[TranscriptSegment],
    face_apps: list[FaceAppearanceRecord],
    pid_by_person_db_id: dict[str, str],
    frame_rows: list[dict],
) -> dict:
    start, end = turn.start_time, turn.end_time
    midpoint = (start + end) / 2.0
    half_window = max((end - start) / 2.0, 0.0) + 2.0
    transcript_snippet = get_transcript_snippet(segments, midpoint, half_window)

    visual_context: list[dict] = []
    for row in frame_rows:
        ts = row["timestamp_s"]
        if ts is None or not (start <= ts <= end):
            continue
        for person_entry in (row["qwen_output"] or {}).get("people") or []:
            pid = _normalize_pid(person_entry.get("pid"), known_pids)
            visual_context.append(
                {
                    "pid": pid,
                    "gaze": person_entry.get("gaze", ""),
                    "pose": person_entry.get("pose", ""),
                    "action": person_entry.get("action", ""),
                    "story_role": person_entry.get("story_role", ""),
                }
            )

    strict_counter: Counter[str] = Counter()
    widened_counter: Counter[str] = Counter()
    wide_start, wide_end = start - 3.0, end + 3.0
    for fa in face_apps:
        if fa.person_id is None or fa.timestamp_s is None:
            continue
        pid = pid_by_person_db_id.get(fa.person_id)
        if not pid:
            continue
        if wide_start <= fa.timestamp_s <= wide_end:
            widened_counter[pid] += 1
        if start <= fa.timestamp_s <= end:
            strict_counter[pid] += 1

    return {
        "turn_id": turn.id,
        "cluster_label": turn.cluster_label,
        "start_time": start,
        "end_time": end,
        "transcript_snippet": transcript_snippet,
        "visual_context": visual_context,
        "face_presence": dict(strict_counter),
        "face_presence_widened": dict(widened_counter),
    }


async def run_speaker_resolution(pool: asyncpg.Pool, video_id: str) -> list[SpeakerTurnRecord]:
    """Resolve every `speaker_turns` row for *video_id* still in
    'unresolved' or 'face_majority' state.

    Never touches 'single_candidate' or 'llm_tiebreak' rows (idempotent —
    `get_unresolved_speaker_turns` already excludes them). Returns the list
    of SpeakerTurnRecord as written back to the DB (only the rows this call
    actually decided on — turns skipped due to an LLM call failure are left
    untouched in the DB and are not included here, so a rerun will retry
    them).
    """
    turns = await get_unresolved_speaker_turns(pool, video_id)
    if not turns:
        logger.info("run_speaker_resolution: no unresolved speaker_turns for video_id=%s", video_id)
        return []

    persons = await get_persons_for_video(pool, video_id)
    if not persons:
        logger.warning(
            "run_speaker_resolution: video_id=%s has no persons — no valid non-null "
            "answer is possible, marking all %d turns llm_unresolved_final",
            video_id,
            len(turns),
        )
        updates = [(t.id, None, None, "llm_unresolved_final") for t in turns]
        await bulk_update_speaker_turn_resolutions(pool, updates)
        await insert_pipeline_alert(
            pool, video_id, 4, "l4_llm_unresolved_final_rate", 1.0,
            _UNRESOLVED_FINAL_RATE_THRESHOLD,
        )
        return [
            SpeakerTurnRecord(
                id=t.id, video_id=t.video_id, cluster_label=t.cluster_label,
                person_id=None, start_time=t.start_time, end_time=t.end_time,
                confidence=None, resolution_method="llm_unresolved_final",
            )
            for t in turns
        ]

    known_pids = {p.pid for p in persons}
    dbid_by_pid = {p.pid: p.id for p in persons}
    pid_by_person_db_id = {p.id: p.pid for p in persons}
    cast_payload = [{"pid": p.pid, "display_name": p.display_name or p.pid} for p in persons]

    segments = await get_transcript_segments_for_video(pool, video_id)
    face_apps = await get_face_appearances_for_video_sampled(pool, video_id)
    frame_rows = await get_frame_analyses_with_timestamps_for_video(pool, video_id)

    contexts_by_id: dict[str, dict] = {
        t.id: _build_turn_context(t, known_pids, segments, face_apps, pid_by_person_db_id, frame_rows)
        for t in turns
    }
    turns_by_id = {t.id: t for t in turns}

    client = _get_groq_client()

    # ------------------------------------------------------------------
    # First pass — cheap/fast tier, batched. Batches are independent (no
    # shared state, no ordering requirement between them) and _call_tool
    # already runs its sync client call via asyncio.to_thread, so dispatching
    # every batch concurrently via asyncio.gather is real concurrency, not
    # just async syntax — this was previously a sequential for-loop awaiting
    # one batch at a time for no reason, wasting wall-clock time on a purely
    # IO-bound API call with no GPU/shared-resource contention to serialize.
    # ------------------------------------------------------------------
    first_pass: dict[str, dict] = {}
    all_contexts = list(contexts_by_id.values())
    first_pass_batches = _chunk(all_contexts, _SPEAKER_BATCH_SIZE)
    first_pass_raws = await asyncio.gather(*[
        _call_tool(
            client, settings.L4_GROUNDING_MODEL, GROUNDING_SPEAKER_SYSTEM_PROMPT,
            RESOLVE_SPEAKER_TURNS_TOOL, {"cast": cast_payload, "turns": batch},
            _SPEAKER_LLM_MAX_TOKENS,
            pool=pool, video_id=video_id, stage="grounding_speaker",
        )
        for batch in first_pass_batches
    ])
    for raw in first_pass_raws:
        if raw is None:
            continue
        for item in _parse_resolutions_lenient(raw):
            if item.turn_id in contexts_by_id:
                first_pass[item.turn_id] = item.model_dump()

    # ------------------------------------------------------------------
    # Bounded escalation — only the confidence < 0.5 subset, ONE retry at
    # the stronger model tier (finalization contract #2).
    # ------------------------------------------------------------------
    low_conf_ids = [
        tid for tid, item in first_pass.items() if item["confidence"] < _LOW_CONFIDENCE_THRESHOLD
    ]
    escalated: dict[str, dict] = {}
    if low_conf_ids:
        low_conf_contexts = [contexts_by_id[tid] for tid in low_conf_ids]
        escalation_batches = _chunk(low_conf_contexts, _SPEAKER_BATCH_SIZE)
        escalation_raws = await asyncio.gather(*[
            _call_tool(
                client, settings.L4_STORY_ARCHITECT_MODEL, GROUNDING_SPEAKER_SYSTEM_PROMPT,
                RESOLVE_SPEAKER_TURNS_TOOL, {"cast": cast_payload, "turns": batch},
                _SPEAKER_LLM_MAX_TOKENS,
                pool=pool, video_id=video_id, stage="grounding_speaker_escalation",
            )
            for batch in escalation_batches
        ])
        for raw in escalation_raws:
            if raw is None:
                continue
            for item in _parse_resolutions_lenient(raw):
                if item.turn_id in contexts_by_id:
                    escalated[item.turn_id] = item.model_dump()

    # ------------------------------------------------------------------
    # Finalize: write-back per CLAUDE.md 1a write-back rule.
    # ------------------------------------------------------------------
    updates: list[tuple[str, str | None, float | None, str]] = []
    results: list[SpeakerTurnRecord] = []
    for turn_id, turn in turns_by_id.items():
        item = first_pass.get(turn_id)
        if item is None:
            # First-pass call failed outright for this turn — leave the row
            # untouched (still 'unresolved'/'face_majority'), safe to retry
            # on the next run (idempotent).
            continue

        if item["confidence"] < _LOW_CONFIDENCE_THRESHOLD and turn_id in escalated:
            item = escalated[turn_id]

        raw_pid = item.get("person_id")
        confidence = item.get("confidence")
        person_db_id = dbid_by_pid.get(raw_pid) if raw_pid else None

        if raw_pid and person_db_id is None:
            logger.warning(
                "run_speaker_resolution: LLM returned pid=%r not in cast for turn_id=%s "
                "— treating as no answer (closed-set validation, rule 12)",
                raw_pid, turn_id,
            )

        if person_db_id is not None and confidence is not None and confidence >= _LOW_CONFIDENCE_THRESHOLD:
            resolution_method = "llm_tiebreak"
            updates.append((turn_id, person_db_id, confidence, resolution_method))
            results.append(SpeakerTurnRecord(
                id=turn_id, video_id=video_id, cluster_label=turn.cluster_label,
                person_id=person_db_id, start_time=turn.start_time, end_time=turn.end_time,
                confidence=confidence, resolution_method=resolution_method,
            ))
        else:
            resolution_method = "llm_unresolved_final"
            updates.append((turn_id, None, confidence, resolution_method))
            results.append(SpeakerTurnRecord(
                id=turn_id, video_id=video_id, cluster_label=turn.cluster_label,
                person_id=None, start_time=turn.start_time, end_time=turn.end_time,
                confidence=confidence, resolution_method=resolution_method,
            ))

    await bulk_update_speaker_turn_resolutions(pool, updates)
    n_unresolved_final = sum(1 for r in results if r.resolution_method == "llm_unresolved_final")
    logger.info(
        "run_speaker_resolution: video_id=%s decided=%d/%d turns (tiebreak=%d, unresolved_final=%d, "
        "left_untouched=%d)",
        video_id, len(results), len(turns),
        sum(1 for r in results if r.resolution_method == "llm_tiebreak"),
        n_unresolved_final,
        len(turns) - len(results),
    )

    # ------------------------------------------------------------------
    # B4 observability: llm_unresolved_final rate + confidence-escalation
    # trigger rate. Computed here (not in finalizer.py) because the
    # escalation trigger count (`low_conf_ids`) only exists in this
    # function's local state — nothing persists "was this turn escalated"
    # as a column, so finalizer.py has no way to reconstruct it after the
    # fact. Denominator is `len(turns)` (all turns this run attempted to
    # resolve), matching the same "rate over what was processed" framing
    # CLAUDE.md 1b uses for the OTHER-bucket check below.
    # ------------------------------------------------------------------
    if turns:
        unresolved_final_rate = n_unresolved_final / len(turns)
        if unresolved_final_rate > _UNRESOLVED_FINAL_RATE_THRESHOLD:
            logger.warning(
                "run_speaker_resolution: llm_unresolved_final rate is %.1f%% "
                "for video_id=%s — signal for human review (pipeline_alerts)",
                unresolved_final_rate * 100, video_id,
            )
            await insert_pipeline_alert(
                pool, video_id, 4, "l4_llm_unresolved_final_rate",
                unresolved_final_rate, _UNRESOLVED_FINAL_RATE_THRESHOLD,
            )

        escalation_trigger_rate = len(low_conf_ids) / len(turns)
        if escalation_trigger_rate > _ESCALATION_TRIGGER_RATE_THRESHOLD:
            logger.warning(
                "run_speaker_resolution: confidence-escalation trigger rate is %.1f%% "
                "for video_id=%s — first-pass model struggling on this video "
                "(pipeline_alerts)",
                escalation_trigger_rate * 100, video_id,
            )
            await insert_pipeline_alert(
                pool, video_id, 4, "l4_confidence_escalation_trigger_rate",
                escalation_trigger_rate, _ESCALATION_TRIGGER_RATE_THRESHOLD,
            )

    return results


# ---------------------------------------------------------------------------
# 1b. Relation canonicalization
# ---------------------------------------------------------------------------


def _humanize_relation(relation: str) -> str:
    """Turn a SCREAMING_SNAKE_CASE predicate into natural-language text —
    bge-large-en-v1.5 was trained on natural text, not identifiers, so this
    materially improves clustering quality."""
    return relation.replace("_", " ").strip().lower()


async def _embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Embed *texts* using the same local BAAI/bge-large-en-v1.5 model
    already used for searchable_facts/scenes (models/embed_model.py) — no
    new embedding dependency, per CLAUDE.md 1b."""
    if not texts:
        return []
    try:
        from models.embed_model import LocalEmbedder

        embedder = LocalEmbedder.get()
        # Real bug found on a live run: L4 runs in its own CPU-only Modal
        # container (l4_image), separate from L3's QwenAnalyserModal, which
        # is the only place that previously called .load() — L4's container
        # never loaded the model at all. .load() is idempotent (no-op if
        # already loaded), so calling it here is always safe and makes this
        # function self-sufficient regardless of orchestration order.
        await asyncio.to_thread(embedder.load)
        embeddings = await asyncio.to_thread(embedder.embed, texts, 64)
        return embeddings  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("L4 relation clustering: local embedding failed (%d texts): %s", len(texts), exc)
        return [None] * len(texts)


def _cosine(a: list[float], b: list[float]) -> float:
    # bge-large-en-v1.5 embeddings are L2-normalised (LocalEmbedder.embed
    # sets normalize_embeddings=True), so dot product == cosine similarity.
    return sum(x * y for x, y in zip(a, b))


def _cluster_by_similarity(
    items: list[str], embeddings: list[list[float] | None], threshold: float
) -> list[list[str]]:
    """Greedy single-pass clustering: each item joins the first existing
    cluster whose centroid similarity clears *threshold*, else starts a new
    cluster. Cheap and order-stable — good enough for the few hundred
    distinct relation strings typical of one video (not a general-purpose
    clustering algorithm)."""
    clusters: list[dict] = []  # [{"members": [...], "centroid": [...]}]
    for item, emb in zip(items, embeddings):
        if emb is None:
            # No embedding available — isolate into its own cluster rather
            # than silently dropping it.
            clusters.append({"members": [item], "centroid": None})
            continue
        best_idx, best_sim = -1, -1.0
        for idx, cluster in enumerate(clusters):
            if cluster["centroid"] is None:
                continue
            sim = _cosine(emb, cluster["centroid"])
            if sim > best_sim:
                best_sim, best_idx = sim, idx
        if best_idx >= 0 and best_sim >= threshold:
            cluster = clusters[best_idx]
            n = len(cluster["members"])
            cluster["centroid"] = [
                (c * n + e) / (n + 1) for c, e in zip(cluster["centroid"], emb)
            ]
            cluster["members"].append(item)
        else:
            clusters.append({"members": [item], "centroid": emb})
    return [c["members"] for c in clusters]


async def run_relation_canonicalization(pool: asyncpg.Pool, video_id: str) -> list[tuple[str, str]]:
    """Assign `kg_edges.canonical_relation` for every distinct raw `relation`
    string used by *video_id*'s edges.

    Returns the list of (raw_relation, canonical_relation) pairs written.
    """
    relations = await get_distinct_kg_edge_relations(pool, video_id)
    if not relations:
        logger.info("run_relation_canonicalization: no kg_edges for video_id=%s", video_id)
        return []

    ontology = await get_ontology_relations(pool, settings.L4_ONTOLOGY_VERSION)
    if not ontology:
        logger.error(
            "run_relation_canonicalization: ontology_relations has no rows for version=%d "
            "— cannot canonicalize video_id=%s",
            settings.L4_ONTOLOGY_VERSION, video_id,
        )
        return []
    ontology_set = set(ontology)

    readable = [_humanize_relation(r) for r in relations]
    embeddings = await _embed_texts(readable)
    clusters = _cluster_by_similarity(relations, embeddings, _RELATION_CLUSTER_SIM_THRESHOLD)

    # Real-video cost finding: sending every raw relation string in a
    # cluster (some clusters have dozens of near-duplicate variants, per
    # CLAUDE.md's own "200+ distinct relation strings observed" note)
    # bloated single calls to 40K+ input tokens for no decision-quality
    # benefit — the LLM only needs enough examples to recognize the
    # pattern, not literally every variant. Cap what's SHOWN to the model;
    # cluster_by_id below keeps the FULL list for write-back (every raw
    # relation still gets mapped to a canonical label, just not
    # individually shown in the prompt).
    cluster_payload = [
        {
            "cluster_id": f"c{i}",
            "raw_relations": members[:_MAX_RELATION_EXAMPLES_PER_CLUSTER],
            "total_count": len(members),
        }
        for i, members in enumerate(clusters)
    ]
    cluster_by_id = {f"c{i}": members for i, members in enumerate(clusters)}

    client = _get_groq_client()
    cluster_batches = _chunk(cluster_payload, _RELATION_BATCH_SIZE)
    batch_raws = await asyncio.gather(*[
        _call_tool(
            client, settings.L4_GROUNDING_MODEL, GROUNDING_RELATION_SYSTEM_PROMPT,
            CANONICALIZE_RELATIONS_TOOL, {"clusters": batch, "ontology": ontology},
            _RELATION_LLM_MAX_TOKENS,
            pool=pool, video_id=video_id, stage="grounding_relation",
        )
        for batch in cluster_batches
    ])

    parsed_results: list[RelationClusterResult] = []
    for batch_idx, raw in enumerate(batch_raws):
        if raw is None:
            logger.error(
                "run_relation_canonicalization: video_id=%s batch %d/%d LLM call failed — "
                "its clusters default to OTHER below (safe to retry on next run)",
                video_id, batch_idx + 1, len(cluster_batches),
            )
            continue
        results_raw = raw.get("results")
        if not isinstance(results_raw, list):
            logger.error(
                "run_relation_canonicalization: video_id=%s batch %d/%d 'results' missing "
                "or not a list in tool output — nothing written for this batch: %r",
                video_id, batch_idx + 1, len(cluster_batches), raw,
            )
            continue
        for i, item_raw in enumerate(results_raw):
            try:
                parsed_results.append(RelationClusterResult.model_validate(item_raw))
            except Exception as exc:
                # Per-item, not per-batch (same fix as _parse_resolutions_lenient
                # above) — one malformed cluster result must not discard every
                # other cluster's canonicalization in the same batch.
                logger.error(
                    "run_relation_canonicalization: video_id=%s batch %d/%d dropping "
                    "malformed result at index %d: %s | raw=%r",
                    video_id, batch_idx + 1, len(cluster_batches), i, exc, item_raw,
                )

    relation_to_canonical: dict[str, str] = {}
    for result in parsed_results:
        canon = result.canonical_relation if result.canonical_relation in ontology_set else "OTHER"
        for raw_rel in cluster_by_id.get(result.cluster_id, []):
            relation_to_canonical[raw_rel] = canon

    # Any cluster the LLM didn't return a result for — default to OTHER
    # rather than leaving canonical_relation NULL (NULL is ambiguous:
    # "not yet processed" vs "reviewed, no match").
    for cluster in cluster_payload:
        for raw_rel in cluster["raw_relations"]:
            relation_to_canonical.setdefault(raw_rel, "OTHER")

    await bulk_update_kg_edges_canonical_relation(pool, video_id, relation_to_canonical)

    if relation_to_canonical:
        other_frac = sum(1 for v in relation_to_canonical.values() if v == "OTHER") / len(relation_to_canonical)
        if other_frac > _OTHER_BUCKET_RATE_THRESHOLD:
            logger.warning(
                "run_relation_canonicalization: OTHER bucket is %.1f%% of distinct relations "
                "for video_id=%s — signal to review and version the ontology (not silent decay)",
                other_frac * 100, video_id,
            )
            await insert_pipeline_alert(
                pool, video_id, 4, "l4_canonical_relation_other_rate",
                other_frac, _OTHER_BUCKET_RATE_THRESHOLD,
            )

    logger.info(
        "run_relation_canonicalization: video_id=%s clustered %d distinct relations into %d clusters, "
        "%d mapped to OTHER",
        video_id, len(relations), len(clusters),
        sum(1 for v in relation_to_canonical.values() if v == "OTHER"),
    )
    return list(relation_to_canonical.items())


# ---------------------------------------------------------------------------
# 1c. Fact dedup — pure embedding cosine similarity, no LLM call
# ---------------------------------------------------------------------------


async def run_fact_dedup(pool: asyncpg.Pool, video_id: str) -> int:
    """Mark near-duplicate `searchable_facts` rows with `superseded_by`
    (never DELETE — rule 13). Returns the number of facts marked.

    Kept fact = earliest by timestamp_s among a near-duplicate cluster.
    Threshold ~0.95 cosine, as spec'd in CLAUDE.md 1c.
    """
    facts = await get_searchable_facts_for_video(pool, video_id)
    facts = [f for f in facts if f.embedding]
    if len(facts) < 2:
        logger.info("run_fact_dedup: %d facts with embeddings for video_id=%s — nothing to dedup", len(facts), video_id)
        return 0

    facts_sorted = sorted(facts, key=lambda f: (f.timestamp_s if f.timestamp_s is not None else 0.0))

    try:
        import numpy as np

        matrix = np.array([f.embedding for f in facts_sorted], dtype=np.float32)
        # bge-large embeddings are already L2-normalised, but re-normalise
        # defensively in case a stored vector wasn't (e.g. legacy rows).
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        n = len(facts_sorted)
        assigned = [False] * n
        mapping: list[tuple[str, str]] = []
        # Chunk the "kept" side to bound peak memory for very large fact
        # counts (full n x n similarity matrix would be n^2 floats).
        chunk_size = 500
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            block = matrix[start:end]  # (b, d)
            sims = block @ matrix.T    # (b, n)
            for local_i, i in enumerate(range(start, end)):
                if assigned[i]:
                    continue
                row = sims[local_i]
                for j in range(i + 1, n):
                    if assigned[j]:
                        continue
                    if row[j] >= _FACT_DEDUP_SIM_THRESHOLD:
                        mapping.append((facts_sorted[j].id, facts_sorted[i].id))
                        assigned[j] = True
    except ImportError:
        logger.warning("run_fact_dedup: numpy unavailable, falling back to pure-python O(n^2) pass")
        n = len(facts_sorted)
        assigned = [False] * n
        mapping = []
        for i in range(n):
            if assigned[i]:
                continue
            emb_i = facts_sorted[i].embedding
            for j in range(i + 1, n):
                if assigned[j] or not facts_sorted[j].embedding:
                    continue
                if _cosine(emb_i, facts_sorted[j].embedding) >= _FACT_DEDUP_SIM_THRESHOLD:
                    mapping.append((facts_sorted[j].id, facts_sorted[i].id))
                    assigned[j] = True

    await bulk_update_fact_superseded_by(pool, mapping)
    logger.info(
        "run_fact_dedup: video_id=%s marked %d/%d facts as superseded (threshold=%.2f)",
        video_id, len(mapping), len(facts_sorted), _FACT_DEDUP_SIM_THRESHOLD,
    )
    return len(mapping)
