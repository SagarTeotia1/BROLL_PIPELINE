"""Level-4 finalization gate.

Implements the "Finalization & source-of-truth contract" from CLAUDE.md
(section "LEVEL 4 — REASONING" -> "Finalization & source-of-truth contract"):
`processing_jobs(level=4)` only reaches DONE after a completeness pass over
Postgres, never just because the Grounding Agent / Story Architect calls
didn't throw. If any check fails, `storylines.status` stays `draft` and the
caller (modal_app.py) marks the job FAILED with the specific reason string
returned here — never a green checkmark over incomplete data.

Postgres is the source of truth. Neo4j (canonical-relation edge correction)
and Pinecone (canonical scenes + storylines projections) are written AFTER
the completeness checks pass, as projections of that Postgres state — same
"write Postgres first, project after" order kb_finalizer.py already uses for
L3. Failures in those projection writes are logged and swallowed (non-fatal):
Postgres correctness + `storylines.status` reaching `final` is what gates
DONE, not that every downstream projection succeeded (CLAUDE.md "Write order
in finalizer.py").
"""
from __future__ import annotations

import asyncio
import logging

import asyncpg

from knowledge_base.postgres.queries import (
    count_unterminal_speaker_turns,
    get_chunks_for_video,
    get_latest_storyline,
    get_scenes_for_video,
    get_shots_for_video,
    update_storyline_status,
)
from shared.types import SceneRecord, StorylineRecord

logger = logging.getLogger(__name__)

# Timeline-coverage checks are float-time comparisons — allow a small epsilon
# so floating point rounding at shot/scene boundaries doesn't register as a
# gap. Anything wider than this is a genuine uncovered range.
_GAP_EPSILON_S = 0.5


# ---------------------------------------------------------------------------
# Interval-coverage helpers (no external deps — pure Python)
# ---------------------------------------------------------------------------


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/touching (start, end) intervals into a sorted, disjoint list."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + _GAP_EPSILON_S:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _uncovered_ranges(
    target: list[tuple[float, float]], covering: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return the sub-ranges of *target*'s union not covered by *covering*'s union.

    Used to detect silent time gaps in the canonical `scenes` timeline: target
    is the union of chunk/shot ranges (what MUST be covered), covering is the
    union of `scenes` row ranges (what IS covered).
    """
    merged_target = _merge_intervals(target)
    merged_cover = _merge_intervals(covering)
    gaps: list[tuple[float, float]] = []

    for t_start, t_end in merged_target:
        cursor = t_start
        for c_start, c_end in merged_cover:
            if c_end <= cursor:
                continue
            if c_start >= t_end:
                break
            if c_start > cursor + _GAP_EPSILON_S:
                gaps.append((cursor, min(c_start, t_end)))
            cursor = max(cursor, c_end)
            if cursor >= t_end:
                break
        if cursor < t_end - _GAP_EPSILON_S:
            gaps.append((cursor, t_end))

    return gaps


# ---------------------------------------------------------------------------
# Completeness checks (CLAUDE.md finalization contract #1)
# ---------------------------------------------------------------------------


async def _check_speaker_turns(pool: asyncpg.Pool, video_id: str) -> str | None:
    """Return a failure reason string, or None if the check passes.

    Every speaker_turns row must be single_candidate / llm_tiebreak /
    llm_unresolved_final — none left as bare unresolved/face_majority
    (those mean "L4 hasn't looked at this yet").
    """
    remaining = await count_unterminal_speaker_turns(pool, video_id)
    if remaining > 0:
        return (
            f"{remaining} speaker_turns row(s) still unresolved/face_majority — "
            "Grounding Agent speaker resolution has not completed for this video"
        )
    return None


async def _check_scene_coverage(pool: asyncpg.Pool, video_id: str) -> str | None:
    """Return a failure reason string, or None if the check passes.

    Every chunks/shots time range must be covered by at least one `scenes`
    row — no silent time gaps in the canonical scene timeline. Shots are
    used as the coverage target when present (finer-grained than chunks);
    falls back to chunks if a video has no shots recorded.
    """
    shots = await get_shots_for_video(pool, video_id)
    target_ranges: list[tuple[float, float]]
    target_label: str
    if shots:
        target_ranges = [(s.start_time, s.end_time) for s in shots if s.start_time is not None and s.end_time is not None]
        target_label = "shots"
    else:
        chunks = await get_chunks_for_video(pool, video_id)
        target_ranges = [(c.start_time, c.end_time) for c in chunks if c.start_time is not None and c.end_time is not None]
        target_label = "chunks"

    if not target_ranges:
        # Nothing to cover (e.g. L1 produced no shots/chunks) — not an L4 problem.
        return None

    scenes = await get_scenes_for_video(pool, video_id)
    scene_ranges = [(sc.start_time, sc.end_time) for sc in scenes]

    gaps = _uncovered_ranges(target_ranges, scene_ranges)
    if gaps:
        total_gap_s = sum(e - s for s, e in gaps)
        first = gaps[0]
        return (
            f"canonical scene timeline has {len(gaps)} time gap(s) totalling "
            f"{total_gap_s:.1f}s not covered by any `scenes` row (checked against "
            f"{target_label}) — first gap [{first[0]:.1f}s, {first[1]:.1f}s)"
        )
    return None


async def _check_storyline(pool: asyncpg.Pool, video_id: str) -> tuple[str | None, StorylineRecord | None]:
    """Return (failure_reason_or_None, latest_storyline_row_or_None).

    The video's storylines row (latest version) must have non-null synopsis
    and beats with length > 0.
    """
    storyline = await get_latest_storyline(pool, video_id)
    if storyline is None:
        return "no `storylines` row found for this video — Story Architect Agent has not run", None
    if not storyline.synopsis or not storyline.synopsis.strip():
        return f"storylines row (video_id={video_id}, version={storyline.version}) has null/empty synopsis", storyline
    if not storyline.beats or len(storyline.beats) == 0:
        return f"storylines row (video_id={video_id}, version={storyline.version}) has zero beats", storyline
    return None, storyline


# ---------------------------------------------------------------------------
# Neo4j + Pinecone propagation (CLAUDE.md "Neo4j + Pinecone propagation")
# Non-fatal: logged and swallowed. Postgres correctness gates DONE, not this.
# ---------------------------------------------------------------------------


async def _embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Embed texts with the local BAAI/bge-large-en-v1.5 model (same model/
    call kb_finalizer.py uses for L3 facts/scenes). L4's embedding volume is
    tiny (~1 vector per canonical scene, ~50/video, + 1 per storyline) so
    this runs fine on CPU when no GPU is present in the calling container —
    unlike L3's per-frame fact embeddings, it is not a throughput-sensitive
    path. Returns None per-position on failure so callers can skip cleanly.
    """
    if not texts:
        return []
    try:
        from models.embed_model import LocalEmbedder  # type: ignore[import]

        embedder = LocalEmbedder.get()
        embeddings: list[list[float]] = await asyncio.to_thread(embedder.embed, texts, 64)
        return embeddings  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("L4 finalizer: local embedding failed (%d texts): %s", len(texts), exc)
        return [None] * len(texts)


async def _propagate_neo4j(video_id: str) -> None:
    try:
        from knowledge_base.neo4j.graph_writer import update_edge_relations_canonical  # type: ignore[import]

        written = await update_edge_relations_canonical(video_id)
        logger.info("L4 finalizer: Neo4j canonical-relation correction wrote %d edges", written)
    except Exception as exc:
        logger.error("L4 finalizer: Neo4j canonical-relation propagation failed (non-fatal): %s", exc)


async def _propagate_pinecone_scenes(pool: asyncpg.Pool, video_id: str) -> None:
    try:
        scenes: list[SceneRecord] = await get_scenes_for_video(pool, video_id)
        if not scenes:
            return
        summaries = [sc.summary or "" for sc in scenes]
        embeddings = await _embed_texts(summaries)

        records = [
            {
                "id": f"{video_id}::{sc.canonical_scene_id}",
                "video_id": video_id,
                "canonical_scene_id": sc.canonical_scene_id,
                "summary": sc.summary or "",
                "embedding": emb,
                "metadata": {
                    "start_time": sc.start_time,
                    "end_time": sc.end_time,
                    "emotional_arc": sc.emotional_arc or "",
                },
            }
            for sc, emb in zip(scenes, embeddings)
            if emb is not None
        ]
        if not records:
            return

        from knowledge_base.pinecone_kb.indexer import upsert_canonical_scenes  # type: ignore[import]

        upserted = await asyncio.to_thread(upsert_canonical_scenes, records)
        logger.info("L4 finalizer: upserted %d canonical scene vectors to Pinecone", upserted)
    except Exception as exc:
        logger.error("L4 finalizer: Pinecone canonical-scenes propagation failed (non-fatal): %s", exc)


async def _propagate_pinecone_storyline(video_id: str, storyline: StorylineRecord) -> None:
    try:
        embeddings = await _embed_texts([storyline.synopsis or ""])
        emb = embeddings[0] if embeddings else None
        if emb is None:
            return

        record = {
            "id": f"{video_id}::v{storyline.version}",
            "video_id": video_id,
            "storyline_id": storyline.id,
            "version": storyline.version,
            "title": storyline.title or "",
            "synopsis": storyline.synopsis or "",
            "embedding": emb,
        }

        from knowledge_base.pinecone_kb.indexer import upsert_storylines  # type: ignore[import]

        upserted = await asyncio.to_thread(upsert_storylines, [record])
        logger.info("L4 finalizer: upserted %d storyline vector(s) to Pinecone", upserted)
    except Exception as exc:
        logger.error("L4 finalizer: Pinecone storyline propagation failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def finalize_level4(pool: asyncpg.Pool, video_id: str) -> dict:
    """Run the L4 completeness gate and, if it passes, propagate to Neo4j /
    Pinecone and flip the video's latest `storylines` row from draft -> final.

    Returns a dict:
      - {"ok": True, "storyline_version": int, "scenes_checked": int}
        on success (storylines.status is now 'final').
      - {"ok": False, "reason": "<specific failed check>"} on failure
        (storylines.status is left untouched — still 'draft' or missing).

    Never raises for expected failure modes (a failing check, or a failing
    projection write) — only for genuinely unexpected errors (e.g. the
    Postgres pool itself being unusable), which the caller (modal_app.py)
    catches and reports as the L4 job's FAILED error_msg.
    """
    # ------------------------------------------------------------------
    # 1. Completeness checks — run independently so a single failure
    #    doesn't hide a second one; report the first failure found in
    #    CLAUDE.md's listed order (speaker turns -> scene coverage -> storyline).
    # ------------------------------------------------------------------
    speaker_reason = await _check_speaker_turns(pool, video_id)
    if speaker_reason:
        logger.warning("L4 finalizer: FAILED for video_id=%s — %s", video_id, speaker_reason)
        return {"ok": False, "reason": speaker_reason}

    scene_reason = await _check_scene_coverage(pool, video_id)
    if scene_reason:
        logger.warning("L4 finalizer: FAILED for video_id=%s — %s", video_id, scene_reason)
        return {"ok": False, "reason": scene_reason}

    storyline_reason, storyline = await _check_storyline(pool, video_id)
    if storyline_reason or storyline is None:
        reason = storyline_reason or "no storylines row found"
        logger.warning("L4 finalizer: FAILED for video_id=%s — %s", video_id, reason)
        return {"ok": False, "reason": reason}

    logger.info(
        "L4 finalizer: all completeness checks passed for video_id=%s (storyline version=%d)",
        video_id, storyline.version,
    )

    # ------------------------------------------------------------------
    # 2. Projections — Postgres is already correct at this point; these are
    #    best-effort mirrors. Failures here do NOT block finalization.
    # ------------------------------------------------------------------
    scenes = await get_scenes_for_video(pool, video_id)
    await _propagate_neo4j(video_id)
    await _propagate_pinecone_scenes(pool, video_id)
    await _propagate_pinecone_storyline(video_id, storyline)

    # ------------------------------------------------------------------
    # 3. Flip storylines.status draft -> final. This is the one write that
    #    makes the storyline visible to L5 (rule 16) — do it last, only
    #    after Postgres checks passed and projections were attempted.
    # ------------------------------------------------------------------
    await update_storyline_status(pool, video_id, storyline.version, "final")
    logger.info(
        "L4 finalizer: storylines(video_id=%s, version=%d) flipped draft -> final",
        video_id, storyline.version,
    )

    return {
        "ok": True,
        "storyline_version": storyline.version,
        "scenes_checked": len(scenes),
    }
