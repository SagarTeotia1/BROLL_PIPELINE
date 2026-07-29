"""Level-4 orchestration wrapper.

Unlike `pipeline.level2.updater` (which does all the Postgres writing
itself), the L4 agents write their own rows directly:
  - `grounding_runner` (Engineer A) updates `speaker_turns`,
    `kg_edges.canonical_relation`, and `searchable_facts.superseded_by`
    internally, one function per sub-task.
  - `story_architect_runner` (Engineer B) writes `scenes` (draft) and one
    `storylines` (draft) row internally.

So `run_level4` here is purely sequencing: call the Grounding Agent's three
sub-tasks, then the Story Architect, then the finalization gate
(`pipeline.level4.finalizer.finalize_level4`) which validates completeness,
propagates to Neo4j/Pinecone, and flips `storylines.status` draft -> final.

Order matters: relation canonicalization / fact dedup / speaker resolution
are independent of scene/storyline synthesis, but the finalizer's
completeness checks (speaker_turns terminal states, scene timeline
coverage, storyline synopsis/beats) require ALL of A's and B's writes to
have landed first — so finalize_level4 always runs last.
"""
from __future__ import annotations

import logging

import asyncpg

from pipeline.level4.finalizer import finalize_level4
from pipeline.level4.grounding_runner import (  # type: ignore[import]
    run_fact_dedup,
    run_relation_canonicalization,
    run_speaker_resolution,
)
from pipeline.level4.story_architect_runner import run_story_architect  # type: ignore[import]

logger = logging.getLogger(__name__)


async def run_level4(pool: asyncpg.Pool, video_id: str) -> dict:
    """Run the full Level-4 reasoning pass for *video_id*: Grounding Agent
    (speaker resolution -> relation canonicalization -> fact dedup), then
    Story Architect Agent, then the finalization gate.

    Each stage writes its own Postgres rows internally (see module
    docstring) — this function does not write to the KB itself, it only
    sequences the calls and returns the finalizer's result dict.

    Args:
        pool:     An open asyncpg connection pool.
        video_id: The video to run Level-4 reasoning for.

    Returns:
        The dict returned by `finalize_level4`:
        `{"ok": True, "storyline_version": int, "scenes_checked": int}` on
        success, or `{"ok": False, "reason": "<specific failed check>"}` if
        the completeness gate did not pass.
    """
    logger.info("[L4] run_level4 start — video_id=%s", video_id)

    # ------------------------------------------------------------------
    # 1. Grounding Agent — speaker resolution, then relation
    #    canonicalization, then fact dedup. Each writes its own rows.
    #    Order between these three doesn't matter for correctness (they
    #    touch disjoint tables), but speaker resolution runs first since
    #    it's the input the Story Architect's speaker_turns join needs.
    # ------------------------------------------------------------------
    await run_speaker_resolution(pool, video_id)
    logger.info("[L4] Grounding Agent 1a (speaker resolution) complete — video_id=%s", video_id)

    await run_relation_canonicalization(pool, video_id)
    logger.info("[L4] Grounding Agent 1b (relation canonicalization) complete — video_id=%s", video_id)

    await run_fact_dedup(pool, video_id)
    logger.info("[L4] Grounding Agent 1c (fact dedup) complete — video_id=%s", video_id)

    # ------------------------------------------------------------------
    # 2. Story Architect Agent — canonical scene timeline + storyline
    #    synthesis (draft). Depends on resolved speaker_turns from step 1.
    # ------------------------------------------------------------------
    await run_story_architect(pool, video_id)
    logger.info("[L4] Story Architect Agent complete — video_id=%s", video_id)

    # ------------------------------------------------------------------
    # 3. Finalization gate — completeness checks, Neo4j/Pinecone
    #    propagation, draft -> final flip. Always runs last.
    # ------------------------------------------------------------------
    result = await finalize_level4(pool, video_id)
    logger.info("[L4] run_level4 complete — video_id=%s result=%s", video_id, result)
    return result
