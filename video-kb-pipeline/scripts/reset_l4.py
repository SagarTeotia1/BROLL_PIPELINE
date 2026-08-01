"""Hard-reset Level-4 data for one video, for a genuine from-zero re-run.

`modal run modal_app.py::reset_level` only resets `processing_jobs.status`
back to 'queued' — it does NOT undo any partial data L4 already wrote. Since
L4's own idempotency design deliberately skips speaker_turns that are
already resolved ("never re-decide a turn this agent already resolved on a
prior run"), re-running after only a status reset would silently process
LESS than a full run — not the clean from-zero timing/cost measurement
this script is for.

This reverts everything L4 (Grounding Agent + Story Architect) wrote for
one video:
  - speaker_turns: rows L4 touched (resolution_method IN
    ('llm_tiebreak', 'llm_unresolved_final')) revert to 'unresolved',
    person_id/confidence cleared. Rows L4 never touched (single_candidate,
    any remaining face_majority/unresolved) are left alone.
  - kg_edges.canonical_relation: cleared for this video.
  - scenes / storylines: deleted for this video (Story Architect's output).
  - correction_events / pipeline_alerts for level=4: cleared for this video
    (cleanup — normally empty for an automatic L4 run, included for safety).
  - processing_jobs(level=4): reset to 'queued', error_msg/meta cleared.

All in one transaction — either the whole reset applies, or none of it does.

Usage:
    python scripts/reset_l4.py --video-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from knowledge_base.postgres.client import get_pool  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-reset L4 data for one video.")
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()
    video_id = args.video_id

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            turns_reset = await conn.execute(
                """
                UPDATE speaker_turns
                SET resolution_method = 'unresolved', person_id = NULL, confidence = NULL
                WHERE video_id = $1::uuid
                  AND resolution_method IN ('llm_tiebreak', 'llm_unresolved_final')
                """,
                video_id,
            )
            edges_reset = await conn.execute(
                "UPDATE kg_edges SET canonical_relation = NULL WHERE video_id = $1::uuid",
                video_id,
            )
            scenes_deleted = await conn.execute(
                "DELETE FROM scenes WHERE video_id = $1::uuid", video_id,
            )
            storylines_deleted = await conn.execute(
                "DELETE FROM storylines WHERE video_id = $1::uuid", video_id,
            )
            corrections_deleted = await conn.execute(
                "DELETE FROM correction_events WHERE video_id = $1::uuid AND level = 4",
                video_id,
            )
            alerts_deleted = await conn.execute(
                "DELETE FROM pipeline_alerts WHERE video_id = $1::uuid AND level = 4",
                video_id,
            )
            job_reset = await conn.execute(
                """
                UPDATE processing_jobs
                SET status = 'queued', error_msg = NULL, meta = '{}'::jsonb
                WHERE video_id = $1::uuid AND level = 4
                """,
                video_id,
            )

    print(f"[reset_l4] video_id={video_id}")
    print(f"  speaker_turns reverted: {turns_reset}")
    print(f"  kg_edges.canonical_relation cleared: {edges_reset}")
    print(f"  scenes deleted: {scenes_deleted}")
    print(f"  storylines deleted: {storylines_deleted}")
    print(f"  correction_events (level=4) deleted: {corrections_deleted}")
    print(f"  pipeline_alerts (level=4) deleted: {alerts_deleted}")
    print(f"  processing_jobs(level=4) reset: {job_reset}")
    print(
        f"\n  Next: modal run modal_app.py::run_level4_modal --video-id {video_id}"
    )


if __name__ == "__main__":
    asyncio.run(main())
