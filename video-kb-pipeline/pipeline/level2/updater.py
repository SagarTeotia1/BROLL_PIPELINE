"""Level-2 knowledge-base writer.

Persists face and color analysis results produced by face_runner and
color_runner into the Postgres knowledge base using the shared query helpers.
"""

from __future__ import annotations

import logging

from knowledge_base.postgres.queries import (
    bulk_insert_face_appearances,
    bulk_insert_face_timeline_events,
    bulk_upsert_color_grades,
    upsert_color_grade,
    upsert_person,
)
from shared.types import (
    ColorGradeRecord,
    FaceAppearanceRecord,
    FaceTimelineEvent,
    PersonRecord,
)

logger = logging.getLogger(__name__)


async def write_level2_to_kb(
    pool,  # asyncpg pool
    video_id: str,
    persons: list[PersonRecord],
    appearances: list[FaceAppearanceRecord],
    timeline_events: list[FaceTimelineEvent],
    color_grades: list[ColorGradeRecord],
) -> None:
    """Persist all Level-2 analysis results to the knowledge base.

    All operations are best-effort per item: a failure on a single record is
    logged and skipped rather than aborting the entire write.

    Args:
        pool:            An asyncpg connection pool.
        video_id:        Stable identifier for the analysed video.
        persons:         Person records from ``run_face_analysis``.
        appearances:     Face appearance records from ``run_face_analysis``.
        timeline_events: Face timeline event records from ``run_face_analysis``.
        color_grades:    Color grade records from ``run_color_grading``.
    """
    # ------------------------------------------------------------------
    # 1. Upsert persons.
    #
    #    upsert_person returns the UUID string of the persisted row.  On
    #    conflict (video_id, pid) Postgres updates the row and returns the
    #    existing id — which may differ from person.id if the person was
    #    first written in an earlier run.
    #
    #    CRITICAL: build a remap table old_uuid → db_uuid so that
    #    face_appearances and timeline_events (which embed the gen_id()
    #    uuid from face_runner) are corrected before insertion.
    # ------------------------------------------------------------------
    uuid_remap: dict[str, str] = {}  # face_runner uuid → db-persisted uuid
    upserted_person_ids: list[str] = []
    for person in persons:
        try:
            returned_id = await upsert_person(pool, person)
            upserted_person_ids.append(returned_id)
            if returned_id != person.id:
                uuid_remap[person.id] = returned_id
        except Exception as exc:
            logger.warning(
                "Failed to upsert person pid=%s (video_id=%s): %s",
                person.pid,
                video_id,
                exc,
            )

    if uuid_remap:
        logger.info(
            "Person UUID remap needed for %d PIDs (ON CONFLICT returned existing UUIDs) — "
            "remapping appearances and timeline events.",
            len(uuid_remap),
        )
        for app in appearances:
            if app.person_id in uuid_remap:
                app.person_id = uuid_remap[app.person_id]
        for ev in timeline_events:
            if ev.person_id in uuid_remap:
                ev.person_id = uuid_remap[ev.person_id]

    logger.info(
        "Upserted %d/%d person records for video %s",
        len(upserted_person_ids),
        len(persons),
        video_id,
    )

    # ------------------------------------------------------------------
    # 2. Insert face appearances — best-effort per record.
    #
    #    bulk_insert_face_appearances uses executemany, which means a
    #    single bad row can abort the whole batch.  We attempt the full
    #    batch first; if that fails we fall back to one-at-a-time inserts
    #    so that valid records are not lost because of one bad entry.
    # ------------------------------------------------------------------
    if appearances:
        inserted_appearances = 0
        try:
            await bulk_insert_face_appearances(pool, appearances)
            inserted_appearances = len(appearances)
        except Exception as bulk_exc:
            logger.warning(
                "bulk_insert_face_appearances failed for video %s (%s) — "
                "falling back to per-record inserts.",
                video_id,
                bulk_exc,
            )
            for app in appearances:
                try:
                    await bulk_insert_face_appearances(pool, [app])
                    inserted_appearances += 1
                except Exception as exc:
                    logger.warning(
                        "Skipping face appearance id=%s (frame=%s, video=%s): %s",
                        app.id,
                        app.frame_index,
                        video_id,
                        exc,
                    )
        logger.info(
            "Inserted %d/%d face appearance records for video %s",
            inserted_appearances,
            len(appearances),
            video_id,
        )
    else:
        logger.info("No face appearances to insert for video %s", video_id)

    # ------------------------------------------------------------------
    # 3. Insert face timeline events — best-effort per record.
    #
    #    Same bulk-then-fallback strategy as appearances.
    # ------------------------------------------------------------------
    if timeline_events:
        inserted_events = 0
        try:
            await bulk_insert_face_timeline_events(pool, timeline_events)
            inserted_events = len(timeline_events)
        except Exception as bulk_exc:
            logger.warning(
                "bulk_insert_face_timeline_events failed for video %s (%s) — "
                "falling back to per-record inserts.",
                video_id,
                bulk_exc,
            )
            for ev in timeline_events:
                try:
                    await bulk_insert_face_timeline_events(pool, [ev])
                    inserted_events += 1
                except Exception as exc:
                    logger.warning(
                        "Skipping face timeline event id=%s "
                        "(actor=%s, video=%s): %s",
                        ev.id,
                        ev.person_id,
                        video_id,
                        exc,
                    )
        logger.info(
            "Inserted %d/%d face timeline events for video %s",
            inserted_events,
            len(timeline_events),
            video_id,
        )
    else:
        logger.info("No face timeline events to insert for video %s", video_id)

    # ------------------------------------------------------------------
    # 4. Upsert color grade records — bulk first, per-record fallback.
    # ------------------------------------------------------------------
    inserted_grades = 0
    if color_grades:
        try:
            await bulk_upsert_color_grades(pool, color_grades)
            inserted_grades = len(color_grades)
        except Exception as bulk_exc:
            logger.warning(
                "bulk_upsert_color_grades failed for video %s (%s) — "
                "falling back to per-record upserts.",
                video_id,
                bulk_exc,
            )
            for grade in color_grades:
                try:
                    await upsert_color_grade(pool, grade)
                    inserted_grades += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to upsert color grade shot_id=%s (video_id=%s): %s",
                        grade.shot_id,
                        video_id,
                        exc,
                    )

    logger.info(
        "Upserted %d/%d color grade records for video %s",
        inserted_grades,
        len(color_grades),
        video_id,
    )

    logger.info(
        "Level-2 KB write complete for video %s — "
        "persons=%d appearances=%d timeline_events=%d color_grades=%d",
        video_id,
        len(upserted_person_ids),
        len(appearances),
        len(timeline_events),
        inserted_grades,
    )
