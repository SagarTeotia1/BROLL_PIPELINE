"""Level-2 knowledge-base writer.

Persists face and color analysis results produced by face_runner and
color_runner into the Postgres knowledge base using the shared query helpers.
"""

from __future__ import annotations

import logging

from knowledge_base.postgres.queries import (
    bulk_insert_face_appearances,
    bulk_insert_face_timeline_events,
    bulk_insert_speaker_turns,
    bulk_upsert_color_grades,
    bulk_upsert_shot_mattes,
    insert_pipeline_alert,
    upsert_color_grade,
    upsert_person,
    upsert_shot_matte,
)
from shared.types import (
    ColorGradeRecord,
    FaceAppearanceRecord,
    FaceTimelineEvent,
    PersonRecord,
    ShotMatteRecord,
    SpeakerTurnRecord,
)

logger = logging.getLogger(__name__)


async def write_level2_to_kb(
    pool,  # asyncpg pool
    video_id: str,
    persons: list[PersonRecord],
    appearances: list[FaceAppearanceRecord],
    timeline_events: list[FaceTimelineEvent],
    color_grades: list[ColorGradeRecord],
    speaker_turns: list[SpeakerTurnRecord] | None = None,
    shot_mattes: list[ShotMatteRecord] | None = None,
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
        speaker_turns:   Speaker turn records from ``fuse_diarization_with_faces``
                         (Stage 0). ``person_id`` on these still refers to the
                         pre-upsert UUIDs from ``run_face_analysis`` — remapped
                         below using the same table used for appearances/events.
        shot_mattes:     Background-matte records from ``run_background_matting``
                         (A1 — runs in parallel with color grading, no
                         cross-references into the person-UUID remap above).
    """
    speaker_turns = speaker_turns or []
    shot_mattes = shot_mattes or []
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
        for turn in speaker_turns:
            if turn.person_id in uuid_remap:
                turn.person_id = uuid_remap[turn.person_id]

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

    # ------------------------------------------------------------------
    # 5. Upsert shot matte records (A1 — background matting, runs in
    #    parallel with color grading) — bulk first, per-record fallback.
    #    Same bulk-then-fallback strategy as color grades.
    # ------------------------------------------------------------------
    inserted_mattes = 0
    if shot_mattes:
        try:
            await bulk_upsert_shot_mattes(pool, shot_mattes)
            inserted_mattes = len(shot_mattes)
        except Exception as bulk_exc:
            logger.warning(
                "bulk_upsert_shot_mattes failed for video %s (%s) — "
                "falling back to per-record upserts.",
                video_id,
                bulk_exc,
            )
            for matte in shot_mattes:
                try:
                    await upsert_shot_matte(pool, matte)
                    inserted_mattes += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to upsert shot matte shot_id=%s (video_id=%s): %s",
                        matte.shot_id,
                        video_id,
                        exc,
                    )

    logger.info(
        "Upserted %d/%d shot matte records for video %s",
        inserted_mattes,
        len(shot_mattes),
        video_id,
    )

    # ------------------------------------------------------------------
    # 6. Insert speaker turns (Stage 0: diarization fused w/ face identity)
    #    — best-effort per record, bulk first, per-record fallback.
    # ------------------------------------------------------------------
    inserted_turns = 0
    if speaker_turns:
        try:
            await bulk_insert_speaker_turns(pool, speaker_turns)
            inserted_turns = len(speaker_turns)
        except Exception as bulk_exc:
            logger.warning(
                "bulk_insert_speaker_turns failed for video %s (%s) — "
                "falling back to per-record inserts.",
                video_id,
                bulk_exc,
            )
            for turn in speaker_turns:
                try:
                    await bulk_insert_speaker_turns(pool, [turn])
                    inserted_turns += 1
                except Exception as exc:
                    logger.warning(
                        "Skipping speaker turn id=%s (video=%s): %s",
                        turn.id,
                        video_id,
                        exc,
                    )
        logger.info(
            "Inserted %d/%d speaker turn records for video %s",
            inserted_turns,
            len(speaker_turns),
            video_id,
        )
    else:
        logger.info("No speaker turns to insert for video %s", video_id)

    logger.info(
        "Level-2 KB write complete for video %s — "
        "persons=%d appearances=%d timeline_events=%d color_grades=%d "
        "shot_mattes=%d speaker_turns=%d",
        video_id,
        len(upserted_person_ids),
        len(appearances),
        len(timeline_events),
        inserted_grades,
        inserted_mattes,
        inserted_turns,
    )

    # ------------------------------------------------------------------
    # 7. B4 observability — pyannote diarization failure signal.
    #
    #    Per CLAUDE.md B8's own framing: "L2 pyannote diarization fails ->
    #    non-fatal -> speaker_turns empty for video". `run_diarization`
    #    (pipeline/level2/diarization_runner.py) returns `[]` on every
    #    failure path (audio extraction, pyannote import, model load,
    #    inference) AND on legitimate "no speech detected" — the two are
    #    not distinguishable from this function's inputs alone (that
    #    would need a new return signal threaded up from
    #    diarization_runner.py through modal_app.py, out of scope for this
    #    pass). We approximate "likely a failure, not just a quiet video"
    #    as: the video clearly has identified people on screen
    #    (`persons` non-empty) but produced zero speaker_turns — a cast
    #    video with visible people almost always has some dialogue.
    # ------------------------------------------------------------------
    if not speaker_turns and persons:
        logger.warning(
            "write_level2_to_kb: video_id=%s has %d persons but zero "
            "speaker_turns — likely pyannote diarization failure "
            "(pipeline_alerts).",
            video_id, len(persons),
        )
        try:
            await insert_pipeline_alert(
                pool, video_id, 2, "l2_pyannote_zero_turns_with_persons",
                0.0, 1.0,
            )
        except Exception as exc:
            logger.warning(
                "write_level2_to_kb: failed to write pipeline_alerts row for "
                "video_id=%s: %s", video_id, exc,
            )
