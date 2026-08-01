from __future__ import annotations

import json
import logging
import uuid

import asyncpg

from shared.types import (
    BackgroundAssignmentRecord,
    ChunkRecord,
    ClientStyleProfileRecord,
    ColorGradeRecord,
    CorrectionEventRecord,
    CutListItemRecord,
    EditPlanRecord,
    EditPlanRevisionRecord,
    EmphasisEffectRecord,
    FaceAppearanceRecord,
    FaceTimelineEvent,
    FrameAnalysisRecord,
    HumanFeedbackRecord,
    KeyframeRecord,
    KGEdge,
    KGNode,
    LayerCompositeRecord,
    LevelStatus,
    PersonRecord,
    ProcessingJob,
    QAReportRecord,
    QwenFrameOutput,
    SceneOverrideRecord,
    SceneRecord,
    SearchableFactRecord,
    SequenceColorAdjustmentRecord,
    ShotMatteRecord,
    ShotRecord,
    SpeakerTurnRecord,
    StockAssetRecord,
    StorylineOverrideRecord,
    StorylineRecord,
    TranscriptSegment,
    VideoMeta,
)

# L7 -- EVALUATION (CLAUDE.md "PIPELINE ADDENDUM 3") additions. Kept as a
# separate import statement (not merged into the block above) so this file
# only ever grows via new lines, never edits to existing ones — merge-safety
# note at the top of this module's task: another implementation effort may
# be adding its own new functions/imports to this same file in parallel.
from shared.types import EvaluationScoreRecord, LLMCallLogRecord  # noqa: E402

logger = logging.getLogger(__name__)


def _vec_to_list(v) -> list[float] | None:
    """Convert a pgvector Vector (or ndarray/list) to list[float].

    pgvector-python returns different types depending on version:
    some return Vector objects, some return numpy arrays directly.
    This helper handles all observed variants without assuming any specific API.
    """
    if v is None:
        return None
    if isinstance(v, list):
        return v
    # numpy array or anything with .tolist()
    if hasattr(v, 'tolist'):
        return v.tolist()
    # pgvector Vector wraps ._value (numpy array)
    if hasattr(v, '_value'):
        inner = v._value
        if hasattr(inner, 'tolist'):
            return inner.tolist()
        return list(inner)
    # direct iteration
    try:
        return list(v)
    except TypeError:
        pass
    # last resort: parse string representation "[0.1,0.2,...]"
    return [float(x) for x in str(v).strip('[]').split(',') if x.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_json(value: dict | list) -> str:
    return json.dumps(value)


def _from_json(value: str | None) -> dict | list:
    if value is None:
        return {}
    return json.loads(value)


def _to_uuid(value: str | None) -> uuid.UUID | None:
    """Convert a string UUID to a :class:`uuid.UUID` object for asyncpg.

    asyncpg requires UUID objects (not strings) when inserting into UUID-typed
    columns.  Returns ``None`` if *value* is ``None`` or an empty string.
    """
    if not value:
        return None
    return uuid.UUID(value)


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


async def upsert_video(pool: asyncpg.Pool, meta: VideoMeta) -> str:
    """Insert or update a video record.  Returns the row ``id``.

    ``client_id`` (migration 016_client_style_profiles.sql) is nullable end
    to end — omitting it (``meta.client_id is None``, the dataclass
    default) leaves it NULL, which is a fully supported "no known client"
    state, not a migration gap.
    """
    row = await pool.fetchrow(
        """
        INSERT INTO videos (id, path, r2_key, duration_s, fps, width, height, client_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (id) DO UPDATE
          SET path       = EXCLUDED.path,
              r2_key     = EXCLUDED.r2_key,
              duration_s = EXCLUDED.duration_s,
              fps        = EXCLUDED.fps,
              width      = EXCLUDED.width,
              height     = EXCLUDED.height,
              client_id  = COALESCE(EXCLUDED.client_id, videos.client_id)
        RETURNING id
        """,
        _to_uuid(meta.id),
        meta.path,
        meta.r2_key,
        meta.duration_s,
        meta.fps,
        meta.width,
        meta.height,
        meta.client_id,
    )
    return str(row["id"])


async def get_video(pool: asyncpg.Pool, video_id: str) -> VideoMeta | None:
    """Return the `videos` row for *video_id*, or None if not found.

    Used by Level-6's Editing Director (`export_xml`/`render_direct`) to
    resolve `fps` and the source file reference (`r2_key` or `path`) needed
    to build an FCPXML asset or drive the FFmpeg render, and by L5 Pass B /
    L6's Color Grading + Caption agents to resolve `client_id` for an
    optional `client_style_profiles` soft-prior lookup (CLAUDE.md
    "PIPELINE ADDENDUM 2" -> "3. Client Style Profiles").
    """
    row = await pool.fetchrow(
        "SELECT * FROM videos WHERE id = $1",
        _to_uuid(video_id),
    )
    if row is None:
        return None
    return VideoMeta(
        id=str(row["id"]),
        path=row["path"],
        r2_key=row["r2_key"],
        duration_s=row["duration_s"],
        fps=row["fps"],
        width=row["width"],
        height=row["height"],
        client_id=row["client_id"] if "client_id" in row else None,
    )


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


async def upsert_chunk(pool: asyncpg.Pool, chunk: ChunkRecord) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO chunks (id, video_id, chunk_index, start_frame, end_frame, start_time, end_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (video_id, chunk_index) DO UPDATE
          SET start_frame = EXCLUDED.start_frame,
              end_frame   = EXCLUDED.end_frame,
              start_time  = EXCLUDED.start_time,
              end_time    = EXCLUDED.end_time
        RETURNING id
        """,
        _to_uuid(chunk.id),
        _to_uuid(chunk.video_id),
        chunk.chunk_index,
        chunk.start_frame,
        chunk.end_frame,
        chunk.start_time,
        chunk.end_time,
    )
    return str(row["id"])


async def bulk_insert_chunks(pool: asyncpg.Pool, chunks: list[ChunkRecord]) -> None:
    """Bulk-upsert chunks using a single executemany call."""
    if not chunks:
        return
    records = [
        (
            _to_uuid(c.id),
            _to_uuid(c.video_id),
            c.chunk_index,
            c.start_frame,
            c.end_frame,
            c.start_time,
            c.end_time,
        )
        for c in chunks
    ]
    await pool.executemany(
        """
        INSERT INTO chunks (id, video_id, chunk_index, start_frame, end_frame, start_time, end_time)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (video_id, chunk_index) DO UPDATE
          SET start_frame = EXCLUDED.start_frame,
              end_frame   = EXCLUDED.end_frame,
              start_time  = EXCLUDED.start_time,
              end_time    = EXCLUDED.end_time
        """,
        records,
    )


async def get_chunks_for_video(pool: asyncpg.Pool, video_id: str) -> list[ChunkRecord]:
    rows = await pool.fetch(
        "SELECT * FROM chunks WHERE video_id = $1 ORDER BY chunk_index",
        _to_uuid(video_id),
    )
    return [
        ChunkRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            chunk_index=r["chunk_index"],
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            start_time=r["start_time"],
            end_time=r["end_time"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Shots
# ---------------------------------------------------------------------------


async def upsert_shot(pool: asyncpg.Pool, shot: ShotRecord) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO shots (id, chunk_id, video_id, shot_index, start_frame, end_frame,
                           start_time, end_time, shot_type, complexity)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (video_id, shot_index) DO UPDATE
          SET chunk_id    = EXCLUDED.chunk_id,
              start_frame = EXCLUDED.start_frame,
              end_frame   = EXCLUDED.end_frame,
              start_time  = EXCLUDED.start_time,
              end_time    = EXCLUDED.end_time,
              shot_type   = EXCLUDED.shot_type,
              complexity  = EXCLUDED.complexity
        RETURNING id
        """,
        _to_uuid(shot.id),
        _to_uuid(shot.chunk_id),
        _to_uuid(shot.video_id),
        shot.shot_index,
        shot.start_frame,
        shot.end_frame,
        shot.start_time,
        shot.end_time,
        shot.shot_type,
        shot.complexity,
    )
    return str(row["id"])


async def bulk_insert_shots(pool: asyncpg.Pool, shots: list[ShotRecord]) -> None:
    """Bulk-upsert shots using a single executemany call."""
    if not shots:
        return
    records = [
        (
            _to_uuid(s.id),
            _to_uuid(s.chunk_id),
            _to_uuid(s.video_id),
            s.shot_index,
            s.start_frame,
            s.end_frame,
            s.start_time,
            s.end_time,
            s.shot_type,
            s.complexity,
        )
        for s in shots
    ]
    await pool.executemany(
        """
        INSERT INTO shots (id, chunk_id, video_id, shot_index, start_frame, end_frame,
                           start_time, end_time, shot_type, complexity)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (video_id, shot_index) DO UPDATE
          SET chunk_id    = EXCLUDED.chunk_id,
              start_frame = EXCLUDED.start_frame,
              end_frame   = EXCLUDED.end_frame,
              start_time  = EXCLUDED.start_time,
              end_time    = EXCLUDED.end_time,
              shot_type   = EXCLUDED.shot_type,
              complexity  = EXCLUDED.complexity
        """,
        records,
    )


async def get_shots_for_video(pool: asyncpg.Pool, video_id: str) -> list[ShotRecord]:
    rows = await pool.fetch(
        "SELECT * FROM shots WHERE video_id = $1 ORDER BY shot_index",
        _to_uuid(video_id),
    )
    return [
        ShotRecord(
            id=str(r["id"]),
            chunk_id=str(r["chunk_id"]),
            video_id=str(r["video_id"]),
            shot_index=r["shot_index"],
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            shot_type=r["shot_type"],
            complexity=r["complexity"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------


async def upsert_keyframe(pool: asyncpg.Pool, kf: KeyframeRecord) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO keyframes (id, shot_id, video_id, frame_index, timestamp_s,
                               r2_key, selection_reason, dino_embedding, siglip_embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (video_id, frame_index) DO UPDATE
          SET timestamp_s      = EXCLUDED.timestamp_s,
              r2_key           = EXCLUDED.r2_key,
              selection_reason = EXCLUDED.selection_reason,
              dino_embedding   = EXCLUDED.dino_embedding,
              siglip_embedding = EXCLUDED.siglip_embedding
        RETURNING id
        """,
        _to_uuid(kf.id),
        _to_uuid(kf.shot_id),
        _to_uuid(kf.video_id),
        kf.frame_index,
        kf.timestamp_s,
        kf.r2_key,
        kf.selection_reason,
        kf.dino_embedding,
        kf.siglip_embedding,
    )
    return str(row["id"])


async def bulk_insert_keyframes(
    pool: asyncpg.Pool, keyframes: list[KeyframeRecord]
) -> None:
    """Bulk-upsert keyframes via executemany — far faster than one-by-one for 1700+ frames."""
    if not keyframes:
        return
    records = [
        (
            _to_uuid(kf.id),
            _to_uuid(kf.shot_id),
            _to_uuid(kf.video_id),
            kf.frame_index,
            kf.timestamp_s,
            kf.r2_key,
            kf.selection_reason,
            kf.dino_embedding,
            kf.siglip_embedding,
        )
        for kf in keyframes
    ]
    await pool.executemany(
        """
        INSERT INTO keyframes (id, shot_id, video_id, frame_index, timestamp_s,
                               r2_key, selection_reason, dino_embedding, siglip_embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (video_id, frame_index) DO UPDATE
          SET timestamp_s      = EXCLUDED.timestamp_s,
              r2_key           = EXCLUDED.r2_key,
              selection_reason = EXCLUDED.selection_reason,
              dino_embedding   = EXCLUDED.dino_embedding,
              siglip_embedding = EXCLUDED.siglip_embedding
        """,
        records,
    )


async def get_keyframes_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[KeyframeRecord]:
    rows = await pool.fetch(
        "SELECT * FROM keyframes WHERE video_id = $1 ORDER BY timestamp_s",
        _to_uuid(video_id),
    )
    return [
        KeyframeRecord(
            id=str(r["id"]),
            shot_id=str(r["shot_id"]),
            video_id=str(r["video_id"]),
            frame_index=r["frame_index"],
            timestamp_s=r["timestamp_s"],
            r2_key=r["r2_key"],
            selection_reason=r["selection_reason"],
            dino_embedding=_vec_to_list(r["dino_embedding"]),
            siglip_embedding=_vec_to_list(r["siglip_embedding"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Transcript segments
# ---------------------------------------------------------------------------


async def bulk_insert_transcript_segments(
    pool: asyncpg.Pool, segs: list[TranscriptSegment]
) -> None:
    """Insert transcript segments in bulk.  Uses executemany for efficiency."""
    if not segs:
        return
    records = [
        (
            _to_uuid(seg.id),
            _to_uuid(seg.video_id),
            _to_uuid(seg.chunk_id),
            seg.text,
            seg.start_time,
            seg.end_time,
            seg.confidence,
            _to_json(seg.words),
        )
        for seg in segs
    ]
    await pool.executemany(
        """
        INSERT INTO transcript_segments
            (id, video_id, chunk_id, text, start_time, end_time, confidence, words)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
        ON CONFLICT (video_id, start_time, end_time) DO UPDATE
          SET text       = EXCLUDED.text,
              confidence = EXCLUDED.confidence,
              words      = EXCLUDED.words,
              chunk_id   = EXCLUDED.chunk_id
        """,
        records,
    )


async def get_transcript_segments_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[TranscriptSegment]:
    rows = await pool.fetch(
        "SELECT * FROM transcript_segments WHERE video_id = $1 ORDER BY start_time",
        _to_uuid(video_id),
    )
    return [
        TranscriptSegment(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            chunk_id=str(r["chunk_id"]) if r["chunk_id"] else None,
            text=r["text"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            confidence=r["confidence"],
            words=json.loads(r["words"]) if r["words"] else [],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Persons
# ---------------------------------------------------------------------------


async def upsert_person(pool: asyncpg.Pool, person: PersonRecord) -> str:
    """Upsert a person record, conflicting on (video_id, pid)."""
    row = await pool.fetchrow(
        """
        INSERT INTO persons (id, video_id, pid, display_name, arcface_embedding)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (video_id, pid) DO UPDATE
          SET display_name      = EXCLUDED.display_name,
              arcface_embedding = EXCLUDED.arcface_embedding
        RETURNING id
        """,
        _to_uuid(person.id),
        _to_uuid(person.video_id),
        person.pid,
        person.display_name,
        person.arcface_embedding,
    )
    return str(row["id"])


async def get_persons_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[PersonRecord]:
    rows = await pool.fetch(
        "SELECT * FROM persons WHERE video_id = $1",
        _to_uuid(video_id),
    )
    return [
        PersonRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            pid=r["pid"],
            display_name=r["display_name"],
            arcface_embedding=_vec_to_list(r["arcface_embedding"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Face appearances
# ---------------------------------------------------------------------------


async def bulk_insert_face_appearances(
    pool: asyncpg.Pool, apps: list[FaceAppearanceRecord]
) -> None:
    if not apps:
        return
    records = [
        (
            _to_uuid(app.id),
            _to_uuid(app.video_id),
            app.frame_index,
            app.timestamp_s,
            _to_uuid(app.person_id),
            app.track_id,
            _to_json(app.bbox) if app.bbox is not None else None,
            app.emotion,
            app.emotion_conf,
        )
        for app in apps
    ]
    await pool.executemany(
        """
        INSERT INTO face_appearances
            (id, video_id, frame_index, timestamp_s, person_id, track_id, bbox, emotion, emotion_conf)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
        ON CONFLICT (id) DO NOTHING
        """,
        records,
    )


async def get_face_appearances_for_video_sampled(
    pool: asyncpg.Pool, video_id: str
) -> list[FaceAppearanceRecord]:
    """Return all face appearance records for a video.

    The face_appearances table is already sampled at approximately 1 fps during
    Level 2 processing (one row per second of video per detected person), so
    returning the full table for a video is safe and does not require additional
    down-sampling here.

    Args:
        pool:     An open asyncpg connection pool.
        video_id: The video's unique identifier.

    Returns:
        All FaceAppearanceRecord rows for the video, ordered by timestamp_s.
    """
    rows = await pool.fetch(
        "SELECT * FROM face_appearances WHERE video_id = $1 ORDER BY timestamp_s",
        _to_uuid(video_id),
    )
    return [
        FaceAppearanceRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            frame_index=r["frame_index"],
            timestamp_s=r["timestamp_s"],
            person_id=str(r["person_id"]) if r["person_id"] else None,
            track_id=r["track_id"],
            bbox=json.loads(r["bbox"]) if r["bbox"] else None,
            emotion=r["emotion"],
            emotion_conf=r["emotion_conf"],
        )
        for r in rows
    ]


async def get_face_appearances_for_frame(
    pool: asyncpg.Pool, video_id: str, frame_index: int
) -> list[FaceAppearanceRecord]:
    rows = await pool.fetch(
        "SELECT * FROM face_appearances WHERE video_id = $1 AND frame_index = $2",
        _to_uuid(video_id),
        frame_index,
    )
    return [
        FaceAppearanceRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            frame_index=r["frame_index"],
            timestamp_s=r["timestamp_s"],
            person_id=str(r["person_id"]) if r["person_id"] else None,
            track_id=r["track_id"],
            bbox=json.loads(r["bbox"]) if r["bbox"] else None,
            emotion=r["emotion"],
            emotion_conf=r["emotion_conf"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Face timeline events
# ---------------------------------------------------------------------------


async def bulk_insert_face_timeline_events(
    pool: asyncpg.Pool, events: list[FaceTimelineEvent]
) -> None:
    if not events:
        return
    records = [
        (
            _to_uuid(ev.id),
            _to_uuid(ev.video_id),
            _to_uuid(ev.person_id),
            ev.emotion,
            ev.start_time,
            ev.end_time,
            ev.confidence,
        )
        for ev in events
    ]
    await pool.executemany(
        """
        INSERT INTO face_timeline_events
            (id, video_id, person_id, emotion, start_time, end_time, confidence)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id) DO NOTHING
        """,
        records,
    )


# ---------------------------------------------------------------------------
# Speaker turns (Level-2 Stage 0: diarization fused with ArcFace identity)
# ---------------------------------------------------------------------------


async def bulk_insert_speaker_turns(
    pool: asyncpg.Pool, turns: list[SpeakerTurnRecord]
) -> None:
    if not turns:
        return
    records = [
        (
            _to_uuid(t.id),
            _to_uuid(t.video_id),
            t.cluster_label,
            _to_uuid(t.person_id),
            t.start_time,
            t.end_time,
            t.confidence,
            t.resolution_method,
        )
        for t in turns
    ]
    await pool.executemany(
        """
        INSERT INTO speaker_turns
            (id, video_id, cluster_label, person_id, start_time, end_time, confidence, resolution_method)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (id) DO NOTHING
        """,
        records,
    )


async def get_speaker_turns_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[SpeakerTurnRecord]:
    rows = await pool.fetch(
        "SELECT * FROM speaker_turns WHERE video_id = $1 ORDER BY start_time",
        _to_uuid(video_id),
    )
    return [
        SpeakerTurnRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            cluster_label=r["cluster_label"],
            person_id=str(r["person_id"]) if r["person_id"] is not None else None,
            start_time=r["start_time"],
            end_time=r["end_time"],
            confidence=r["confidence"],
            resolution_method=r["resolution_method"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Color grades
# ---------------------------------------------------------------------------


async def upsert_color_grade(pool: asyncpg.Pool, grade: ColorGradeRecord) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO color_grades
            (id, video_id, shot_id, frame_index, timestamp_s, parameters, style_tags)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        ON CONFLICT (id) DO UPDATE
          SET parameters = EXCLUDED.parameters,
              style_tags = EXCLUDED.style_tags
        RETURNING id
        """,
        _to_uuid(grade.id),
        _to_uuid(grade.video_id),
        _to_uuid(grade.shot_id),
        grade.frame_index,
        grade.timestamp_s,
        _to_json(grade.parameters),
        grade.style_tags,
    )
    return str(row["id"])


async def bulk_upsert_color_grades(
    pool: asyncpg.Pool, grades: list[ColorGradeRecord]
) -> None:
    """Upsert all color grade records in a single executemany call."""
    if not grades:
        return
    records = [
        (
            _to_uuid(g.id),
            _to_uuid(g.video_id),
            _to_uuid(g.shot_id),
            g.frame_index,
            g.timestamp_s,
            _to_json(g.parameters),
            g.style_tags,
        )
        for g in grades
    ]
    await pool.executemany(
        """
        INSERT INTO color_grades
            (id, video_id, shot_id, frame_index, timestamp_s, parameters, style_tags)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        ON CONFLICT (id) DO UPDATE
          SET parameters = EXCLUDED.parameters,
              style_tags = EXCLUDED.style_tags
        """,
        records,
    )


async def get_color_grades_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[ColorGradeRecord]:
    """Return all color_grades rows for a video, ordered by timestamp.

    Used by the Story Architect Agent's deterministic `usability_score`
    rollup (technical-quality half of the formula — see
    `pipeline.level4.story_architect_runner`).
    """
    rows = await pool.fetch(
        "SELECT * FROM color_grades WHERE video_id = $1 ORDER BY timestamp_s",
        _to_uuid(video_id),
    )
    return [
        ColorGradeRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            shot_id=str(r["shot_id"]) if r["shot_id"] else None,
            frame_index=r["frame_index"],
            timestamp_s=r["timestamp_s"],
            parameters=json.loads(r["parameters"]) if isinstance(r["parameters"], str) else (r["parameters"] or {}),
            style_tags=list(r["style_tags"] or []),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Shot mattes (Level-2, A1 — background matting)
# ---------------------------------------------------------------------------


async def upsert_shot_matte(pool: asyncpg.Pool, matte: ShotMatteRecord) -> str:
    """Upsert one ``shot_mattes`` row.

    Conflicts on ``(shot_id, model_version)`` — not ``id`` — so re-running
    matting for the same shot with the same model version is a true UPSERT
    (rule 9: idempotent) rather than accumulating a new row per run.
    """
    row = await pool.fetchrow(
        """
        INSERT INTO shot_mattes
            (id, shot_id, video_id, r2_key, model_version)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (shot_id, model_version) DO UPDATE
          SET r2_key = EXCLUDED.r2_key
        RETURNING id
        """,
        _to_uuid(matte.id),
        _to_uuid(matte.shot_id),
        _to_uuid(matte.video_id),
        matte.r2_key,
        matte.model_version,
    )
    return str(row["id"])


async def bulk_upsert_shot_mattes(
    pool: asyncpg.Pool, mattes: list[ShotMatteRecord]
) -> None:
    """Upsert all shot matte records in a single executemany call."""
    if not mattes:
        return
    records = [
        (
            _to_uuid(m.id),
            _to_uuid(m.shot_id),
            _to_uuid(m.video_id),
            m.r2_key,
            m.model_version,
        )
        for m in mattes
    ]
    await pool.executemany(
        """
        INSERT INTO shot_mattes
            (id, shot_id, video_id, r2_key, model_version)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (shot_id, model_version) DO UPDATE
          SET r2_key = EXCLUDED.r2_key
        """,
        records,
    )


async def get_shot_mattes_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[ShotMatteRecord]:
    """Return all shot_mattes rows for a video."""
    rows = await pool.fetch(
        "SELECT * FROM shot_mattes WHERE video_id = $1",
        _to_uuid(video_id),
    )
    return [
        ShotMatteRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            shot_id=str(r["shot_id"]),
            r2_key=r["r2_key"],
            model_version=r["model_version"],
        )
        for r in rows
    ]


async def get_shot_mattes_for_shots(
    pool: asyncpg.Pool, shot_ids: list[str]
) -> dict[str, ShotMatteRecord]:
    """Return {shot_id: ShotMatteRecord} for the given *shot_ids*.

    Used by A4's `materialize_layer_composite_ops` (Level-6 Editing
    Director) to look up the matting artifact for whichever shot a
    `background_swap` cut_list_item's clip falls inside — targeted lookup
    by id list rather than pulling every shot_matte for the whole video.
    When a shot has multiple `model_version` rows (re-matted with a newer
    model), the most recently inserted row for that shot wins (`ORDER BY
    id DESC` — `id` is a nanoid, not time-sortable, so this is a
    best-effort "last write wins" tiebreak, not a true recency guarantee;
    acceptable here since `bulk_upsert_shot_mattes` already UPSERTs on
    `(shot_id, model_version)`, so duplicates per shot only occur across
    genuinely different model versions).
    """
    if not shot_ids:
        return {}
    uuids = [_to_uuid(sid) for sid in shot_ids]
    rows = await pool.fetch(
        "SELECT * FROM shot_mattes WHERE shot_id = ANY($1::uuid[])",
        uuids,
    )
    result: dict[str, ShotMatteRecord] = {}
    for r in rows:
        shot_id = str(r["shot_id"])
        if shot_id in result:
            continue  # keep first seen — see docstring's tiebreak note
        result[shot_id] = ShotMatteRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            shot_id=shot_id,
            r2_key=r["r2_key"],
            model_version=r["model_version"],
        )
    return result


# ---------------------------------------------------------------------------
# Frame analyses
# ---------------------------------------------------------------------------


async def upsert_frame_analysis(
    pool: asyncpg.Pool, analysis: FrameAnalysisRecord
) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO frame_analyses
            (id, keyframe_id, video_id, scene_id, scene_change,
             qwen_output, caption, beat_type, scene_mood, tension_level, tags)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
        ON CONFLICT (keyframe_id) DO UPDATE
          SET scene_id      = EXCLUDED.scene_id,
              scene_change  = EXCLUDED.scene_change,
              qwen_output   = EXCLUDED.qwen_output,
              caption       = EXCLUDED.caption,
              beat_type     = EXCLUDED.beat_type,
              scene_mood    = EXCLUDED.scene_mood,
              tension_level = EXCLUDED.tension_level,
              tags          = EXCLUDED.tags
        RETURNING id
        """,
        _to_uuid(analysis.id),
        _to_uuid(analysis.keyframe_id),
        _to_uuid(analysis.video_id),
        analysis.scene_id,
        analysis.scene_change,
        _to_json(analysis.qwen_output),
        analysis.caption,
        analysis.beat_type,
        analysis.scene_mood,
        analysis.tension_level,
        analysis.tags,
    )
    return str(row["id"])


async def bulk_insert_frame_analyses(
    pool: asyncpg.Pool, analyses: list[FrameAnalysisRecord]
) -> None:
    """Bulk-upsert frame analyses — far faster than one-by-one for 1700+ frames."""
    if not analyses:
        return
    records = [
        (
            _to_uuid(a.id),
            _to_uuid(a.keyframe_id),
            _to_uuid(a.video_id),
            a.scene_id,
            a.scene_change,
            _to_json(a.qwen_output),
            a.caption,
            a.beat_type,
            a.scene_mood,
            a.tension_level,
            a.tags,
        )
        for a in analyses
    ]
    await pool.executemany(
        """
        INSERT INTO frame_analyses
            (id, keyframe_id, video_id, scene_id, scene_change,
             qwen_output, caption, beat_type, scene_mood, tension_level, tags)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
        ON CONFLICT (keyframe_id) DO UPDATE
          SET scene_id      = EXCLUDED.scene_id,
              scene_change  = EXCLUDED.scene_change,
              qwen_output   = EXCLUDED.qwen_output,
              caption       = EXCLUDED.caption,
              beat_type     = EXCLUDED.beat_type,
              scene_mood    = EXCLUDED.scene_mood,
              tension_level = EXCLUDED.tension_level,
              tags          = EXCLUDED.tags
        """,
        records,
    )


async def bulk_upsert_kg_nodes(
    pool: asyncpg.Pool, nodes: list[KGNode]
) -> dict[tuple[str, str], str]:
    """Bulk-upsert KG nodes and return {(node_type, ref_id): db_id} map.

    Uses a fetch-per-batch approach: inserts all nodes, then queries back the
    resulting IDs by (video_id, node_type, ref_id) — avoiding 17K round-trips.
    """
    if not nodes:
        return {}

    video_id = nodes[0].video_id

    records = [
        (
            _to_uuid(node.id),
            _to_uuid(node.video_id),
            node.node_type,
            node.ref_id,
            node.label,
            _to_json(node.properties),
            node.embedding,
        )
        for node in nodes
    ]
    await pool.executemany(
        """
        INSERT INTO kg_nodes
            (id, video_id, node_type, ref_id, label, properties, embedding)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        ON CONFLICT (video_id, node_type, ref_id) DO UPDATE
          SET label      = EXCLUDED.label,
              properties = EXCLUDED.properties,
              embedding  = EXCLUDED.embedding
        """,
        records,
    )

    rows = await pool.fetch(
        "SELECT id, node_type, ref_id FROM kg_nodes WHERE video_id = $1",
        _to_uuid(video_id),
    )
    return {(r["node_type"], r["ref_id"]): str(r["id"]) for r in rows}


# ---------------------------------------------------------------------------
# Searchable facts
# ---------------------------------------------------------------------------


async def bulk_insert_searchable_facts(
    pool: asyncpg.Pool, facts: list[SearchableFactRecord]
) -> None:
    if not facts:
        return
    records = [
        (
            _to_uuid(fact.id),
            _to_uuid(fact.video_id),
            _to_uuid(fact.frame_id),
            fact.fact_text,
            fact.timestamp_s,
            fact.embedding,
            fact.legacy_vector_id,
        )
        for fact in facts
    ]
    await pool.executemany(
        """
        INSERT INTO searchable_facts
            (id, video_id, frame_id, fact_text, timestamp_s, embedding, legacy_vector_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (video_id, fact_text) DO UPDATE
          SET timestamp_s = EXCLUDED.timestamp_s,
              embedding   = EXCLUDED.embedding,
              legacy_vector_id = EXCLUDED.legacy_vector_id
        """,
        records,
    )


async def get_existing_frame_analyses(
    pool: asyncpg.Pool, keyframe_ids: list[str]
) -> dict[str, QwenFrameOutput]:
    """Return {keyframe_id: QwenFrameOutput} for already-analysed keyframes.

    Used to skip Qwen re-inference when frame_analyses already exist in DB.
    Keyframes with unparseable qwen_output are silently excluded.
    """
    if not keyframe_ids:
        return {}
    uuids = [_to_uuid(kid) for kid in keyframe_ids]
    rows = await pool.fetch(
        "SELECT keyframe_id, qwen_output FROM frame_analyses WHERE keyframe_id = ANY($1::uuid[])",
        uuids,
    )
    result: dict[str, QwenFrameOutput] = {}
    for row in rows:
        try:
            raw = row["qwen_output"]
            data = json.loads(raw) if isinstance(raw, str) else raw
            output = QwenFrameOutput.model_validate(data)
            result[str(row["keyframe_id"])] = output
        except Exception as exc:
            logger.warning("Could not parse stored qwen_output for keyframe %s: %s", row["keyframe_id"], exc)
    return result


async def update_searchable_fact_embedding(
    pool: asyncpg.Pool,
    fact_id: str,
    embedding: list[float],
    legacy_vector_id: str | None = None,
) -> None:
    """Update the embedding for a searchable fact.

    Called after embeddings are (re)computed so the vector is persisted to
    Postgres — the sole store for fact search since B7 dropped Pinecone.

    Args:
        pool:             An open asyncpg connection pool.
        fact_id:          The fact's unique identifier (nanoid string).
        embedding:        The 1024-dim float vector to persist.
        legacy_vector_id: Optional legacy id (was the Pinecone vector id);
            kept only for historical-row bookkeeping, no functional use.
    """
    await pool.execute(
        """
        UPDATE searchable_facts
        SET embedding = $2,
            legacy_vector_id = COALESCE($3, legacy_vector_id)
        WHERE id = $1
        """,
        _to_uuid(fact_id),
        embedding,
        legacy_vector_id,
    )


# ---------------------------------------------------------------------------
# Knowledge graph nodes
# ---------------------------------------------------------------------------


async def upsert_kg_node(pool: asyncpg.Pool, node: KGNode) -> str:
    """Upsert a KG node, conflicting on (video_id, node_type, ref_id)."""
    row = await pool.fetchrow(
        """
        INSERT INTO kg_nodes
            (id, video_id, node_type, ref_id, label, properties, embedding)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        ON CONFLICT (video_id, node_type, ref_id) DO UPDATE
          SET label      = EXCLUDED.label,
              properties = EXCLUDED.properties,
              embedding  = EXCLUDED.embedding
        RETURNING id
        """,
        _to_uuid(node.id),
        _to_uuid(node.video_id),
        node.node_type,
        node.ref_id,
        node.label,
        _to_json(node.properties),
        node.embedding,
    )
    return str(row["id"])


# ---------------------------------------------------------------------------
# Knowledge graph edges
# ---------------------------------------------------------------------------


async def bulk_insert_kg_edges(pool: asyncpg.Pool, edges: list[KGEdge]) -> None:
    if not edges:
        return
    records = [
        (
            _to_uuid(edge.id),
            _to_uuid(edge.video_id),
            _to_uuid(edge.source_id),
            _to_uuid(edge.target_id),
            edge.relation,
            edge.weight,
            _to_json(edge.properties),
        )
        for edge in edges
    ]
    await pool.executemany(
        """
        INSERT INTO kg_edges
            (id, video_id, source_id, target_id, relation, weight, properties)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        ON CONFLICT (id) DO NOTHING
        """,
        records,
    )


# ---------------------------------------------------------------------------
# Processing jobs
# ---------------------------------------------------------------------------


async def upsert_job(pool: asyncpg.Pool, job: ProcessingJob) -> str:
    """Upsert a processing job, conflicting on (video_id, level)."""
    row = await pool.fetchrow(
        """
        INSERT INTO processing_jobs
            (id, video_id, level, status, error_msg, meta)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (video_id, level) DO UPDATE
          SET status    = EXCLUDED.status,
              error_msg = EXCLUDED.error_msg,
              meta      = EXCLUDED.meta
        RETURNING id
        """,
        _to_uuid(job.id),
        _to_uuid(job.video_id),
        job.level,
        job.status.value,
        job.error_msg,
        _to_json(job.meta),
    )
    return str(row["id"])


async def update_job_status(
    pool: asyncpg.Pool,
    video_id: str,
    level: int,
    status: LevelStatus,
    error_msg: str | None = None,
) -> None:
    """Update the status (and optionally error_msg) of a job identified by
    (video_id, level).  Sets ``started_at`` when transitioning to RUNNING and
    ``completed_at`` when transitioning to DONE or FAILED.
    """
    await pool.execute(
        """
        UPDATE processing_jobs
        SET status       = $3,
            error_msg    = $4,
            started_at   = CASE WHEN $3 = 'running'              THEN NOW() ELSE started_at   END,
            completed_at = CASE WHEN $3 IN ('done', 'failed')    THEN NOW() ELSE completed_at END
        WHERE video_id = $1
          AND level    = $2
        """,
        _to_uuid(video_id),
        level,
        status.value,
        error_msg,
    )


async def update_job_meta(
    pool: asyncpg.Pool,
    video_id: str,
    level: int,
    meta: dict,
) -> None:
    """Merge *meta* into the existing processing_jobs.meta JSONB for (video_id, level).

    Uses PostgreSQL ``||`` operator so existing keys are preserved and new keys
    are added / overwritten.
    """
    await pool.execute(
        """
        UPDATE processing_jobs
        SET meta = COALESCE(meta, '{}'::jsonb) || $3::jsonb
        WHERE video_id = $1
          AND level    = $2
        """,
        _to_uuid(video_id),
        level,
        _to_json(meta),
    )


async def get_level_status(
    pool: asyncpg.Pool, video_id: str, level: int
) -> LevelStatus | None:
    """Return the current status of a processing job, or None if not found."""
    row = await pool.fetchrow(
        "SELECT status FROM processing_jobs WHERE video_id = $1 AND level = $2",
        _to_uuid(video_id),
        level,
    )
    if row is None:
        return None
    return LevelStatus(row["status"])


# ---------------------------------------------------------------------------
# Knowledge graph node/edge queries (Postgres structured cache)
# For graph traversal queries use knowledge_base.neo4j.graph_writer.query_graph
# ---------------------------------------------------------------------------


async def get_kg_nodes_for_video(pool: asyncpg.Pool, video_id: str) -> list[KGNode]:
    """Get all KG nodes for a video from the Postgres structured cache.

    Use this for structured filtering (e.g. WHERE node_type='Person').
    For graph traversal queries (multi-hop, path finding) use Neo4j directly.

    Args:
        pool:     An open asyncpg connection pool.
        video_id: The video's unique identifier.

    Returns:
        All KGNode rows for the video, ordered by node_type and label.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM kg_nodes WHERE video_id=$1 ORDER BY node_type, label",
            _to_uuid(video_id),
        )
        return [
            KGNode(
                id=str(row["id"]),
                video_id=str(row["video_id"]),
                node_type=row["node_type"],
                ref_id=row["ref_id"],
                label=row["label"] or "",
                properties=json.loads(row["properties"]) if row["properties"] else {},
                embedding=_vec_to_list(row["embedding"]),
            )
            for row in rows
        ]


async def get_unresolved_speaker_turns(
    pool: asyncpg.Pool, video_id: str
) -> list[SpeakerTurnRecord]:
    """Speaker turns not yet reasoned about by the Grounding Agent.

    Only 'unresolved' and 'face_majority' rows — never 'single_candidate' or
    'llm_tiebreak' (L4 never downgrades L2 certainty, rule 11/12).
    """
    rows = await pool.fetch(
        """
        SELECT * FROM speaker_turns
        WHERE video_id = $1
          AND resolution_method IN ('unresolved', 'face_majority')
        ORDER BY start_time
        """,
        _to_uuid(video_id),
    )
    return [
        SpeakerTurnRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            cluster_label=r["cluster_label"],
            person_id=str(r["person_id"]) if r["person_id"] is not None else None,
            start_time=r["start_time"],
            end_time=r["end_time"],
            confidence=r["confidence"],
            resolution_method=r["resolution_method"],
        )
        for r in rows
    ]


async def bulk_update_speaker_turn_resolutions(
    pool: asyncpg.Pool,
    updates: list[tuple[str, str | None, float | None, str]],
) -> None:
    """Write back Grounding Agent 1a resolutions.

    Each tuple is (turn_id, person_id, confidence, resolution_method).
    person_id is the DB persons.id UUID (already mapped from `pid` by the
    caller) or None for llm_unresolved_final. Never touches rows outside
    this explicit id list (idempotency is enforced by the caller only
    selecting unresolved/face_majority rows in the first place).
    """
    if not updates:
        return
    records = [
        (_to_uuid(turn_id), _to_uuid(person_id), confidence, resolution_method)
        for turn_id, person_id, confidence, resolution_method in updates
    ]
    await pool.executemany(
        """
        UPDATE speaker_turns
        SET person_id = $2,
            confidence = $3,
            resolution_method = $4
        WHERE id = $1
        """,
        records,
    )


# ---------------------------------------------------------------------------
# Frame analyses joined with keyframe timestamps (Level-4 context building)
# ---------------------------------------------------------------------------


async def get_frame_analyses_with_timestamps_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[dict]:
    """Return [{keyframe_id, timestamp_s, qwen_output}, ...] for a video.

    Used by the Grounding Agent to pull Qwen `people[]` entries that fall
    inside a speaker turn's time window (visual_context) without touching
    raw frame_analyses anywhere else in L4.
    """
    rows = await pool.fetch(
        """
        SELECT k.id AS keyframe_id, k.timestamp_s, fa.qwen_output
        FROM frame_analyses fa
        JOIN keyframes k ON k.id = fa.keyframe_id
        WHERE fa.video_id = $1
        ORDER BY k.timestamp_s
        """,
        _to_uuid(video_id),
    )
    result = []
    for r in rows:
        raw = r["qwen_output"]
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            data = {}
        result.append(
            {
                "keyframe_id": str(r["keyframe_id"]),
                "timestamp_s": r["timestamp_s"],
                "qwen_output": data or {},
            }
        )
    return result


# ---------------------------------------------------------------------------
# Relation canonicalization (Level-4 Grounding Agent 1b)
# ---------------------------------------------------------------------------


async def get_distinct_kg_edge_relations(
    pool: asyncpg.Pool, video_id: str
) -> list[str]:
    rows = await pool.fetch(
        "SELECT DISTINCT relation FROM kg_edges WHERE video_id = $1 AND relation IS NOT NULL",
        _to_uuid(video_id),
    )
    return [r["relation"] for r in rows]


async def get_ontology_relations(pool: asyncpg.Pool, version: int) -> list[str]:
    rows = await pool.fetch(
        "SELECT relation FROM ontology_relations WHERE version = $1 ORDER BY relation",
        version,
    )
    return [r["relation"] for r in rows]


async def bulk_update_kg_edges_canonical_relation(
    pool: asyncpg.Pool,
    video_id: str,
    relation_to_canonical: dict[str, str],
) -> None:
    """Set kg_edges.canonical_relation for every edge whose raw `relation`
    matches a key in *relation_to_canonical*. Raw `relation` is left
    untouched for audit (per CLAUDE.md 1b write-back rule)."""
    if not relation_to_canonical:
        return
    records = [
        (_to_uuid(video_id), raw, canonical)
        for raw, canonical in relation_to_canonical.items()
    ]
    await pool.executemany(
        """
        UPDATE kg_edges
        SET canonical_relation = $3
        WHERE video_id = $1 AND relation = $2
        """,
        records,
    )


# ---------------------------------------------------------------------------
# Fact dedup (Level-4 Grounding Agent 1c)
# ---------------------------------------------------------------------------


async def get_searchable_facts_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[SearchableFactRecord]:
    """Return all searchable_facts rows for a video that have an embedding
    and are not already marked superseded — the fact-dedup pass only needs
    to consider active (non-superseded) facts."""
    rows = await pool.fetch(
        """
        SELECT * FROM searchable_facts
        WHERE video_id = $1 AND embedding IS NOT NULL AND superseded_by IS NULL
        ORDER BY timestamp_s
        """,
        _to_uuid(video_id),
    )
    return [
        SearchableFactRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            fact_text=r["fact_text"],
            frame_id=str(r["frame_id"]) if r["frame_id"] else None,
            timestamp_s=r["timestamp_s"],
            embedding=_vec_to_list(r["embedding"]),
            legacy_vector_id=r["legacy_vector_id"],
        )
        for r in rows
    ]


async def search_searchable_facts_by_embedding(
    pool: asyncpg.Pool,
    embedding: list[float],
    limit: int = 10,
    video_id: str | None = None,
) -> list[SearchableFactRecord]:
    """Cosine-similarity search over `searchable_facts.embedding` (B7 —
    replaces the Pinecone `facts` namespace). `video_id` is optional so
    callers can either scope to one video or search across all of them.
    """
    if video_id is not None:
        rows = await pool.fetch(
            """
            SELECT * FROM searchable_facts
            WHERE embedding IS NOT NULL AND video_id = $2
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            embedding,
            _to_uuid(video_id),
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT * FROM searchable_facts
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            embedding,
            limit,
        )
    return [
        SearchableFactRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            fact_text=r["fact_text"],
            frame_id=str(r["frame_id"]) if r["frame_id"] else None,
            timestamp_s=r["timestamp_s"],
            embedding=_vec_to_list(r["embedding"]),
            legacy_vector_id=r["legacy_vector_id"],
        )
        for r in rows
    ]


async def bulk_update_fact_superseded_by(
    pool: asyncpg.Pool, mapping: list[tuple[str, str]]
) -> None:
    """Mark near-duplicate facts with a `superseded_by` pointer.

    Each tuple is (fact_id, kept_fact_id). Additive only — never DELETEs a
    row (rule 13).
    """
    if not mapping:
        return
    records = [(_to_uuid(fact_id), _to_uuid(kept_id)) for fact_id, kept_id in mapping]
    await pool.executemany(
        "UPDATE searchable_facts SET superseded_by = $2 WHERE id = $1",
        records,
    )


async def get_kg_edges_for_video(pool: asyncpg.Pool, video_id: str) -> list[KGEdge]:
    """Get all KG edges for a video from the Postgres structured cache.

    Use this for structured filtering (e.g. WHERE relation='BELONGS_TO').
    For graph traversal queries (multi-hop, path finding) use Neo4j directly.

    Args:
        pool:     An open asyncpg connection pool.
        video_id: The video's unique identifier.

    Returns:
        All KGEdge rows for the video.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM kg_edges WHERE video_id=$1",
            _to_uuid(video_id),
        )
        return [
            KGEdge(
                id=str(row["id"]),
                video_id=str(row["video_id"]),
                source_id=str(row["source_id"]),
                target_id=str(row["target_id"]),
                relation=row["relation"],
                weight=float(row["weight"]),
                properties=json.loads(row["properties"]) if row["properties"] else {},
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# Canonical relation → Neo4j propagation (Level-4)
# ---------------------------------------------------------------------------


async def get_kg_edges_with_canonical_relation(
    pool: asyncpg.Pool, video_id: str
) -> list[dict]:
    """Return edges that have a non-null `canonical_relation`, joined against
    kg_nodes to resolve each endpoint's (node_type, ref_id) — everything
    `knowledge_base.neo4j.graph_writer.update_edge_relations_canonical`
    needs to MERGE the corrected relationship type without touching edges
    that haven't been canonicalized yet.

    Each dict has keys: db_id, weight, raw_relation, canonical_relation,
    src_type, src_ref_id, tgt_type, tgt_ref_id.
    """
    rows = await pool.fetch(
        """
        SELECT
            e.id             AS db_id,
            e.weight         AS weight,
            e.relation       AS raw_relation,
            e.canonical_relation AS canonical_relation,
            ns.node_type     AS src_type,
            ns.ref_id        AS src_ref_id,
            nt.node_type     AS tgt_type,
            nt.ref_id        AS tgt_ref_id
        FROM kg_edges e
        JOIN kg_nodes ns ON ns.id = e.source_id
        JOIN kg_nodes nt ON nt.id = e.target_id
        WHERE e.video_id = $1 AND e.canonical_relation IS NOT NULL
        """,
        _to_uuid(video_id),
    )
    return [
        {
            "db_id": str(r["db_id"]),
            "weight": float(r["weight"]),
            "raw_relation": r["raw_relation"],
            "canonical_relation": r["canonical_relation"],
            "src_type": r["src_type"],
            "src_ref_id": r["src_ref_id"],
            "tgt_type": r["tgt_type"],
            "tgt_ref_id": r["tgt_ref_id"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Scenes (Level-4 Story Architect Agent — canonical scene timeline)
# ---------------------------------------------------------------------------


async def get_scenes_for_video(pool: asyncpg.Pool, video_id: str) -> list[SceneRecord]:
    """Return all canonical `scenes` rows for a video, ordered by start_time.

    Used by `pipeline.level4.finalizer` to check that chunks/shots time
    ranges are fully covered (no silent gaps in the scene timeline), and by
    the Pinecone propagation pass to embed one vector per canonical scene.
    """
    rows = await pool.fetch(
        "SELECT * FROM scenes WHERE video_id = $1 ORDER BY start_time",
        _to_uuid(video_id),
    )
    return [_row_to_scene(r) for r in rows]


async def get_scene_by_id(pool: asyncpg.Pool, scene_id: str) -> SceneRecord | None:
    """Fetch one `scenes` row by id. Used by `pipeline/feedback/correction_logger.py`
    to capture `original_value` before an override overwrites a field."""
    row = await pool.fetchrow("SELECT * FROM scenes WHERE id = $1", _to_uuid(scene_id))
    if row is None:
        return None
    return _row_to_scene(row)


def _row_to_scene(r) -> SceneRecord:
    return SceneRecord(
        id=str(r["id"]),
        video_id=str(r["video_id"]),
        canonical_scene_id=r["canonical_scene_id"],
        discarded_aliases=list(r["discarded_aliases"] or []),
        start_time=r["start_time"],
        end_time=r["end_time"],
        participants=list(r["participants"] or []),
        summary=r["summary"],
        emotional_arc=r["emotional_arc"],
        causal_link_to_next=r["causal_link_to_next"],
        usability_score=r["usability_score"],
        embedding=_vec_to_list(r["embedding"]) if "embedding" in r else None,
    )


async def delete_scenes_for_video(pool: asyncpg.Pool, video_id: str) -> int:
    """Delete all `scenes` rows for *video_id* — call before a Story Architect
    re-run writes fresh ones (see `bulk_upsert_scenes` docstring for why this
    is required, not just a courtesy).

    Real-video finding: `bulk_upsert_scenes`'s UPSERT key is
    `(video_id, canonical_scene_id)`, and `canonical_scene_id` is derived
    from Qwen's own per-frame `scene_id` (via `_SceneGroup.majority_scene_id`
    in story_architect_runner.py) — exactly the unstable, non-deterministic
    text CLAUDE.md's "Why L4 exists" section documents as the reason L4
    exists in the first place. Across two runs of the same video (different
    model, different temperature draw, or just LLM non-determinism), the
    same real-world scene can get a different majority label each time, so
    the UPSERT conflict key never matches and rows never overwrite — they
    accumulate. Observed on video_id=97199656: ~5 near-duplicate `scenes`
    rows all spanning the same ~127s intro segment, ~8 for a "COMING SOON"
    segment, ~10 for a Netflix-logo segment, surviving under different
    canonical_scene_id text from different runs. rule 15's "safe UPSERT for
    scenes/intermediate state" was the right intent; delete-then-insert is
    what actually delivers it, since `scenes` (unlike `storylines`) has no
    version column to disambiguate old vs new by.
    """
    result = await pool.execute(
        "DELETE FROM scenes WHERE video_id = $1::uuid", _to_uuid(video_id)
    )
    # asyncpg execute() returns a status string like "DELETE 43"
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


async def bulk_upsert_scenes(pool: asyncpg.Pool, scenes: list[SceneRecord]) -> None:
    """Upsert canonical scene rows, keyed on (video_id, canonical_scene_id).

    Callers MUST call `delete_scenes_for_video` first on a re-run — see that
    function's docstring for why the UPSERT key alone does not make this
    idempotent in practice (the key is derived from non-deterministic LLM
    text, so the "overwrites its own prior draft output" claim only holds
    within a single run, not across runs).
    """
    if not scenes:
        return
    records = [
        (
            _to_uuid(s.id),
            _to_uuid(s.video_id),
            s.canonical_scene_id,
            s.discarded_aliases,
            s.start_time,
            s.end_time,
            s.participants,
            s.summary,
            s.emotional_arc,
            s.causal_link_to_next,
            s.usability_score,
            s.embedding,
        )
        for s in scenes
    ]
    await pool.executemany(
        """
        INSERT INTO scenes
            (id, video_id, canonical_scene_id, discarded_aliases, start_time,
             end_time, participants, summary, emotional_arc, causal_link_to_next,
             usability_score, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (video_id, canonical_scene_id) DO UPDATE
          SET discarded_aliases   = EXCLUDED.discarded_aliases,
              start_time          = EXCLUDED.start_time,
              end_time            = EXCLUDED.end_time,
              participants        = EXCLUDED.participants,
              summary             = EXCLUDED.summary,
              emotional_arc       = EXCLUDED.emotional_arc,
              causal_link_to_next = EXCLUDED.causal_link_to_next,
              usability_score     = EXCLUDED.usability_score,
              embedding           = COALESCE(EXCLUDED.embedding, scenes.embedding)
        """,
        records,
    )


async def update_scene_embeddings(
    pool: asyncpg.Pool, updates: list[tuple[str, list[float]]]
) -> None:
    """Bulk-update `scenes.embedding` for a batch of (scene_id, embedding)
    pairs (B7 — replaces the Pinecone canonical-scenes propagation pass in
    `pipeline/level4/finalizer.py`). Caller batches in groups of 100 (rule 8).
    """
    if not updates:
        return
    records = [(_to_uuid(scene_id), emb) for scene_id, emb in updates]
    await pool.executemany(
        "UPDATE scenes SET embedding = $2 WHERE id = $1",
        records,
    )


async def search_scenes_by_embedding(
    pool: asyncpg.Pool,
    embedding: list[float],
    limit: int = 10,
    video_id: str | None = None,
) -> list[SceneRecord]:
    """Cosine-similarity search over `scenes.embedding` (B7 — replaces the
    Pinecone `scenes` namespace's canonical-scene variant).
    """
    if video_id is not None:
        rows = await pool.fetch(
            """
            SELECT * FROM scenes
            WHERE embedding IS NOT NULL AND video_id = $2
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            embedding,
            _to_uuid(video_id),
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT * FROM scenes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            embedding,
            limit,
        )
    return [_row_to_scene(r) for r in rows]


# ---------------------------------------------------------------------------
# Storylines (Level-4 Story Architect Agent — versioned, immutable-once-final)
# ---------------------------------------------------------------------------


def _storyline_from_row(r) -> StorylineRecord:
    beats_raw = r["beats"]
    cast_raw = r["cast_members"]
    return StorylineRecord(
        id=str(r["id"]),
        video_id=str(r["video_id"]),
        version=r["version"],
        status=r["status"],
        title=r["title"],
        synopsis=r["synopsis"],
        cast_members=(json.loads(cast_raw) if isinstance(cast_raw, str) else (cast_raw or {})),
        beats=(json.loads(beats_raw) if isinstance(beats_raw, str) else (beats_raw or [])),
        embedding=_vec_to_list(r["embedding"]) if "embedding" in r else None,
    )


async def get_storyline_by_id(pool: asyncpg.Pool, storyline_id: str) -> StorylineRecord | None:
    """Fetch one `storylines` row by id. Used by
    `pipeline/feedback/correction_logger.py` to capture `original_value`
    before an override overwrites a field."""
    row = await pool.fetchrow("SELECT * FROM storylines WHERE id = $1", _to_uuid(storyline_id))
    if row is None:
        return None
    return _storyline_from_row(row)


async def get_latest_storyline(pool: asyncpg.Pool, video_id: str) -> StorylineRecord | None:
    """Return the highest-`version` `storylines` row for a video (draft or
    final — the finalizer decides whether it's eligible to flip to final),
    or None if the Story Architect Agent hasn't written one yet."""
    row = await pool.fetchrow(
        """
        SELECT * FROM storylines
        WHERE video_id = $1
        ORDER BY version DESC
        LIMIT 1
        """,
        _to_uuid(video_id),
    )
    if row is None:
        return None
    return _storyline_from_row(row)


async def upsert_storyline_draft(pool: asyncpg.Pool, storyline: StorylineRecord) -> str:
    """Insert or update a `draft` storylines row for (video_id, version).

    Only ever call this with storyline.status == 'draft' — a re-run of the
    Story Architect Agent on the same video_id/version overwrites its own
    prior draft (idempotent, rule 9/15). This function must never be used
    to write a 'final' row in place; `update_storyline_status` is the only
    write path that flips draft -> final, and only after the finalizer's
    completeness checks pass (rule 4 / rule 17). Callers are responsible for
    choosing a version that does not collide with an existing 'final' row
    for this video — see `get_latest_storyline`.
    """
    row = await pool.fetchrow(
        """
        INSERT INTO storylines
            (id, video_id, version, status, title, synopsis, cast_members, beats)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb)
        ON CONFLICT (video_id, version) DO UPDATE
          SET title        = EXCLUDED.title,
              synopsis     = EXCLUDED.synopsis,
              cast_members = EXCLUDED.cast_members,
              beats        = EXCLUDED.beats
          WHERE storylines.status = 'draft'
        RETURNING id
        """,
        _to_uuid(storyline.id),
        _to_uuid(storyline.video_id),
        storyline.version,
        storyline.status,
        storyline.title,
        storyline.synopsis,
        _to_json(storyline.cast_members),
        _to_json(storyline.beats),
    )
    if row is None:
        # ON CONFLICT ... WHERE guard skipped the update because the
        # existing row at this version is already 'final' — never overwrite
        # it. Surface this loudly rather than silently doing nothing.
        raise RuntimeError(
            f"Refusing to overwrite final storyline for video_id={storyline.video_id} "
            f"version={storyline.version}"
        )
    return str(row["id"])


async def update_storyline_status(
    pool: asyncpg.Pool, video_id: str, version: int, status: str
) -> None:
    """Flip a `storylines` row's status (draft -> final), keyed by
    (video_id, version). Never call with status='final' unless the
    finalization gate's completeness checks have already passed — this is
    the one write that makes a storyline visible to L5 (rule 16)."""
    await pool.execute(
        """
        UPDATE storylines
        SET status = $3
        WHERE video_id = $1 AND version = $2
        """,
        _to_uuid(video_id),
        version,
        status,
    )


async def get_final_storyline_for_video(pool: asyncpg.Pool, video_id: str) -> StorylineRecord | None:
    """Return the highest-`version` `storylines` row for a video with
    `status = 'final'`, or None if L4 hasn't finalized a storyline for this
    video yet.

    `get_latest_storyline` (above) intentionally returns the latest row
    regardless of status, for L4's own finalizer use. L5 must never read a
    `draft` row (CLAUDE.md read contract / rule 16) — this is the variant
    that enforces that at the query level so callers in `pipeline/level5/`
    can't accidentally bypass it.
    """
    row = await pool.fetchrow(
        """
        SELECT * FROM storylines
        WHERE video_id = $1 AND status = 'final'
        ORDER BY version DESC
        LIMIT 1
        """,
        _to_uuid(video_id),
    )
    if row is None:
        return None
    return _storyline_from_row(row)


async def update_storyline_embedding(
    pool: asyncpg.Pool, storyline_id: str, embedding: list[float]
) -> None:
    """Set `storylines.embedding` for one row (B7 — replaces the Pinecone
    storyline propagation pass in `pipeline/level4/finalizer.py`; one vector
    per video's finalized synopsis).
    """
    await pool.execute(
        "UPDATE storylines SET embedding = $2 WHERE id = $1",
        _to_uuid(storyline_id),
        embedding,
    )


async def search_storylines_by_embedding(
    pool: asyncpg.Pool,
    embedding: list[float],
    limit: int = 10,
    status: str | None = "final",
) -> list[StorylineRecord]:
    """Cosine-similarity search over `storylines.embedding` (B7 — replaces
    the planned Pinecone `storylines` namespace; enables cross-video plot/
    theme search). Defaults to `status='final'` — matches the read contract
    (L5 and any downstream search-by-plot consumer should only see finalized
    storylines); pass `status=None` to search drafts too.
    """
    if status is not None:
        rows = await pool.fetch(
            """
            SELECT * FROM storylines
            WHERE embedding IS NOT NULL AND status = $2
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            embedding,
            status,
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT * FROM storylines
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            embedding,
            limit,
        )
    return [_storyline_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Edit plans (Level-5 Planning — Selection & Scoring / Sequencing & Pacing)
# ---------------------------------------------------------------------------


def _edit_plan_from_row(r) -> EditPlanRecord:
    ops_raw = r["operations"]
    return EditPlanRecord(
        id=str(r["id"]),
        video_id=str(r["video_id"]),
        storyline_id=str(r["storyline_id"]) if r["storyline_id"] is not None else None,
        user_prompt=r["user_prompt"],
        target_duration_s=r["target_duration_s"],
        platform=r["platform"],
        status=r["status"],
        version=r["version"],
        operations=(json.loads(ops_raw) if isinstance(ops_raw, str) else (ops_raw or [])),
        achieved_duration_s=r["achieved_duration_s"],
    )


async def insert_edit_plan(pool: asyncpg.Pool, plan: EditPlanRecord) -> str:
    """Insert (or, if re-running the same video_id/version while it's still
    'draft', update) an `edit_plans` row. Mirrors `upsert_storyline_draft`'s
    guard: refuses to silently overwrite a row that has moved past 'draft'
    (reviewed/applied/superseded) — those are append-only via a new version,
    never mutated in place.
    """
    row = await pool.fetchrow(
        """
        INSERT INTO edit_plans
            (id, video_id, storyline_id, user_prompt, target_duration_s,
             platform, status, version, operations, achieved_duration_s)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
        ON CONFLICT (video_id, version) DO UPDATE
          SET storyline_id        = EXCLUDED.storyline_id,
              user_prompt         = EXCLUDED.user_prompt,
              target_duration_s   = EXCLUDED.target_duration_s,
              platform            = EXCLUDED.platform,
              status              = EXCLUDED.status,
              operations          = EXCLUDED.operations,
              achieved_duration_s = EXCLUDED.achieved_duration_s
          WHERE edit_plans.status = 'draft'
        RETURNING id
        """,
        _to_uuid(plan.id),
        _to_uuid(plan.video_id),
        _to_uuid(plan.storyline_id) if plan.storyline_id else None,
        plan.user_prompt,
        plan.target_duration_s,
        plan.platform,
        plan.status,
        plan.version,
        _to_json(plan.operations),
        plan.achieved_duration_s,
    )
    if row is None:
        raise RuntimeError(
            f"Refusing to overwrite non-draft edit_plan for video_id={plan.video_id} "
            f"version={plan.version}"
        )
    return str(row["id"])


async def get_edit_plan(pool: asyncpg.Pool, edit_plan_id: str) -> EditPlanRecord | None:
    row = await pool.fetchrow(
        "SELECT * FROM edit_plans WHERE id = $1",
        _to_uuid(edit_plan_id),
    )
    if row is None:
        return None
    return _edit_plan_from_row(row)


async def get_latest_edit_plan_for_video(pool: asyncpg.Pool, video_id: str) -> EditPlanRecord | None:
    """Return the highest-`version` `edit_plans` row for a video (any
    status), or None if no plan has been written yet. Used to decide the
    next `version` number for a new planning run (see
    `pipeline/level5/planner_runner.py` for the versioning-semantics note)."""
    row = await pool.fetchrow(
        """
        SELECT * FROM edit_plans
        WHERE video_id = $1
        ORDER BY version DESC
        LIMIT 1
        """,
        _to_uuid(video_id),
    )
    if row is None:
        return None
    return _edit_plan_from_row(row)


async def insert_edit_plan_revision(pool: asyncpg.Pool, revision: EditPlanRevisionRecord) -> str:
    """Insert one `edit_plan_revisions` row — `diff_operations` only, never
    the full plan again (CLAUDE.md rule 21)."""
    row = await pool.fetchrow(
        """
        INSERT INTO edit_plan_revisions
            (id, edit_plan_id, user_feedback, diff_operations)
        VALUES ($1, $2, $3, $4::jsonb)
        RETURNING id
        """,
        _to_uuid(revision.id),
        _to_uuid(revision.edit_plan_id),
        revision.user_feedback,
        _to_json(revision.diff_operations),
    )
    return str(row["id"])


# ---------------------------------------------------------------------------
# Overrides + correction events (Addendum 2, item 2 — correction feedback loop)
#
# insert_scene_override/insert_storyline_override are raw table writes only.
# Callers should go through pipeline/feedback/correction_logger.py::
# log_scene_correction()/log_storyline_correction() instead of calling these
# directly, so every override write also gets a matching correction_events
# row (rule 25) — these two functions exist so that logger has something to
# call, not as a second public entry point.
# ---------------------------------------------------------------------------


async def insert_scene_override(pool: asyncpg.Pool, override: SceneOverrideRecord) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO scene_overrides (id, scene_id, field, new_value, reason, created_by)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6)
        RETURNING id
        """,
        _to_uuid(override.id),
        _to_uuid(override.scene_id),
        override.field,
        _to_json(override.new_value),
        override.reason,
        override.created_by,
    )
    return str(row["id"])


async def insert_storyline_override(pool: asyncpg.Pool, override: StorylineOverrideRecord) -> str:
    row = await pool.fetchrow(
        """
        INSERT INTO storyline_overrides (id, storyline_id, field, new_value, reason, created_by)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6)
        RETURNING id
        """,
        _to_uuid(override.id),
        _to_uuid(override.storyline_id),
        override.field,
        _to_json(override.new_value),
        override.reason,
        override.created_by,
    )
    return str(row["id"])


async def insert_correction_event(pool: asyncpg.Pool, event: CorrectionEventRecord) -> str:
    """Insert one `correction_events` row — the corrections dataset (Addendum
    2, item 2). Every override write and every `edit_plan_revisions` write
    must go through `pipeline/feedback/correction_logger.py`, which calls
    this alongside the override/revision insert (rule 25) — no code path
    should write one without the other."""
    row = await pool.fetchrow(
        """
        INSERT INTO correction_events
            (id, video_id, level, entity_type, entity_id, field,
             original_value, corrected_value, correction_source, reason)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10)
        RETURNING id
        """,
        _to_uuid(event.id),
        _to_uuid(event.video_id),
        event.level,
        event.entity_type,
        _to_uuid(event.entity_id),
        event.field,
        _to_json(event.original_value),
        _to_json(event.corrected_value),
        event.correction_source,
        event.reason,
    )
    return str(row["id"])


# ---------------------------------------------------------------------------
# Client style profiles (Addendum 2, item 3 — migration 016)
# ---------------------------------------------------------------------------


def _row_to_client_style_profile(r) -> ClientStyleProfileRecord:
    return ClientStyleProfileRecord(
        id=str(r["id"]),
        client_id=r["client_id"],
        caption_style=_from_json(r["caption_style"]),
        pacing_preference=r["pacing_preference"],
        brand_colors=_from_json(r["brand_colors"]),
        default_platform=r["default_platform"],
        notes=r["notes"],
    )


async def get_client_style_profile(
    pool: asyncpg.Pool, client_id: str
) -> ClientStyleProfileRecord | None:
    """Return the `client_style_profiles` row for *client_id*, or None if no
    profile exists yet. Read-only soft-prior lookup (CLAUDE.md rule 26) —
    callers (L5 Pass B, L6 Color Grading Agent, L6 Caption/Text Overlay
    Agent) must treat a None result exactly like "no client system in use
    at all", never as an error."""
    row = await pool.fetchrow(
        "SELECT * FROM client_style_profiles WHERE client_id = $1",
        client_id,
    )
    if row is None:
        return None
    return _row_to_client_style_profile(row)


async def upsert_client_style_profile(
    pool: asyncpg.Pool, profile: ClientStyleProfileRecord
) -> str:
    """Insert or update the one `client_style_profiles` row for
    `profile.client_id` (unique index, v1 — one profile per client, no
    versioning yet). Returns the row `id`."""
    row = await pool.fetchrow(
        """
        INSERT INTO client_style_profiles
            (id, client_id, caption_style, pacing_preference, brand_colors,
             default_platform, notes, updated_at)
        VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6, $7, NOW())
        ON CONFLICT (client_id) DO UPDATE
          SET caption_style     = EXCLUDED.caption_style,
              pacing_preference = EXCLUDED.pacing_preference,
              brand_colors      = EXCLUDED.brand_colors,
              default_platform  = EXCLUDED.default_platform,
              notes             = EXCLUDED.notes,
              updated_at        = NOW()
        RETURNING id
        """,
        _to_uuid(profile.id),
        profile.client_id,
        _to_json(profile.caption_style),
        profile.pacing_preference,
        _to_json(profile.brand_colors),
        profile.default_platform,
        profile.notes,
    )
    return str(row["id"])


# ---------------------------------------------------------------------------
# Cut list items (Level-6 Editing Director — deterministic snapped cut list)
# ---------------------------------------------------------------------------


async def delete_cut_list_items_for_edit_plan(pool: asyncpg.Pool, edit_plan_id: str) -> int:
    """Delete all `cut_list_items` rows for *edit_plan_id* — call before a
    fresh `run_level6` writes new ones (see `bulk_insert_cut_list_items`
    docstring: ids are freshly generated per run via `gen_id()`, so
    `ON CONFLICT (id) DO NOTHING` never matches a prior run's rows, and
    "the caller's responsibility to clear stale rows" was never actually
    implemented anywhere until this function existed).

    Real-video finding, first live L6 run: without this, two calls to
    `run_level6` for the same `edit_plan_id` (e.g. a render that failed at
    the ffmpeg step and got retried) left the FIRST run's cut_list_items in
    place while the SECOND run added a full new set on top — 7 real
    SELECT_CLIP ops turned into 21 `cut_list_items` rows, and
    `sequence_color_adjustments` (also insert-only, same ids-not-matching
    issue) went from 14 to 22 rows including several color deltas stacked
    onto the SAME clip. `render_direct`'s filter_complex then built a
    filter graph over all of them, producing chains with the same
    colortemperature/eq/colorbalance sequence applied twice — and ffmpeg's
    filtergraph parser ran out of memory trying to execute it
    ("Cannot allocate memory"). `sequence_color_adjustments`,
    `emphasis_effects`, and `layer_composites` all have
    `ON DELETE CASCADE` from `cut_list_items` (migrations 007/010/011), so
    deleting here alone clears every downstream L6 output table too — no
    separate delete needed for those three.
    """
    result = await pool.execute(
        "DELETE FROM cut_list_items WHERE edit_plan_id = $1::uuid", _to_uuid(edit_plan_id)
    )
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


async def bulk_insert_cut_list_items(
    pool: asyncpg.Pool, items: list[CutListItemRecord]
) -> None:
    """Bulk-insert `cut_list_items` rows for a finalized `edit_plan`.

    Callers MUST call `delete_cut_list_items_for_edit_plan` first on a
    re-run — see that function's docstring for the real-video duplication
    bug this fixes (the `ON CONFLICT (id) DO NOTHING` clause below is not,
    by itself, enough for idempotency: freshly generated ids never conflict
    with a prior run's rows).
    """
    if not items:
        return
    records = [
        (
            _to_uuid(item.id),
            _to_uuid(item.edit_plan_id),
            item.op_id,
            item.sequence_index,
            item.source_start,
            item.source_end,
            item.audio_lead_ms,
            item.video_lead_ms,
            item.transition,
        )
        for item in items
    ]
    await pool.executemany(
        """
        INSERT INTO cut_list_items
            (id, edit_plan_id, op_id, sequence_index, source_start, source_end,
             audio_lead_ms, video_lead_ms, transition)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (id) DO NOTHING
        """,
        records,
    )


async def get_cut_list_items_for_edit_plan(
    pool: asyncpg.Pool, edit_plan_id: str
) -> list[CutListItemRecord]:
    """Return all `cut_list_items` rows for an edit plan, ordered by
    `sequence_index` — the order both `export_xml` and `render_direct`
    consume to assemble the final sequence."""
    rows = await pool.fetch(
        "SELECT * FROM cut_list_items WHERE edit_plan_id = $1 ORDER BY sequence_index",
        _to_uuid(edit_plan_id),
    )
    return [
        CutListItemRecord(
            id=str(r["id"]),
            edit_plan_id=str(r["edit_plan_id"]),
            op_id=r["op_id"],
            sequence_index=r["sequence_index"],
            source_start=r["source_start"],
            source_end=r["source_end"],
            audio_lead_ms=r["audio_lead_ms"],
            video_lead_ms=r["video_lead_ms"],
            transition=r["transition"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Sequence color adjustments (Level-6 Color Grading Agent)
# ---------------------------------------------------------------------------


async def bulk_insert_sequence_color_adjustments(
    pool: asyncpg.Pool, adjustments: list[SequenceColorAdjustmentRecord]
) -> None:
    """Bulk-insert `sequence_color_adjustments` rows.

    Follows the `bulk_insert_cut_list_items` pattern: `ON CONFLICT (id) DO
    NOTHING` — callers generate a fresh `id` per adjustment
    (`pipeline.level6.color_grading_runner.run_color_grading`), so a re-run
    that reuses the same ids is a no-op. A re-run of the color grading agent
    for an edit_plan_id that should replace prior adjustments is the
    caller's responsibility (e.g. delete-then-reinsert), mirroring the
    additive-only discipline already used for cut_list_items.
    """
    if not adjustments:
        return
    records = [
        (
            _to_uuid(a.id),
            _to_uuid(a.edit_plan_id),
            _to_uuid(a.cut_list_item_id),
            _to_json(a.base_parameters),
            _to_json(a.sequence_delta),
            a.rationale,
        )
        for a in adjustments
    ]
    await pool.executemany(
        """
        INSERT INTO sequence_color_adjustments
            (id, edit_plan_id, cut_list_item_id, base_parameters, sequence_delta, rationale)
        VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6)
        ON CONFLICT (id) DO NOTHING
        """,
        records,
    )


async def get_sequence_color_adjustments_for_edit_plan(
    pool: asyncpg.Pool, edit_plan_id: str
) -> list[SequenceColorAdjustmentRecord]:
    """Return all `sequence_color_adjustments` rows for an edit plan.

    Used by the render pipeline (Engineer A's `render_direct` extension
    point) to look up the final delta for each `cut_list_item.id` when
    building the FFmpeg filter graph — see
    `pipeline.level6.color_grading_runner.build_ffmpeg_color_filters`.
    """
    rows = await pool.fetch(
        "SELECT * FROM sequence_color_adjustments WHERE edit_plan_id = $1",
        _to_uuid(edit_plan_id),
    )
    return [
        SequenceColorAdjustmentRecord(
            id=str(r["id"]),
            edit_plan_id=str(r["edit_plan_id"]),
            cut_list_item_id=str(r["cut_list_item_id"]),
            base_parameters=(
                json.loads(r["base_parameters"])
                if isinstance(r["base_parameters"], str)
                else (r["base_parameters"] or {})
            ),
            sequence_delta=(
                json.loads(r["sequence_delta"])
                if isinstance(r["sequence_delta"], str)
                else (r["sequence_delta"] or {})
            ),
            rationale=r["rationale"],
        )
        for r in rows
    ]


async def count_unterminal_speaker_turns(pool: asyncpg.Pool, video_id: str) -> int:
    """Count speaker_turns rows still in a pre-L4 state (`unresolved` or
    `face_majority`) — used by the finalization gate's completeness check.
    A non-zero count means the Grounding Agent hasn't finished (or hasn't
    run) for this video yet."""
    row = await pool.fetchrow(
        """
        SELECT COUNT(*) AS n FROM speaker_turns
        WHERE video_id = $1 AND resolution_method IN ('unresolved', 'face_majority')
        """,
        _to_uuid(video_id),
    )
    return int(row["n"]) if row else 0


# ---------------------------------------------------------------------------
# Stock assets (A2 — knowledge_base/stock_assets/, standalone infrastructure,
# built once, queried later by L6's Compositing Agent). See CLAUDE.md
# "PIPELINE ADDENDUM" → A2.
# ---------------------------------------------------------------------------


async def upsert_stock_asset(pool: asyncpg.Pool, asset: StockAssetRecord) -> str:
    """Insert or update one `stock_assets` row.  Returns the row ``id``."""
    row = await pool.fetchrow(
        """
        INSERT INTO stock_assets
            (id, source, external_id, description, tags, license_type,
             embedding, r2_cache_key)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (id) DO UPDATE
          SET source        = EXCLUDED.source,
              external_id   = EXCLUDED.external_id,
              description   = EXCLUDED.description,
              tags          = EXCLUDED.tags,
              license_type  = EXCLUDED.license_type,
              embedding     = EXCLUDED.embedding,
              r2_cache_key  = EXCLUDED.r2_cache_key
        RETURNING id
        """,
        _to_uuid(asset.id),
        asset.source,
        asset.external_id,
        asset.description,
        asset.tags,
        asset.license_type,
        asset.embedding,
        asset.r2_cache_key,
    )
    return str(row["id"])


async def bulk_upsert_stock_assets(
    pool: asyncpg.Pool, assets: list[StockAssetRecord]
) -> None:
    """Upsert `stock_assets` rows via `executemany`, batched in groups of 100
    by the caller (`knowledge_base/stock_assets/indexer.py`) — rule 8's
    "Pinecone upserts batched in 100s" principle applied here to the
    Postgres-only A2 subsystem (B7: no Pinecone for this new table).

    Safe to call with more than 100 records — `executemany` itself does not
    need pre-chunking to be correct, but the indexer chunks anyway to keep
    each transaction small and to log progress per batch.
    """
    if not assets:
        return
    records = [
        (
            _to_uuid(a.id),
            a.source,
            a.external_id,
            a.description,
            a.tags,
            a.license_type,
            a.embedding,
            a.r2_cache_key,
        )
        for a in assets
    ]
    await pool.executemany(
        """
        INSERT INTO stock_assets
            (id, source, external_id, description, tags, license_type,
             embedding, r2_cache_key)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (id) DO UPDATE
          SET source        = EXCLUDED.source,
              external_id   = EXCLUDED.external_id,
              description   = EXCLUDED.description,
              tags          = EXCLUDED.tags,
              license_type  = EXCLUDED.license_type,
              embedding     = EXCLUDED.embedding,
              r2_cache_key  = EXCLUDED.r2_cache_key
        """,
        records,
    )


def _row_to_stock_asset(r) -> StockAssetRecord:
    return StockAssetRecord(
        id=str(r["id"]),
        source=r["source"],
        license_type=r["license_type"],
        external_id=r["external_id"],
        description=r["description"],
        tags=list(r["tags"] or []),
        embedding=_vec_to_list(r["embedding"]),
        r2_cache_key=r["r2_cache_key"],
    )


async def get_stock_asset_by_source_external_id(
    pool: asyncpg.Pool, source: str, external_id: str
) -> StockAssetRecord | None:
    """Look up an existing `stock_assets` row by `(source, external_id)`.

    Used by the indexer to decide whether to re-embed/re-upsert an asset
    already ingested from a prior Pexels search (idempotent re-runs — same
    "safe to rerun" guarantee used throughout the pipeline).
    """
    row = await pool.fetchrow(
        "SELECT * FROM stock_assets WHERE source = $1 AND external_id = $2",
        source,
        external_id,
    )
    return _row_to_stock_asset(row) if row is not None else None


async def get_stock_asset_by_id(pool: asyncpg.Pool, asset_id: str) -> StockAssetRecord | None:
    """Look up one `stock_assets` row by its primary key.

    Used at A4 render time (`pipeline/level6/editing_director.py::
    get_compositing_render_extras`) to resolve a `background_assignments`
    pick's `asset_id` back to its `r2_cache_key` for the actual FFmpeg
    background-swap composite — re-checked at render time rather than
    trusted from materialize time, since the asset could in principle have
    been removed/re-indexed between when the Compositing Agent picked it
    and when this edit_plan is rendered.
    """
    row = await pool.fetchrow(
        "SELECT * FROM stock_assets WHERE id = $1",
        _to_uuid(asset_id),
    )
    return _row_to_stock_asset(row) if row is not None else None


async def search_stock_assets(
    pool: asyncpg.Pool,
    embedding: list[float],
    limit: int = 10,
    license_type: str | None = None,
) -> list[StockAssetRecord]:
    """Cosine-similarity search over `stock_assets.embedding`.

    This is the entry point the L6 Compositing Agent (`background_selector.py`,
    built separately) calls to retrieve top-k candidate stock assets for a
    scene, before the one LLM-worthy pick. `license_type` is optional —
    filtering is available now even though enforcement is not yet wired into
    any calling agent (see CLAUDE.md "Before building A3: stock licensing").

    Args:
        pool: asyncpg connection pool.
        embedding: 1024-dim query vector (BAAI/bge-large-en-v1.5), typically
            an embedding of scene transcript + `scene_mood`/tags.
        limit: Max number of results, ordered by ascending cosine distance
            (closest/most similar first).
        license_type: If given, restrict results to this exact license type.

    Returns:
        List of :class:`StockAssetRecord`, closest match first.
    """
    if license_type is not None:
        rows = await pool.fetch(
            """
            SELECT * FROM stock_assets
            WHERE embedding IS NOT NULL AND license_type = $2
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            embedding,
            license_type,
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT * FROM stock_assets
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            embedding,
            limit,
        )
    return [_row_to_stock_asset(r) for r in rows]


# ---------------------------------------------------------------------------
# Frame analyses — scene-signal fields for a video (Level-6 Compositing
# Agent: background_selector.py aggregates scene_mood/tags per `scenes`
# window, emphasis_selector.py aggregates beat_type/tension_level per
# `cut_list_items` window). Distinct from `get_frame_analyses_with_
# timestamps_for_video` (L4, qwen_output only) — this returns the flat
# scene-signal columns L4's Story Architect already wrote to `frame_analyses`
# directly, no qwen_output JSON parsing needed by callers.
# ---------------------------------------------------------------------------


async def get_frame_analyses_fields_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[dict]:
    """Return [{keyframe_id, timestamp_s, scene_id, beat_type, scene_mood,
    tension_level, tags, caption}, ...] for a video, ordered by timestamp.

    `scene_id` here is frame_analyses' own loose per-frame text label (NOT
    the canonical `scenes.id` UUID FK) — kept only as a debugging aid for
    callers that want to see which raw alias a frame carried; do not use it
    to join against `scenes`, use time-range overlap instead (the whole
    reason the canonical `scenes` table exists — see CLAUDE.md "Why L4
    exists").
    """
    rows = await pool.fetch(
        """
        SELECT k.id AS keyframe_id, k.timestamp_s, fa.scene_id, fa.beat_type,
               fa.scene_mood, fa.tension_level, fa.tags, fa.caption
        FROM frame_analyses fa
        JOIN keyframes k ON k.id = fa.keyframe_id
        WHERE fa.video_id = $1
        ORDER BY k.timestamp_s
        """,
        _to_uuid(video_id),
    )
    return [
        {
            "keyframe_id": str(r["keyframe_id"]),
            "timestamp_s": r["timestamp_s"],
            "scene_id": r["scene_id"],
            "beat_type": r["beat_type"],
            "scene_mood": r["scene_mood"],
            "tension_level": r["tension_level"],
            "tags": list(r["tags"] or []),
            "caption": r["caption"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Compositing Agent — A3 background_assignments (scene-level, L6
# background_selector.py) + A5-decision emphasis_effects (cut-order-level,
# L6 emphasis_selector.py). See CLAUDE.md "PIPELINE ADDENDUM" -> A3, A5 and
# migration `010_compositing.sql`.
# ---------------------------------------------------------------------------


async def bulk_upsert_background_assignments(
    pool: asyncpg.Pool, assignments: list[BackgroundAssignmentRecord]
) -> None:
    """Upsert `background_assignments` rows, keyed on `scene_id` (unique
    index in `010_compositing.sql`) — a background pick is a property of
    the scene, so re-running `run_background_selection` for a video safely
    overwrites its own prior pick rather than accumulating duplicates
    (rule 9/15 idempotency, same pattern as `bulk_upsert_scenes`)."""
    if not assignments:
        return
    records = [
        (
            _to_uuid(a.id),
            _to_uuid(a.scene_id),
            _to_uuid(a.asset_id),
            a.start_offset,
            a.loop,
            a.rationale,
        )
        for a in assignments
    ]
    await pool.executemany(
        """
        INSERT INTO background_assignments
            (id, scene_id, asset_id, start_offset, loop, rationale)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (scene_id) DO UPDATE
          SET asset_id     = EXCLUDED.asset_id,
              start_offset = EXCLUDED.start_offset,
              loop         = EXCLUDED.loop,
              rationale    = EXCLUDED.rationale
        """,
        records,
    )


async def get_background_assignments_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[BackgroundAssignmentRecord]:
    """Return all `background_assignments` rows for a video's scenes,
    joined through `scenes.video_id` (the table itself carries no
    video_id column — a background pick is scoped to `scene_id` only)."""
    rows = await pool.fetch(
        """
        SELECT ba.* FROM background_assignments ba
        JOIN scenes s ON s.id = ba.scene_id
        WHERE s.video_id = $1
        """,
        _to_uuid(video_id),
    )
    return [
        BackgroundAssignmentRecord(
            id=str(r["id"]),
            scene_id=str(r["scene_id"]),
            asset_id=str(r["asset_id"]),
            start_offset=r["start_offset"],
            loop=r["loop"],
            rationale=r["rationale"],
        )
        for r in rows
    ]


async def bulk_insert_emphasis_effects(
    pool: asyncpg.Pool, effects: list[EmphasisEffectRecord]
) -> None:
    """Insert `emphasis_effects` rows. `ON CONFLICT (id) DO NOTHING` — same
    pattern as `bulk_insert_cut_list_items`/`bulk_insert_sequence_color_
    adjustments`: callers generate a fresh id per row, so a rerun with new
    ids is additive, not a true UPSERT — a caller wanting a clean re-decide
    for an edit_plan should clear stale rows for it first (same documented
    caveat as the color-grading table)."""
    if not effects:
        return
    records = [
        (
            _to_uuid(e.id),
            _to_uuid(e.cut_list_item_id),
            e.effect_type,
            _to_json(e.parameters),
            e.rationale,
        )
        for e in effects
    ]
    await pool.executemany(
        """
        INSERT INTO emphasis_effects
            (id, cut_list_item_id, effect_type, parameters, rationale)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        ON CONFLICT (id) DO NOTHING
        """,
        records,
    )


async def get_emphasis_effects_for_edit_plan(
    pool: asyncpg.Pool, edit_plan_id: str
) -> list[EmphasisEffectRecord]:
    """Return all `emphasis_effects` rows for an edit plan's cut list,
    joined through `cut_list_items.edit_plan_id` (the table itself carries
    no edit_plan_id column, mirroring `sequence_color_adjustments`'
    cut_list_item_id-only keying)."""
    rows = await pool.fetch(
        """
        SELECT ee.* FROM emphasis_effects ee
        JOIN cut_list_items cli ON cli.id = ee.cut_list_item_id
        WHERE cli.edit_plan_id = $1
        """,
        _to_uuid(edit_plan_id),
    )
    result = []
    for r in rows:
        params = r["parameters"]
        result.append(
            EmphasisEffectRecord(
                id=str(r["id"]),
                cut_list_item_id=str(r["cut_list_item_id"]),
                effect_type=r["effect_type"],
                parameters=json.loads(params) if isinstance(params, str) else (params or {}),
                rationale=r["rationale"],
            )
        )
    return result


# ---------------------------------------------------------------------------
# Layer composites (A4 — Level-6 Editing Director's materialized mechanics
# record of a LAYER_COMPOSITE render, see migration `011_layer_composites.sql`
# and `shared/types.py::LayerCompositeRecord` for why this table exists
# separately from `background_assignments`/`emphasis_effects`.)
# ---------------------------------------------------------------------------


async def bulk_upsert_layer_composites(
    pool: asyncpg.Pool, composites: list[LayerCompositeRecord]
) -> None:
    """Upsert `layer_composites` rows, keyed on `(cut_list_item_id,
    layer_type)` (unique index in `011_layer_composites.sql`) — re-running
    `materialize_layer_composite_ops` for an edit_plan safely overwrites its
    own prior mechanics record for the same clip+layer_type rather than
    accumulating duplicates (rule 9/15, same idempotency pattern as
    `bulk_upsert_background_assignments`)."""
    if not composites:
        return
    records = [
        (
            _to_uuid(c.id),
            _to_uuid(c.cut_list_item_id),
            c.layer_type,
            _to_uuid(c.source_ref) if c.source_ref else None,
            _to_json(c.position) if c.position is not None else None,
            c.opacity,
            c.z_index,
        )
        for c in composites
    ]
    await pool.executemany(
        """
        INSERT INTO layer_composites
            (id, cut_list_item_id, layer_type, source_ref, position, opacity, z_index)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
        ON CONFLICT (cut_list_item_id, layer_type) DO UPDATE
          SET source_ref = EXCLUDED.source_ref,
              position   = EXCLUDED.position,
              opacity    = EXCLUDED.opacity,
              z_index    = EXCLUDED.z_index
        """,
        records,
    )


async def get_layer_composites_for_edit_plan(
    pool: asyncpg.Pool, edit_plan_id: str
) -> list[LayerCompositeRecord]:
    """Return all `layer_composites` rows for an edit plan's cut list,
    joined through `cut_list_items.edit_plan_id` (mirrors
    `get_emphasis_effects_for_edit_plan`'s join shape)."""
    rows = await pool.fetch(
        """
        SELECT lc.* FROM layer_composites lc
        JOIN cut_list_items cli ON cli.id = lc.cut_list_item_id
        WHERE cli.edit_plan_id = $1
        """,
        _to_uuid(edit_plan_id),
    )
    result = []
    for r in rows:
        pos = r["position"]
        result.append(
            LayerCompositeRecord(
                id=str(r["id"]),
                cut_list_item_id=str(r["cut_list_item_id"]),
                layer_type=r["layer_type"],
                source_ref=str(r["source_ref"]) if r["source_ref"] else None,
                position=json.loads(pos) if isinstance(pos, str) else pos,
                opacity=r["opacity"],
                z_index=r["z_index"],
            )
        )
    return result


# ---------------------------------------------------------------------------
# Pipeline alerts (CLAUDE.md "PART B — Hardening (resolved decisions)" -> B4
# "Observability beyond the OTHER-bucket alert"). Shared, level-agnostic
# table — see migrations/013_pipeline_alerts.sql for the DDL and rationale
# for NOT upserting (alerts are append-only events, not decision state).
# ---------------------------------------------------------------------------


async def insert_pipeline_alert(
    pool: asyncpg.Pool,
    video_id: str,
    level: int,
    alert_type: str,
    value: float | None,
    threshold: float | None,
) -> str:
    """Insert one `pipeline_alerts` row. Returns the new row's `id`.

    Called by a level's finalizer/updater when a threshold trips (B4's
    table: L2 pyannote failure rate, L4 `llm_unresolved_final` rate, L4
    confidence-escalation trigger rate, L4 `canonical_relation='OTHER'`
    rate). Never raises for "no alert needed" — callers only call this once
    they've already decided the threshold tripped; this function just
    writes the row.
    """
    row = await pool.fetchrow(
        """
        INSERT INTO pipeline_alerts (id, video_id, level, alert_type, value, threshold)
        VALUES (gen_random_uuid(), $1, $2, $3, $4, $5)
        RETURNING id
        """,
        _to_uuid(video_id),
        level,
        alert_type,
        value,
        threshold,
    )
    return str(row["id"])


# ---------------------------------------------------------------------------
# QA reports (PIPELINE ADDENDUM 2, item 1 — pipeline/level6/qa_agent.py).
# See migrations/015_qa_reports.sql for the DDL/rationale.
# ---------------------------------------------------------------------------


async def insert_qa_report(pool: asyncpg.Pool, report: QAReportRecord) -> str:
    """Insert one `qa_reports` row. Returns the new row's `id`.

    Called once per `run_qa_agent` invocation (the last step of
    `pipeline/level6/updater.py::run_level6`) — QA never edits an existing
    report or the edit_plan/cut_list_items it reviewed (rule 24: QA
    reports, it doesn't edit), so this is always a fresh insert, never an
    upsert."""
    row = await pool.fetchrow(
        """
        INSERT INTO qa_reports
            (id, edit_plan_id, video_id, status, deterministic_checks, llm_review, llm_status)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
        RETURNING id
        """,
        _to_uuid(report.id),
        _to_uuid(report.edit_plan_id),
        _to_uuid(report.video_id),
        report.status,
        _to_json(report.deterministic_checks),
        report.llm_review,
        report.llm_status,
    )
    return str(row["id"])


async def get_qa_report_for_edit_plan(
    pool: asyncpg.Pool, edit_plan_id: str
) -> QAReportRecord | None:
    """Return the most recent `qa_reports` row for *edit_plan_id*, or None
    if QA has never run for this plan. Most-recent-first since a plan can
    theoretically be QA'd more than once (e.g. a manual re-run after a
    render fix) — `created_at DESC` picks the latest verdict."""
    row = await pool.fetchrow(
        """
        SELECT * FROM qa_reports
        WHERE edit_plan_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        _to_uuid(edit_plan_id),
    )
    if row is None:
        return None
    return QAReportRecord(
        id=str(row["id"]),
        edit_plan_id=str(row["edit_plan_id"]),
        video_id=str(row["video_id"]),
        status=row["status"],
        deterministic_checks=(
            json.loads(row["deterministic_checks"])
            if isinstance(row["deterministic_checks"], str)
            else (row["deterministic_checks"] or {})
        ),
        llm_review=row["llm_review"],
        llm_status=row["llm_status"],
    )


# ---------------------------------------------------------------------------
# Human feedback (PIPELINE ADDENDUM 3, LEVEL 8, item 8b — migration
# 018_l8_human_feedback.sql). Holistic/qualitative feedback, distinct from
# `correction_events` (field-level "this value was X, should be Y").
# `category`/`sentiment` are closed enums (rule 28) so this is aggregatable
# later by L9's `reward_signals` (not built here — L8 only produces the raw
# signal). Appended per this repo's "only append to queries.py" convention
# so concurrent additions from other in-flight work don't conflict.
# ---------------------------------------------------------------------------


async def insert_human_feedback(pool: asyncpg.Pool, feedback: HumanFeedbackRecord) -> str:
    """Insert one `human_feedback` row. Returns the new row's `id`.

    Called by `scripts/log_human_feedback.py` (CLI entry point — this
    pipeline has no admin UI yet, same pattern as every other L4-L8 write
    path). Always a fresh insert, never an upsert — each piece of feedback
    is its own event, not a field to overwrite."""
    row = await pool.fetchrow(
        """
        INSERT INTO human_feedback
            (id, video_id, edit_plan_id, scene_id, sentiment, category,
             free_text, rating, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        """,
        _to_uuid(feedback.id),
        _to_uuid(feedback.video_id),
        _to_uuid(feedback.edit_plan_id),
        _to_uuid(feedback.scene_id),
        feedback.sentiment,
        feedback.category,
        feedback.free_text,
        feedback.rating,
        feedback.source,
    )
    return str(row["id"])


async def get_human_feedback_for_video(
    pool: asyncpg.Pool, video_id: str
) -> list[HumanFeedbackRecord]:
    """Return all `human_feedback` rows for *video_id*, newest first."""
    rows = await pool.fetch(
        """
        SELECT * FROM human_feedback
        WHERE video_id = $1
        ORDER BY created_at DESC
        """,
        _to_uuid(video_id),
    )
    return [
        HumanFeedbackRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            edit_plan_id=str(r["edit_plan_id"]) if r["edit_plan_id"] else None,
            scene_id=str(r["scene_id"]) if r["scene_id"] else None,
            sentiment=r["sentiment"],
            category=r["category"],
            free_text=r["free_text"],
            rating=r["rating"],
            source=r["source"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# LEVEL 7 -- EVALUATION (CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 7 --
# EVALUATION"). See migrations/017_l7_evaluation.sql for the DDL/rationale.
# ONLY new functions appended below this line for L7 -- nothing above this
# point in the file was modified (merge-safety: a parallel L8/L9
# implementation may also be appending new functions to this same file).
# ---------------------------------------------------------------------------


async def insert_evaluation_score(pool: asyncpg.Pool, score: EvaluationScoreRecord) -> str:
    """Insert one `evaluation_scores` row (7b). Returns the new row's `id`.

    Called once per `run_qa_agent` invocation, right after the existing
    intent-match LLM pass, when the rubric-scoring call succeeded. Always a
    fresh insert (same "QA reports, never edits" discipline as
    `insert_qa_report` — rule 24/27), never an upsert."""
    row = await pool.fetchrow(
        """
        INSERT INTO evaluation_scores
            (id, qa_report_id, edit_plan_id, intent_match, narrative_coherence,
             pacing_consistency, technical_cleanliness, rationale)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
        RETURNING id
        """,
        _to_uuid(score.id),
        _to_uuid(score.qa_report_id),
        _to_uuid(score.edit_plan_id),
        score.intent_match,
        score.narrative_coherence,
        score.pacing_consistency,
        score.technical_cleanliness,
        _to_json(score.rationale),
    )
    return str(row["id"])


async def get_evaluation_score_for_qa_report(
    pool: asyncpg.Pool, qa_report_id: str
) -> EvaluationScoreRecord | None:
    """Return the `evaluation_scores` row for *qa_report_id*, or None if
    the rubric-scoring pass never ran / never succeeded for that report
    (e.g. no OPENROUTER_API_KEY configured — same non-fatal-degrade
    contract as the existing intent-match pass)."""
    row = await pool.fetchrow(
        "SELECT * FROM evaluation_scores WHERE qa_report_id = $1 ORDER BY created_at DESC LIMIT 1",
        _to_uuid(qa_report_id),
    )
    if row is None:
        return None
    rationale = row["rationale"]
    return EvaluationScoreRecord(
        id=str(row["id"]),
        qa_report_id=str(row["qa_report_id"]),
        edit_plan_id=str(row["edit_plan_id"]),
        intent_match=row["intent_match"],
        narrative_coherence=row["narrative_coherence"],
        pacing_consistency=row["pacing_consistency"],
        technical_cleanliness=row["technical_cleanliness"],
        rationale=json.loads(rationale) if isinstance(rationale, str) else (rationale or {}),
    )


async def insert_llm_call_log(
    pool: asyncpg.Pool,
    *,
    video_id: str | None,
    level: int,
    stage: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cost_usd: float | None,
    latency_ms: int | None,
) -> str:
    """Insert one `llm_call_log` row (7c). Returns the new row's `id`.

    Called from `shared/llm_client.py::log_llm_call` right after every
    LLM-consuming call site across L4-L6 reads its `response.usage` object
    (the same object those call sites already log via `logger.info`) — this
    just also persists it durably. `video_id` may be None for a call with
    no clean single-video association; the column allows NULL."""
    row = await pool.fetchrow(
        """
        INSERT INTO llm_call_log
            (id, video_id, level, stage, model, prompt_tokens, completion_tokens,
             cost_usd, latency_ms)
        VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
        """,
        _to_uuid(video_id),
        level,
        stage,
        model,
        prompt_tokens,
        completion_tokens,
        cost_usd,
        latency_ms,
    )
    return str(row["id"])


async def get_llm_call_log_for_video(pool: asyncpg.Pool, video_id: str) -> list[LLMCallLogRecord]:
    """Return every `llm_call_log` row for *video_id*, most recent first —
    used by the dashboard / cost audits, not by any pipeline stage itself."""
    rows = await pool.fetch(
        "SELECT * FROM llm_call_log WHERE video_id = $1 ORDER BY created_at DESC",
        _to_uuid(video_id),
    )
    return [
        LLMCallLogRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]) if r["video_id"] else None,
            level=r["level"],
            stage=r["stage"],
            model=r["model"],
            prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"],
            cost_usd=r["cost_usd"],
            latency_ms=r["latency_ms"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Reward signals (PIPELINE ADDENDUM 3, LEVEL 9 -- migration
# 019_l9_reward_signals.sql). Appended per this file's "only append, never
# modify existing lines" convention -- `RewardSignalRecord` is imported
# locally inside each function below instead of touching the existing
# top-of-file import block (this file already has multiple concurrent
# append-only editors this session; a shared top import line is exactly the
# kind of edit that would collide).
# ---------------------------------------------------------------------------


async def upsert_reward_signal(
    pool: asyncpg.Pool,
    scope_type: str,
    scope_key: str,
    reward_score: float,
    sample_count: int,
) -> str:
    """UPSERT one `reward_signals` row on (scope_type, scope_key) -- 9a.
    Called by `scripts/compute_reward_signals.py`, safe to rerun. Unlike
    `pipeline_alerts` (append-only event log), `reward_signals` is a derived
    cache -- each call fully replaces the prior reward_score/sample_count
    for that scope, matching "recomputed periodically" rollup semantics."""
    row = await pool.fetchrow(
        """
        INSERT INTO reward_signals (id, scope_type, scope_key, reward_score, sample_count, updated_at)
        VALUES (gen_random_uuid(), $1, $2, $3, $4, NOW())
        ON CONFLICT (scope_type, scope_key) DO UPDATE
        SET reward_score = EXCLUDED.reward_score,
            sample_count = EXCLUDED.sample_count,
            updated_at = NOW()
        RETURNING id
        """,
        scope_type,
        scope_key,
        reward_score,
        sample_count,
    )
    return str(row["id"])


async def get_reward_signals(
    pool: asyncpg.Pool, scope_type: str | None = None
) -> list[RewardSignalRecord]:
    """Return `reward_signals` rows, optionally filtered by scope_type,
    highest reward_score first. Read-only -- used by dashboards/audits and
    by the 9b/9c call sites below (rule 29: never mutates behavior itself,
    only informs human-reviewed downstream steps)."""
    from shared.types import RewardSignalRecord

    if scope_type is not None:
        rows = await pool.fetch(
            "SELECT * FROM reward_signals WHERE scope_type = $1 ORDER BY reward_score DESC",
            scope_type,
        )
    else:
        rows = await pool.fetch("SELECT * FROM reward_signals ORDER BY reward_score DESC")
    return [
        RewardSignalRecord(
            id=str(r["id"]),
            scope_type=r["scope_type"],
            scope_key=r["scope_key"],
            reward_score=r["reward_score"],
            sample_count=r["sample_count"],
        )
        for r in rows
    ]


async def get_reward_signal(
    pool: asyncpg.Pool, scope_type: str, scope_key: str
) -> RewardSignalRecord | None:
    """Return the single `reward_signals` row for (scope_type, scope_key),
    or None if never computed / no underlying data. Used by 9b (client
    few-shot gate in `story_architect_runner.py`/`planner_runner.py`) and by
    `scripts/compute_reward_signals.py`'s own 9c alert-condition check."""
    from shared.types import RewardSignalRecord

    row = await pool.fetchrow(
        "SELECT * FROM reward_signals WHERE scope_type = $1 AND scope_key = $2",
        scope_type,
        scope_key,
    )
    if row is None:
        return None
    return RewardSignalRecord(
        id=str(row["id"]),
        scope_type=row["scope_type"],
        scope_key=row["scope_key"],
        reward_score=row["reward_score"],
        sample_count=row["sample_count"],
    )


async def get_all_evaluation_scores_with_video(pool: asyncpg.Pool) -> list[dict]:
    """Return every `evaluation_scores` row joined to its `video_id` (via
    `edit_plans`), for `scripts/compute_reward_signals.py`'s aggregation
    pass. Each dict: {edit_plan_id, video_id, intent_match,
    narrative_coherence, pacing_consistency, technical_cleanliness}. Read-
    only -- does not touch `evaluation_scores`' own schema/query functions."""
    rows = await pool.fetch(
        """
        SELECT es.edit_plan_id, ep.video_id, es.intent_match,
               es.narrative_coherence, es.pacing_consistency,
               es.technical_cleanliness
        FROM evaluation_scores es
        JOIN edit_plans ep ON ep.id = es.edit_plan_id
        """
    )
    return [dict(r) for r in rows]


async def get_all_human_feedback(pool: asyncpg.Pool) -> list[dict]:
    """Return every `human_feedback` row across all videos, for
    `scripts/compute_reward_signals.py`'s aggregation pass. Each dict:
    {id, video_id, edit_plan_id, sentiment, category, rating}. Read-only --
    does not touch `human_feedback`'s own schema/query functions."""
    rows = await pool.fetch(
        "SELECT id, video_id, edit_plan_id, sentiment, category, rating FROM human_feedback"
    )
    return [dict(r) for r in rows]


async def get_distinct_canonical_relations_for_video(pool: asyncpg.Pool, video_id: str) -> list[str]:
    """Return the distinct non-null `kg_edges.canonical_relation` values
    present for *video_id*. Used by `scripts/compute_reward_signals.py` to
    attribute a video-level evaluation/feedback contribution to every
    canonical relation that video's knowledge graph actually used -- a
    coarse, video-level (not scene/edge-level) attribution, documented as a
    known simplification in the script's own module docstring."""
    rows = await pool.fetch(
        "SELECT DISTINCT canonical_relation FROM kg_edges "
        "WHERE video_id = $1 AND canonical_relation IS NOT NULL",
        _to_uuid(video_id),
    )
    return [r["canonical_relation"] for r in rows]


async def get_video_ids_with_client(pool: asyncpg.Pool) -> dict[str, str]:
    """Return {video_id: client_id} for every video with a non-null
    `client_id` -- used by `scripts/compute_reward_signals.py` to scope
    evaluation/feedback contributions to scope_type='client' (9a/9b)."""
    rows = await pool.fetch("SELECT id, client_id FROM videos WHERE client_id IS NOT NULL")
    return {str(r["id"]): r["client_id"] for r in rows}


async def get_top_scoring_scenes_for_client(
    pool: asyncpg.Pool, client_id: str, limit: int = 5
) -> list[dict]:
    """Return up to *limit* `scenes` rows (canonical_scene_id + summary +
    emotional_arc) for videos belonging to *client_id*, ranked by that
    scene's video's average `evaluation_scores` dimension (highest first).

    This is the actual few-shot EXAMPLE content for 9b's
    `client_style_examples` prompt field -- `reward_signals`
    (scope_type='client') is only the gate that decides WHETHER to inject
    examples (via `sample_count`); this query supplies what to inject.
    Bounded per rule 23 -- callers pass limit=3-5, never unbounded."""
    rows = await pool.fetch(
        """
        SELECT s.canonical_scene_id, s.summary, s.emotional_arc, ev.avg_score
        FROM scenes s
        JOIN videos v ON v.id = s.video_id
        JOIN edit_plans ep ON ep.video_id = v.id
        JOIN (
            SELECT edit_plan_id,
                   (COALESCE(intent_match, 0) + COALESCE(narrative_coherence, 0)
                    + COALESCE(pacing_consistency, 0) + COALESCE(technical_cleanliness, 0)) / 4.0 AS avg_score
            FROM evaluation_scores
        ) ev ON ev.edit_plan_id = ep.id
        WHERE v.client_id = $1 AND s.summary IS NOT NULL
        ORDER BY ev.avg_score DESC
        LIMIT $2
        """,
        client_id,
        limit,
    )
    return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# Editor style profile learning (CLAUDE.md "PIPELINE ADDENDUM 4")
# ---------------------------------------------------------------------------


async def set_video_client_id(pool: asyncpg.Pool, video_id: str, client_id: str) -> None:
    """Tag `videos.id = video_id` with `client_id` -- used by
    `scripts/build_editor_profile.py` (Addendum 4, Phase 1) so that every
    exemplar video an editor supplies for style-profile extraction is
    attributed to that client, which is what makes L9's `compute_reward_
    signals.py` (`scope_type='client'`) and 9b's few-shot injection able to
    scope by `client_id` for future edits of that same video, and future
    videos tagged with the same `client_id`, downstream. Idempotent --
    re-running with the same (video_id, client_id) is a no-op UPDATE."""
    await pool.execute(
        "UPDATE videos SET client_id = $1 WHERE id = $2",
        client_id,
        _to_uuid(video_id),
    )
