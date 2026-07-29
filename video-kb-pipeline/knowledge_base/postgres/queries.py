from __future__ import annotations

import json
import logging
import uuid

import asyncpg

from shared.types import (
    ChunkRecord,
    ColorGradeRecord,
    CutListItemRecord,
    EditPlanRecord,
    EditPlanRevisionRecord,
    FaceAppearanceRecord,
    FaceTimelineEvent,
    FrameAnalysisRecord,
    KeyframeRecord,
    KGEdge,
    KGNode,
    LevelStatus,
    PersonRecord,
    ProcessingJob,
    QwenFrameOutput,
    SceneRecord,
    SearchableFactRecord,
    SequenceColorAdjustmentRecord,
    ShotRecord,
    SpeakerTurnRecord,
    StorylineRecord,
    TranscriptSegment,
    VideoMeta,
)

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
    """Insert or update a video record.  Returns the row ``id``."""
    row = await pool.fetchrow(
        """
        INSERT INTO videos (id, path, r2_key, duration_s, fps, width, height)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE
          SET path       = EXCLUDED.path,
              r2_key     = EXCLUDED.r2_key,
              duration_s = EXCLUDED.duration_s,
              fps        = EXCLUDED.fps,
              width      = EXCLUDED.width,
              height     = EXCLUDED.height
        RETURNING id
        """,
        _to_uuid(meta.id),
        meta.path,
        meta.r2_key,
        meta.duration_s,
        meta.fps,
        meta.width,
        meta.height,
    )
    return str(row["id"])


async def get_video(pool: asyncpg.Pool, video_id: str) -> VideoMeta | None:
    """Return the `videos` row for *video_id*, or None if not found.

    Used by Level-6's Editing Director (`export_xml`/`render_direct`) to
    resolve `fps` and the source file reference (`r2_key` or `path`) needed
    to build an FCPXML asset or drive the FFmpeg render.
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
            fact.pinecone_id,
        )
        for fact in facts
    ]
    await pool.executemany(
        """
        INSERT INTO searchable_facts
            (id, video_id, frame_id, fact_text, timestamp_s, embedding, pinecone_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (video_id, fact_text) DO UPDATE
          SET timestamp_s = EXCLUDED.timestamp_s,
              embedding   = EXCLUDED.embedding,
              pinecone_id = EXCLUDED.pinecone_id
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
    pinecone_id: str | None = None,
) -> None:
    """Update the embedding (and optionally pinecone_id) for a searchable fact.

    Called after OpenAI embeddings are computed so that the vector is persisted
    back to Postgres in addition to being upserted to Pinecone.

    Args:
        pool:        An open asyncpg connection pool.
        fact_id:     The fact's unique identifier (nanoid string).
        embedding:   The 1024-dim float vector to persist.
        pinecone_id: The Pinecone vector ID used for this fact, if known.
    """
    await pool.execute(
        """
        UPDATE searchable_facts
        SET embedding   = $2,
            pinecone_id = COALESCE($3, pinecone_id)
        WHERE id = $1
        """,
        _to_uuid(fact_id),
        embedding,
        pinecone_id,
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
            pinecone_id=r["pinecone_id"],
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
    return [
        SceneRecord(
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
        )
        for r in rows
    ]


async def bulk_upsert_scenes(pool: asyncpg.Pool, scenes: list[SceneRecord]) -> None:
    """Upsert canonical scene rows, keyed on (video_id, canonical_scene_id).

    Safe to call repeatedly for the same video_id (rule 9 / rule 15 —
    scenes are UPSERT, not INSERT, so a re-run of the Story Architect Agent
    on the same video overwrites its own prior draft output rather than
    accumulating duplicates).
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
        )
        for s in scenes
    ]
    await pool.executemany(
        """
        INSERT INTO scenes
            (id, video_id, canonical_scene_id, discarded_aliases, start_time,
             end_time, participants, summary, emotional_arc, causal_link_to_next,
             usability_score)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (video_id, canonical_scene_id) DO UPDATE
          SET discarded_aliases   = EXCLUDED.discarded_aliases,
              start_time          = EXCLUDED.start_time,
              end_time            = EXCLUDED.end_time,
              participants        = EXCLUDED.participants,
              summary             = EXCLUDED.summary,
              emotional_arc       = EXCLUDED.emotional_arc,
              causal_link_to_next = EXCLUDED.causal_link_to_next,
              usability_score     = EXCLUDED.usability_score
        """,
        records,
    )


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
    )


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
# Cut list items (Level-6 Editing Director — deterministic snapped cut list)
# ---------------------------------------------------------------------------


async def bulk_insert_cut_list_items(
    pool: asyncpg.Pool, items: list[CutListItemRecord]
) -> None:
    """Bulk-insert `cut_list_items` rows for a finalized `edit_plan`.

    Follows the `bulk_insert_speaker_turns` pattern: `ON CONFLICT (id) DO
    NOTHING` — callers generate a fresh `id` per item (see
    `pipeline.level6.editing_director.snap_cut_points`), so a re-run that
    reuses the same ids is a no-op rather than a duplicate row, while a
    re-run with freshly generated ids still requires the caller to clear
    stale rows for the edit_plan_id first if idempotent overwrite is
    desired (mirrors the additive-only discipline used elsewhere in L6).
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
