from __future__ import annotations

import json
import logging
import uuid

import asyncpg

from shared.types import (
    ChunkRecord,
    ColorGradeRecord,
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
    SearchableFactRecord,
    ShotRecord,
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
