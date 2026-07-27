-- ─────────────────────────────────────────────────────────────────────────────
-- Migration 002: Deduplicate all tables and add UNIQUE constraints.
--
-- Full FK dependency tree (→ = references, no cascade unless noted):
--
--   chunks
--     ├── shots.chunk_id            (ON DELETE CASCADE)
--     │     ├── keyframes.shot_id   (ON DELETE CASCADE)
--     │     │     ├── frame_analyses.keyframe_id  (ON DELETE CASCADE)
--     │     │     └── searchable_facts.frame_id   (NO cascade — manual delete)
--     │     └── color_grades.shot_id              (NO cascade — manual delete)
--     └── transcript_segments.chunk_id            (NO cascade — manual delete)
--
-- Delete order per block: deepest child first, then parent.
-- ─────────────────────────────────────────────────────────────────────────────

-- ── 1. chunks ────────────────────────────────────────────────────────────────
-- Identify duplicate chunks: keep the one with the highest id per (video_id, chunk_index).
-- Must delete ALL descendants before deleting the duplicate chunk rows.

-- 1a. searchable_facts referencing keyframes of shots of duplicate chunks
DELETE FROM searchable_facts
WHERE frame_id IN (
    SELECT kf.id FROM keyframes kf
    JOIN shots sh ON sh.id = kf.shot_id
    WHERE sh.chunk_id IN (
        SELECT id FROM chunks
        WHERE id NOT IN (
            SELECT DISTINCT ON (video_id, chunk_index) id
            FROM chunks ORDER BY video_id, chunk_index, id DESC
        )
    )
);

-- 1b. color_grades referencing shots of duplicate chunks
DELETE FROM color_grades
WHERE shot_id IN (
    SELECT id FROM shots
    WHERE chunk_id IN (
        SELECT id FROM chunks
        WHERE id NOT IN (
            SELECT DISTINCT ON (video_id, chunk_index) id
            FROM chunks ORDER BY video_id, chunk_index, id DESC
        )
    )
);

-- 1c. keyframes (frame_analyses cascade from keyframes via ON DELETE CASCADE)
DELETE FROM keyframes
WHERE shot_id IN (
    SELECT id FROM shots
    WHERE chunk_id IN (
        SELECT id FROM chunks
        WHERE id NOT IN (
            SELECT DISTINCT ON (video_id, chunk_index) id
            FROM chunks ORDER BY video_id, chunk_index, id DESC
        )
    )
);

-- 1d. shots (keyframes already gone; color_grades already gone)
DELETE FROM shots
WHERE chunk_id IN (
    SELECT id FROM chunks
    WHERE id NOT IN (
        SELECT DISTINCT ON (video_id, chunk_index) id
        FROM chunks ORDER BY video_id, chunk_index, id DESC
    )
);

-- 1e. transcript_segments referencing duplicate chunks
DELETE FROM transcript_segments
WHERE chunk_id IN (
    SELECT id FROM chunks
    WHERE id NOT IN (
        SELECT DISTINCT ON (video_id, chunk_index) id
        FROM chunks ORDER BY video_id, chunk_index, id DESC
    )
);

-- 1f. duplicate chunks (all children gone)
DELETE FROM chunks
WHERE id NOT IN (
    SELECT DISTINCT ON (video_id, chunk_index) id
    FROM chunks ORDER BY video_id, chunk_index, id DESC
);

ALTER TABLE chunks
    ADD CONSTRAINT chunks_video_chunk_key UNIQUE (video_id, chunk_index);

-- ── 2. shots ─────────────────────────────────────────────────────────────────

-- 2a. searchable_facts referencing keyframes of duplicate shots
DELETE FROM searchable_facts
WHERE frame_id IN (
    SELECT kf.id FROM keyframes kf
    WHERE kf.shot_id IN (
        SELECT id FROM shots
        WHERE id NOT IN (
            SELECT DISTINCT ON (video_id, shot_index) id
            FROM shots ORDER BY video_id, shot_index, id DESC
        )
    )
);

-- 2b. color_grades referencing duplicate shots
DELETE FROM color_grades
WHERE shot_id IN (
    SELECT id FROM shots
    WHERE id NOT IN (
        SELECT DISTINCT ON (video_id, shot_index) id
        FROM shots ORDER BY video_id, shot_index, id DESC
    )
);

-- 2c. keyframes (frame_analyses cascade)
DELETE FROM keyframes
WHERE shot_id IN (
    SELECT id FROM shots
    WHERE id NOT IN (
        SELECT DISTINCT ON (video_id, shot_index) id
        FROM shots ORDER BY video_id, shot_index, id DESC
    )
);

-- 2d. duplicate shots
DELETE FROM shots
WHERE id NOT IN (
    SELECT DISTINCT ON (video_id, shot_index) id
    FROM shots ORDER BY video_id, shot_index, id DESC
);

ALTER TABLE shots
    ADD CONSTRAINT shots_video_shot_key UNIQUE (video_id, shot_index);

-- ── 3. keyframes ─────────────────────────────────────────────────────────────

-- 3a. searchable_facts referencing duplicate keyframes
DELETE FROM searchable_facts
WHERE frame_id IN (
    SELECT id FROM keyframes
    WHERE id NOT IN (
        SELECT DISTINCT ON (video_id, frame_index) id
        FROM keyframes ORDER BY video_id, frame_index, id DESC
    )
);

-- 3b. frame_analyses referencing duplicate keyframes (has ON DELETE CASCADE but be explicit)
DELETE FROM frame_analyses
WHERE keyframe_id IN (
    SELECT id FROM keyframes
    WHERE id NOT IN (
        SELECT DISTINCT ON (video_id, frame_index) id
        FROM keyframes ORDER BY video_id, frame_index, id DESC
    )
);

-- 3c. duplicate keyframes
DELETE FROM keyframes
WHERE id NOT IN (
    SELECT DISTINCT ON (video_id, frame_index) id
    FROM keyframes ORDER BY video_id, frame_index, id DESC
);

ALTER TABLE keyframes
    ADD CONSTRAINT keyframes_video_frame_key UNIQUE (video_id, frame_index);

-- ── 4. frame_analyses ────────────────────────────────────────────────────────
-- searchable_facts.frame_id references keyframes (not frame_analyses), so no child cleanup needed.

DELETE FROM frame_analyses
WHERE id NOT IN (
    SELECT DISTINCT ON (keyframe_id) id
    FROM frame_analyses ORDER BY keyframe_id, id DESC
);

ALTER TABLE frame_analyses
    ADD CONSTRAINT frame_analyses_keyframe_id_key UNIQUE (keyframe_id);

-- ── 5. transcript_segments ───────────────────────────────────────────────────

DELETE FROM transcript_segments
WHERE id NOT IN (
    SELECT DISTINCT ON (video_id, start_time, end_time) id
    FROM transcript_segments ORDER BY video_id, start_time, end_time, id DESC
);

ALTER TABLE transcript_segments
    ADD CONSTRAINT transcript_segments_video_time_key
    UNIQUE (video_id, start_time, end_time);

-- ── 6. searchable_facts ──────────────────────────────────────────────────────

DELETE FROM searchable_facts
WHERE id NOT IN (
    SELECT DISTINCT ON (video_id, fact_text) id
    FROM searchable_facts ORDER BY video_id, fact_text, id DESC
);

ALTER TABLE searchable_facts
    ADD CONSTRAINT searchable_facts_video_fact_key UNIQUE (video_id, fact_text);
