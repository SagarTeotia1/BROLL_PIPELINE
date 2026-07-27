-- Deduplicate frame_analyses: keep newest row per keyframe_id, drop rest.
-- Then add UNIQUE constraint so ON CONFLICT (keyframe_id) works.

-- Step 1: delete duplicate rows (keep the latest id per keyframe_id)
DELETE FROM frame_analyses
WHERE id NOT IN (
    SELECT DISTINCT ON (keyframe_id) id
    FROM frame_analyses
    ORDER BY keyframe_id, id DESC
);

-- Step 2: also deduplicate searchable_facts per (video_id, fact_text) while we're here
DELETE FROM searchable_facts
WHERE id NOT IN (
    SELECT DISTINCT ON (video_id, fact_text) id
    FROM searchable_facts
    ORDER BY video_id, fact_text, id DESC
);

-- Step 3: add UNIQUE constraint on frame_analyses.keyframe_id
ALTER TABLE frame_analyses
    ADD CONSTRAINT frame_analyses_keyframe_id_key UNIQUE (keyframe_id);
