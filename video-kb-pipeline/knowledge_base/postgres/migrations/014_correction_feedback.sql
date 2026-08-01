-- Addendum 2, item 2: correction feedback loop.
-- Additive only (migrations/README.md policy): one new table, no changes to
-- existing columns. scene_overrides/storyline_overrides (005) are unchanged;
-- this migration only gives the codebase a durable, structured record of
-- every correction made through them (and through edit_plan_revisions),
-- which nothing wrote to before this migration.

BEGIN;

CREATE TABLE IF NOT EXISTS correction_events (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id           UUID REFERENCES videos(id) ON DELETE CASCADE,
  level              SMALLINT NOT NULL,          -- which level's output was corrected (2, 4, 5, 6)
  entity_type        TEXT NOT NULL,               -- speaker_turn | canonical_relation | scene | storyline | edit_plan | qa_report
  entity_id          UUID NOT NULL,
  field              TEXT NOT NULL,
  original_value     JSONB NOT NULL,
  corrected_value    JSONB NOT NULL,
  correction_source  TEXT NOT NULL,               -- client | internal_editor | qa_agent_flag
  reason             TEXT,
  created_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_correction_events_video ON correction_events(video_id);
CREATE INDEX IF NOT EXISTS idx_correction_events_level_entity ON correction_events(level, entity_type);

COMMIT;
