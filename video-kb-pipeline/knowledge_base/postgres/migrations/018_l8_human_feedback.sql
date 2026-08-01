-- PIPELINE ADDENDUM 3, LEVEL 8: Human Feedback (item 8b).
-- Additive only (migrations/README.md policy): one new table, no changes to
-- existing columns. `correction_events` (migration 014) is field-level
-- ("this exact value was X, should be Y"); `human_feedback` is holistic/
-- qualitative ("this cut felt jarring", "I love how this scene flows") —
-- there's no specific field to diff for that kind of feedback. `category`
-- is a closed enum (rule 28) so `reward_signals` (L9, not built in this
-- migration) can aggregate it later. Scoped strictly to L8 per the task
-- split with the parallel L7 worktree (evaluation_scores/llm_call_log are
-- NOT added here — those belong to a separate L7 migration owned elsewhere).

BEGIN;

CREATE TABLE IF NOT EXISTS human_feedback (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  edit_plan_id  UUID REFERENCES edit_plans(id) ON DELETE CASCADE,   -- nullable — feedback can be on the video generally
  scene_id      UUID REFERENCES scenes(id) ON DELETE CASCADE,        -- nullable — feedback can be scene-scoped or general
  sentiment     TEXT NOT NULL CHECK (sentiment IN ('positive', 'negative', 'neutral')),
  category      TEXT NOT NULL CHECK (category IN
                  ('pacing', 'color', 'caption', 'speaker_attribution',
                   'narrative', 'music_audio', 'b_roll', 'other')),
  free_text     TEXT NOT NULL,
  rating         SMALLINT CHECK (rating BETWEEN 1 AND 5),  -- optional, nullable
  source        TEXT NOT NULL DEFAULT 'client',       -- client | internal_editor
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_human_feedback_video ON human_feedback(video_id);
CREATE INDEX IF NOT EXISTS idx_human_feedback_edit_plan ON human_feedback(edit_plan_id);

COMMIT;
