-- Level-5 planning: EditPlan (Selection & Scoring pass + Sequencing & Pacing
-- pass). See CLAUDE.md "LEVEL 5 — PLANNING (Editing Director's Plan)" for the
-- full design.

CREATE TABLE IF NOT EXISTS edit_plans (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id            UUID REFERENCES videos(id) ON DELETE CASCADE,
  storyline_id        UUID REFERENCES storylines(id),   -- pins to a specific final version
  user_prompt         TEXT NOT NULL,
  target_duration_s   FLOAT,
  platform            TEXT,                              -- reel | full_cut | youtube | ...
  status              TEXT NOT NULL DEFAULT 'draft'
                       CHECK (status IN ('draft', 'reviewed', 'applied', 'superseded')),
  version             INT NOT NULL DEFAULT 1,
  operations          JSONB NOT NULL,                     -- ordered EditOperation[]
  achieved_duration_s FLOAT,                              -- computed post-hoc, never trusted from LLM
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (video_id, version)
);

CREATE INDEX IF NOT EXISTS idx_edit_plans_video_status ON edit_plans(video_id, status);

-- one row per user round of feedback, so "re-plan" is a diff, not a regenerate
CREATE TABLE IF NOT EXISTS edit_plan_revisions (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edit_plan_id    UUID REFERENCES edit_plans(id) ON DELETE CASCADE,
  user_feedback   TEXT NOT NULL,
  diff_operations JSONB NOT NULL,     -- only what changed, not the full plan again
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_edit_plan_revisions_plan ON edit_plan_revisions(edit_plan_id);
