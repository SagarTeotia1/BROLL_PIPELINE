-- CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 7 -- EVALUATION" (7a-7d).
-- Additive only (migrations/README.md policy): two new tables, no changes
-- to any existing table/column. Scope note: this migration deliberately
-- covers ONLY L7 (`evaluation_scores`, `llm_call_log`) -- L8's
-- `human_feedback` and L9's `reward_signals` are owned by a parallel
-- implementation and intentionally NOT included here (see CLAUDE.md's
-- combined `017_l7_l8_l9_evaluation.sql` naming in the spec; split here to
-- keep the two implementation efforts from colliding on one file).
--
-- 7b: `evaluation_scores` -- one row per `qa_reports` row, 1:1 extension.
-- Rubric scoring (intent_match/narrative_coherence/pacing_consistency/
-- technical_cleanliness) is ADDITIVE to `qa_reports.status` (rule 27) --
-- this table is never read by the gate that decides pass/warn/fail.
--
-- 7c: `llm_call_log` -- durable cost/latency tracking (closes audit gap
-- #9). Queryable columns, not a `processing_jobs.meta` JSONB blob (B2 was
-- speced that way but never implemented; this supersedes that plan with a
-- real table, per 7c's own "queryable columns > buried JSON" rationale).

BEGIN;

CREATE TABLE IF NOT EXISTS evaluation_scores (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  qa_report_id          UUID REFERENCES qa_reports(id) ON DELETE CASCADE,
  edit_plan_id          UUID REFERENCES edit_plans(id) ON DELETE CASCADE,
  intent_match          FLOAT,
  narrative_coherence   FLOAT,
  pacing_consistency    FLOAT,
  technical_cleanliness FLOAT,
  rationale             JSONB NOT NULL DEFAULT '{}',
  created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluation_scores_qa_report ON evaluation_scores(qa_report_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_scores_edit_plan ON evaluation_scores(edit_plan_id);

CREATE TABLE IF NOT EXISTS llm_call_log (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id          UUID REFERENCES videos(id) ON DELETE CASCADE,
  level             SMALLINT NOT NULL,
  stage             TEXT NOT NULL,
  model             TEXT NOT NULL,
  prompt_tokens     INT,
  completion_tokens INT,
  cost_usd          FLOAT,
  latency_ms        INT,
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_call_log_video_level ON llm_call_log(video_id, level);

COMMIT;
