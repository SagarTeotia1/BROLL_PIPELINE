-- PIPELINE ADDENDUM 2, item 1: QA / Validation Agent.
-- Additive only (migrations/README.md policy): one new table, no changes to
-- existing columns. Per CLAUDE.md "1. QA / Validation Agent" — L6, tail —
-- deterministic checks (blackdetect/silencedetect/loudness/caption-drift/
-- color-clipping) plus one narrow Groq intent-match pass, aggregated into
-- one gate status. `qa_reports.id` is the `entity_id` a future
-- `correction_events` row for `entity_type='qa_report'` would reference
-- (Addendum 2, item 2) when a human later overrides a QA flag — that
-- reverse write path is not wired yet, this migration only adds the table
-- the QA Agent itself writes to.

BEGIN;

CREATE TABLE IF NOT EXISTS qa_reports (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edit_plan_id          UUID REFERENCES edit_plans(id) ON DELETE CASCADE,
  video_id              UUID REFERENCES videos(id) ON DELETE CASCADE,
  status                TEXT NOT NULL CHECK (status IN ('pass', 'warn', 'fail')),
  deterministic_checks  JSONB NOT NULL,   -- {black_frames: [...], silences: [...], loudness_range: {...}, caption_drift: [...], clipping: [...]}
  llm_review            TEXT,
  llm_status            TEXT CHECK (llm_status IN ('pass', 'warn', 'fail') OR llm_status IS NULL),
  created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_qa_reports_edit_plan ON qa_reports(edit_plan_id);
CREATE INDEX IF NOT EXISTS idx_qa_reports_video ON qa_reports(video_id);

COMMIT;
