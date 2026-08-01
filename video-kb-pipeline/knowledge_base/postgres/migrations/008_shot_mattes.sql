-- Level-2 background matting (A1): per-shot alpha matte, run in parallel
-- with color grading. See CLAUDE.md "PIPELINE ADDENDUM" -> A1 for the full
-- design.

CREATE TABLE IF NOT EXISTS shot_mattes (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  shot_id       UUID REFERENCES shots(id) ON DELETE CASCADE,
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  r2_key        TEXT NOT NULL,   -- alpha-matte video or PNG sequence
  model_version TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shot_mattes_video ON shot_mattes(video_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_shot_mattes_shot_model ON shot_mattes(shot_id, model_version);
