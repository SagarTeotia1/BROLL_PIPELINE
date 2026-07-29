-- Level-6 specialized action agents: Editing Director (cut list), Color
-- Grading (sequence-aware delta adjustments). See CLAUDE.md
-- "LEVEL 6 — SPECIALIZED ACTION AGENTS" for the full design.

CREATE TABLE IF NOT EXISTS cut_list_items (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edit_plan_id   UUID REFERENCES edit_plans(id) ON DELETE CASCADE,
  op_id          TEXT NOT NULL,               -- references EditOperation.op_id
  sequence_index INT NOT NULL,
  source_start   FLOAT NOT NULL,               -- snapped, may differ slightly from plan's start_time
  source_end     FLOAT NOT NULL,
  audio_lead_ms  INT DEFAULT 0,                -- J-cut offset
  video_lead_ms  INT DEFAULT 0,                -- L-cut offset
  transition     TEXT DEFAULT 'cut'
);

CREATE INDEX IF NOT EXISTS idx_cut_list_items_plan ON cut_list_items(edit_plan_id, sequence_index);

CREATE TABLE IF NOT EXISTS sequence_color_adjustments (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edit_plan_id     UUID REFERENCES edit_plans(id) ON DELETE CASCADE,
  cut_list_item_id UUID REFERENCES cut_list_items(id) ON DELETE CASCADE,
  base_parameters  JSONB NOT NULL,       -- from color_grades, unchanged
  sequence_delta   JSONB NOT NULL,       -- {param: adjustment} to harmonize with neighbors
  rationale        TEXT
);

CREATE INDEX IF NOT EXISTS idx_sequence_color_adjustments_plan ON sequence_color_adjustments(edit_plan_id);
