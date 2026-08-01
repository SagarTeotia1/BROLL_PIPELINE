-- Level-6 Compositing Agent: A3 (background/B-roll selection) + A5-decision
-- half (zoom/highlight emphasis WHAT, not the FFmpeg mechanics — those land
-- in a later phase of editing_director.py). See CLAUDE.md "PIPELINE
-- ADDENDUM — Motion Graphics/Compositing + Hardening" -> A3, A5.

-- A3 — scene-level, not cut-order-level: a background pick is a property
-- of the canonical `scenes` row (L4), reusable across any edit plan built
-- from that scene, hence no edit_plan_id/cut_list_item_id here.
CREATE TABLE IF NOT EXISTS background_assignments (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scene_id     UUID REFERENCES scenes(id) ON DELETE CASCADE,
  asset_id     UUID REFERENCES stock_assets(id),
  start_offset FLOAT DEFAULT 0,
  loop         BOOLEAN DEFAULT false,
  rationale    TEXT
);

CREATE INDEX IF NOT EXISTS idx_background_assignments_scene ON background_assignments(scene_id);

-- One background pick per scene per run; re-running background_selector.py
-- for a video overwrites its own prior pick rather than accumulating
-- duplicates (rule 9/15 — safe UPSERT on rerun).
CREATE UNIQUE INDEX IF NOT EXISTS idx_background_assignments_scene_unique
  ON background_assignments(scene_id);

-- A5 (decision half) — cut-order-level: FK to cut_list_items, same as
-- sequence_color_adjustments, since WHAT to zoom/highlight on is inherently
-- about a specific clip in the finalized cut, not the source shot.
CREATE TABLE IF NOT EXISTS emphasis_effects (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cut_list_item_id  UUID REFERENCES cut_list_items(id) ON DELETE CASCADE,
  effect_type       TEXT NOT NULL,   -- zoom | highlight
  parameters        JSONB NOT NULL,
  rationale         TEXT,
  CONSTRAINT emphasis_effects_effect_type_check CHECK (effect_type IN ('zoom', 'highlight'))
);

CREATE INDEX IF NOT EXISTS idx_emphasis_effects_cut_list_item ON emphasis_effects(cut_list_item_id);
