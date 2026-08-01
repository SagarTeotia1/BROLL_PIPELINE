-- Level-6 Editing Director: A4 (layering/compositing mechanics) materialized
-- ops. See CLAUDE.md "PIPELINE ADDENDUM — Motion Graphics/Compositing +
-- Hardening" -> A4 ("Layering/compositing mechanics").
--
-- This table is written by editing_director.py's
-- `materialize_layer_composite_ops` — it derives one row per cut_list_item
-- that has a resolvable `background_assignments` pick (via that clip's
-- SELECT_CLIP op's scene_id) or an explicit PiP/overlay request already
-- present in the finalized edit_plan.operations. It is NEVER written by the
-- Compositing Agent (background_selector.py/emphasis_selector.py) directly —
-- those only decide WHAT (background_assignments/emphasis_effects); this
-- table is the mechanics layer's own record of HOW it rendered that
-- decision, kept for audit/idempotency, distinct from A5's emphasis_effects
-- (zoom/highlight) which stay mechanics-free by design (their render-time
-- parameters — start_rect/end_rect/target_bbox timing — are computed
-- on-the-fly from emphasis_effects.parameters + cut_list_items timing +
-- videos.width/height and are not persisted separately).

CREATE TABLE IF NOT EXISTS layer_composites (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cut_list_item_id  UUID REFERENCES cut_list_items(id) ON DELETE CASCADE,
  layer_type        TEXT NOT NULL,  -- background_swap | pip | overlay
  source_ref        UUID,           -- background_assignments.id or another cut_list_item_id
  position           JSONB,
  opacity           FLOAT DEFAULT 1.0,
  z_index           INT DEFAULT 0,
  CONSTRAINT layer_composites_layer_type_check CHECK (layer_type IN ('background_swap', 'pip', 'overlay'))
);

CREATE INDEX IF NOT EXISTS idx_layer_composites_cut_list_item ON layer_composites(cut_list_item_id);

-- One materialized layer-composite row per (cut_list_item_id, layer_type) —
-- re-running materialization for the same edit_plan overwrites its own
-- prior mechanics record rather than accumulating duplicates (rule 9/15,
-- same idempotency pattern as background_assignments' scene_id unique
-- index).
CREATE UNIQUE INDEX IF NOT EXISTS idx_layer_composites_cut_item_type_unique
  ON layer_composites(cut_list_item_id, layer_type);
