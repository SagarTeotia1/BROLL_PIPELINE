-- Observability beyond the OTHER-bucket log line. See CLAUDE.md "PART B —
-- Hardening (resolved decisions)" -> B4 ("Observability beyond the
-- OTHER-bucket alert"). Each level's finalizer/updater writes a row here
-- when a threshold trips, per B4's table:
--   L2 pyannote failure rate, L4 llm_unresolved_final rate,
--   L4 confidence-escalation trigger rate, L4 canonical_relation='OTHER'
--   rate (already logged as a warning in grounding_runner.py — this
--   migration gives that warning a real row to land in, per B4's "already
--   specced" note).
--
-- Shared, level-agnostic table (not one table per level) — alert_type is
-- the free-text discriminator, same pattern as emphasis_effects.effect_type
-- / layer_composites.layer_type elsewhere in this schema. Purely additive:
-- no existing table/column touched.

CREATE TABLE IF NOT EXISTS pipeline_alerts (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id    UUID REFERENCES videos(id) ON DELETE CASCADE,
  level       SMALLINT NOT NULL,
  alert_type  TEXT NOT NULL,
  value       FLOAT,
  threshold   FLOAT,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_alerts_video ON pipeline_alerts(video_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_alerts_type ON pipeline_alerts(alert_type);

-- No uniqueness constraint: unlike layer_composites/background_assignments
-- (mechanics/decision state that must not accumulate duplicates on rerun),
-- an alert is an append-only observability event — rerunning a level that
-- trips the same threshold again is a genuine second occurrence worth its
-- own row (e.g. tracking whether an OTHER-bucket problem got better or
-- worse across reprocessing runs), not something to upsert away.
