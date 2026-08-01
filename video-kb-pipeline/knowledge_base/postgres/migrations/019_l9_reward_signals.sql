-- CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 9 -- REWARD & PUNISHMENT" (9a).
-- Additive only (migrations/README.md policy): one new table, no changes to
-- any existing table/column. `reward_signals` is a computed rollup over
-- `evaluation_scores` (L7) + `human_feedback` (L8) -- it does not store raw
-- feedback itself (that stays in those two tables, untouched by this
-- migration per the task constraint "don't touch evaluation_scores/
-- llm_call_log/human_feedback schemas").
--
-- Recomputed periodically by `scripts/compute_reward_signals.py` (a script,
-- not a live trigger, per the spec's own "not raw storage... recomputed
-- periodically" framing) -- UPSERT on (scope_type, scope_key), so reruns are
-- safe (rule 15's "idempotent" discipline extended to L9).
--
-- scope_type is a free-text discriminator (same pattern as
-- pipeline_alerts.alert_type / emphasis_effects.effect_type elsewhere in
-- this schema) -- not a CHECK-constrained enum, since the spec names four
-- illustrative scopes ('canonical_relation' | 'pacing_style' | 'color_style'
-- | 'client') without closing the set to exactly those four.
--
-- rule 29: reward_signals never mutates pipeline behavior directly -- it is
-- read-only input to 9b (prompt few-shot injection) and 9c (alert trigger),
-- both human-reviewed downstream steps. Nothing in this migration or its
-- consumers writes back to evaluation_scores/human_feedback/kg_edges/etc.

BEGIN;

CREATE TABLE IF NOT EXISTS reward_signals (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scope_type    TEXT NOT NULL,      -- 'canonical_relation' | 'pacing_style' | 'color_style' | 'client'
  scope_key     TEXT NOT NULL,      -- e.g. the relation name, or a client_id
  reward_score  FLOAT NOT NULL,     -- confidence-shrunk rolling average, -1..1
  sample_count  INT NOT NULL,       -- raw (pre-shrinkage) number of contributing evaluation/feedback rows
  updated_at    TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (scope_type, scope_key)
);

CREATE INDEX IF NOT EXISTS idx_reward_signals_scope_type ON reward_signals(scope_type);

COMMIT;
