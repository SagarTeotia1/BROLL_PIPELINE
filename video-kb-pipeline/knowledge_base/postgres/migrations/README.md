# Migration policy

- Additive/backward-compatible only. New tables, new nullable/defaulted
  columns. No `DROP COLUMN`, no `ALTER ... SET NOT NULL` against an existing
  populated column without a backfill step in the same migration file.
- Each migration runs inside one transaction. Failure rolls back the whole
  file — never partial-apply. No migration disables this.
- Numbered sequentially, never reordered or edited after merge. A mistake
  gets a new migration that corrects it, not an edit to an old file.
- Backfill note: `005_l4_reasoning.sql` added `kg_edges.canonical_relation`.
  Videos processed before this migration have it `NULL` — L4's Grounding
  Agent relation-canonicalization step must be re-run per-video to backfill.
  `finalizer.py`'s quality gate must not require `canonical_relation` on
  pre-005 videos unless they are explicitly reprocessed.
