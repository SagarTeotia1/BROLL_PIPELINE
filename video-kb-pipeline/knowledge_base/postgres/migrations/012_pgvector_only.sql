-- B7 (Vendor consolidation — CLAUDE.md "PART B — Hardening (resolved
-- decisions)" -> B7): drop Pinecone entirely, consolidate all vector search
-- on Postgres pgvector. This migration adds the columns/indexes that used to
-- be Pinecone-only data:
--   - `scenes.embedding`      — replaces the Pinecone `scenes` namespace's
--                               canonical-scene variant (L4 propagation).
--   - `storylines.embedding`  — replaces the planned Pinecone `storylines`
--                               namespace (per-video synopsis search).
--   - `kg_nodes` ivfflat index — CLAUDE.md's original schema notes claimed
--                               this index already existed; it did not
--                               (only `searchable_facts` had one). Added here
--                               so `kg_nodes.embedding` (used for L3's loose
--                               per-frame Scene node) is actually queryable
--                               via cosine search, not just stored.
-- All changes are additive/nullable — no backfill required (migration
-- policy B5), existing rows simply have NULL embedding until re-embedded.

ALTER TABLE scenes ADD COLUMN IF NOT EXISTS embedding vector(1024);
CREATE INDEX IF NOT EXISTS idx_scenes_embedding
  ON scenes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

ALTER TABLE storylines ADD COLUMN IF NOT EXISTS embedding vector(1024);
CREATE INDEX IF NOT EXISTS idx_storylines_embedding
  ON storylines USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_embedding
  ON kg_nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- `searchable_facts.pinecone_id` is renamed, not dropped (RENAME COLUMN
-- preserves existing data — safe/additive per migration policy B5, unlike a
-- DROP). It was the Pinecone vector id; now it's just a legacy per-fact
-- external id, no longer written by any live code path but kept for
-- historical rows / manual lookups.
DO $$ BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'searchable_facts' AND column_name = 'pinecone_id'
  ) THEN
    ALTER TABLE searchable_facts RENAME COLUMN pinecone_id TO legacy_vector_id;
  END IF;
END $$;
