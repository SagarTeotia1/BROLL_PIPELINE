-- A2 — Stock asset library + embedding index. Standalone infrastructure
-- under knowledge_base/stock_assets/, built once, queried later by L6's
-- Compositing Agent (A3, built separately). See CLAUDE.md "PIPELINE
-- ADDENDUM — Motion Graphics/Compositing + Hardening" → A2.
--
-- license_type is present and filterable now even though enforcement is
-- deferred — see CLAUDE.md "Before building A3: stock licensing".

CREATE TABLE IF NOT EXISTS stock_assets (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source        TEXT NOT NULL,        -- pexels | storyblocks | internal
  external_id   TEXT,
  description   TEXT,
  tags          TEXT[],
  license_type  TEXT NOT NULL,        -- see licensing note in CLAUDE.md A2
  embedding     vector(1024),         -- BAAI/bge-large-en-v1.5, same model as searchable_facts/kg_nodes
  r2_cache_key  TEXT
);

CREATE INDEX IF NOT EXISTS idx_stock_assets_embedding
  ON stock_assets USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- Lookups used by the indexer (upsert de-dup) and the Compositing Agent
-- (filter by source/license before or alongside the vector search).
CREATE UNIQUE INDEX IF NOT EXISTS idx_stock_assets_source_external_id
  ON stock_assets (source, external_id) WHERE external_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_stock_assets_license_type ON stock_assets (license_type);
