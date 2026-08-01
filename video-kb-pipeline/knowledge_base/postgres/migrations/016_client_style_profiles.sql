-- CLAUDE.md "PIPELINE ADDENDUM 2" -> "3. Client Style Profiles".
-- Additive only (migrations/README.md policy): one new nullable column on
-- the existing `videos` table (existing rows get NULL, no backfill), plus
-- one new table. Read contract: L5 Pass B + L6 Color Grading Agent + L6
-- Caption/Text Overlay Agent read `client_style_profiles` as a SOFT PRIOR
-- only (rule 26) — a video/client with no row here behaves exactly as
-- today, zero prompt change.

BEGIN;

-- No client/user system exists yet in this schema (see CLAUDE.md's own
-- note) — client_id is a free-text external identifier, not a FK to a
-- users table. Nullable: existing videos and any video processed without a
-- known client stay NULL forever, which is a fully supported state, not a
-- migration gap.
ALTER TABLE videos ADD COLUMN IF NOT EXISTS client_id TEXT;

CREATE TABLE IF NOT EXISTS client_style_profiles (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id          TEXT NOT NULL,
  caption_style      JSONB DEFAULT '{}',
  pacing_preference  TEXT,
  brand_colors       JSONB DEFAULT '{}',
  default_platform   TEXT,
  notes              TEXT,
  created_at         TIMESTAMPTZ DEFAULT NOW(),
  updated_at         TIMESTAMPTZ DEFAULT NOW()
);

-- One profile per client, v1 (no versioning yet — see CLAUDE.md item 3's
-- "simplest thing that could work" framing). This is also what makes
-- `upsert_client_style_profile` a true upsert (ON CONFLICT (client_id)).
CREATE UNIQUE INDEX IF NOT EXISTS idx_client_style_profiles_client_id
  ON client_style_profiles(client_id);

CREATE INDEX IF NOT EXISTS idx_videos_client_id ON videos(client_id);

COMMIT;
