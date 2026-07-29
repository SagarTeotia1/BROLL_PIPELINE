-- Level-4 reasoning: Grounding Agent (speaker resolution + relation
-- canonicalization + fact dedup) and Story Architect Agent (canonical scene
-- timeline + storyline synthesis). See CLAUDE.md "LEVEL 4 — REASONING" for
-- the full design and finalization contract.

ALTER TABLE kg_edges ADD COLUMN IF NOT EXISTS canonical_relation TEXT;

ALTER TABLE searchable_facts ADD COLUMN IF NOT EXISTS superseded_by UUID REFERENCES searchable_facts(id);

-- Terminal vs. pending unresolved distinction (finalization contract #3).
ALTER TABLE speaker_turns DROP CONSTRAINT IF EXISTS speaker_turns_resolution_method_check;
ALTER TABLE speaker_turns ADD CONSTRAINT speaker_turns_resolution_method_check
  CHECK (resolution_method IN ('face_majority', 'single_candidate', 'llm_tiebreak', 'llm_unresolved_final', 'unresolved'));

-- Versioned canonical relation ontology — not frozen. If the OTHER bucket
-- exceeds ~5% of a video's edges, that's a signal to review and bump version.
CREATE TABLE IF NOT EXISTS ontology_relations (
  version    INT NOT NULL,
  relation   TEXT NOT NULL,
  PRIMARY KEY (version, relation)
);

INSERT INTO ontology_relations (version, relation) VALUES
  (1, 'SPEAKS_TO'), (1, 'ADDRESSES'), (1, 'DISCUSSES'), (1, 'EXPLAINS'),
  (1, 'REACTS_TO'), (1, 'AGREES_WITH'), (1, 'DISAGREES_WITH'), (1, 'CRITICIZES'),
  (1, 'PRAISES'), (1, 'ASKS'), (1, 'ANSWERS'), (1, 'HAS_EMOTION'),
  (1, 'GESTURES_TOWARD'), (1, 'LOOKS_AT'), (1, 'IS_SEATED_NEAR'), (1, 'IS_POSITIONED_NEAR'),
  (1, 'WEARS'), (1, 'HOLDS'), (1, 'CAUSES'), (1, 'FOLLOWS'),
  (1, 'PART_OF_SCENE'), (1, 'APPEARS_IN'), (1, 'CONTAINS_OBJECT'), (1, 'IN_LOCATION'),
  (1, 'ILLUSTRATES_THEME'), (1, 'MENTIONS'), (1, 'OTHER')
ON CONFLICT (version, relation) DO NOTHING;

-- Canonical scene timeline — first-class, replaces the loose kg_nodes Scene grouping.
CREATE TABLE IF NOT EXISTS scenes (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id            UUID REFERENCES videos(id) ON DELETE CASCADE,
  canonical_scene_id  TEXT NOT NULL,
  discarded_aliases   TEXT[] DEFAULT '{}',
  start_time          FLOAT NOT NULL,
  end_time            FLOAT NOT NULL,
  participants        TEXT[] DEFAULT '{}',       -- array of pid
  summary             TEXT,
  emotional_arc       TEXT,
  causal_link_to_next TEXT,
  usability_score     FLOAT,                     -- deterministic, from color_grades + transcript confidence (L5 Part 1 fix #4)
  UNIQUE (video_id, canonical_scene_id)
);

CREATE INDEX IF NOT EXISTS idx_scenes_video_time ON scenes(video_id, start_time);

-- Versioned, immutable-once-final storyline document — L5's sole input contract.
CREATE TABLE IF NOT EXISTS storylines (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id    UUID REFERENCES videos(id) ON DELETE CASCADE,
  version     INT NOT NULL DEFAULT 1,
  status      TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'final')),
  title        TEXT,
  synopsis     TEXT,
  cast_members JSONB DEFAULT '{}',
  beats        JSONB DEFAULT '[]',
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (video_id, version)
);

CREATE INDEX IF NOT EXISTS idx_storylines_video_status ON storylines(video_id, status);

-- Cheap corrections that don't require re-running L4 (finalization contract #6).
CREATE TABLE IF NOT EXISTS scene_overrides (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scene_id    UUID REFERENCES scenes(id) ON DELETE CASCADE,
  field       TEXT NOT NULL,          -- e.g. 'summary', 'participants'
  new_value   JSONB NOT NULL,
  reason      TEXT,
  created_by  TEXT,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS storyline_overrides (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  storyline_id UUID REFERENCES storylines(id) ON DELETE CASCADE,
  field        TEXT NOT NULL,
  new_value    JSONB NOT NULL,
  reason       TEXT,
  created_by   TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);
