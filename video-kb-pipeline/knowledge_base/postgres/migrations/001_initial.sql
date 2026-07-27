-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Videos
CREATE TABLE IF NOT EXISTS videos (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  path          TEXT NOT NULL,
  r2_key        TEXT,
  duration_s    FLOAT,
  fps           FLOAT,
  width         INT,
  height        INT,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Processing jobs
CREATE TABLE IF NOT EXISTS processing_jobs (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  level         SMALLINT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'queued',
  started_at    TIMESTAMPTZ,
  completed_at  TIMESTAMPTZ,
  error_msg     TEXT,
  meta          JSONB DEFAULT '{}'
);

-- Chunks (ASHFS output, time segments)
CREATE TABLE IF NOT EXISTS chunks (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  chunk_index   INT NOT NULL,
  start_frame   INT,
  end_frame     INT,
  start_time    FLOAT,
  end_time      FLOAT
);

-- Shots (within chunks)
CREATE TABLE IF NOT EXISTS shots (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chunk_id      UUID REFERENCES chunks(id) ON DELETE CASCADE,
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  shot_index    INT NOT NULL,
  start_frame   INT,
  end_frame     INT,
  start_time    FLOAT,
  end_time      FLOAT,
  shot_type     TEXT,
  complexity    FLOAT
);

-- Keyframes (selected by ASHFS)
CREATE TABLE IF NOT EXISTS keyframes (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  shot_id       UUID REFERENCES shots(id) ON DELETE CASCADE,
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  frame_index   INT NOT NULL,
  timestamp_s   FLOAT NOT NULL,
  r2_key        TEXT,
  selection_reason TEXT,
  dino_embedding vector(768),
  siglip_embedding vector(1152)
);

-- Transcript segments (Whisper output)
CREATE TABLE IF NOT EXISTS transcript_segments (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  chunk_id      UUID REFERENCES chunks(id),
  text          TEXT NOT NULL,
  start_time    FLOAT NOT NULL,
  end_time      FLOAT NOT NULL,
  confidence    FLOAT,
  words         JSONB DEFAULT '[]'
);

-- Persons (identity from ArcFace)
CREATE TABLE IF NOT EXISTS persons (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  pid           TEXT NOT NULL,
  display_name  TEXT,
  arcface_embedding vector(512)
);

-- Face appearances (per-frame)
CREATE TABLE IF NOT EXISTS face_appearances (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  frame_index   INT NOT NULL,
  timestamp_s   FLOAT NOT NULL,
  person_id     UUID REFERENCES persons(id),
  track_id      INT,
  bbox          JSONB,
  emotion       TEXT,
  emotion_conf  FLOAT
);

-- Face timeline events (emotion spans)
CREATE TABLE IF NOT EXISTS face_timeline_events (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  person_id     UUID REFERENCES persons(id),
  emotion       TEXT NOT NULL,
  start_time    FLOAT NOT NULL,
  end_time      FLOAT NOT NULL,
  confidence    FLOAT
);

-- Color grade records (per shot)
CREATE TABLE IF NOT EXISTS color_grades (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  shot_id       UUID REFERENCES shots(id),
  frame_index   INT,
  timestamp_s   FLOAT,
  parameters    JSONB NOT NULL,
  style_tags    TEXT[]
);

-- Frame analyses (Qwen3-VL output)
CREATE TABLE IF NOT EXISTS frame_analyses (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  keyframe_id   UUID REFERENCES keyframes(id) ON DELETE CASCADE,
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  scene_id      TEXT,
  scene_change  BOOLEAN,
  qwen_output   JSONB NOT NULL,
  caption       TEXT,
  beat_type     TEXT,
  scene_mood    TEXT,
  tension_level TEXT,
  tags          TEXT[]
);

-- Searchable facts (from Qwen searchable_facts field)
CREATE TABLE IF NOT EXISTS searchable_facts (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  frame_id      UUID REFERENCES keyframes(id),
  fact_text     TEXT NOT NULL,
  timestamp_s   FLOAT,
  embedding     vector(1024),
  pinecone_id   TEXT
);

-- Knowledge graph nodes
CREATE TABLE IF NOT EXISTS kg_nodes (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  node_type     TEXT NOT NULL,
  ref_id        TEXT NOT NULL,
  label         TEXT,
  properties    JSONB DEFAULT '{}',
  embedding     vector(1024)
);

-- Knowledge graph edges
CREATE TABLE IF NOT EXISTS kg_edges (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id) ON DELETE CASCADE,
  source_id     UUID REFERENCES kg_nodes(id) ON DELETE CASCADE,
  target_id     UUID REFERENCES kg_nodes(id) ON DELETE CASCADE,
  relation      TEXT NOT NULL,
  weight        FLOAT DEFAULT 1.0,
  properties    JSONB DEFAULT '{}'
);

-- Unique constraints required for ON CONFLICT upsert logic in queries.py
-- ADD CONSTRAINT IF NOT EXISTS is not valid PostgreSQL syntax; use DO blocks.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_persons_video_pid') THEN
    ALTER TABLE persons ADD CONSTRAINT uq_persons_video_pid UNIQUE (video_id, pid);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_jobs_video_level') THEN
    ALTER TABLE processing_jobs ADD CONSTRAINT uq_jobs_video_level UNIQUE (video_id, level);
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_kg_nodes_video_type_ref') THEN
    ALTER TABLE kg_nodes ADD CONSTRAINT uq_kg_nodes_video_type_ref UNIQUE (video_id, node_type, ref_id);
  END IF;
END $$;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_keyframes_video_ts ON keyframes(video_id, timestamp_s);
CREATE INDEX IF NOT EXISTS idx_face_appearances_video ON face_appearances(video_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_face_events_video_person ON face_timeline_events(video_id, person_id);
CREATE INDEX IF NOT EXISTS idx_frame_analyses_video_scene ON frame_analyses(video_id, scene_id);
CREATE INDEX IF NOT EXISTS idx_searchable_facts_video ON searchable_facts(video_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(source_id, relation);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(target_id, relation);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_shots_video ON shots(video_id, shot_index);
CREATE INDEX IF NOT EXISTS idx_persons_video_pid ON persons(video_id, pid);
CREATE INDEX IF NOT EXISTS idx_transcript_video_time ON transcript_segments(video_id, start_time);
CREATE INDEX IF NOT EXISTS idx_jobs_video_level ON processing_jobs(video_id, level);

-- pgvector IVFFlat indexes (approximate nearest neighbour)
CREATE INDEX IF NOT EXISTS idx_keyframes_dino ON keyframes USING ivfflat (dino_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_facts_embedding ON searchable_facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_persons_arcface ON persons USING ivfflat (arcface_embedding vector_cosine_ops) WITH (lists = 50);
