# VIDEO UNDERSTANDING PIPELINE — COMPLETE PLAN

## STATUS: AWAITING USER REVIEW — DO NOT IMPLEMENT YET

---

## What We're Building

A 3-level video understanding pipeline that runs on Modal.com (L40s GPU) and produces a rich, queryable knowledge base combining a PostgreSQL graph database, Pinecone vector index, and Cloudflare R2 object store.

Three existing production-ready modules wire together:
- **ASHFS** (`/ashfs/`) — adaptive keyframe sampler (6-stage pipeline, DINOv2, shot detection)
- **color-grading** (`/color-grading/`) — 45-parameter color analysis engine (OpenCV, CuPy/NumPy)
- **face_analysis-model** (`/face_analysis-model/`) — ArcFace + HSEmotion + ByteTrack (ONNX, multithreaded)

---

## Pipeline Architecture

```
Video Input (R2 URL or local path)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  LEVEL 1 — PARALLEL (both run simultaneously)        │
│                                                      │
│  Branch A: ASHFS                                     │
│    shot detection → DINOv2 embeddings →              │
│    complexity scoring → keyframe selection →         │
│    hierarchical pruning → manifest.json              │
│                                                      │
│  Branch B: Whisper                                   │
│    audio extraction → Whisper Large-v3 →             │
│    word-level timestamps → transcript segments       │
│                                                      │
│  After both complete → align transcript to chunks    │
│  → write Level 1 KB records                         │
└──────────────────────────────────────────────────────┘
        │  (Level 1 complete)
        ▼
┌──────────────────────────────────────────────────────┐
│  LEVEL 2 — SEQUENTIAL (full video, not keyframes)   │
│                                                      │
│  Pass 1: Face Analysis (face_analysis-model)         │
│    SCRFD detect → ByteTrack → ArcFace identity →    │
│    HSEmotion → timeline events                      │
│    → who is in which frame/shot/chunk               │
│                                                      │
│  Pass 2: Color Grading (color-grading engine)        │
│    sample 1 representative frame per shot →          │
│    45-parameter analysis → grade recommendations     │
│                                                      │
│  → update KB with L2 records                        │
└──────────────────────────────────────────────────────┘
        │  (Level 2 complete)
        ▼
┌──────────────────────────────────────────────────────┐
│  LEVEL 3 — QWEN3-VL + KNOWLEDGE GRAPH               │
│                                                      │
│  vLLM serving Qwen3-VL 8B on L40s (48GB VRAM)       │
│  Batch keyframes (from ASHFS manifest)               │
│  Per frame: inject transcript snippet +              │
│    person IDs (from ArcFace) + prev scene summary    │
│  → structured JSON per frame (prompt V2 below)       │
│                                                      │
│  After all frames analysed:                          │
│  Build final knowledge graph                         │
│    nodes: Video, Chunk, Shot, Frame, Person,         │
│           Scene, Object, Theme, Emotion              │
│    edges: CONTAINS, APPEARS_IN, FOLLOWS,             │
│           CAUSES, HAS_EMOTION, RELATES_TO            │
│  → embed searchable_facts → push to Pinecone         │
│  → push graph to PostgreSQL (Apache AGE extension)  │
└──────────────────────────────────────────────────────┘
```

---

## New Folder Structure

Create new project root: `/video-kb-pipeline/` (sibling to existing module folders)

```
video-kb-pipeline/
│
├── CLAUDE.md                        ← this file
├── pyproject.toml                   ← all deps, one project
├── .env.example                     ← all required env vars documented
│
├── modal_app.py                     ← Modal app entry point, all Images + functions
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py              ← top-level: receive video → run L1→L2→L3 → return
│   │
│   ├── level1/
│   │   ├── __init__.py
│   │   ├── ashfs_runner.py          ← thin wrapper: calls ASHFS sampler_pipeline
│   │   ├── whisper_runner.py        ← audio extract + Whisper inference + word timestamps
│   │   └── aligner.py              ← merge transcript segments onto ASHFS chunks/shots
│   │
│   ├── level2/
│   │   ├── __init__.py
│   │   ├── face_runner.py           ← wraps face_analysis-model pipeline (headless CLI mode)
│   │   ├── color_runner.py          ← wraps color-grading engine, 1 frame per shot
│   │   └── updater.py              ← write L2 results to PostgreSQL KB
│   │
│   └── level3/
│       ├── __init__.py
│       ├── qwen_runner.py           ← vLLM client: batch keyframes → call Qwen3-VL
│       ├── context_builder.py      ← assemble per-frame context: transcript + pids + summary
│       ├── graph_builder.py        ← extract nodes/edges from all frame JSON → build graph
│       └── kb_finalizer.py         ← write graph to PG + embed facts to Pinecone
│
├── knowledge_base/
│   ├── __init__.py
│   │
│   ├── postgres/
│   │   ├── client.py               ← asyncpg connection pool (from env)
│   │   ├── schema.py               ← all CREATE TABLE DDL (run once at startup)
│   │   ├── queries.py              ← typed query functions, no inline SQL elsewhere
│   │   └── migrations/             ← versioned SQL migration files
│   │       └── 001_initial.sql
│   │
│   ├── pinecone_kb/
│   │   ├── client.py               ← Pinecone client init (from env)
│   │   └── indexer.py              ← upsert facts/scenes/persons to named namespaces
│   │
│   └── r2/
│       ├── client.py               ← boto3 S3-compat client pointing at R2 endpoint
│       └── uploader.py             ← upload keyframe PNGs, manifests, JSON exports
│
├── prompts/
│   ├── qwen_v2.py                  ← Qwen prompt V2 (full, final — see section below)
│   └── scene_summary.py            ← rolling scene summary prompt (1-2 sentence compress)
│
├── models/
│   ├── whisper_model.py            ← faster-whisper wrapper (beam=5, word_timestamps=True)
│   └── qwen_vllm.py               ← vLLM AsyncLLMEngine wrapper + batch dispatcher
│
├── shared/
│   ├── types.py                    ← ALL shared dataclasses (VideoMeta, ChunkRecord, etc.)
│   ├── config.py                   ← pydantic-settings: load + validate all env vars at import
│   └── utils.py                    ← ffmpeg audio extract, frame to bytes, retry decorator
│
└── modules/                        ← pip-installed editable or sys.path added
    ← ASHFS, color-grading, face_analysis-model imported directly from sibling dirs
    ← No copy/symlink — add parent to PYTHONPATH in Modal Image build
```

---

## Data Models (PostgreSQL Schema)

### Core Tables

```sql
-- ─── VIDEO ───────────────────────────────────────────────────────────────────
CREATE TABLE videos (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  path          TEXT NOT NULL,
  r2_key        TEXT,                        -- R2 object key for source video
  duration_s    FLOAT,
  fps           FLOAT,
  width         INT,
  height        INT,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ─── PROCESSING STATE ────────────────────────────────────────────────────────
CREATE TABLE processing_jobs (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  level         SMALLINT NOT NULL,           -- 1, 2, or 3
  status        TEXT NOT NULL,               -- queued | running | done | failed
  started_at    TIMESTAMPTZ,
  completed_at  TIMESTAMPTZ,
  error_msg     TEXT,
  meta          JSONB DEFAULT '{}'
);

-- ─── LEVEL 1: STRUCTURE ──────────────────────────────────────────────────────
CREATE TABLE chunks (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  chunk_index   INT NOT NULL,
  start_frame   INT,
  end_frame     INT,
  start_time    FLOAT,
  end_time      FLOAT
);

CREATE TABLE shots (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chunk_id      UUID REFERENCES chunks(id),
  video_id      UUID REFERENCES videos(id),
  shot_index    INT NOT NULL,
  start_frame   INT,
  end_frame     INT,
  start_time    FLOAT,
  end_time      FLOAT,
  shot_type     TEXT,                        -- hard_cut | soft | motion
  complexity    FLOAT
);

CREATE TABLE keyframes (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  shot_id       UUID REFERENCES shots(id),
  video_id      UUID REFERENCES videos(id),
  frame_index   INT NOT NULL,
  timestamp_s   FLOAT NOT NULL,
  r2_key        TEXT,                        -- PNG stored in R2
  selection_reason TEXT,                     -- common | unique | diversity
  dino_embedding vector(768),               -- pgvector: DINOv2 embedding
  siglip_embedding vector(1152)             -- pgvector: SigLIP2 embedding (nullable)
);

-- ─── LEVEL 1: TRANSCRIPT ─────────────────────────────────────────────────────
CREATE TABLE transcript_segments (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  chunk_id      UUID REFERENCES chunks(id),  -- nullable if not aligned yet
  text          TEXT NOT NULL,
  start_time    FLOAT NOT NULL,
  end_time      FLOAT NOT NULL,
  confidence    FLOAT,
  words         JSONB                        -- [{word, start, end, prob}]
);

-- ─── LEVEL 2: PERSONS ────────────────────────────────────────────────────────
CREATE TABLE persons (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  pid           TEXT NOT NULL,               -- P1, P2, ... (assigned by face pipeline)
  display_name  TEXT,                        -- from cast DB if registered
  arcface_embedding vector(512)             -- for cross-video identity matching
);

CREATE TABLE face_appearances (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  frame_index   INT NOT NULL,
  timestamp_s   FLOAT NOT NULL,
  person_id     UUID REFERENCES persons(id),
  track_id      INT,
  bbox          JSONB,                       -- {x,y,w,h}
  emotion       TEXT,
  emotion_conf  FLOAT
);

CREATE TABLE face_timeline_events (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  person_id     UUID REFERENCES persons(id),
  emotion       TEXT NOT NULL,
  start_time    FLOAT NOT NULL,
  end_time      FLOAT NOT NULL,
  confidence    FLOAT
);

-- ─── LEVEL 2: COLOR ──────────────────────────────────────────────────────────
CREATE TABLE color_grades (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  shot_id       UUID REFERENCES shots(id),
  frame_index   INT,
  timestamp_s   FLOAT,
  parameters    JSONB NOT NULL,              -- all 45 params: {name: {current, recommended, delta}}
  style_tags    TEXT[]
);

-- ─── LEVEL 3: QWEN ANALYSIS ──────────────────────────────────────────────────
CREATE TABLE frame_analyses (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  keyframe_id   UUID REFERENCES keyframes(id),
  video_id      UUID REFERENCES videos(id),
  scene_id      TEXT,
  scene_change  BOOLEAN,
  qwen_output   JSONB NOT NULL,              -- full Qwen V2 JSON response
  caption       TEXT,
  beat_type     TEXT,
  scene_mood    TEXT,
  tension_level TEXT,
  tags          TEXT[]
);

CREATE TABLE searchable_facts (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  frame_id      UUID REFERENCES keyframes(id),
  fact_text     TEXT NOT NULL,
  timestamp_s   FLOAT,
  embedding     vector(1536),               -- embedded by text-embedding-3-small or similar
  pinecone_id   TEXT                        -- mirrored Pinecone vector ID
);

-- ─── LEVEL 3: KNOWLEDGE GRAPH ────────────────────────────────────────────────
-- Using Apache AGE extension for Cypher queries on PostgreSQL
-- Graph name per video: 'video_{video_id_short}'

-- Also maintain adjacency tables for fast traversal without AGE:
CREATE TABLE kg_nodes (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  node_type     TEXT NOT NULL,               -- Video|Chunk|Shot|Frame|Person|Scene|Object|Theme
  ref_id        TEXT NOT NULL,               -- id of the referenced entity
  label         TEXT,
  properties    JSONB DEFAULT '{}',
  embedding     vector(1536)                -- for semantic node search
);

CREATE TABLE kg_edges (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  source_id     UUID REFERENCES kg_nodes(id),
  target_id     UUID REFERENCES kg_nodes(id),
  relation      TEXT NOT NULL,               -- CONTAINS|APPEARS_IN|FOLLOWS|CAUSES|HAS_EMOTION
  weight        FLOAT DEFAULT 1.0,
  properties    JSONB DEFAULT '{}'
);

-- ─── INDEXES ─────────────────────────────────────────────────────────────────
CREATE INDEX ON keyframes(video_id, timestamp_s);
CREATE INDEX ON face_appearances(video_id, frame_index);
CREATE INDEX ON face_timeline_events(video_id, person_id);
CREATE INDEX ON frame_analyses(video_id, scene_id);
CREATE INDEX ON searchable_facts(video_id);
CREATE INDEX ON kg_edges(source_id, relation);
CREATE INDEX ON kg_edges(target_id, relation);

-- pgvector indexes (IVFFlat for approximate nearest neighbour)
CREATE INDEX ON keyframes USING ivfflat (dino_embedding vector_l2_ops) WITH (lists = 100);
CREATE INDEX ON searchable_facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON persons USING ivfflat (arcface_embedding vector_cosine_ops) WITH (lists = 50);
```

---

## Pinecone Namespaces

```
Index name: video-kb

Namespaces:
  frames          dim=768   metric=cosine   → DINOv2 keyframe embeddings
                  metadata: {video_id, frame_index, timestamp, shot_id}

  facts           dim=1536  metric=cosine   → searchable_facts text embeddings
                  metadata: {video_id, frame_id, timestamp, tags}

  scenes          dim=1536  metric=cosine   → per-scene caption embeddings
                  metadata: {video_id, scene_id, beat_type, mood}

  persons         dim=512   metric=cosine   → ArcFace embeddings for cross-video search
                  metadata: {video_id, person_id, pid, display_name}
```

---

## Modal.com Deployment Strategy

### GPU Selection
- **L40s** (48 GB VRAM, Ada Lovelace) for ALL levels
- L1 + L2: L40s.medium or standard — don't need full VRAM, but consistent hardware
- L3 (vLLM): needs ≥24 GB VRAM → L40s standard (Qwen3-VL 8B in bfloat16 ≈ 16 GB)

### Modal Functions

```python
# Level 1A — ASHFS (spawned in parallel with 1B)
@app.function(gpu="L40S", timeout=3600, image=ashfs_image)
def run_ashfs(video_path: str, video_id: str) -> dict: ...

# Level 1B — Whisper (spawned in parallel with 1A)  
@app.function(gpu="L40S", timeout=3600, image=whisper_image)
def run_whisper(video_path: str, video_id: str) -> list[dict]: ...

# Level 2A — Face analysis
@app.function(gpu="L40S", timeout=7200, image=face_image)
def run_face_analysis(video_path: str, video_id: str, cast_db_r2_key: str) -> dict: ...

# Level 2B — Color grading (CPU or GPU)
@app.function(gpu="L40S", timeout=3600, image=color_image)
def run_color_grading(video_path: str, video_id: str, shot_frames: list) -> list[dict]: ...

# Level 3 — Qwen3-VL vLLM serving
@app.cls(gpu="L40S", timeout=7200, image=qwen_image)
class QwenAnalyser:
    @modal.enter()
    def load_model(self): ...          # start vLLM engine once, reuse across calls
    
    @modal.method()
    def analyse_batch(self, frames: list[dict]) -> list[dict]: ...

# Orchestrator — no GPU needed, coordinates the above
@app.function(timeout=14400)
def run_pipeline(video_id: str, video_r2_key: str): ...
```

### Modal Images

```python
# ashfs_image: torch, PySceneDetect, opencv-python, timm, numpy
# whisper_image: faster-whisper, ffmpeg-python, torch
# face_image: torch, onnxruntime-gpu, opencv-python, pyav, numpy
# color_image: numpy, cupy-cuda12x, opencv-python, scipy, scikit-learn, numba
# qwen_image: vllm>=0.6, transformers, accelerate, pillow, torch
# base_image (shared): asyncpg, pinecone, boto3, pydantic-settings, nanoid
```

---

## vLLM Settings for Qwen3-VL 8B

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-VL-7B-Instruct",   # use Qwen3-VL when released on HF
    dtype="bfloat16",
    tensor_parallel_size=1,                  # 1× L40s (48GB) — no TP needed for 8B
    gpu_memory_utilization=0.90,
    max_model_len=16384,                     # fits long prompt + image tokens
    max_num_seqs=8,                          # concurrent requests in flight
    enable_prefix_caching=True,             # cache shared prompt prefix across frames
    limit_mm_per_prompt={"image": 1},       # 1 image per frame request
    mm_processor_kwargs={
        "max_pixels": 1003520,              # ~1000×1000 effective res
        "min_pixels": 256 * 28 * 28,
    },
)
```

**Batch strategy:**
- Sort keyframes by timestamp
- Group into batches of 8 (= max_num_seqs)
- Submit all 8 concurrently via asyncio.gather
- Rolling scene summary updated after each shot boundary
- Pinecone upserts batched in groups of 100 after Qwen completes

---

## Qwen Prompt V2 (Final — Use This)

File: `prompts/qwen_v2.py`

This is the prompt with visual emotion inference, transcript context, and rolling scene summary. Full text documented in `prompts/qwen_v2.py`. Key differences from V1:
- `emotion_inferred`: VLM determines emotion from face/posture/transcript, NOT trusting automated HSEmotion label
- `emotion_source`: `"visual_read" | "transcript_tone" | "fallback_automated_guess" | "not_determinable"`
- `dialogue_subtitle`: grounded in actual transcript snippet, not just visual
- `causality` and `continuity` informed by rolling scene summary

HSEmotion output is still passed (as `automated emotion guess, low reliability, last resort only`) so VLM can use it as fallback when face is obscured/too small.

---

## Data Flow: How video_id Maps to Everything

```
video_id
  └── chunks[]
        ├── transcript_segments[]     (Whisper, aligned to chunk time range)
        └── shots[]
              ├── face_appearances[]  (who is present, which frames)
              ├── color_grades[]      (1 representative frame per shot)
              └── keyframes[]
                    └── frame_analyses[]
                          ├── searchable_facts[] (embedded → Pinecone)
                          └── story_beat, causality, relations...

Persons (P1, P2, ...) → face_timeline_events (emotion spans)
                       → face_appearances (frame-level)
                       → appear in frame_analyses.people[].pid

Knowledge Graph:
  Video → CONTAINS → Chunks → CONTAINS → Shots → CONTAINS → Keyframes
  Keyframes → HAS_ANALYSIS → FrameAnalyses
  Persons → APPEARS_IN → Shots (derived from face_appearances)
  Scenes → FOLLOWS → Scenes (causality chain)
  Themes → PRESENT_IN → Scenes
```

---

## Environment Variables Required

```bash
# PostgreSQL (you will provide)
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/dbname

# Pinecone (you will provide)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=video-kb
PINECONE_ENVIRONMENT=

# Cloudflare R2 (you will provide)
R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=

# Embedding model for text (searchable_facts → Pinecone)
OPENAI_API_KEY=          # if using text-embedding-3-small
# OR
EMBED_MODEL=local        # use sentence-transformers locally on Modal

# Modal (auto-configured via modal token set)
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=

# Optional: HuggingFace token for Qwen model download
HF_TOKEN=
```

---

## Implementation Order (After Review Approval)

### Step 1 — Scaffold & Config
- Create `video-kb-pipeline/` folder
- `pyproject.toml` with all deps
- `shared/config.py` (pydantic-settings, all env vars validated at import)
- `shared/types.py` (all shared dataclasses)

### Step 2 — Database
- `knowledge_base/postgres/schema.py` — all DDL
- `knowledge_base/postgres/client.py` — asyncpg pool
- `knowledge_base/postgres/queries.py` — typed query functions
- `knowledge_base/postgres/migrations/001_initial.sql`
- `knowledge_base/pinecone_kb/client.py` + `indexer.py`
- `knowledge_base/r2/client.py` + `uploader.py`
- Run migration, verify schema

### Step 3 — Level 1
- `models/whisper_model.py` — faster-whisper, word timestamps
- `pipeline/level1/ashfs_runner.py` — import ASHFS, call sampler_pipeline
- `pipeline/level1/whisper_runner.py` — extract audio via ffmpeg, run Whisper
- `pipeline/level1/aligner.py` — merge transcript onto chunk/shot time ranges
- Write L1 records to PostgreSQL

### Step 4 — Level 2
- `pipeline/level2/face_runner.py` — call face_analysis-model in headless mode
- `pipeline/level2/color_runner.py` — call color-grading engine per shot
- `pipeline/level2/updater.py` — write persons, appearances, grades to PostgreSQL
- Verify ArcFace PID mapping stored and queryable

### Step 5 — Level 3
- `prompts/qwen_v2.py` — full prompt template
- `prompts/scene_summary.py` — rolling summary compressor
- `models/qwen_vllm.py` — AsyncLLMEngine with settings above
- `pipeline/level3/context_builder.py` — assemble per-frame context
- `pipeline/level3/qwen_runner.py` — batch dispatch + collect results
- `pipeline/level3/graph_builder.py` — extract nodes/edges from all JSON
- `pipeline/level3/kb_finalizer.py` — write to PostgreSQL + Pinecone

### Step 6 — Modal Integration
- `modal_app.py` — all Images, @app.function decorators, orchestrator
- Level 1 parallel spawn (asyncio.gather on two Modal functions)
- Level 2 sequential after Level 1 done
- Level 3 sequential after Level 2 done
- Full end-to-end test on a short video (~5 min)

### Step 7 — Audit & Hardening
- Spawn agents to review each module for bugs
- Type checking: pyright strict pass
- Error handling: every external call wrapped (R2 upload retry, PG connection retry, vLLM timeout)
- Verify no business logic in Modal function bodies (delegate to pipeline modules)
- Verify Qwen output parsed and validated with pydantic before DB write
- Load test: 60-min video

---

## Key Engineering Rules (Non-Negotiable)

1. **Env validated at import** — `shared/config.py` crashes at boot if any var missing, not at runtime
2. **No inline SQL** — all queries in `knowledge_base/postgres/queries.py`
3. **No raw Qwen calls outside `models/qwen_vllm.py`**
4. **Pydantic model for every Qwen response** — never write raw `qwen_output["field"]` without schema
5. **R2 key format**: `videos/{video_id}/frames/{frame_index:06d}.png` — consistent, queryable
6. **PID stability**: person PIDs assigned by face_analysis-model must be stable across the same video — use track_id + cast_db actor_id as source of truth, remap to P1/P2/... only at KB write time
7. **Rolling summary**: update after every completed shot (not every frame) to avoid prompt bloat
8. **Pinecone upserts**: batch in 100s, never one-by-one
9. **All Modal functions idempotent**: re-running a level on same video_id is safe (UPSERT not INSERT)
10. **Apache AGE**: install as PG extension, create graph per video — `SELECT * FROM cypher('video_{id}', $$...$$) AS (...)` pattern

---

## Open Questions (Resolve Before Implementation)

1. **Qwen3-VL model ID**: Is `Qwen/Qwen3-VL-8B` available on HuggingFace, or do we use `Qwen/Qwen2.5-VL-7B-Instruct`? → User to confirm
2. **Cast DB for face analysis**: Do you have a pre-built cast SQLite DB with actor embeddings, or does the pipeline auto-assign PIDs to unknown faces?
3. **Apache AGE vs pure relational graph**: Do you want full Cypher query support (Apache AGE), or is the adjacency table (`kg_nodes` + `kg_edges`) sufficient for your queries?
4. **Text embedding model**: OpenAI `text-embedding-3-small` (1536 dim, API cost) or local `BAAI/bge-large-en-v1.5` (1024 dim, free, Modal)?
5. **Color grading sampling**: 1 frame per shot (representative, fast) or every N-th frame across full video? → Currently planning 1 per shot.
6. **Video input source**: Videos already in R2, or uploaded locally → pipeline uploads to R2 first?
