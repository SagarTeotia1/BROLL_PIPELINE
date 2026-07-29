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
│  LEVEL 2 — 3-WAY PARALLEL (full video, not keyframes)│
│  [IMPLEMENTED — differs from original plan below]   │
│                                                      │
│  Pass A: Face Analysis (face_analysis-model)         │
│    SCRFD detect → ByteTrack → ArcFace identity →    │
│    HSEmotion → timeline events                      │
│    → who is in which frame/shot/chunk               │
│                                                      │
│  Pass B: Color Grading (color-grading engine)        │
│    sample 1 representative frame per shot →          │
│    45-parameter analysis → grade recommendations     │
│                                                      │
│  Pass C: Stage 0 — Speaker Diarization (pyannote)    │
│    audio → speaker turns (cluster_label, start,end)  │
│    num_speakers hint = len(cast_list) when known     │
│    → fuse_diarization_with_faces(): cluster→pid via  │
│      face co-presence majority vote (Pass A output)  │
│    → speaker_turns table, resolution_method per turn:│
│      single_candidate | face_majority | unresolved   │
│    (non-fatal — pyannote failure never blocks L2)    │
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

  facts           dim=1024  metric=cosine   → searchable_facts text embeddings
                  metadata: {video_id, frame_id, timestamp, tags}
                  [IMPLEMENTED — knowledge_base/pinecone_kb/indexer.py upsert_facts()]
                  NOTE: dim is 1024 (BAAI/bge-large-en-v1.5, local), not 1536 as
                  originally planned — corrected here to match actual code.

  scenes          dim=1024  metric=cosine   → per-scene caption embeddings
                  metadata: {video_id, scene_id, beat_type, mood}
                  [IMPLEMENTED — upsert_scenes(), currently fed from L3's loose
                  per-frame Scene grouping. L4 re-points this at the canonical
                  `scenes` table instead — see LEVEL 4 → Neo4j + Pinecone propagation.]

  persons         dim=512   metric=cosine   → ArcFace embeddings for cross-video search
                  metadata: {video_id, person_id, pid, display_name}
                  [PLANNED — not yet implemented in indexer.py]

  storylines      dim=1024  metric=cosine   → per-video synopsis embedding (one vector/video)
                  metadata: {video_id, storyline_id, version, title}
                  [PLANNED — new namespace, added by L4 for cross-video plot/theme search]
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

---

## LEVEL 4 — REASONING (Grounding Agent + Story Architect Agent)

### STATUS: IMPLEMENTED — not yet run end-to-end (no live pipeline run has
executed L4 yet; code is complete and self-consistent per a reconciliation
pass, but real-world runtime bugs — model-response shape drift, rate limits,
etc. — should be expected on the first real run, same pattern as L2's
diarization rollout)

L1–L3 are implemented and running (Neo4j is the actual graph store in use, not
Apache AGE — that part of the original plan above is superseded). Stage 0
(speaker diarization) was pulled *into* L2 rather than being its own level —
see the updated L2 box above.

**LLM provider: Groq, not Anthropic.** All prompt/tool-schema text below was
originally written against Anthropic's Messages API shape (`input_schema` as
a top-level tool field). The actual implementation in `pipeline/level4/`
targets **Groq** (`groq` Python SDK, OpenAI-compatible `chat.completions`
interface) running **`qwen/qwen3.6-27b`** — Groq's current Qwen model (131k
context, function calling; `qwen3-32b` deprecated 2026-06-17). Groq's Qwen
lineup is single-tier right now, so both the Grounding Agent (originally
speced as cheap-tier) and the Story Architect Agent (originally speced as
strong-tier) run the **same model** — there is no cheap/strong split on Groq
today the way there was on Anthropic (Haiku vs Opus). Tool schemas in code
use the OpenAI-compatible `{"type": "function", "function": {"name", ...,
"parameters": {...}}}` shape, not the Anthropic `input_schema` shape shown
in the prompt blocks below — the blocks are conceptually accurate (what
fields, what decision logic) but the exact JSON schema wrapper differs from
what's actually in `prompts/grounding_speaker.py` /
`prompts/grounding_relation.py` / `prompts/story_architect.py`. Config:
`shared/config.py` → `GROQ_API_KEY`, `L4_GROUNDING_MODEL`,
`L4_STORY_ARCHITECT_MODEL` (both default `qwen/qwen3.6-27b`).

### Why L4 exists

L1–L3 produce raw, per-frame, sometimes-contradictory data:
- `speaker_turns` — most turns resolved to a person via face co-presence, but
  some are `unresolved` (no face visible during that turn) or low-confidence
  `face_majority` (multiple faces present, majority vote only).
- `kg_edges.relation` — Qwen invents a new free-text predicate almost every
  frame (`IS_LAUGHING_WHILE_ADDRESSING`, `EXPLAINS_TACTICAL_ADJUSTMENTS_IN_
  FOOTBALL_MATCHES_USING_THE_SENE`, ...). 200+ distinct strings observed on a
  single video. Not a queryable ontology.
- `frame_analyses.scene_id` — inconsistent naming per frame
  (`studio_podcast_01`, `scene_001`, `interview_studio_01`,
  `world_cup_special_opening` all appear in the same video). No canonical
  scene timeline exists yet, only loosely-grouped `Scene` kg_nodes.
- `searchable_facts` — near-duplicate facts across adjacent frames.

L4 does not re-derive anything L1–L3 already got right. It resolves what's
genuinely ambiguous, canonicalizes vocabulary, and compresses per-frame data
into a per-scene / per-video narrative structure. Its output is the *only*
thing L5 (Planning) reads — L5 never touches raw `frame_analyses` directly.

### Two agents, not one monolith

Splitting into two agents (rather than one big reasoning pass) because the
two jobs have opposite cost/context shapes:

| | Grounding Agent | Story Architect Agent |
|---|---|---|
| Model tier | cheap/fast (Haiku-tier) | strong reasoning (Sonnet/Opus-tier) |
| Call volume | high — one call per unresolved item | low — one call per scene (~50/video) |
| Context per call | small, single item | rolling summary + one scene's frames |
| Job type | closed-set classification / disambiguation | narrative synthesis |
| Ground truth constraint | must pick from closed cast set or null | must not invent facts absent from input |

A single large-context pass over all 1700+ frame_analyses per video would be
slower, more expensive, and less consistent than this split — same reasoning
that already shaped L3's rolling-scene-summary design.

### Agent 1 — Grounding Agent

Three sub-tasks, same agent, different prompt templates, all "resolve the
ambiguous case from a closed set" in shape:

**1a. Speaker turn resolution** — only for `speaker_turns` rows where
`resolution_method IN ('unresolved', 'face_majority')` **and**
`resolution_method != 'llm_tiebreak'` (idempotency — never re-decide a turn
this agent already resolved on a prior run of the same `video_id`).

**Batched, not one-call-per-turn.** All unresolved turns for a video go in
ONE call as a list — cost/latency scale with videos, not with unresolved-turn
count. Uses structured tool-output (function-calling schema), not
prompt-only "output JSON" — far fewer parse failures/retries than free-text
JSON instructions.

Added signal vs. the first draft: **track_id continuity**. A face can be
absent in the exact turn window but present on the same `track_id`
immediately before/after it — that's real identity evidence deterministic
fusion didn't use because it only looked inside `[start_time, end_time]`.
Feed a ±3s widened window of `face_appearances` (track_id + pid) alongside
the strict in-window `face_presence`.

```
SYSTEM PROMPT — Grounding Agent: Speaker Resolution
────────────────────────────────────────────────────
You resolve a BATCH of ambiguous speaker turns for one video's knowledge
base in a single pass. You do not re-derive anything already certain — you
only decide what deterministic fusion could not.

Input:
  cast: [{pid, display_name}, ...]   — the ONLY valid non-null answers
  turns: [
    {
      turn_id, cluster_label, start_time, end_time,
      transcript_snippet: transcript text overlapping [start-2s, end+2s],
      visual_context: Qwen frame people[] entries in-window
        (pid, gaze, pose, action, story_role) — may be empty,
      face_presence: {pid: frame_count} strictly inside [start, end] — may be
        empty (that is WHY this turn is unresolved),
      face_presence_widened: {pid: frame_count} inside [start-3s, end+3s] on
        a continuous track_id — identity evidence just outside the strict
        window, may still be empty
    },
    ...
  ]

For EACH turn, decide independently (do not let one turn's answer bias
another) using this priority order:
  1. Explicit self-reference, or being addressed by name, in transcript_snippet
  2. Turn-taking pattern — who was addressed/asked a question in the prior turn
  3. face_presence_widened — same track_id continuous across the turn boundary
  4. Visual cues — gaze direction toward camera/mic, "speaking"/"addressing"
     action tags in visual_context
  5. If signals conflict or none are present, return null for that turn. Do
     not guess.

Return one result per turn_id, via the resolve_speaker_turns tool call.
Never output a pid not present in `cast`. Never invent a new speaker.
```

Tool/function schema (structured output, not parsed free text):
```json
{
  "name": "resolve_speaker_turns",
  "parameters": {
    "type": "object",
    "properties": {
      "resolutions": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "turn_id": {"type": "string"},
            "person_id": {"type": ["string", "null"]},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"}
          },
          "required": ["turn_id", "person_id", "confidence", "reasoning"]
        }
      }
    },
    "required": ["resolutions"]
  }
}
```

Write-back: `UPDATE speaker_turns SET person_id=<result>, confidence=<result>,
resolution_method='llm_tiebreak' WHERE id=<turn_id>` — only when
`person_id` is non-null; leave `unresolved` rows alone otherwise (never
downgrade a value the deterministic pass already set).

**1b. Relation canonicalization** — batch job, not per-edge. Cluster the
distinct `kg_edges.relation` strings for a video by embedding similarity
first (cheap, local `BAAI/bge-large-en-v1.5`, already used for
`searchable_facts`), then one LLM call for ALL clusters in that video
(structured tool output again, one array in one call).

Ontology is **versioned, not frozen** — `ontology_relations` table
(`version INT, relation TEXT`) seeded with the list below. If the `OTHER`
bucket exceeds ~5% of a video's edges, that's a signal to review and mint a
new canonical relation + bump the version — not silently let `OTHER` become
a junk drawer that grows forever.

```
SYSTEM PROMPT — Grounding Agent: Relation Canonicalization
────────────────────────────────────────────────────────────
You assign ONE canonical relation type to each cluster of near-duplicate
free-text relation strings extracted from one video's frame analysis.

Input:
  clusters: [{cluster_id, raw_relations: [...]}, ...]
  ontology: the current allowed canonical relations (below) — you may ONLY
    choose from this list, never invent a new one.

Canonical ontology (v1):
  SPEAKS_TO, ADDRESSES, DISCUSSES, EXPLAINS, REACTS_TO, AGREES_WITH,
  DISAGREES_WITH, CRITICIZES, PRAISES, ASKS, ANSWERS, HAS_EMOTION,
  GESTURES_TOWARD, LOOKS_AT, IS_SEATED_NEAR, IS_POSITIONED_NEAR, WEARS,
  HOLDS, CAUSES, FOLLOWS, PART_OF_SCENE, APPEARS_IN, CONTAINS_OBJECT,
  IN_LOCATION, ILLUSTRATES_THEME, MENTIONS, OTHER

Return one result per cluster_id via the canonicalize_relations tool call.
If nothing fits, use OTHER — do not force a bad match.
```

Write-back: `kg_edges` gets a new `canonical_relation TEXT` column
(migration `005_kg_relation_canonical.sql`); raw `relation` is kept
untouched for audit. Neo4j graph_writer picks up `canonical_relation` for
edge type on next graph rebuild, keeping `relation` as an edge property.

**1c. Fact dedup** — embedding-similarity threshold pass over
`searchable_facts` per video (no LLM call needed — pure cosine similarity on
existing embeddings, mark near-duplicates with a `superseded_by` pointer
rather than deleting, so nothing is destructively lost).

### Agent 2 — Story Architect Agent

**Windowed batches, not one-call-per-scene.** A strict per-scene loop with a
rolling summary is inherently serial (each call needs the previous call's
summary) — for a ~50-scene video that's 50 sequential round-trips, all
latency, no parallelism possible. Instead: batch 3-5 consecutive scenes per
call, rolling summary syncs once per batch instead of once per scene. Cuts
call count 3-5x; narrative coherence is unaffected at that granularity since
scenes within one batch already share full mutual context in a single
prompt (no summary needed *between* them, only carried *into* the next
batch).

```
SYSTEM PROMPT — Story Architect Agent
────────────────────────────────────────────────────
You are the final reasoning pass before planning agents consume this video's
knowledge base. You write FOR a planner, not for a human reader — be
structured and grounded, not literary.

Input (one batch of 3-5 consecutive scenes, chronological):
  rolling_summary: 1-3 sentence summary of everything established before
    this batch (empty on the first batch)
  scenes: [
    {
      scene_frames: PRUNED frame_analyses for this scene — caption,
        causality, continuity, people[] (pid+story_role+action only, NOT
        full gaze/pose/clothing/apparel detail), beat_type, scene_mood,
        tension_level. NEVER feed full raw qwen_output JSON here — a dense
        video can have 30+ frames/scene at ~300 tokens/frame raw; the pruned
        fields above run ~80-100 tokens/frame, a ~3x cut that matters a lot
        at scene-batch scale.
      speaker_turns: resolved dialogue for this scene's time range
        [{person_id, start_time, end_time, transcript_text}, ...]
    },
    ...
  ]
  cast: {pid: display_name}

For EACH scene in the batch:
  1. Merge that scene's frame-level scene_id variants into ONE canonical
     scene record. Trust the majority scene_id string across its
     scene_frames; list discarded aliases.
  2. Write one scene beat, grounded ONLY in that scene's scene_frames +
     speaker_turns — never invent a detail absent from the input. Use the
     OTHER scenes in this same batch as context for causal_link_to_next
     where relevant (they are already in front of you, no need to guess).

Then, once for the whole batch:
  3. Update rolling_summary for the NEXT batch. Keep it to 3 sentences max —
     this is what gets carried forward, not the full history. Do not let it
     grow unbounded across batches.

Return one scene_beat per input scene, via the write_scene_beats tool call.
```

Tool/function schema (structured output):
```json
{
  "name": "write_scene_beats",
  "parameters": {
    "type": "object",
    "properties": {
      "scene_beats": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "canonical_scene_id": {"type": "string"},
            "discarded_aliases": {"type": "array", "items": {"type": "string"}},
            "start_time": {"type": "number"},
            "end_time": {"type": "number"},
            "participants": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "emotional_arc": {"type": "string"},
            "causal_link_to_next": {"type": ["string", "null"]}
          }
        }
      },
      "updated_rolling_summary": {"type": "string"}
    },
    "required": ["scene_beats", "updated_rolling_summary"]
  }
}
```

Write-back: new `scenes` table (canonical, first-class — currently scenes
only exist loosely as `kg_nodes` rows) and a `storylines` table
(`video_id, title, synopsis, cast_members JSONB, beats JSONB, created_at`) — the
assembled beats become one `storylines` row per video. This is L5's actual
input contract: small, structured, one document per video, never the raw
KB dump.

### Observability & evaluation

Neither agent ships without these — otherwise L4's accuracy and cost are
both unknown until something downstream breaks:

- **Cost tracking**: log input/output token counts + $ per video per stage
  (`grounding_speaker`, `grounding_relation`, `story_architect`), same
  `StepTimer` pattern already used for L1-L3 timings — extend
  `processing_jobs.meta` with a `tokens`/`cost_usd` breakdown per stage.
- **Accuracy spot-check**: sample N `llm_tiebreak` speaker resolutions per
  video for manual review; track agreement rate over time. Without this,
  "the Grounding Agent resolved 18/31 turns" tells you nothing about whether
  those 18 are actually correct.
- **OTHER-bucket monitoring**: alert when `canonical_relation = 'OTHER'`
  exceeds ~5% of a video's edges — signal to review and version the
  ontology, not silent decay into a junk drawer.

### Finalization & source-of-truth contract

The whole point of L4 is that L5/L6 never have to reach back into raw KB
data, and nobody has to manually notice-and-rerun L4 later because it was
silently half-correct. That requires structural guarantees, not convention:

**1. Quality gate before DONE — not just "the calls didn't throw".**
`processing_jobs(level=4)` only flips to `DONE` after a validation pass:
  - every `speaker_turns` row for the video is either `single_candidate`,
    `llm_tiebreak`, or `llm_unresolved_final` (see #3) — none left as bare
    `unresolved`/`face_majority` (those mean "not yet reasoned about").
  - every `chunks`/`shots` time range is covered by at least one `scenes`
    row — no silent time gaps in the canonical scene timeline.
  - the video's `storylines` row has non-null `synopsis` and `beats` with
    length > 0.
  If any check fails, the job stays `FAILED` with the specific reason in
  `error_msg` — never a green checkmark over incomplete data.

**2. Confidence-gated escalation, not blind single-pass trust.**
Cheap-model output below a confidence threshold (e.g. `< 0.5`) is not
accepted as final on the first pass. Those items get a second pass with a
stronger model (Sonnet/Opus-tier) in a small follow-up batch — bounded,
only the uncertain subset, not a full re-run. This is what actually removes
the "notice it's wrong weeks later, rerun everything" failure mode: the
system escalates uncertainty *before* declaring done, instead of after
someone finds a bad answer downstream.

**3. Terminal vs. pending unresolved — a real distinction.**
`speaker_turns.resolution_method` gains a 5th value: `llm_unresolved_final`
— set when the Grounding Agent (both passes) genuinely cannot determine a
speaker. This is different from the pre-L4 `unresolved` state (`= "L4
hasn't looked at this yet"`). L5 can safely treat `llm_unresolved_final` as
"this is permanently ambiguous, plan around it" rather than "maybe try
again."

**4. Storylines are versioned and immutable once finalized.**
`storylines` gets `status` (`draft` → `final`) and `version INT`. L4 writes
`draft` while running; the finalization gate above flips it to `final` only
after all checks pass. **L5 only ever reads `status = 'final'`.** If L4
re-runs later (new cast info, ontology bump, whatever), it inserts a NEW
row with `version = prior + 1` — never overwrites a `final` row in place.
L5 can pin to a specific version; nothing mutates underneath an in-flight
planning run.

**5. One-way read contract — no bypass, ever.**
L5 (Planner) reads `storylines(status='final')` + `persons` only. L6
(specialized action agents) reads L5's plan output only — never reaches
back into `storylines`, `scenes`, `frame_analyses`, or `kg_edges` directly.
This is what makes L4 an actual boundary instead of a suggestion: if L6
needs something L5's plan didn't carry forward, that's a signal to fix L5's
output shape, not a license to reach around it.

**6. Cheap corrections don't require re-running L4.**
For the rare genuinely-wrong case a human catches, `scene_overrides` /
`storyline_overrides` tables let a correction be patched in as a layer L5
reads on top of the `final` row, instead of burning another full LLM pass
over the whole video to fix one sentence.

### New tables (migration `005_l4_reasoning.sql`)

```sql
ALTER TABLE kg_edges ADD COLUMN IF NOT EXISTS canonical_relation TEXT;

ALTER TABLE searchable_facts ADD COLUMN IF NOT EXISTS superseded_by UUID REFERENCES searchable_facts(id);

-- Add the terminal state alongside the existing 4 (see finalization contract #3)
ALTER TABLE speaker_turns DROP CONSTRAINT IF EXISTS speaker_turns_resolution_method_check;
ALTER TABLE speaker_turns ADD CONSTRAINT speaker_turns_resolution_method_check
  CHECK (resolution_method IN ('face_majority', 'single_candidate', 'llm_tiebreak', 'llm_unresolved_final', 'unresolved'));

CREATE TABLE IF NOT EXISTS ontology_relations (
  version    INT NOT NULL,
  relation   TEXT NOT NULL,
  PRIMARY KEY (version, relation)
);

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
  causal_link_to_next TEXT
);

CREATE TABLE IF NOT EXISTS storylines (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id    UUID REFERENCES videos(id) ON DELETE CASCADE,
  version     INT NOT NULL DEFAULT 1,
  status      TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'final')),
  title       TEXT,
  synopsis    TEXT,
  cast_members JSONB DEFAULT '{}',  -- "cast" is a reserved SQL keyword
  beats       JSONB DEFAULT '[]',
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (video_id, version)
);

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
```

### Folder structure

```
pipeline/level4/
├── __init__.py
├── grounding_runner.py       ← 1a/1b/1c dispatch, batches unresolved items,
│                                confidence-gated escalation to strong model
├── story_architect_runner.py ← windowed-batch calls (3-5 scenes), rolling summary
├── finalizer.py               ← quality-gate validation → flips storylines
│                                draft → final, or fails the job with a reason
└── updater.py                ← writes speaker_turns updates, kg_edges.canonical_relation,
                                  scenes, storylines (draft), ontology_relations, plus the
                                  Neo4j edge-type correction pass and Pinecone
                                  upsert_scenes()/upsert_storylines() projections

prompts/
├── grounding_speaker.py       ← 1a system prompt (above)
├── grounding_relation.py      ← 1b system prompt (above)
└── story_architect.py         ← Agent 2 system prompt (above)
```

### Neo4j + Pinecone propagation

L4 doesn't only write Postgres — the two systems L3 already populates need
correction/extension passes too, or L4's cleanup never reaches the graph or
the vector index and both stay stuck with the pre-L4 mess.

**Neo4j (`knowledge_base/neo4j/graph_writer.py`).** `write_edges()` currently
builds the Neo4j relationship type directly from the raw `edge.relation`
string (sanitized into a Cypher-safe identifier) — this is *why* the
200+ freeform Qwen predicates became 200+ distinct Neo4j edge types, not
just a Postgres problem. L4's `updater.py` (1b relation canonicalization)
must re-run `write_edges()` for the affected edges using
`canonical_relation` as the relationship type instead of `relation` once
it's set — a graph correction pass, not a full graph rebuild. Until an
edge has a `canonical_relation`, its Neo4j edge type is unchanged (no
regression, just not yet cleaned up).

**Pinecone (`knowledge_base/pinecone_kb/indexer.py`).** `upsert_scenes()`
already exists and is currently fed from L3's loose per-frame Scene
grouping (the inconsistent `scene_id` aliases documented in "Why L4
exists"). The Story Architect Agent's `updater.py` re-points this at the
new canonical `scenes` table instead — one embedding per canonical scene
(from `scenes.summary`), not one per frame-level alias. This directly
fixes the search quality problem: querying "scenes" today can return
near-duplicate hits for `studio_podcast_01` / `scene_001` /
`interview_studio_01` when they're the same scene.

New function needed: `upsert_storylines()` in `indexer.py`, mirroring
`upsert_scenes()`'s shape — embeds `storylines.synopsis` (one vector per
`storylines` row, only for `status='final'`) into the new `storylines`
Pinecone namespace (see Pinecone Namespaces above). This is what makes
cross-video plot/theme search possible later — "find videos where someone
gets ambushed on a football podcast" — which no existing namespace
supports (facts/scenes are per-frame/per-scene granularity, not
per-video narrative).

**Write order in `finalizer.py`:** Postgres writes (scenes, storylines,
canonical_relation) happen first and are the source of truth; Neo4j and
Pinecone writes happen after, as projections of that Postgres state —
same pattern L3's `kb_finalizer.py` already uses. If a Neo4j or Pinecone
write fails, log and continue (non-fatal, matches rule "L2 failure is
non-fatal" precedent) — Postgres being correct and `storylines.status`
reaching `final` is what the finalization gate checks, not that every
projection succeeded. A retry/backfill pass for graph/vector projections
is a fine future addition, not a blocker for L4 v1.

### Engineering rules for L4 (extends the non-negotiable list above)

11. **L4 never downgrades L2 certainty** — `single_candidate` speaker_turns
    are never re-touched by the Grounding Agent, only `unresolved` /
    `face_majority` rows.
12. **Closed-set outputs only** — Grounding Agent must never emit a `pid` or
    `canonical_relation` outside the given closed list. Validate every LLM
    response with pydantic before any write; reject and log on schema
    mismatch, never write raw text to a typed column.
13. **Dedup is additive, not destructive** — `superseded_by` pointer, never
    `DELETE FROM searchable_facts`.
14. **Rolling summary bounded** — Story Architect's `updated_rolling_summary`
    must not grow past ~3 sentences call-to-batch; truncate defensively if a
    call ignores the instruction.
15. **L4 is idempotent AND versioned per video** — re-running on the same
    `video_id` is a safe UPSERT for `scenes`/intermediate state, but never
    overwrites a `storylines` row with `status='final'` — it inserts a new
    `version`. Same "safe to rerun" guarantee as L1–L3, extended with an
    audit trail instead of silent overwrite.
16. **L5 reads `storylines(status='final')` only** — planning agents never
    query `frame_analyses`/`kg_edges`/`scenes` directly, and never read a
    `draft` row. If L5 needs something L4 didn't capture, that's a signal to
    extend L4's output shape, not to bypass it.
17. **Nothing is DONE until the finalizer says so** — `processing_jobs
    (level=4)` only reaches `DONE` after `finalizer.py`'s completeness
    checks pass (see Finalization contract above). A job that "ran without
    exceptions" but left gaps is `FAILED`, not `DONE`.

---

## LEVEL 5 — PLANNING (Editing Director's Plan)

### STATUS: IMPLEMENTED — not yet run end-to-end (`pipeline/level5/planner_runner.py`
complete: Pass A, Pass B, `enforce_duration`, `validate_plan`,
`run_level5_planning`, `apply_revision` — all `py_compile`/import-clean;
`edit_plans`/`edit_plan_revisions` tables via migration
`006_l5_planning.sql`, applied to the live DB. Never executed against a
real Groq call or a real `storylines`/`scenes` row — L4 hasn't run live
yet either, so there's no finalized storyline for L5 to plan against.)

**LLM provider: Groq**, same as L4 — `groq` Python SDK, OpenAI-compatible
function calling, model `qwen/qwen3.6-27b` for both passes (config:
`shared/config.py` → `L5_SELECTION_MODEL`, `L5_SEQUENCING_MODEL`, both
default `qwen/qwen3.6-27b`). Tool schemas in code use the OpenAI-compatible
shape, not the Anthropic-shaped pseudo-JSON below — same caveat as L4.

### What L5 is

L5 turns a user's editing intent ("cut this into a 60s reel focused on
Person X's commentary", "give me the highlights", "full cut, tighter
pacing") into a **structured, executable edit plan** — an ordered list of
operations referencing real scene/time data, that L6 agents can act on
without re-interpreting intent themselves. L5 is the only place natural-
language judgment happens about *what* goes in the video and in what
order; L6 agents receive decisions, not ambiguity.

The user should be able to see a plan, react to it, and get a re-plan —
not regenerate from scratch.

### Read contract (rule 16, restated as a hard boundary)

```
L5 reads:       storylines(status='final'), scenes, persons, scene_overrides
L5 NEVER reads: frame_analyses, kg_edges, speaker_turns, raw searchable_facts
```

If L5 needs something not in `storylines`/`scenes`, extend L4's output
shape — never let L5 reach back into raw data.

### Two-pass structure

Same split reasoning as L4's Grounding/Story-Architect divide — separate
"what's relevant" (needs breadth, cheap-ish) from "how should it flow"
(needs strong reasoning, narrow).

**Pass A — Selection & scoring** (cheap/fast tier).
Input: `storylines(final)` beats + `scenes` (incl. `usability_score`, see
L4 finalizer addition below) + user intent + hard constraints (target
duration, must-include persons/topics, must-exclude content).
Output: ranked candidate scenes with relevance score + short rationale,
grounded only in beat/scene data already present — same closed-set
discipline as L4.

**Batch cap, same reasoning as the L4 Grounding Agent:** a dense/long
video can produce hundreds of candidate scenes — cap Pass A at ~30-40
scenes per call, sub-batch, merge scores. Do not feed an unbounded list
into one call; LLM accuracy on item #180 of a list degrades silently
relative to item #15.

**Pass B — Sequencing & pacing** (strong-reasoning tier, one call).
Input: Pass A's ranked candidates (already pruned — context bounded
regardless of source video length) + `causal_link_to_next` chains from
storylines + target duration/platform + pacing preference.
Output: the `EditPlan` — ordered operations with rationale traceable back
to specific beat IDs (every cut explainable as "because storyline beat X
said Y").

### EditPlan schema

```sql
CREATE TABLE edit_plans (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id            UUID REFERENCES videos(id),
  storyline_id        UUID REFERENCES storylines(id),   -- pins to a specific final version
  user_prompt         TEXT NOT NULL,
  target_duration_s   FLOAT,
  platform            TEXT,                              -- reel | full_cut | youtube | ...
  status              TEXT NOT NULL DEFAULT 'draft'
                       CHECK (status IN ('draft', 'reviewed', 'applied', 'superseded')),
  version             INT NOT NULL DEFAULT 1,
  operations          JSONB NOT NULL,                     -- ordered EditOperation[]
  achieved_duration_s FLOAT,                              -- computed post-hoc, never trusted from LLM
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (video_id, version)
);

-- one row per user round of feedback, so "re-plan" is a diff, not a regenerate
CREATE TABLE edit_plan_revisions (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edit_plan_id    UUID REFERENCES edit_plans(id),
  user_feedback   TEXT NOT NULL,
  diff_operations JSONB NOT NULL,     -- only what changed, not the full plan again
  created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

`EditOperation` shape (JSONB array element):

```json
{
  "op_id": "op_001",
  "type": "SELECT_CLIP",
  "scene_id": "uuid-of-scenes-row",
  "start_time": 142.3,
  "end_time": 158.9,
  "sequence_index": 0,
  "rationale": "beat_id:xyz — Person X's key stat callout, high relevance to prompt",
  "transition_in": "cut",
  "downstream_ops": ["TEXT_OVERLAY:op_014", "COLOR_MATCH:op_020"]
}
```

Operation types cover both L5's own decisions (`SELECT_CLIP`, `TRIM`,
`REORDER`, `CUT_TO`) and dispatch requests for L6
(`COLOR_MATCH_REQUEST`, `TEXT_OVERLAY_REQUEST`, `AUDIO_DUCK_REQUEST`,
`B_ROLL_INSERT_REQUEST`). L5 decides *that* a color match is needed at a
cut point; it does not decide the LUT parameters — that's L6's job, same
separation of concerns as the Grounding/Story-Architect split.

### Hard vs. soft constraints — don't trust the LLM with arithmetic

Target duration is a **hard constraint enforced programmatically**, not
something hoped for from the model. After Pass B returns operations, sum
`(end_time - start_time)` across `SELECT_CLIP` ops, compare to
`target_duration_s`. If over/under beyond tolerance (e.g. ±10%), feed
back the specific overage/shortfall and ask the model to trim/extend from
the *existing* selected set — same bounded-escalation pattern as L4.

**Bounded retry, same principle as L4's escalation budget:** cap this
correction loop at 2-3 attempts. If still out of tolerance after that,
fall back to a programmatic trim (drop the lowest-relevance clip(s) until
duration fits) rather than looping the LLM indefinitely.

### Validation before a plan is usable

Same discipline as L4's finalizer:
- every `scene_id` referenced actually exists in `scenes` for this `video_id`
- every `start_time`/`end_time` falls inside that scene's bounds (no hallucinated timestamps)
- `sequence_index` is contiguous, no gaps/dupes
- reject with reason, never silently coerce

### L4 addition needed to support L5: `usability_score`

Flagged during L4 planning, not deferred: nothing in L1-L4 currently
captures take quality — audio clipping, focus/exposure issues, an
interrupted/restarted take, or which of several takes of "the same beat"
to prefer. An editing tool's core value is picking the *good* take;
without this signal, L5's Pass A has no way to rank near-duplicate scenes
covering the same moment.

Doesn't need a new agent — `color_grades` already runs per-frame and can
emit a cheap technical-quality score (exposure/focus/audio peak) as a
byproduct; Whisper's `confidence`/word-level probs are already in
`transcript_segments`. Add to L4's finalizer: roll these into a
per-scene `usability_score FLOAT` column on `scenes`, computed
deterministically (no LLM call) from existing color_grades + transcript
confidence data already sitting in the KB.

---

## LEVEL 6 — SPECIALIZED ACTION AGENTS

### STATUS: IMPLEMENTED — not yet run end-to-end against a real video

All four pieces built (`pipeline/level6/`): Editing Director
(`editing_director.py` — cut snapping, FCPXML export, direct FFmpeg
render with a stream-copy fast path + filter_complex path), Color
Grading (`color_grading_runner.py` — Groq sequence-delta reasoning +
FFmpeg filter mapping, params clamped against the real
`color-grading/color_analyzer/analyzer/schema.py` bounds at runtime),
Audio Sync (`audio_sync.py` — pure DSP, no LLM), Caption/Text Overlay
(`caption_overlay.py` — Groq for style only, text always sourced from
real transcript data, drift-checked). Orchestrated by
`pipeline/level6/updater.py::run_level6`.

**Verified live** (not just `py_compile`): Editing Director's cut
snapping + FCPXML export + both render paths were smoke-tested against a
real synthetic video with real ffmpeg (8.1.2) — a real bug was caught and
fixed there (`ffprobe`'s `pkt_pts_time` field is deprecated on modern
ffmpeg, only `pts_time` exists now). Color Grading's filter-building was
verified against real `color_grades` rows. `run_color_grading` end-to-end
and the full `run_level6` orchestration were **not** live-tested — no
video has real `edit_plans`/`cut_list_items` rows yet (L5 hasn't run
live either).

**Known gaps, honestly flagged (not silently papered over):**
- **FCPXML never test-imported into Resolve/Premiere** — matches the
  documented FCPXML 1.9 spec and parses as valid XML, but "parses" ≠
  "imports correctly." Needs a real-NLE round-trip test before trusting
  it in production. Also doesn't handle NTSC drop-frame rates
  (23.976/29.97/59.94), only integer-ish fps.
- **Color conversion formulas are unverified judgment calls** — the
  stops→`eq`-brightness and temp/strength→`colorbalance` conversions are
  reasonable engineering approximations, not checked against a
  color-science reference or a real graded frame. Spot-check visually
  before trusting the grade output looks right.
- **Audio ducking is a real no-op** — `compute_ducking_filter` always
  returns `None`. This pipeline's schema has no music-bed/secondary-
  audio-track concept for `sidechaincompress` to duck against — not a
  bug, a genuine missing schema concept. `AUDIO_DUCK_REQUEST` ops surface
  a note in `run_level6`'s result instead of silently doing nothing.
  Fixing this for real needs a schema addition (a secondary audio asset
  reference) before it's meaningful — v-next, not v1.
- **Audio normalization integration gap found during reconciliation and
  fixed**: `render_direct`'s `extra_filters` only ever wired into the
  per-clip *video* label — there was no path for `loudnorm` (a
  sequence-level filter) to reach the render at all. Fixed by adding a
  `final_audio_filter` param to `render_direct`, applied post-concat on
  the `[outa]` label; `run_level6` now passes `compute_loudnorm_filter`'s
  output through it instead of just returning it unapplied.
- **Multicam sync is a documented no-op** — no `camera`/`sync_group`
  field anywhere in the schema for a single-source-video pipeline; not a
  v1 blocker per the original design note.
- **`modal_app.py` wiring not done** — L6 is user-triggered per finalized
  `edit_plan`, not part of the automatic L1-L5 batch pipeline, so this is
  lower priority than L1-L4's automatic wiring was; a standalone
  on-demand `run_level6_modal` Modal function still needs to be added
  when this is ready to actually run on Modal instead of locally.

**Rendering approach, decided after inspecting the actual sibling
`color-grading` module (it is analysis-only — `color_analyzer/`, no
`apply_grade`/LUT/render function exists anywhere in it; L6 has to build
the pixel-application layer itself, not reuse one):**

- **Two-tier output.** (1) XML/EDL export (Resolve/Premiere) from
  `cut_list_items` — primary path, matches the "Cursor for video"
  framing: user gets an editable project, not a locked render. (2) Direct
  FFmpeg render — secondary path, for quick preview / non-technical users
  who want an MP4 immediately. Both consume the same `cut_list_items` +
  `sequence_color_adjustments` tables.
- **Color: no LUT baking needed for v1.** `color_grades.parameters`
  (`knowledge_base/../color-grading/color_analyzer/analyzer/schema.py`)
  is DaVinci-Resolve-shaped: `white_balance.{temperature,tint}`,
  `primary.{exposure,contrast,gamma}`, `presence.saturation`,
  `tone_curve.{shadow_lift,contrast_strength}`,
  `color_wheels.{lift,gamma,gain}.{temp,strength}` (classic 3-way
  lift/gamma/gain color corrector). This maps close to 1:1 onto FFmpeg's
  built-in filters — apply directly, no intermediate LUT:
  - `white_balance.temperature` → `colortemperature` filter
  - `primary.{exposure,contrast,gamma}` + `presence.saturation` →
    `eq=contrast=:brightness=:saturation=:gamma=`
  - `color_wheels.lift/gamma/gain` → `colorbalance=rs=:gs=:bs=:rm=:gm=
    :bm=:rh=:gh=:bh=` (shadows/midtones/highlights RGB offsets — direct
    match to a 3-way corrector)
  - `tone_curve.*` → `curves` filter
  A LUT (`.cube`) export can be added later as a *secondary* artifact for
  NLE portability, once v1's direct-filter path is proven — not blocking
  v1.
- **Cut mechanics.** Stream-copy (`-c copy`) a `SELECT_CLIP` when its
  snapped boundary lands on a keyframe (fast, lossless); re-encode when
  it doesn't (frame-accurate cuts mid-GOP require it — check keyframe
  alignment per clip, branch per-clip, never blanket re-encode). Concat
  via `filter_complex` (`trim`+`setpts`/`asetpts` per input + `concat`)
  when any per-clip color/audio filter is applied (it will be) — one
  ffmpeg invocation, not N+1 intermediate files. J/L-cut
  (`audio_lead_ms`/`video_lead_ms`) via `adelay`/`atrim` in the same
  filter graph.
- **Audio.** FFmpeg `loudnorm` two-pass (measure, then apply) targeting
  -14 LUFS (streaming default) or -16 (podcast), platform-dependent from
  `edit_plans.platform`. Ducking via `sidechaincompress`. Multicam sync
  is a v-next item — nothing in the current schema carries multi-camera
  source info for a single pipeline run, not a v1 blocker.

### Design principle

L6 agents are **narrow and mostly deterministic** — they consume specific
`EditOperation` types from a finalized `edit_plan`, do one job well, and
write results back keyed to `op_id`. LLM judgment is used only where a
judgment call genuinely can't be reduced to a formula (pacing snap
points, color-match targets, caption phrasing) — everything else is
code. Same discipline already applied at L4: don't let an agent do what a
deterministic pass can do more cheaply and reliably.

Orchestration mirrors the existing Modal pattern: some L6 agents run in
parallel (color grading and captioning don't depend on each other); one
dependency matters — **continuity-aware agents (color, audio) need the
final assembled sequence order from L5, not the original video's shot
order**, since they reason about adjacency in the *cut*, not the source
footage.

```
edit_plan (finalized)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  L6 — PARALLEL (all consume the same finalized edit_plan)  │
│                                                             │
│  Editing Director   →  builds the actual cut list/XML      │
│  Color Grading      →  per-shot params, sequence-aware      │
│  Audio Sync         →  levels, ducking, multicam align      │
│  Caption/Text Agent →  on-screen text per TEXT_OVERLAY op   │
└───────────────────────────────────────────────────────────┘
        │  (all agents write results keyed to op_id)
        ▼
   Deterministic render (FFmpeg / XML export) — not an agent
```

### Editing Director

Mostly execution, not reasoning: turns `SELECT_CLIP`/`TRIM`/`REORDER`
operations into the actual FFmpeg edit list / Resolve-Premiere XML, per
the existing "no business logic in Modal function bodies" rule — the XML
writer is a pure function of `operations`.

The one place it needs judgment: **exact cut-point snapping**. `EditPlan`
gives approximate in/out points from scene boundaries; the Director
should snap to natural pause/breath points using data already computed
(VAD, `words[].start/end` from transcript) rather than hard-cutting
mid-word or mid-breath. A small deterministic search over an existing
signal, not an LLM call.

```sql
CREATE TABLE cut_list_items (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edit_plan_id   UUID REFERENCES edit_plans(id),
  op_id          TEXT NOT NULL,               -- references EditOperation.op_id
  sequence_index INT NOT NULL,
  source_start   FLOAT NOT NULL,               -- snapped, may differ slightly from plan's start_time
  source_end     FLOAT NOT NULL,
  audio_lead_ms  INT DEFAULT 0,                -- J-cut offset
  video_lead_ms  INT DEFAULT 0,                -- L-cut offset
  transition     TEXT DEFAULT 'cut'
);
```

### Color Grading Agent

The hard part is already built: 45-param analysis per shot in
`color_grades`. The L6 job is **not** re-analyzing color — it's solving
continuity across a sequence assembled from non-adjacent source shots,
which the original per-shot analysis never had to consider.

Input: `color_grades` rows for every shot in the finalized cut order (via
`cut_list_items` sequence), optional mood target (from
`scenes.emotional_arc` or explicit user request, e.g. "warmer",
"cinematic"). Job: for each shot, compute a delta from its neighbors in
*cut order* (not source order) on key params (white balance, exposure,
saturation) and recommend an adjustment that pulls outliers toward
sequence consistency — not just toward some absolute "ideal" grade. This
is the part that justifies an LLM/reasoning pass over pure per-shot
heuristics.

```sql
CREATE TABLE sequence_color_adjustments (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edit_plan_id     UUID REFERENCES edit_plans(id),
  cut_list_item_id UUID REFERENCES cut_list_items(id),
  base_parameters  JSONB NOT NULL,       -- from color_grades, unchanged
  sequence_delta   JSONB NOT NULL,       -- {param: adjustment} to harmonize with neighbors
  rationale        TEXT
);
```

### Audio Sync Agent

Handles LUFS normalization across the assembled cut (shots from
different source takes have different loudness — needs sequence-level,
not per-shot, normalization), ducking under overlaid text/music, and
multicam sync-group alignment when a cut pulls from multiple camera
angles of the same moment. **Mostly deterministic DSP — no LLM needed
here at all.** Flag this explicitly so it doesn't get over-built.

### Caption/Text Overlay Agent

Consumes `TEXT_OVERLAY_REQUEST` ops + the underlying `dialogue_subtitle`
already produced at L3 (grounded in real transcript, not re-generated) +
platform target (reel vs. full cut changes caption style/size/timing
conventions). Styling judgment (word-by-word karaoke captions for reels
vs. static lower-thirds for a full cut) is the one LLM-worthy decision;
the text itself should almost always come from existing transcript data,
never be invented.

### Motion Graphics Agent — explicitly deferred

Flagged as future roadmap, not built. Left out of L6 v1 entirely rather
than stubbed — same "don't build for the level you don't have yet"
discipline applied throughout.

### Engineering rules for L5/L6 (extends the list above)

18. **L6 never re-interprets intent** — if an L6 agent finds itself
    needing to decide *what* the user wants rather than *how* to execute
    a decision L5 already made, that's a sign the operation's
    rationale/parameters are underspecified in the `EditPlan` — fix L5's
    output shape, don't let L6 guess (rule 16, applied one level down).
19. **Sequence-aware agents (color, audio) key off cut order, not
    source order** — the one place L6 genuinely differs from L1-L4's
    per-shot/per-frame processing model; don't reuse per-shot logic
    unmodified and assume it generalizes.
20. **Deterministic where possible, LLM only for the specific judgment
    call** — Editing Director's XML writer, Audio Sync's DSP, and
    duration enforcement in L5 are pure functions; keep them that way
    rather than routing through a model call because the rest of the
    pipeline is agentic.
21. **`edit_plan_revisions` are diffs, not regenerates** — "make it
    faster-paced" feeds the existing plan + feedback back in; it does
    not re-run the full two-pass L5 from scratch.
22. **L5's correction loops are bounded, same as L4's escalation
    budget** — duration-fit retries cap at 2-3 attempts, then fall back
    to a programmatic trim. No unbounded LLM retry loops anywhere in the
    pipeline.
23. **Batch caps apply everywhere an LLM sees a list that scales with
    video length** — L4's Grounding Agent (speaker turns), L5's Pass A
    (candidate scenes) both cap around 30-40 items/call and sub-batch.
    This is now a standing rule, not a one-off fix — apply it to any
    future stage that feeds a per-item list into one call.
