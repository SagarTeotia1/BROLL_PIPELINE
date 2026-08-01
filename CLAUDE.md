# VIDEO UNDERSTANDING PIPELINE — COMPLETE PLAN

## STATUS: L1-L9 IMPLEMENTED — L1-L6 VERIFIED END-TO-END ON REAL DATA, L7-L9 VERIFIED AGAINST LIVE DB

Superseded from the original "AWAITING USER REVIEW — DO NOT IMPLEMENT YET" —
that was true when this doc was first written; it stopped being true partway
through this project's life. See "IMPLEMENTATION STATUS SNAPSHOT" immediately
below for the real, current, per-level state (what's built, what's verified,
what model runs where, what's still a known gap) as of **2026-07-31**. The
rest of this document (architecture, schemas, prompts, engineering rules) is
still the accurate design reference — this snapshot is the "where are we
against that design, for real" answer, kept at the top so it's the first
thing read, not buried after 2000+ lines of design history.

---

## IMPLEMENTATION STATUS SNAPSHOT (2026-07-31)

### Per-level status

| Level | What it does | Code status | Verified how |
|---|---|---|---|
| **L1** | ASHFS keyframe sampling + Whisper transcript | Implemented | Real Modal runs — `processing_jobs`: 11 `done`, 1 `failed` across 12 real videos |
| **L2** | Face (ArcFace/HSEmotion/ByteTrack) + Color (45-param) + Speaker diarization (pyannote) + matting | Implemented | Real Modal runs — 6 `done`, 5 `failed` (pyannote/HF_TOKEN issues are the common failure, non-fatal by design) |
| **L3** | Qwen3-VL frame analysis + knowledge graph build (Postgres + Neo4j) | Implemented | Real Modal runs — 2 `done`, 2 `failed` |
| **L4** | Grounding Agent (speaker resolution, relation canon, fact dedup) + Story Architect Agent (scenes, storyline) | Implemented | Run live multiple times this session on video `97199656`; real bugs found+fixed: cross-run scene accumulation, empty fallback summaries starving L5 relevance scoring |
| **L5** | Selection (Pass A) + Sequencing (Pass B) + duration enforcement + validation | Implemented | Run live multiple times this session; real bugs found+fixed: whole-plan rejection on one incomplete field, `user_prompt` never reaching the sequencing pass, whole-plan rejection on one malformed op |
| **L6** | Editing Director (cuts/XML/render) + Color Grading + Caption/Text Overlay + Compositing (A3/A5) + Audio Sync + QA Agent | Implemented | **Real render produced and verified**: `out.mp4`, 63.1MB, 131.5s, 1920×1080 h264/aac, zero decode errors full-file scan. Real bugs found+fixed: Compositing Agent crash (uncaught missing dep) taking down the whole render, `cut_list_items`/color-adjustment accumulation across reruns, cut-snapping collapsing to one wrong shared timestamp, ffmpeg `Cannot allocate memory` on reordered clips (shared-decode filter_complex → per-clip extraction + concat-demuxer rewrite) |
| **L7** | Evaluation — golden-set test, rubric scoring, cost/latency tracking, alerts panel | Implemented | Real DB rows: `evaluation_scores` (1), `llm_call_log` (1); `tests/integration/test_e2e_golden.py` 7/7 passing |
| **L8** | Human Feedback — `human_feedback` table + fixed dead L4 correction entry point | Implemented | Real DB rows: `human_feedback` (2), real `correction_events`/`scene_overrides` rows proving the previously-dead `log_scene_correction` now works |
| **L9** | Reward/Punishment — `reward_signals` aggregation, few-shot injection (L4/L5), negative-reward alert trigger | Implemented | Real DB rows: `reward_signals` (21), idempotent rerun confirmed. 9b/9c wired correctly but **not yet live-firing** — no video has `client_id` populated yet (9b), no relation has crossed the negative-reward+sample-count threshold yet (9c) |

Full test suite: **183/183 passing** (`pytest tests/`) — includes fixing all 6 pre-existing failures an audit found (5 stale test fixtures, 1 test asserting behavior a real bug fix intentionally changed), not just avoiding new ones.

### Model-per-level (what actually calls what, today)

| Level | Component | Model | Provider | Notes |
|---|---|---|---|---|
| L1 | ASHFS keyframes | — | local (DINOv2 + PySceneDetect) | no LLM |
| L1 | Transcript | `large-v3` | local (faster-whisper) | no LLM |
| L2 | Face | ArcFace + HSEmotion + ByteTrack | local (ONNX) | no LLM |
| L2 | Color | — | local (OpenCV/CuPy, 45-param deterministic) | no LLM |
| L2 | Diarization | `pyannote/speaker-diarization-3.1` | local (gated HF model) | no LLM |
| L3 | Frame analysis + graph extraction | `Qwen/Qwen3-VL-8B-Instruct` | self-hosted vLLM on Modal (L40S/A100) | no external API |
| L4 | Grounding Agent (1a speaker, 1b relation, 1c fact dedup) | `deepseek/deepseek-v3.2` | OpenRouter | `reasoning.enabled=False` — closed-set classification, no benefit from reasoning |
| L4 | Story Architect Agent | `deepseek/deepseek-v3.2` | OpenRouter | `reasoning.enabled=True` — narrative synthesis, low call volume, worth the cost |
| L5 | Pass A — Selection & Scoring | `qwen/qwen3-30b-a3b-instruct-2507` | OpenRouter | cheap/fast tier, high call volume |
| L5 | Pass B — Sequencing & Pacing | `qwen/qwen3-235b-a22b-2507` | OpenRouter | strong tier, one call/video |
| L6 | Color Grading Agent (sequence deltas) | `qwen/qwen3-30b-a3b-instruct-2507` | OpenRouter | numeric-delta reasoning, cheap tier |
| L6 | Caption/Text Overlay Agent (style only) | `qwen/qwen3-30b-a3b-instruct-2507` | OpenRouter | text always from real transcript, never invented |
| L6 | Compositing Agent (A3 background pick, A5 emphasis) | `qwen/qwen3-30b-a3b-instruct-2507` | OpenRouter | closed-set pick from pre-retrieved candidates |
| L6 | QA Agent — intent-match pass | `qwen/qwen3-235b-a22b-2507` | **Groq** (not OpenRouter — a real pre-existing bug, see below) | strong tier, one call/plan |
| L7 | QA Agent — rubric scoring (4-dim, extends QA Agent) | `qwen/qwen3-235b-a22b-2507` | OpenRouter | same model as QA intent-match, correct provider |
| Embeddings (L3 facts/scenes, L4 relation clustering, L4 scene/storyline vectors) | — | `BAAI/bge-large-en-v1.5` | local (`sentence-transformers`) | **currently broken** — package installed but `transformers`→`tensorflow`→`protobuf` version conflict in this environment breaks the import chain; every embedding write this session degraded non-fatally to `NULL`, never blocked a run |
| L9 | `reward_signals` computation | — | local (deterministic aggregation script) | no LLM |

**Known model-routing bug, found but not yet fixed**: L6's QA Agent's *existing* intent-match pass gates on `GROQ_API_KEY` even though every other L4-L6 call site (including this session's new L7 rubric-scoring pass, same file) uses the OpenRouter client. Since most environments only have `OPENROUTER_API_KEY` set, the intent-match pass silently no-ops (`llm_status=NULL`) while the deterministic checks + new rubric pass still run fine. Left untouched by the L7 implementation per its "extend, don't replace" scope — worth a one-line fix (swap the gate to `OPENROUTER_API_KEY`, same as its own new sibling call) next time someone's in that file.

### Real data in the live DB right now (Neon Postgres, `DATABASE_URL` in `.env`)

`videos`=12, `scenes`=15, `storylines`=3, `edit_plans`=4, `cut_list_items`=7, `qa_reports`=2, `evaluation_scores`=1, `llm_call_log`=1, `human_feedback`=2, `reward_signals`=21, `pipeline_alerts`=7. The one fully-verified end-to-end video is `video_id=97199656-d176-46aa-88b4-026670be4576` — real L1→L6 output, snapshotted as this project's first golden test fixture (`tests/fixtures/golden_97199656.json`).

### What's still a known, honest gap (not hidden, not yet fixed)

- **Local embeddings broken** (see table above) — real functionality loss (no `scenes.embedding`/`searchable_facts.embedding` vector search works today), not fatal to any pipeline stage.
- **`sentence-transformers` env conflict** is the same root cause that also crashed L6's Compositing Agent before the non-fatal-catch fix — the crash is fixed, the underlying missing capability (A3 background selection) is still non-functional until the dependency conflict is resolved.
- **QA Agent's intent-match Groq-gating bug** (above) — narrow, one-line fix, not yet applied.
- **L9's 9b/9c are correctly wired but not yet live-firing** — need `videos.client_id` populated (9b) and more `evaluation_scores`/`human_feedback` volume (9c) before they do anything, by design (confidence floors, not a bug).
- **Modal wiring for L5/L6 doesn't exist** — both run only as local CLI scripts (`scripts/run_l5.py`, `scripts/run_l6.py`), not as `@app.function`s. Fine for the current dev/verification stage, a real gap before this could run unattended in production.
- **No editor has ever watched the rendered output** — every verification this session was structural (ffprobe, decode-scan, ffmpeg exit code, DB row counts, pytest). Nothing here substitutes for a human judging whether the edit is actually *good* — see this session's earlier "editor-level assessment" discussion for why that distinction matters.

---

## What We're Building

A 3-level video understanding pipeline that runs on Modal.com (L40s GPU) and produces a rich, queryable knowledge base combining a PostgreSQL graph database, Pinecone vector index, and Cloudflare R2 object store.

**Note (2026-07-31): this intro paragraph describes the ORIGINAL 3-level plan.**
The pipeline grew to 9 levels (L1-L9) over this project's life — see
"IMPLEMENTATION STATUS SNAPSHOT" above for the current shape. Pinecone was
also dropped entirely (B7) in favor of pgvector. Left as-is below for
historical accuracy of the original design intent, same convention this doc
already uses for every other superseded section.

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

### STATUS: SUPERSEDED (B7, PART B — Hardening → resolved) — Pinecone was
dropped entirely; `knowledge_base/pinecone_kb/` deleted. All vector search
now runs on Postgres pgvector (ivfflat indexes), consolidated as follows:

| Old Pinecone namespace | Now lives at | Search entry point |
|---|---|---|
| `frames` (dim=768, DINOv2) | `keyframes.dino_embedding` | already pgvector — was never Pinecone-only |
| `facts` (dim=1024) | `searchable_facts.embedding` | `knowledge_base.postgres.queries.search_searchable_facts_by_embedding` |
| `scenes` (dim=1024) | `scenes.embedding` (canonical, L4) + `kg_nodes.embedding` (loose per-frame alias, L3) | `search_scenes_by_embedding` |
| `persons` (dim=512, planned) | `persons.arcface_embedding` | already pgvector — was never Pinecone-only |
| `storylines` (dim=1024, planned) | `storylines.embedding` | `search_storylines_by_embedding` |

Columns/indexes added by migration `012_pgvector_only.sql`. Kept below
unedited for historical context (matches the pattern used elsewhere in this
doc — e.g. L1's original architecture box — of marking superseded sections
rather than deleting them).

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
  clusters: [{cluster_id, raw_relations: [...] (a representative sample of
    up to 8 near-duplicate strings from this cluster, NOT the exhaustive
    list — every member still gets mapped to your answer, only what's shown
    to you is capped — real-video cost finding: some clusters had dozens of
    near-duplicate variants, bloating input tokens with no decision-quality
    benefit), total_count (the real number of members in this cluster)}, ...]
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
      scene_index: 0-based position of this scene within THIS batch — echo
        back unchanged on the matching scene_beat so it can be matched to
        the correct input scene.
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
Every scene_beat MUST include the scene_index of the input scene it answers.
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
            "scene_index": {"type": "integer"},
            "canonical_scene_id": {"type": "string"},
            "discarded_aliases": {"type": "array", "items": {"type": "string"}},
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

`start_time`/`end_time` deliberately absent from this schema (removed after
a real-video finding): write-back always uses the input group's real frame
timestamps, never the LLM's — requiring these as output fields caused
validation failures when the model returned null for values it doesn't
actually know, dropping entire otherwise-valid windows for zero benefit.

`scene_index` added after a real-video finding (first live L4 run): the
model sometimes returns fewer `scene_beats` than input scenes (e.g. 3
beats for 4 scenes in one batch). The original design matched beats to
input scenes by list position (`zip`), which silently dropped whichever
scene got cut — a real gap in scene time coverage, contradicting the
finalizer's own "every chunk/shot time range covered by ≥1 scenes row"
rule. Each input scene now carries a `scene_index` (0-based position in
its batch) that the model must echo back on its matching beat; the runner
matches by that index instead of position. Any input scene with no
matching beat still gets a scene row — a deterministic fallback record
built from frame timestamps alone (`summary: "(no beat returned by story
architect for this scene)"`, empty participants) — so time coverage is
never gapped even when the model under-returns.

Second real-video finding, same first live run, different bug: even with
`scene_index` matching working correctly, the finalizer's coverage check
still failed — "6 time gap(s) totalling 17.0s not covered by any `scenes`
row (checked against shots)". Root cause: `_SceneGroup.start_time`/
`end_time` were computed as the min/max timestamp of the group's own
Qwen-analyzed keyframes — but keyframes are a sparse ASHFS-selected subset
of a shot's frames, not every frame, so a scene's raw keyframe span is
narrower than the shot(s) it actually covers. Fix: `_snap_group_boundaries()`
(`pipeline/level4/story_architect_runner.py`), called right after grouping,
widens each group's bounds to be contiguous with its neighbors (boundary =
midpoint between adjacent groups' raw keyframe endpoints) and snaps the
first/last group out to the overall min/max shot start/end fetched via
`get_shots_for_video`. Pure widening, so it cannot invalidate anything
already correct — `_scene_speaker_turns`' overlap matching and
`_compute_usability_score`'s color-grade/transcript-confidence matching use
these same bounds and only gain from a wider window, never lose data.

Third finding, same audit pass: whole-window failure (LLM call failed, or
`WriteSceneBeatsOutput` failed pydantic validation) used to `continue` with
**zero** scene records written for every scene in that window — same
coverage-gap class as the scene_index bug, just not closed by that fix
(which only handles a partially-short response, not a wholly-missing one).
Fixed by extracting `_fallback_scene_record()` and calling it for every
group in the window on both failure paths, not just the per-scene
missing-beat case. All three drop points (window-call-failed,
window-validation-failed, scene-missing-from-response) now funnel through
the same fallback, so "the model didn't answer" can never mean "this time
range has no `scenes` row at all."

Fourth finding (next live run after the above three fixes, gap shrank from
17.0s/6 gaps to 7.4s/2 gaps but did not reach zero): `bulk_upsert_scenes`
UPSERTs on `(video_id, canonical_scene_id)`. When two DIFFERENT, non-
adjacent groups end up with the same canonical scene label (Qwen reusing
scene_id strings across genuinely separate scenes is exactly the disease
L4 exists to cure — see "Why L4 exists" above), the UPSERT silently
overwrote one scene's row with the other's, dropping its time range even
though `_snap_group_boundaries` had already made the *input* groups
gap-free — the loss happened at write time, not at grouping time. Fixed
via `_disambiguate_canonical_scene_ids()`, called right before
`bulk_upsert_scenes` in `run_story_architect`: first occurrence of a label
keeps it (idempotent reruns per rule 15), every later occurrence in the
same batch gets a `__at_{start_time}s` suffix, which is always unique
since two distinct scenes never share both bounds.

Sixth finding, `grounding_runner.py` (1b, relation canonicalization) —
recurring `Expecting ',' delimiter` malformed-JSON errors on real videos,
worsening across reruns (OTHER-bucket rate climbing 15.5% -> 19.7% ->
29.6%). Root cause: `run_relation_canonicalization` sent EVERY distinct
relation cluster in ONE uncapped call — `CANONICALIZE_RELATIONS_TOOL`
requires a `reasoning` string per cluster, and with ~150+ clusters typical
for a video (per this doc's own "200+ distinct relation strings observed"
note), output routinely exceeded `_RELATION_LLM_MAX_TOKENS` (4096) and got
cut off mid-JSON — a real truncation bug wearing the same symptom as the
"weak backend" malformed-JSON case the retry logic was built for, but not
fixable by retrying alone (truncation recurs, just at a different random
cutoff each time — explaining the climbing OTHER rate: whichever clusters
got cut varied per run and silently defaulted to OTHER). This was also a
standing violation of rule 23 (batch caps apply everywhere a list scales
with video length) — 1b only capped `raw_relations` *shown per cluster*,
never the cluster *count*. Fixed: `_RELATION_BATCH_SIZE = 25`, clusters
sub-batched and dispatched concurrently via `asyncio.gather` (same pattern
1a's `_SPEAKER_BATCH_SIZE` already used), results merged across batches
before the OTHER-bucket / write-back step.

Fifth finding, same run: `beat.participants` sometimes come back as
`"P1 (Samay Raina)"` instead of the bare pid `"P1"` given in
`cast_payload` keys — exact-match validation dropped every participant in
the scene as "not in cast," even though the identity was unambiguous.
Fixed via `_resolve_participant_pid()`: tries exact pid match, then the
text before a `"("`, then a case-insensitive match against cast display
names — still closed-set (rule 12), never invents a pid outside
`cast_payload`, just tolerant of the model's formatting.

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

STATUS: the Pinecone half below is SUPERSEDED by B7 (see "Pinecone
Namespaces" section) — `finalizer.py` now writes scene/storyline embeddings
directly to `scenes.embedding`/`storylines.embedding` via
`update_scene_embeddings`/`update_storyline_embedding` instead of calling
`knowledge_base/pinecone_kb/indexer.py` (deleted). The Neo4j half is
unchanged/current. Left below for historical context.

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

---

## PIPELINE ADDENDUM — Motion Graphics/Compositing + Hardening

### STATUS: B1/B2/B5/B6/B8 RESOLVED (doc-only, below). A1-A5 + B3/B4/B7 PLANNED, phased build in progress.

### PART A — Motion Graphics / Compositing Features

L1-L4 (understand + store) unchanged. New surface lands in **L2** (one new
deterministic stage, matting) and **L6** (one new agent — Compositing Agent —
+ new `EditOperation` types the Editing Director executes).

#### A1. Background matting (green screen / background removal)

**Level: L2** — per-shot, parallel with Color Grading (no cut-order dependency).

- New file: `pipeline/level2/matting_runner.py` — wraps RVM (Robust Video
  Matting) or MODNet, per-shot inference
- Modal Image: add matting model deps to L2 image build in `modal_app.py`
- New table:
  ```sql
  CREATE TABLE shot_mattes (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shot_id       UUID REFERENCES shots(id),
    video_id      UUID REFERENCES videos(id),
    r2_key        TEXT NOT NULL,   -- alpha-matte video or PNG sequence
    model_version TEXT NOT NULL
  );
  ```
- `pipeline/level2/updater.py` — extend to write `shot_mattes` rows alongside `color_grades`

#### A2. Stock asset library + embedding index

**New subsystem under `knowledge_base/`** — infrastructure, build once, query from L6.

- New folder: `knowledge_base/stock_assets/`
  - `client.py` — Pexels/Storyblocks API client (or curated internal library)
  - `indexer.py` — embeds description/tags with `bge-large-en-v1.5` (same model
    already used for `searchable_facts` and L4 relation clustering — no second
    embedding model)
- New table:
  ```sql
  CREATE TABLE stock_assets (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source        TEXT NOT NULL,        -- pexels | storyblocks | internal
    external_id   TEXT,
    description   TEXT,
    tags          TEXT[],
    license_type  TEXT NOT NULL,        -- see licensing note below
    embedding     vector(1024),
    r2_cache_key  TEXT
  );
  CREATE INDEX ON stock_assets USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
  ```

#### A3. Background/B-roll selection (the judgment call)

**Level: L6** — new agent, same tier logic as Color Grading/Caption agents:
retrieval cheap+deterministic, the *pick* is the one LLM-worthy step.

- New file: `pipeline/level6/compositing_agent.py`
  - submodule `background_selector.py` — embed scene transcript + `scene_mood`/
    `tags` (already in `frame_analyses`) → top-k `stock_assets` candidates →
    one LLM call picks + times it (loop? trim? offset?)
- New prompt: `prompts/background_selection.py`
- New table:
  ```sql
  CREATE TABLE background_assignments (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scene_id     UUID REFERENCES scenes(id),
    asset_id     UUID REFERENCES stock_assets(id),
    start_offset FLOAT DEFAULT 0,
    loop         BOOLEAN DEFAULT false,
    rationale    TEXT
  );
  ```

#### A4. Layering/compositing mechanics (video-on-video, PiP, background swap)

**Level: L6, Editing Director** — pure mechanics once A1-A3 decided *what*
goes where. No new agent — extend the one that already turns operations into FFmpeg.

- Extend `shared/types.py`: add `LAYER_COMPOSITE` to `EditOperation` enum —
  `{base_layer, overlay_layer, position, opacity, blend_mode, z_index}`
- Extend `pipeline/level6/editing_director.py` — new branch in XML/FFmpeg
  writer for `LAYER_COMPOSITE` ops (`filter_complex overlay`, alpha compositing)
- New table:
  ```sql
  CREATE TABLE layer_composites (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cut_list_item_id  UUID REFERENCES cut_list_items(id),
    layer_type        TEXT NOT NULL,  -- background_swap | pip | overlay
    source_ref        UUID,           -- background_assignments.id or another cut_list_item_id
    position          JSONB,
    opacity           FLOAT DEFAULT 1.0,
    z_index           INT DEFAULT 0
  );
  ```

#### A5. Zoom in/out and highlight/emphasis effects

**Split, same pattern as A1-A4.**

- **Mechanics (deterministic)** → `pipeline/level6/editing_director.py` —
  `zoompan`/`crop` for Ken-Burns motion, overlay shapes (circle/arrow/underline)
  for callouts. Extend `EditOperation` enum: `ZOOM_EMPHASIS {start_rect,
  end_rect, easing, duration}`, `HIGHLIGHT_CALLOUT {shape, target_bbox,
  start_time, duration}`
- **Decision (what to zoom/highlight on)** → `pipeline/level6/compositing_agent.py`,
  submodule `emphasis_selector.py` — reuses existing signal, no new data
  collection: `beat_type`/`tension_level`/`emotion` from `frame_analyses`,
  face bboxes from `face_appearances`. Rule-first (e.g. "zoom into speaker
  face when tension_level spikes"), LLM only if a rule can't decide.
- New table:
  ```sql
  CREATE TABLE emphasis_effects (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cut_list_item_id  UUID REFERENCES cut_list_items(id),
    effect_type       TEXT NOT NULL,   -- zoom | highlight
    parameters        JSONB NOT NULL,
    rationale         TEXT
  );
  ```

#### Updated L6 diagram

```
edit_plan (finalized)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  L6 — PARALLEL (all consume the same finalized edit_plan)   │
│                                                               │
│  Editing Director   →  cut list/XML + layer/zoom/highlight   │
│                         op execution (mechanics)              │
│  Compositing Agent  →  background selection + emphasis        │
│                         selection (judgment)                  │
│  Color Grading      →  per-shot params, sequence-aware        │
│  Audio Sync         →  levels, ducking, multicam align         │
│  Caption/Text Agent →  on-screen text per TEXT_OVERLAY op      │
└────────────────────────────────────────────────────────────┘
```

Dependency: Compositing Agent's *decisions* (A3, A5) feed Editing Director's
*execution* (A4, A5 mechanics) — same one-way relationship as L5→L6 generally.
Matting (A1) is the one exception living in L2, not L6 — no cut-order dependency.

#### Before building A3: stock licensing

Confirm license tier of whichever API (Pexels/Storyblocks/etc.) actually
covers commercial redistribution in a paid client's final delivered video —
not just personal/internal use. `license_type` on `stock_assets` (above) lets
this be filtered at selection time even if not enforced day one.

### PART B — Hardening (resolved decisions)

**B1 — Postgres vs Neo4j duality (RESOLVED).** PostgreSQL is the single
source of truth for the knowledge graph (`kg_nodes`/`kg_edges`, plus L4's
`scenes`/`storylines`). Neo4j is a **derived projection**, rebuilt/patched
from Postgres state — never written to independently. This already matches
what L4's `updater.py` does (`canonical_relation` correction pass rewrites
Neo4j edge types from Postgres, never the reverse); stating it here removes
any remaining ambiguity for new code. Any future Neo4j write must originate
from a Postgres read, never the other way.

**B2 — Cost & Latency Budgets.**

| Level | Target cost / video-minute | Target latency / video-minute |
|---|---|---|
| L1 (ASHFS + Whisper) | ~$0.01 (GPU time only, no paid API) | ~15s |
| L2 (Face + Color + Diarization) | ~$0.01 (GPU time only) | ~20s |
| L3 (Qwen3-VL) | ~$0.03 (self-hosted vLLM, GPU time) | ~40s |
| L4 (Grounding + Story Architect, Groq) | ~$0.02 (Groq token cost) | ~10s |
| L5 (Planning, Groq) | ~$0.01 (Groq token cost, one-time per plan not per minute) | ~15s per plan |
| L6 (Color/Caption/Compositing agents + render) | ~$0.01 (Groq) + render time | render-bound, not agent-bound |

These are targets, not SLAs — tune after first real end-to-end run. Tracked
per-video via `processing_jobs.meta` JSONB: every level's `updater.py`/
`finalizer.py` writes `{cost_usd, duration_s}` into it (extends the existing
`StepTimer` pattern already used for L1-L3 timings).

**B3 — Test plan.** See `tests/` (added this pass): `tests/fixtures/` (2-3
golden reference videos + hand-verified expected output — to be recorded),
`tests/unit/` (cut-snapping, LUFS calc, PID remap, matting — pure functions
per the "deterministic where possible" design principle), `tests/integration/`
(pydantic schema-drift tests for Qwen/Groq responses — response-shape drift
is an already-documented expected failure mode for L3/L4/L5/L6). Replaces
the old Step 7 "spawn agents to audit" plan.

**B4 — Observability beyond the OTHER-bucket alert.** Shared table:
```sql
CREATE TABLE pipeline_alerts (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id    UUID REFERENCES videos(id),
  level       SMALLINT NOT NULL,
  alert_type  TEXT NOT NULL,
  value       FLOAT,
  threshold   FLOAT,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);
```
Each level's `finalizer.py` writes a row when a threshold trips: L2 pyannote
failure rate, L4 `llm_unresolved_final` rate, L4/L5 confidence-escalation
trigger rate, L4 `canonical_relation = 'OTHER'` rate (already specced).

**B5 — Migration/rollback strategy (RESOLVED).** Policy: every migration
under `knowledge_base/postgres/migrations/` must be additive/backward-
compatible (new tables/columns, nullable or defaulted — never a destructive
`DROP`/`ALTER ... NOT NULL` against existing populated columns without a
backfill step in the same migration). A failed migration on a live KB rolls
back via the enclosing transaction — never partial-apply. Documented in
`knowledge_base/postgres/migrations/README.md` (added this pass). Backfill
note: videos processed before migration `005_kg_relation_canonical.sql` have
`kg_edges.canonical_relation IS NULL` — L4's Grounding Agent relation-
canonicalization step must be re-run per-video to backfill, not assumed
retroactively populated; `finalizer.py`'s quality gate should not require
`canonical_relation` on pre-005 videos unless they're explicitly re-run.

**B6 — Cross-video scope (RESOLVED).** v1 is explicitly scoped to
**single-video** identity — `persons.pid` (P1, P2, ...) is only stable
*within* one video's `videos.id`, not across videos. Cross-video identity
resolution (recognizing the same real person across two different videos)
is **out of scope for v1**. The real mechanism, when built: `person_identities
(id, canonical_name, embedding vector(512))` with `persons.identity_id` FK,
resolved via `arcface_embedding` cosine similarity against
`person_identities.embedding` at a confidence threshold, human-reviewed
below it. Not built this pass — v-next, same as Motion Graphics Agent.

**B7 — Vendor consolidation (Pinecone vs pgvector).** Deferred to end of
phased build per suggested order (cleanup, not a blocker). When done: drop
Pinecone, rely solely on existing `ivfflat` indexes on `searchable_facts`/
`kg_nodes`; delete `knowledge_base/pinecone_kb/`; update `indexer.py` call
sites in L3/L4/L5's `updater.py` files to write Postgres only.

**B8 — Failure Mode Reference.**

| Level | Failure | Fatal? | Retry policy | Who/what intervenes |
|---|---|---|---|---|
| L2 | pyannote diarization fails | No — non-fatal | None, `speaker_turns` empty for video | L4 Grounding Agent has nothing to resolve; scenes still built without speaker attribution |
| L3 | Qwen response shape drift | No — per-frame | Reject + log, frame skipped | Manual review if skip rate high |
| L4 | Speaker turn unresolved by both passes | No | Confidence-gated escalation: cheap pass → strong pass (bounded, 1 extra call) | Marked `llm_unresolved_final`, L5 plans around it |
| L4 | `canonical_relation` = OTHER > 5% of video's edges | No, but alerted | None automatic | Human reviews `pipeline_alerts`, may version ontology |
| L4 | Finalizer completeness check fails | Yes — job `FAILED` | None automatic | `error_msg` set, video reprocessed after fix |
| L5 | Plan duration out of tolerance after 2-3 LLM correction attempts | No | Falls back to programmatic trim (drop lowest-relevance clips) | None — deterministic fallback always succeeds |
| L5 | Referenced `scene_id`/timestamp invalid | Yes — plan rejected | None automatic | User/L5 caller must re-request plan |
| L6 | Neo4j projection write fails | No — non-fatal | None automatic | Logged; Postgres remains source of truth; retry/backfill is v-next. (Pinecone row superseded — B7 dropped Pinecone entirely, projection is Neo4j-only now.) |
| L6 | `AUDIO_DUCK_REQUEST` op with no secondary audio asset | No — documented no-op | N/A | Surfaced as a note in `run_level6`'s result, not silent |

---

## PIPELINE ADDENDUM 2 — QA Agent + Correction Feedback Loop + Client Style Profiles

### STATUS: IMPLEMENTED — not yet run end-to-end (same caveat as every other
level in this doc: code is complete, `py_compile`-clean, 175/175 pytest
passing, migrations 001-016 sequential with no gaps — but no real video has
exercised the QA Agent, correction feedback loop, or style-profile priors
yet, because no real video has exercised L1-L6 itself yet). User overrode
the original gating recommendation ("build after first real client video")
and asked for all three built now — done. The original rationale below is
preserved for why the gate existed, not as a currently-blocking status.

### Why gated, and why the feedback loop is the exception

L1-L6 plus Addendum 1 are fully coded but per their own STATUS lines have
never run end-to-end on a real paying job. Adding more agents before that
happens repeats the same failure mode already flagged in this doc's own
design discipline: sophistication added without a feedback signal from
reality to say whether it's the *right* sophistication. Concretely — a QA
agent's checks and thresholds (what counts as "black frame," what loudness
range is acceptable, what "drifted caption" means in practice) are informed
by what actually breaks on a real render, which doesn't exist yet. Building
it now means guessing thresholds twice: once now, once for real after the
first client video exposes what actually goes wrong.

The feedback loop doesn't have this problem — `scene_overrides` and
`storyline_overrides` already exist (Addendum 1, L4), so wiring a write path
into them costs nothing to have ready, and every real correction from here
forward becomes data instead of being lost. This is also the actual moat:
a system that improves from usage (corrections → future fine-tuning/few-shot)
beats a system that only gets more features.

**No new numbered level for any of these three.** QA sits at the tail of
L6 (same "specialized action agent" tier as Color/Caption/Compositing).
The feedback loop is a write path into L4/L5's existing override machinery,
not a stage with its own input/output contract. Style profiles are a table
L5/L6 read as a soft prior, same shape as reading `persons`/`scenes` — not
a processing stage at all.

### 1. QA / Validation Agent — IMPLEMENTED

`pipeline/level6/qa_agent.py` + `prompts/qa_review.py` + migration
`015_qa_reports.sql`. Wired as the final step of `run_level6`, after render
+ all other L6 agents. Deterministic checks (blackdetect/silencedetect via
FFmpeg, loudness re-measured against the actual delivered render via
`audio_sync.py`'s own filter-builder, caption drift vs L3's
`dialogue_subtitle`, color clipping via `color_grading_runner.py`'s existing
clamp bounds — all reused, nothing re-invented) plus one Groq intent-match
pass. Surfaces `qa_status`/`qa_report_id` in `run_level6`'s result without
raising or blocking (rule 24 — reports, never edits). Spec below is the
as-built design.

**Level: L6, tail** — runs after Editing Director's render, before delivery.
Split same as every other L6 agent: deterministic checks first (cheap, no
LLM), one narrow LLM pass only for what a formula can't catch.

**Deterministic checks (no LLM, run first, always):**
- Black-frame detection — FFmpeg `blackdetect` filter over the rendered output
- Silence/audio drop-out detection — FFmpeg `silencedetect`
- Loudness range check — reuse `audio_sync.py`'s existing `loudnorm`
  measure-pass output (already computed for normalization, don't recompute)
- Caption/transcript drift check — compare each `TEXT_OVERLAY_REQUEST`'s
  rendered caption text against its source `dialogue_subtitle` (already
  grounded in real transcript per L6 rules) — pure string diff, not LLM
- Color clipping check — reuse `color_grades`/`sequence_color_adjustments`
  data already computed, flag if the applied FFmpeg `eq`/`colorbalance`
  params would clip (out-of-range values), don't re-analyze pixels

**LLM pass (Groq `qwen/qwen3.6-27b`, one call per delivered plan, text-only —
no vision call, since sampled-frame visual review isn't worth a second model
tier for v1):** compare the `edit_plan.user_prompt` (stated intent) against
the assembled `storylines`/`scenes` text actually selected, flag if the
delivered cut's content doesn't match what was asked for (e.g. user asked
for "Person X's commentary only," QA checks `cut_list_items` participants
against that). Closed-set discipline same as every other L6 agent — reports
a pass/fail + reasoning, never edits the plan itself (rule 18: L6 doesn't
re-interpret intent, it flags for a human).

New table:
```sql
CREATE TABLE qa_reports (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  edit_plan_id   UUID REFERENCES edit_plans(id),
  video_id       UUID REFERENCES videos(id),
  status         TEXT NOT NULL CHECK (status IN ('pass', 'warn', 'fail')),
  deterministic_checks JSONB NOT NULL,   -- {black_frames: [...], silences: [...], loudness_range, caption_drift: [...], clipping: [...]}
  llm_review     TEXT,
  llm_status     TEXT CHECK (llm_status IN ('pass', 'warn', 'fail', NULL)),
  created_at     TIMESTAMPTZ DEFAULT NOW()
);
```

`processing_jobs`-style gate: a `fail` status blocks delivery (surfaced to
whatever delivers the video to the client, human-in-the-loop review), a
`warn` delivers with a flagged note, `pass` delivers clean. Never silently
auto-fix — QA reports, it doesn't edit (same one-way-boundary discipline as
L5→L6 generally).

File: `pipeline/level6/qa_agent.py`, prompt: `prompts/qa_review.py`.

### 2. Correction Feedback Loop — BUILD NOW

**What exists already (Addendum 1, L4):** `scene_overrides` and
`storyline_overrides` tables let a human-caught correction patch a `final`
storyline/scene without burning a full L4 re-run. What's missing: nothing
currently *writes* to them from a real correction event, and nothing logs
the correction anywhere durable for future fine-tuning — a fix today is a
one-off patch, not data.

**New table — the actual corrections dataset:**
```sql
CREATE TABLE correction_events (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id           UUID REFERENCES videos(id),
  level              SMALLINT NOT NULL,        -- which level's output was corrected (2, 4, 5, 6)
  entity_type        TEXT NOT NULL,            -- speaker_turn | canonical_relation | scene | storyline | edit_plan | qa_report
  entity_id          UUID NOT NULL,
  field              TEXT NOT NULL,
  original_value     JSONB NOT NULL,
  corrected_value    JSONB NOT NULL,
  correction_source  TEXT NOT NULL,            -- client | internal_editor | qa_agent_flag
  reason             TEXT,
  created_at         TIMESTAMPTZ DEFAULT NOW()
);
```

**Write path, additive to code that already exists:**
- `scene_overrides`/`storyline_overrides` inserts (Addendum 1, already
  planned but never wired to a real trigger) now always also insert a
  matching `correction_events` row — one function,
  `pipeline/feedback/correction_logger.py::log_correction()`, called from
  both override-insert sites so there's exactly one place this can drift.
- `edit_plan_revisions` (L5, already exists — "diffs, not regenerates," rule
  21) gets the same treatment: every revision insert also logs a
  `correction_events` row with `level=5`, `entity_type='edit_plan'`,
  `original_value`/`corrected_value` derived from the diff.
- L4 speaker-turn / relation corrections (if a human overrides an
  `llm_tiebreak` or `canonical_relation` after the fact) — same pattern,
  needs a small new admin write path since none exists yet for L4 outputs
  specifically (L4 currently has no post-hoc human-correction entry point
  at all — this is a real gap this item fixes, not just instrumentation).

**Why this is the moat, concretely:** `correction_events` rows are the
first-class dataset for (a) few-shot examples injected into L4/L5 prompts
for repeat clients — "here's how this client's speaker attributions usually
get corrected," (b) eventual fine-tuning data once volume justifies it, (c)
the QA Agent's own threshold tuning once it exists (real corrections tell
you what the deterministic checks should actually flag). None of this
requires L1-L6 architecture changes — it's a table and one logging function.

File: `pipeline/feedback/correction_logger.py`, `pipeline/feedback/__init__.py`.
Migration: `014_correction_feedback.sql` (adds `correction_events`, and the
missing L4 human-correction write path table if needed — TBD at
implementation time whether L4 corrections reuse `scene_overrides`/
`storyline_overrides` field-level shape or need their own `speaker_turn_
overrides`/`relation_overrides` tables; decide when actually wiring the L4
entry point, not speculatively now).

### 3. Client Style Profiles — IMPLEMENTED

`videos.client_id` (nullable) + `client_style_profiles` table, migration
`016_client_style_profiles.sql`. Read as a soft prior by L5 Pass B
(`pipeline/level5/planner_runner.py`, `pacing_preference` →
`client_style_pacing_preference` in `prompts/l5_sequencing.py`, explicitly
subordinate to the user's per-edit `pacing_preference` and to
`causal_link_to_next` on conflict) and L6's Color Grading Agent
(`brand_colors` → `target_brand_bias`) + Caption/Text Overlay Agent
(`caption_style` → `client_style_prior`, never touches caption `text`
itself). No `client_id`, or a `client_id` with no profile row, produces
byte-identical prompt payloads to pre-Addendum-2 behavior — verified by
construction, every lookup gates on truthiness before injecting anything.
Spec below is the as-built design.

**New table:**
```sql
CREATE TABLE client_style_profiles (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id          TEXT NOT NULL,             -- external client identifier, not a users table FK (no user system speced yet)
  caption_style      JSONB DEFAULT '{}',        -- font, size, position, karaoke vs static preference
  pacing_preference  TEXT,                       -- fast | moderate | slow, informs L5 Pass B cut density
  brand_colors       JSONB DEFAULT '{}',         -- target color-grade bias, informs L6 Color Grading Agent
  default_platform   TEXT,                       -- reel | full_cut | youtube | ...
  notes              TEXT,
  created_at         TIMESTAMPTZ DEFAULT NOW(),
  updated_at         TIMESTAMPTZ DEFAULT NOW()
);
```

**Read contract, additive not required:** L5 Pass B (sequencing/pacing) and
L6's Color Grading Agent + Caption/Text Overlay Agent read
`client_style_profiles` for the video's `client_id` **if a row exists** —
soft prior injected into the existing prompts (e.g. Color Grading Agent's
sequence-delta prompt gets a "target brand bias" field alongside neighbor
deltas), never a hard override of scene-grounded reasoning. No row = current
behavior unchanged, exactly as today. This is why it's safe to defer: it's
additive to already-working prompts, not a prerequisite for them.

Where `client_id` comes from is intentionally left open — no client/user
system exists in the current schema (`videos` has no `client_id` column
yet). Adding one is part of this item's implementation, not decided now.

### Engineering rules extending the list (24-26)

24. **QA Agent reports, never edits** — a `fail`/`warn` status blocks or
    flags delivery for human review; it never mutates `edit_plans`,
    `cut_list_items`, or triggers an automatic re-render. Same one-way
    boundary as every other cross-level read contract in this doc.
25. **Every override write is also a correction_events write** — no code
    path inserts into `scene_overrides`/`storyline_overrides`/
    `edit_plan_revisions` without also logging to `correction_events` via
    `log_correction()`. One function, not duplicated logic per call site,
    so the corrections dataset can't silently drift out of sync with the
    override tables it's supposed to mirror.
26. **Style profiles are a prior, never a constraint** — `client_style_
    profiles` rows inform prompt context; they must never hard-block a
    scene-grounded decision (e.g. a pacing preference doesn't force a cut
    that violates `causal_link_to_next`). If a future version needs hard
    constraints, that's a new field with its own validation path, not an
    overload of this table's soft-prior semantics.

---

## PIPELINE ADDENDUM 3 — L7 Evaluation, L8 Human Feedback, L9 Reward/Punishment

### STATUS: SPEC'D, AWAITING USER REVIEW — DO NOT IMPLEMENT WITHOUT SIGN-OFF

Same convention as the original plan at the top of this doc: written before code, reviewed before build. This addendum exists because a real audit (agent-run, against live code and the live Postgres DB, not doc claims) scored the pipeline's evaluation/guardrail coverage **4.3/10** — see "Audit findings" below. L7-L9 are the direct, gap-driven response, not speculative feature-building. Every item below traces to a specific numbered gap the audit found.

### Audit findings (2026-07-31, against live code + live DB — 12 videos processed at the time)

| # | Area | Score | Real finding |
|---|---|---:|---|
| 1 | Schema/structural validation | 8/10 | Real pydantic gates + closed-set enums exist. But 6/176 `tests/integration` schema-drift tests **currently fail** against live code — contradicts this doc's own earlier "175/175 passing" claim (Addendum 2 STATUS line), which was accurate when written and has since drifted. |
| 2 | Completeness gates | 7/10 | `finalize_level4` genuinely checks speaker-turn terminality / scene coverage / non-null synopsis before flipping `storylines.status`→`final`. Real, not docstring-only. But live `processing_jobs` has **zero rows at level 4/5/6** — the gate has never fired via the real orchestrator, only via direct function calls in dev/debug sessions. |
| 3 | Confidence-based escalation | 7/10 | Real, wired, bounded (`grounding_runner.py`). Never exercised to completion on a real orchestrated run. |
| 4 | Deterministic quality checks | 7/10 | Real, reused (blackdetect/silencedetect/loudness/caption-drift/color-clip in `qa_agent.py`). One real `qa_reports` row exists in the live DB (`status='warn'`). |
| 5 | LLM-as-judge checks | 4/10 | Code exists (Groq intent-match pass in `qa_agent.py`) but the one real row in the DB has `llm_status=NULL` — didn't actually run to completion on the only real execution so far (missing `GROQ_API_KEY` in that run's environment). Narrow scope (intent-match only) even when it does run. |
| 6 | Offline/golden-set testing | 1/10 | `tests/fixtures/README.md` states plainly: no video assets are committed. Zero golden cases exist. |
| 7 | Human-in-the-loop feedback capture | 3/10 | `correction_events` + `correction_logger.py` real and wired for **L5 only**. `log_scene_correction`/`log_storyline_correction` (L4) have **zero callers anywhere in the codebase** — dead code, no entry point exists to actually log an L4 correction. Live table: 0 rows. |
| 8 | Regression/drift tracking | 5/10 | `pipeline_alerts` real and written to (7 live rows, `l4_canonical_relation_other_rate` breaching its 0.05 threshold at 0.17-0.30 on every real run). Nothing reads or surfaces these — write-only log. |
| 9 | Cost/latency observability | 2/10 | Only `logger.info(...)` of token counts — no durable table. `processing_jobs.meta` was speced (B2) to carry `{tokens, cost_usd}` and never implemented. |
| 10 | End-to-end/integration testing | 2/10 | "Integration" tests are schema-drift/pydantic unit tests, not full-pipeline runs. Nothing asserts on output *quality* — only "didn't throw." |

**Weighted overall: 4.3/10** (completeness gates, LLM-judge, human-feedback loop, and offline-testing weighted higher — these are the ones that catch *substantive* quality regressions, not just structural bugs).

### Design principle for L7-L9

Same discipline as every other level in this doc: **deterministic where possible, LLM only for the specific judgment call that can't be reduced to a formula** (rule 20, applied one level up). L7 mostly extends things that already exist and work (`qa_agent.py`'s deterministic checks, `pipeline_alerts`) rather than inventing new machinery. L8 is schema + one missing entry point, not a new agent. L9 is explicitly bounded — a reward/punishment *signal* and its two concrete consumers (few-shot injection, ontology-versioning triggers), **not** an RL training loop or a fine-tuned reward model. This pipeline doesn't have the data volume for that yet (same "don't build for the level you don't have" call already made for Motion Graphics, cross-video identity, and the Style Learning System's embedding phase).

---

### LEVEL 7 — EVALUATION

Closes gaps #1, #3, #4, #6, #8, #9, #10.

**7a. Golden-set bootstrap (closes gap #6).** `tests/fixtures/` gets its first real case: the video/edit-plan pair already produced and manually verified in this session — `video_id=97199656-d176-46aa-88b4-026670be4576`, `edit_plan_id=b276be9d-c803-4cba-b86e-e3da624c479f`, rendered output confirmed structurally valid (full-file ffmpeg decode scan, zero errors) and matching the plan's requested clip order. Snapshot: the `storylines`/`scenes` rows (JSON dump, same shape as the L4 export already produced this session), the `edit_plans.operations` array, and the rendered output's `ffprobe` metadata (duration/codec/resolution) as the expected-output baseline. `tests/integration/test_e2e_golden.py` asserts a rerun of L4→L5→L6 against this video reproduces output within tolerance (scene count ±2, achieved duration ±5%, no `qa_status='fail'`) — not byte-identical (LLM non-determinism is expected and already documented extensively in this doc), but structurally equivalent. This is the FIRST end-to-end quality test this pipeline will have (closes gap #10 partially — see 7d for the rest).

**7b. Rubric scoring — extend `qa_agent.py`, don't replace it (closes gap #1, partially #5).** The existing Groq intent-match pass becomes one dimension of a multi-dimension rubric, not the whole check:

```
SYSTEM PROMPT — QA Agent: Rubric Review (extends existing intent-match prompt)
────────────────────────────────────────────────────────────────────────────
Score this delivered edit against FOUR dimensions, 0-10 each, grounded ONLY
in the storylines/scenes/edit_plan data already provided (same closed-set
discipline as every other agent in this pipeline — never invent a judgment
about footage you weren't shown):

  intent_match      — does the assembled content match user_prompt
                       (existing check, unchanged)
  narrative_coherence — do the selected scenes in this order tell a
                       coherent story per their causal_link_to_next chains
  pacing_consistency  — do cut durations follow a defensible rhythm, or
                       does the sequence feel arbitrary (grounded in
                       sequence_color_adjustments/cut_list_items durations
                       already computed, not re-analyzing pixels)
  technical_cleanliness — cross-check against the deterministic checks
                       already run (blackdetect/silencedetect/caption-drift/
                       clipping) — does the LLM's read agree with what the
                       formulas already found, flag if not (a real
                       disagreement here is itself a signal worth logging)

Return one score + one-sentence rationale per dimension via the
score_edit_rubric tool call. Never invent a score dimension not listed.
```

New table `evaluation_scores` (one row per `qa_reports` row, 1:1 extension):

```sql
CREATE TABLE evaluation_scores (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  qa_report_id          UUID REFERENCES qa_reports(id) ON DELETE CASCADE,
  edit_plan_id          UUID REFERENCES edit_plans(id),
  intent_match          FLOAT,
  narrative_coherence   FLOAT,
  pacing_consistency    FLOAT,
  technical_cleanliness FLOAT,
  rationale             JSONB NOT NULL DEFAULT '{}',  -- {dimension: sentence}
  created_at            TIMESTAMPTZ DEFAULT NOW()
);
```

Written by `qa_agent.py` right after the existing intent-match call — same non-fatal discipline as the rest of L6 (a failed rubric call leaves `evaluation_scores` absent, doesn't block delivery — `qa_status` is still gated by the deterministic checks + existing intent-match pass, rubric scores are additive signal for L9, not a new blocking gate; rule 24 still applies, unchanged).

**7c. Durable cost/latency tracking (closes gap #9).** New table, not a `processing_jobs.meta` JSONB blob (queryable columns > buried JSON for something this doc already wants to alert on):

```sql
CREATE TABLE llm_call_log (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id         UUID REFERENCES videos(id),
  level            SMALLINT NOT NULL,           -- 4, 5, or 6
  stage            TEXT NOT NULL,                -- e.g. 'grounding_speaker', 'story_architect', 'l5_selection'
  model            TEXT NOT NULL,
  prompt_tokens    INT,
  completion_tokens INT,
  cost_usd         FLOAT,                        -- computed from OpenRouter's per-model pricing, best-effort
  latency_ms       INT,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON llm_call_log(video_id, level);
```

One `log_llm_call()` helper in `shared/llm_client.py`, called from every `_call_tool`/`_call_groq_tool` site right after each `chat.completions.create` response (the `usage` object is already being read for the existing `logger.info` calls — this just also persists it). Non-fatal: a failed insert logs and continues, never blocks the actual LLM-consuming call.

**7d. `pipeline_alerts` gets a consumer (closes gap #8).** Extend the dashboard's existing L4 Reasoning tab (built earlier this session) with an "Alerts" panel — a simple table of `pipeline_alerts` rows for the selected video, red-highlighted if `value > threshold`. This alone closes the "write-only log" problem — no new alerting infra needed, the data already exists, it just needs a screen. (A push-notification/Slack-webhook consumer is a reasonable v-next, not required for this addendum.)

---

### LEVEL 8 — HUMAN FEEDBACK

Closes gap #7 fully, and is the direct answer to "negative prompt / negative feedback."

**8a. Fix the dead L4 correction entry point.** `log_scene_correction`/`log_storyline_correction` exist in `pipeline/feedback/correction_logger.py` but have zero callers. Add the missing call sites: a small CLI script `scripts/log_l4_correction.py` (same shape as `scripts/run_l5.py`/`run_l6.py` — this pipeline already has no admin UI, CLI is the established pattern) that takes `--video-id --entity-type (scene|storyline) --entity-id --field --corrected-value --reason` and calls the existing (already-correct, just never-invoked) logger function. This is the smallest possible fix for the biggest single gap the audit found — no new logic, just a way to actually call code that already works.

**8b. `human_feedback` — holistic/qualitative feedback, distinct from `correction_events`.** `correction_events` is field-level: "this exact value was X, should be Y." That can't express "this cut felt jarring" or "I love how this scene flows" — there's no specific field to diff. This is the actual gap behind "negative prompt / negative feedback":

```sql
CREATE TABLE human_feedback (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_id      UUID REFERENCES videos(id),
  edit_plan_id  UUID REFERENCES edit_plans(id),      -- nullable — feedback can be on the video generally
  scene_id      UUID REFERENCES scenes(id),           -- nullable — feedback can be scene-scoped or general
  sentiment     TEXT NOT NULL CHECK (sentiment IN ('positive', 'negative', 'neutral')),
  category      TEXT NOT NULL CHECK (category IN
                  ('pacing', 'color', 'caption', 'speaker_attribution',
                   'narrative', 'music_audio', 'b_roll', 'other')),
  free_text     TEXT NOT NULL,
  rating         SMALLINT CHECK (rating BETWEEN 1 AND 5),  -- optional, nullable
  source        TEXT NOT NULL DEFAULT 'client',       -- client | internal_editor
  created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON human_feedback(video_id);
CREATE INDEX ON human_feedback(edit_plan_id);
```

Same CLI-first pattern as 8a: `scripts/log_human_feedback.py --video-id --sentiment --category --text [--edit-plan-id] [--scene-id] [--rating]`. A dashboard form is a reasonable v-next (the dashboard already reads this DB — adding a write form is additive, not required for this addendum to be useful).

**8c. Negative feedback is not just "delete/ignore" — it's a first-class signal for L9.** This is the point of splitting `human_feedback` from `correction_events`: a correction says "fix this specific thing," negative feedback says "this pattern is wrong even though I can't point at one field" — e.g. "the pacing always feels too slow for this client's reels." L9 reads `human_feedback.sentiment='negative'` rows aggregated by `category` as its punishment signal (see below) — this is the entire reason category is a closed enum here, not free text: it needs to be aggregatable.

---

### LEVEL 9 — REWARD & PUNISHMENT

Explicitly bounded scope — a signal + two consumers, not a training system.

**9a. `reward_signals` — the aggregation layer.** Not raw storage (that's `evaluation_scores` + `human_feedback`, already real tables) — this is a computed rollup, recomputed periodically (a script, not a live trigger — `scripts/compute_reward_signals.py`, safe to rerun, UPSERT on `(scope_type, scope_key)`):

```sql
CREATE TABLE reward_signals (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scope_type    TEXT NOT NULL,      -- 'canonical_relation' | 'pacing_style' | 'color_style' | 'client'
  scope_key     TEXT NOT NULL,      -- e.g. the relation name, or a client_id
  reward_score  FLOAT NOT NULL,     -- rolling average of evaluation_scores dims + human_feedback sentiment, -1..1
  sample_count  INT NOT NULL,
  updated_at    TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (scope_type, scope_key)
);
```

`reward_score` formula (deterministic, no LLM — same "deterministic where possible" discipline): normalize `evaluation_scores` dimensions to -1..1, map `human_feedback.sentiment` to {positive: +1, neutral: 0, negative: -1} weighted by `rating` when present, average per `scope_key` with a minimum `sample_count` floor (don't trust an average of 1-2 samples — same confidence-weighted-shrinkage principle already used for the Style Learning System discussion). This directly extends that earlier design conversation — `reward_signals` scoped by `client` is the numeric backbone the Style Learning System's "creator profile" would eventually read from.

**9b. Reward → few-shot injection (the "moat" mechanism Addendum 2 proposed but never built).** When `L4_STORY_ARCHITECT_MODEL`/`L5_SEQUENCING_MODEL` calls run for a video with a known `client_id`, look up `reward_signals WHERE scope_type='client' AND scope_key=client_id AND sample_count >= 5` — if present, inject the top-N highest-`reward_score` past `scenes.summary`/`storylines.beats` for that client as few-shot examples in the prompt (bounded, same batch-cap discipline as rule 23 — cap at 3-5 examples, not an unbounded dump). Soft prior only (rule 26 still applies) — never overrides grounded data, only nudges style/phrasing toward what's historically scored well for that client.

**9c. Punishment → ontology/prompt versioning trigger (extends the existing OTHER-bucket mechanism, doesn't invent a new one).** `reward_signals WHERE scope_type='canonical_relation' AND reward_score < -0.3 AND sample_count >= 10` is a new alert condition feeding the SAME `pipeline_alerts` table + review workflow already speced for the OTHER-bucket rate (CLAUDE.md's existing "signal to review and version the ontology" language) — a canonical relation that's technically well-classified (not `OTHER`) but consistently scores badly in human feedback is just as real a signal for ontology review as the OTHER-bucket rate is. Same human-review-then-version response, not automated remapping — rule 12 (closed-set outputs only, human decides new categories) still applies.

**9d. Explicitly out of scope for this addendum** (v-next, same discipline as Motion Graphics/cross-video identity/style embeddings): fine-tuning any model, automated prompt rewriting, RL training loops, automated threshold adjustment without human review. `reward_signals` is a **reporting and retrieval** table, not a control system that changes pipeline behavior without a human in the loop reading `pipeline_alerts` first.

---

### New tables summary (migration `017_l7_l8_l9_evaluation.sql`)

`evaluation_scores`, `llm_call_log`, `human_feedback`, `reward_signals` — all additive per the existing migration policy (B5 — never a destructive change to existing tables). No existing table's schema changes in this addendum.

### Engineering rules extending the list (27-30)

27. **L7's rubric scoring is additive to `qa_status`, never a new blocking gate on its own** — `evaluation_scores` informs L9, it does not change whether `qa_agent.py` marks a delivery `pass`/`warn`/`fail`. That gate stays owned by the deterministic checks + existing intent-match pass (rule 24 unchanged).
28. **`human_feedback.category` is a closed enum, not free text** — same closed-set discipline as everything else in this pipeline (rule 12's principle applied to a new area); `reward_signals` aggregation requires it to be a stable, small set of keys, not an open-ended string that can't be rolled up.
29. **`reward_signals` never mutates pipeline behavior directly** — it's read by prompt-construction code (9b) and alert logic (9c), both human-reviewed downstream steps, same one-way-boundary discipline as every cross-level read contract in this doc (L5 reads L4's `final` output only, L6 reads L5's plan only, etc. — L9's signal is consumed the same way, never a live feedback loop that changes behavior without a human step in between).
30. **Every new table in this addendum must have a real caller before STATUS moves past "implemented"** — the audit's single biggest finding was code that exists but is never invoked (`log_scene_correction`, the rubric-adjacent `llm_status=NULL` row). Any implementation PR for L7-L9 must include the call site, not just the table/function, and must be verified against a real run (same standard this session's live L4/L5/L6 bug fixes were held to) before claiming done.

---

## PIPELINE ADDENDUM 4 — Editor Style Profile Learning (extends Addendum 2, item 3)

### STATUS: SPEC'D — building now, see IMPLEMENTATION STATUS SNAPSHOT at top of doc for live progress

**Not a new numbered level.** This closes a gap this doc already identified twice: Addendum 2 item 3 ("Client Style Profiles") built `client_style_profiles` as a soft-prior consumption point, but every field on it was always meant to be *hand-set* — "learned from exemplar videos" was explicitly deferred. This addendum is that deferred piece, and it plugs into levels that already exist rather than adding one:

```
INGEST  (L1+L2, unchanged)  →  EXTRACT (new, deterministic)  →  STORE (client_style_profiles, already exists)  →  CONSUME (L5 Pass B + L6 Color Grading, already reading it)  →  IMPROVE (L9 9b/9c, already built, just unwired)
```

### Why this was cheap to spec: almost everything already exists

| Piece needed | Status before this addendum |
|---|---|
| Shot-boundary detection | Already real — ASHFS, L1 |
| Per-shot 45-param color analysis | Already real — `color_grades` table, L2 |
| Word-level transcript timing | Already real — `transcript_segments`, L1 |
| `videos.client_id` column | Already exists (migration 016), nullable, just never populated on any real video |
| `client_style_profiles` table + upsert function | Already exists (migration 016) — `upsert_client_style_profile`, `ON CONFLICT (client_id) DO UPDATE` |
| L5 Pass B reading `client_style_pacing_preference` as soft prior | Already wired (Addendum 2) |
| L6 Color Grading reading `brand_colors` as soft prior | Already wired (Addendum 2) |
| L9 9b: reward-scored few-shot injection per `client_id` | Already built this session — was reported "not live-reachable, no video has client_id set" |
| L9 9c: per-client reward tracking | Already built |

**The only genuinely new code**: a deterministic metric-extraction script, and wiring `client_id` onto real video rows. Everything downstream already consumes the shape this produces.

### Phase 0 — Ingest (reuse L1+L2, no new code)

Editor supplies 3-5 of their own already-edited/finished videos. Each runs through **L1 (ASHFS + Whisper) + L2 (color grading; face/diarization optional, not needed for style analysis)** — the same Modal functions already processing the other 12 real videos in this DB. No L3-L6 needed — we are analyzing *existing* style, not generating a new edit for these.

### Phase 1 — Deterministic metric extraction (`scripts/build_editor_profile.py`, new)

Pure functions, **no LLM call** — same "deterministic where possible" discipline as `usability_score` (L4), cut-snapping (L6), and every other formula-not-model decision in this pipeline:

- **Pacing**: avg shot length, median shot length, shot-length variance, and the full cut-duration *sequence* (not just a mean — a "tight-tight-tight-hold" rhythm signature is lost if you only keep the average), computed from `shots`.
- **Color**: mean + variance per `color_grades` parameter across all shots in all 3-5 videos — the 45 parameters already exist per shot, this is pure aggregation, no new analysis.
- **Audio**: loudness target + dynamic range via an `ffmpeg loudnorm` measure-only pass — reuses the same measurement `pipeline/level6/audio_sync.py::compute_loudnorm_filter` already does, just run in measure mode against the finished video instead of a rendered output.
- **Caption style: explicitly NOT extracted.** This pipeline transcribes *spoken* audio; it has no OCR step for on-screen burned-in text, so caption density/style cannot be inferred from a finished video today. If caption style matters, it has to be a manual field on `client_style_profiles.caption_style` (already a JSONB column, already accepts hand-entered data) — stated here so nobody later assumes this addendum silently covers it.

Output: one aggregated row upserted into `client_style_profiles` via the already-existing `upsert_client_style_profile`. New helper `set_video_client_id(pool, video_id, client_id)` (queries.py, appended, not modifying `upsert_video`) tags each of the 3-5 source videos with the given `client_id` so future reward-signal aggregation (Phase 3) has something to scope by.

### Phase 2 — Confidence, not false authority

3-5 videos is a thin sample — already flagged in this session's earlier Style Learning System discussion. No new mechanism needed: `client_style_profiles` fields are already read as a **soft prior, never a hard constraint** (rule 26, unchanged) by L5/L6. `build_editor_profile.py` records `sample_count` in `notes` (existing free-text column) so it's visible on inspection, but does not gate/withhold writing the profile — a thin-but-present prior is still better than none, per the same reasoning already used for `reward_signals`' shrinkage-not-omission design.

### Phase 3 — Continuous improvement (already built — this addendum just turns it on)

With `client_id` now populated on real videos: every future edit for that client produces `evaluation_scores` + optional `human_feedback`, L9's existing `compute_reward_signals.py` rolls those into `reward_signals(scope_type='client', scope_key=client_id)`, and L9 9b (already wired into `story_architect_runner.py`/`planner_runner.py`, already tested as correctly-inert on empty data) starts actually injecting top-scoring past examples once `sample_count` crosses its floor. Nothing new to build here — this phase is "the missing link gets connected," not new machinery.

### New code, precisely (no new table, no new migration)

- `scripts/build_editor_profile.py` — the extraction script (Phase 1).
- `knowledge_base/postgres/queries.py` — append `set_video_client_id`.
- No schema change: `videos.client_id` and `client_style_profiles` already exist from migration 016.

### Where this sits, restated plainly for anyone skimming

Not L10. It is: **new data flowing into L1/L2's existing output tables → a new deterministic extraction step reading those tables → the existing Addendum 2 storage table → the existing L5/L6 soft-prior consumption → the existing L9 reward loop.** The diagram immediately below shows exactly where.

---

## END-TO-END ARCHITECTURE — WHAT'S DOING WHAT (2026-07-31)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  SOURCE VIDEO (R2 URL or local path)                                                  │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ L1 — STRUCTURE (parallel)                          MODELS: none (local, deterministic) │
│   ASHFS: shot detect → DINOv2 → keyframes    →  shots, keyframes, chunks              │
│   Whisper large-v3 (local faster-whisper)    →  transcript_segments                   │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ L2 — PER-SHOT SIGNAL (3-way parallel)              MODELS: local only                  │
│   Face: ArcFace+HSEmotion+ByteTrack (ONNX)   →  persons, face_appearances             │
│   Color: 45-param OpenCV/CuPy analysis       →  color_grades                          │
│   Diarization: pyannote-3.1 (gated HF model) →  speaker_turns                         │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────────┐
        ▼                                                │  (Addendum 4, editor's OWN
┌──────────────────────────────────────────────────┐     │   finished videos only —
│ L3 — VISION UNDERSTANDING                         │     │   does NOT continue past L2)
│   Qwen/Qwen3-VL-8B-Instruct (self-hosted vLLM,    │     ▼
│   Modal L40S/A100)                          →  frame_analyses, kg_nodes, kg_edges     ┌───────────────────────┐
└──────────────────────────────────────────────────┘     │ ADDENDUM 4 — deterministic  │
        │                                                 │ metric extraction (no LLM) │
        ▼                                                 │ pacing/color/audio stats   │
┌──────────────────────────────────────────────────┐      │  → client_style_profiles   │
│ L4 — REASONING                MODEL: deepseek/deepseek-v3.2 (OpenRouter)              │
│   Grounding Agent  (reasoning=False, closed-set)  →  speaker_turns resolved,          │
│                                                       kg_edges.canonical_relation      │
│   Story Architect  (reasoning=True, synthesis)    →  scenes, storylines(final)        │
└──────────────────────────────────────────────────┘      └────────────┬──────────────┘
        │                                                               │ soft prior
        ▼                                                               │ (rule 26)
┌──────────────────────────────────────────────────┐                    │
│ L5 — PLANNING                                                          │
│   Pass A Selection: qwen/qwen3-30b-a3b-instruct-2507 (OpenRouter)      │
│   Pass B Sequencing: qwen/qwen3-235b-a22b-2507 (OpenRouter) ◄──────────┤ client_style_pacing_preference
│                                                    →  edit_plans, cut_list_items       │  + client_style_examples (L9 9b)
└──────────────────────────────────────────────────┘                    │
        │                                                                │
        ▼                                                                │
┌──────────────────────────────────────────────────┐                    │
│ L6 — ACTION AGENTS (parallel) + RENDER                                 │
│   Editing Director: deterministic (cut snap, per-clip extract+concat) │
│   Color Grading: qwen/qwen3-30b-a3b-instruct-2507  ◄────────────────────┤ brand_colors soft prior
│   Caption/Text Overlay: qwen/qwen3-30b-a3b-instruct-2507               │
│   Compositing (A3/A5): qwen/qwen3-30b-a3b-instruct-2507                │
│   Audio Sync: deterministic DSP                                        │
│   QA Agent: deterministic checks + qwen/qwen3-235b-a22b-2507 (intent)  │
│                                                    →  out.mp4, qa_reports              │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│ L7 — EVALUATION                    MODEL: qwen/qwen3-235b-a22b-2507 (OpenRouter,       │
│   Rubric scoring (4-dim, extends QA Agent)         same call as QA intent-match)       │
│   Cost/latency log (deterministic)                                                     │
│                                                    →  evaluation_scores, llm_call_log  │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│ L8 — HUMAN FEEDBACK                MODEL: none (CLI + tables)                          │
│   log_l4_correction.py / log_human_feedback.py    →  correction_events,               │
│                                                       human_feedback                   │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│ L9 — REWARD & PUNISHMENT           MODEL: none (deterministic aggregation)             │
│   compute_reward_signals.py  →  reward_signals                                        │
│     ├─ 9b → few-shot injection back into L4 Story Architect + L5 Pass B (above)        │
│     └─ 9c → pipeline_alerts (negative-reward relation types → human ontology review)   │
└──────────────────────────────────────────────────┘

STORES:  Postgres (Neon) — source of truth for everything above (pgvector for embeddings,
         B7 dropped Pinecone).  Neo4j — derived graph projection, never written independently
         (B1).  Cloudflare R2 — keyframe/video object storage.

PROVIDERS:  OpenRouter — all L4-L7 LLM calls (deepseek-v3.2, qwen3-30b, qwen3-235b).
            Groq — configured as optional fallback only, not used by default anywhere.
            Self-hosted vLLM (Modal GPU) — L3 only, Qwen3-VL-8B.
            Local (no API) — L1/L2 models, embeddings (BAAI/bge-large-en-v1.5, currently
            broken in this environment — see IMPLEMENTATION STATUS SNAPSHOT), L6/L9
            deterministic DSP/aggregation.
```

---