from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    # Store the URL exactly as provided (postgresql+asyncpg://...).
    # Use the asyncpg_url property to get the form asyncpg accepts.
    DATABASE_URL: str

    @property
    def asyncpg_url(self) -> str:
        """Return the DATABASE_URL with the asyncpg-compatible scheme.

        asyncpg expects ``postgresql://`` (or ``postgres://``), not
        ``postgresql+asyncpg://``.  This property does the substitution
        so callers never have to remember.
        """
        return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

    # ------------------------------------------------------------------
    # Vector search: pgvector only (B7 — Pinecone dropped; all embeddings
    # live in Postgres columns with ivfflat indexes — searchable_facts,
    # kg_nodes, scenes, storylines, stock_assets).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cloudflare R2
    # ------------------------------------------------------------------
    R2_ENDPOINT_URL: str
    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_BUCKET_NAME: str

    # ------------------------------------------------------------------
    # Embedding model (local, runs on same GPU as Qwen)
    # ------------------------------------------------------------------
    # BAAI/bge-large-en-v1.5 produces 1024-dim vectors and runs on-device.
    # No external API calls are made for embeddings.
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"

    # ------------------------------------------------------------------
    # Neo4j (Graph DB for knowledge graph traversal)
    # ------------------------------------------------------------------
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"

    # ------------------------------------------------------------------
    # Speaker diarization (Level-2 Stage 0)
    # ------------------------------------------------------------------
    # Gated model on HuggingFace — requires HF_TOKEN with accepted license
    # for pyannote/speaker-diarization-3.1 (and its dependency segmentation-3.0).
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization-3.1"

    # ------------------------------------------------------------------
    # Background matting (Level-2, A1 — see CLAUDE.md PIPELINE ADDENDUM)
    # ------------------------------------------------------------------
    # torch.hub repo + variant for Robust Video Matting. Cached under
    # TORCH_HOME (already set to a persistent volume for face/color — see
    # modal_app.py face_color_image) so the repo/checkpoint download only
    # happens once per volume, not once per video.
    MATTING_HUB_REPO: str = "PeterL1n/RobustVideoMatting"
    MATTING_MODEL_VARIANT: str = "mobilenetv3"
    # Real-video measurement: unbatched every-native-fps-frame RVM inference
    # over a full shot list was the dominant L2 cost (~20% GPU util, mostly
    # idle — Python-loop-bound single-frame calls, not compute-bound). Only
    # every Nth frame gets real inference; skipped frames hold the last
    # computed alpha (see matting_runner.py::MattingModel.matte_clip).
    MATTING_FRAME_STRIDE: int = 3

    # ------------------------------------------------------------------
    # Groq (Level-4 reasoning + Level-5 planning — Grounding Agent, Story
    # Architect Agent, and L5's Selection/Sequencing passes all run on Groq).
    # ------------------------------------------------------------------
    # Provider: OpenRouter (OpenAI-compatible API, base_url in
    # shared/llm_client.py). Real two-tier split — cheap/fast for high-call-
    # volume closed-set decisions, strong for narrative/synthesis/gating
    # decisions — using OpenRouter's actual Qwen lineup (Groq only hosted a
    # single qwen3.6-27b tier, which meant "cheap vs strong" shared one model
    # and never got a real split; that limitation is gone now).
    #
    # L4 switched to deepseek/deepseek-v3.2 for both tiers (was the Qwen
    # split above) — same OpenAI-compatible tool-calling shape, no call-site
    # changes needed (see shared/llm_client.py docstring). Reasoning behavior
    # is controlled per-call via `reasoning: {"enabled": bool}` in
    # extra_body — grounding_runner.py passes False (fast closed-set
    # classification), story_architect_runner.py passes True (needs actual
    # reasoning for narrative synthesis / scene merging). 164K context is
    # not a limiting factor — L4's batch caps (rule 23) keep every call well
    # under that regardless of video length. Qwen model IDs kept below,
    # commented, in case of rollback.
    # qwen cheap/fast: qwen/qwen3-30b-a3b-instruct-2507 — $0.04815/$0.1931/M
    # qwen strong: qwen/qwen3-235b-a22b-2507 — $0.09/$0.55/M
    OPENROUTER_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None  # kept as an optional fallback provider — see shared/llm_client.py
    L4_GROUNDING_MODEL: str = "deepseek/deepseek-v3.2"
    L4_STORY_ARCHITECT_MODEL: str = "deepseek/deepseek-v3.2"
    L4_ONTOLOGY_VERSION: int = 1
    L5_SELECTION_MODEL: str = "qwen/qwen3-30b-a3b-instruct-2507"
    L5_SEQUENCING_MODEL: str = "qwen/qwen3-235b-a22b-2507"
    # Level-6 Caption/Text Overlay Agent — the one LLM-worthy decision in L6
    # (caption STYLE only, never the text itself). Cheap tier — style pick is
    # a closed-set decision, not narrative synthesis.
    L6_CAPTION_MODEL: str = "qwen/qwen3-30b-a3b-instruct-2507"
    # Level-6 Color Grading Agent — sequence-delta computation (see
    # pipeline/level6/color_grading_runner.py). Cheap tier — numeric-delta
    # reasoning over neighbor shots, not narrative synthesis. Kept as its own
    # setting (rather than reusing L5_SEQUENCING_MODEL) so L6 agents can be
    # retuned independently of planning, per the existing per-stage pattern.
    L6_COLOR_MODEL: str = "qwen/qwen3-30b-a3b-instruct-2507"
    # Level-6 Compositing Agent (A3 background_selector.py pick, A5-decision
    # emphasis_selector.py fallback) — cheap tier, closed-set pick from a
    # pre-retrieved candidate list. Kept as its own setting for the same
    # independent-retuning reason as L6_COLOR_MODEL/L6_CAPTION_MODEL above.
    L6_COMPOSITING_MODEL: str = "qwen/qwen3-30b-a3b-instruct-2507"
    # Level-6 QA/Validation Agent (PIPELINE ADDENDUM 2, item 1) — the one
    # LLM-worthy QA decision: does the delivered cut's assembled content
    # (storylines/scenes text actually selected) match `edit_plan.user_prompt`.
    # Strong tier — this is a judgment call gating delivery, not a closed-set
    # pick, and it's a low-call-volume check (once per delivered plan).
    # Switched from qwen/qwen3-235b-a22b-2507 to deepseek/deepseek-v3.2:
    # real-video finding — the QA Agent's intent-match check (order
    # verification against an explicit-sequence user_prompt) failed 3
    # consecutive real attempts on qwen3-235b even after fixing a genuine
    # data bug (added explicit playback_position field) and 2 rounds of
    # prompt clarification — the model's own final rationale was
    # self-contradictory (described the correct order, then called it a
    # violation). deepseek-v3.2 already demonstrated stronger reasoning on
    # comparable judgment tasks in this pipeline (L4 Story Architect scene
    # synthesis) — same OpenRouter provider, same cost class.
    L6_QA_MODEL: str = "deepseek/deepseek-v3.2"

    # ------------------------------------------------------------------
    # Stock asset library (A2 — knowledge_base/stock_assets/)
    # ------------------------------------------------------------------
    # Pexels API key for stock video search (https://api.pexels.com/videos/search).
    # Optional/nullable — the "internal" curated-library source does not need
    # an API key, and this subsystem is built once, queried later by L6, not
    # required for L1-L5 to run.
    PEXELS_API_KEY: str | None = None

    # ------------------------------------------------------------------
    # Optional integrations
    # ------------------------------------------------------------------
    # OPENAI_API_KEY is kept for future use only — it is NOT used for
    # embeddings (local model via sentence-transformers is used instead).
    OPENAI_API_KEY: str | None = None
    HF_TOKEN: str | None = None
    MODAL_TOKEN_ID: str | None = None
    MODAL_TOKEN_SECRET: str | None = None


# Module-level singleton — crashes at import if required vars are missing,
# which is the desired behaviour (fail fast at startup, not at runtime).
settings = Settings()
