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
    # Pinecone
    # ------------------------------------------------------------------
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "video-kb"
    PINECONE_ENVIRONMENT: str = "us-east-1"

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
    # Groq (Level-4 reasoning + Level-5 planning — Grounding Agent, Story
    # Architect Agent, and L5's Selection/Sequencing passes all run on Groq).
    # ------------------------------------------------------------------
    # qwen/qwen3.6-27b is the only current Groq-hosted Qwen model (qwen3-32b
    # deprecated 2026-06-17) — single tier, so cheap/strong-tier agents that
    # used separate models on Anthropic now share this one model on Groq.
    GROQ_API_KEY: str | None = None
    L4_GROUNDING_MODEL: str = "qwen/qwen3.6-27b"
    L4_STORY_ARCHITECT_MODEL: str = "qwen/qwen3.6-27b"
    L4_ONTOLOGY_VERSION: int = 1
    L5_SELECTION_MODEL: str = "qwen/qwen3.6-27b"
    L5_SEQUENCING_MODEL: str = "qwen/qwen3.6-27b"
    # Level-6 Caption/Text Overlay Agent — the one LLM-worthy decision in L6
    # (caption STYLE only, never the text itself). Same single-tier Groq Qwen
    # model as L4/L5 — see note above.
    L6_CAPTION_MODEL: str = "qwen/qwen3.6-27b"
    # Level-6 Color Grading Agent — sequence-delta computation (see
    # pipeline/level6/color_grading_runner.py). Same single-tier Groq Qwen
    # model as L4/L5 — kept as its own setting (rather than reusing
    # L5_SEQUENCING_MODEL) so L6 agents can be retuned to a different model
    # independently of planning, per the existing per-stage settings pattern.
    L6_COLOR_MODEL: str = "qwen/qwen3.6-27b"

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
