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
