"""Local embedding model using sentence-transformers.

Provides a singleton LocalEmbedder that wraps BAAI/bge-large-en-v1.5
(1024-dim) running on the same GPU as the Qwen vLLM engine.  This replaces
the previous OpenAI text-embedding-3-small dependency — no external API calls
are made for embeddings.

Usage::

    from models.embed_model import LocalEmbedder

    embedder = LocalEmbedder.get()
    embedder.load()                     # called once at container startup
    vectors = embedder.embed(texts)     # list[list[float]], len == len(texts)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed — local embeddings unavailable. "
        "Install with: pip install sentence-transformers>=3.0"
    )

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST


EMBED_DIM = 1024  # BAAI/bge-large-en-v1.5 output dimension


class LocalEmbedder:
    """Singleton local embedding model using sentence-transformers.

    The model is loaded lazily via :meth:`load` and then reused for all
    subsequent :meth:`embed` / :meth:`embed_one` calls.  Designed to be
    loaded once at Modal container startup alongside the Qwen vLLM engine.

    Thread safety: :meth:`embed` is synchronous and GPU-bound.  Async callers
    should wrap calls in ``asyncio.to_thread``::

        embeddings = await asyncio.to_thread(embedder.embed, texts, 64)
    """

    _instance: LocalEmbedder | None = None
    _model: "_ST | None" = None

    @classmethod
    def get(cls) -> LocalEmbedder:
        """Return the module-level singleton, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str | None = None) -> None:
        """Load the embedding model.

        Safe to call multiple times — subsequent calls are no-ops.

        Args:
            model_name: HuggingFace model identifier.  Defaults to
                ``BAAI/bge-large-en-v1.5`` (1024-dim, best-in-class for
                retrieval tasks, fits comfortably alongside Qwen on an L40S).
            device: Torch device string. Defaults to None, which
                auto-detects via ``torch.cuda.is_available()`` — this class
                is used both in L3's GPU container (alongside Qwen) and in
                L4's CPU-only container (``l4_image`` installs a CPU torch
                wheel, per CLAUDE.md's own "CPU-only, no GPU" design for
                L4) — hardcoding "cuda" here would crash the L4 container,
                which has no CUDA device at all.

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        if not ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers>=3.0"
            )
        if self._model is not None:
            logger.debug("LocalEmbedder.load() called but model already loaded — skipping.")
            return
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        logger.info("LocalEmbedder loaded: %s (dim=%d, device=%s)", model_name, EMBED_DIM, device)

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of texts, returning one 1024-dim vector per text.

        Embeddings are L2-normalised (required for bge-large cosine similarity
        to work correctly via dot-product).

        Args:
            texts:      Input strings to embed.  Empty list returns ``[]``.
            batch_size: GPU batch size.  64 is safe for L40S with bge-large.

        Returns:
            List of float vectors, same length as *texts*.

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # bge-large requires L2 normalisation
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string.  Convenience wrapper around :meth:`embed`."""
        return self.embed([text])[0]
