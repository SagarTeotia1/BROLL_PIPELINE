from __future__ import annotations

import logging
import time

from pinecone import Pinecone, ServerlessSpec

from shared.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton index handle
# ---------------------------------------------------------------------------

_pinecone_client: Pinecone | None = None
_index_handle = None  # pinecone.Index — typed loosely to avoid import-time issues

_DIMENSION = 1024          # BAAI/bge-large-en-v1.5 output dimension (local model)
_METRIC = "cosine"
_READY_POLL_INTERVAL = 2   # seconds between readiness checks
_READY_TIMEOUT = 120       # seconds before giving up


def _get_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pinecone_client


def _ensure_index_exists(pc: Pinecone) -> None:
    """Create the Pinecone index if it does not already exist, then wait until ready."""
    index_name = settings.PINECONE_INDEX_NAME
    existing = {idx.name for idx in pc.list_indexes().indexes}

    if index_name not in existing:
        logger.info(
            "Pinecone index '%s' not found — creating (dim=%d, metric=%s, region=%s).",
            index_name,
            _DIMENSION,
            _METRIC,
            settings.PINECONE_ENVIRONMENT,
        )
        pc.create_index(
            name=index_name,
            dimension=_DIMENSION,
            metric=_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.PINECONE_ENVIRONMENT,
            ),
        )
    else:
        logger.debug("Pinecone index '%s' already exists.", index_name)

    # Wait until the index reports as ready.
    deadline = time.monotonic() + _READY_TIMEOUT
    while True:
        description = pc.describe_index(index_name)
        # In Pinecone v3 the status field is a dict-like object.
        # Support both attribute access and dict access defensively.
        status = description.status
        is_ready = status["ready"] if isinstance(status, dict) else getattr(status, "ready", False)
        if is_ready:
            logger.info("Pinecone index '%s' is ready.", index_name)
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Pinecone index '{index_name}' did not become ready within "
                f"{_READY_TIMEOUT}s."
            )
        logger.debug("Waiting for Pinecone index to become ready…")
        time.sleep(_READY_POLL_INTERVAL)


def get_index():  # -> pinecone.Index
    """Return the singleton Pinecone index handle.

    Creates the index if it does not exist and waits for it to be ready on the
    first call.  Subsequent calls return the cached handle immediately.

    Returns:
        A :class:`pinecone.Index` object connected to
        ``settings.PINECONE_INDEX_NAME``.
    """
    global _index_handle

    if _index_handle is not None:
        return _index_handle

    pc = _get_client()
    _ensure_index_exists(pc)
    _index_handle = pc.Index(settings.PINECONE_INDEX_NAME)
    logger.info("Pinecone index handle acquired for '%s'.", settings.PINECONE_INDEX_NAME)
    return _index_handle
