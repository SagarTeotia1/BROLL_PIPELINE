from __future__ import annotations

import logging
from typing import Any

from shared.types import SearchableFactRecord
from shared.utils import retry
from knowledge_base.pinecone_kb.client import get_index

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BATCH_SIZE = 100

# Pinecone namespaces used by this pipeline.
# DINOv2 (768-dim) and ArcFace (512-dim) embeddings live in pgvector only.
# Pinecone stores 1024-dim text embeddings (BAAI/bge-large-en-v1.5) for semantic search.
_NS_FACTS = "facts"
_NS_SCENES = "scenes"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _batch(items: list, size: int):
    """Yield successive *size*-length chunks from *items*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Public upsert functions
# ---------------------------------------------------------------------------


@retry(max_attempts=3, wait_seconds=2.0)
def upsert_facts(facts: list[SearchableFactRecord]) -> int:
    """Upsert searchable facts into the ``facts`` Pinecone namespace.

    Facts whose ``embedding`` field is ``None`` are silently skipped because
    Pinecone requires a vector for every record.

    Args:
        facts: List of :class:`~shared.types.SearchableFactRecord` instances.

    Returns:
        The total number of vectors successfully upserted.
    """
    index = get_index()
    eligible = [f for f in facts if f.embedding is not None]

    if not eligible:
        logger.debug("upsert_facts: no eligible records (all embeddings are None).")
        return 0

    def _trunc(s: str, limit: int = 500) -> str:
        return s[:limit] if len(s.encode()) > limit else s

    total = 0
    for batch in _batch(eligible, _BATCH_SIZE):
        vectors: list[dict[str, Any]] = [
            {
                "id": f.pinecone_id or f.id,
                "values": f.embedding,
                "metadata": {
                    "video_id": f.video_id,
                    "fact_text": _trunc(f.fact_text),
                    "timestamp_s": f.timestamp_s,
                    "frame_id": str(f.frame_id) if f.frame_id else "",
                    "db_id": f.id,
                },
            }
            for f in batch
        ]
        index.upsert(vectors=vectors, namespace=_NS_FACTS)
        total += len(vectors)
        logger.debug("upsert_facts: upserted batch of %d → total %d", len(vectors), total)

    logger.info("upsert_facts: %d vectors upserted to namespace '%s'.", total, _NS_FACTS)
    return total


@retry(max_attempts=3, wait_seconds=2.0)
def upsert_scenes(scene_records: list[dict]) -> int:
    """Upsert scene caption embeddings into the ``scenes`` Pinecone namespace.

    Each record in *scene_records* must be a dict with:
    - ``id`` (str): unique identifier used as the Pinecone vector ID
    - ``video_id`` (str)
    - ``scene_id`` (str): logical scene identifier within the video
    - ``caption`` (str): the scene caption text
    - ``embedding`` (list[float]): 1024-dim text embedding of the caption
    - ``metadata`` (dict, optional): extra metadata fields to store

    Records whose ``embedding`` key is missing or ``None`` are skipped.

    Args:
        scene_records: List of scene dicts as described above.

    Returns:
        The total number of vectors successfully upserted.
    """
    index = get_index()
    eligible = [r for r in scene_records if r.get("embedding") is not None]

    if not eligible:
        logger.debug("upsert_scenes: no eligible records.")
        return 0

    total = 0
    for batch in _batch(eligible, _BATCH_SIZE):
        vectors: list[dict[str, Any]] = []
        def _trunc(s: str, limit: int = 500) -> str:
            return s[:limit] if len(s.encode()) > limit else s

        for r in batch:
            extra_meta: dict[str, Any] = {
                k: v for k, v in (r.get("metadata", {}) or {}).items()
                if isinstance(v, (str, int, float, bool))
            }
            vectors.append(
                {
                    "id": r["id"],
                    "values": r["embedding"],
                    "metadata": {
                        "video_id": r.get("video_id", ""),
                        "scene_id": r.get("scene_id", ""),
                        "caption": _trunc(r.get("caption", "")),
                        **extra_meta,
                    },
                }
            )
        index.upsert(vectors=vectors, namespace=_NS_SCENES)
        total += len(vectors)
        logger.debug("upsert_scenes: upserted batch of %d → total %d", len(vectors), total)

    logger.info("upsert_scenes: %d vectors upserted to namespace '%s'.", total, _NS_SCENES)
    return total
