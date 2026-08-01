"""Stock asset indexer (A2 — knowledge_base/stock_assets/).

Embeds `NormalizedStockAsset.description` (+ tags) with the same local
`BAAI/bge-large-en-v1.5` model already used for `searchable_facts` and L4
relation clustering (`models/embed_model.LocalEmbedder`) — CLAUDE.md is
explicit: "no second embedding model for this". Upserts into the
`stock_assets` Postgres table (migration `009_stock_assets.sql`) in batches
of 100, matching rule 8's "Pinecone upserts batched in 100s, never
one-by-one" principle applied to this Postgres-only subsystem — B7 already
decided stock_assets is pgvector-only, Pinecone is never touched here.

Usage::

    from models.embed_model import LocalEmbedder
    from knowledge_base.stock_assets.client import get_pexels_client
    from knowledge_base.stock_assets.indexer import index_assets

    embedder = LocalEmbedder.get()
    embedder.load()                                   # once at startup

    assets = get_pexels_client().search("rain city night", per_page=40)
    count = await index_assets(pool, assets, embedder)
"""
from __future__ import annotations

import logging

import asyncpg

from knowledge_base.postgres.queries import (
    bulk_upsert_stock_assets,
    get_stock_asset_by_source_external_id,
)
from knowledge_base.stock_assets.client import NormalizedStockAsset
from shared.types import StockAssetRecord
from shared.utils import gen_id

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


def _batch(items: list, size: int):
    """Yield successive *size*-length chunks from *items*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _embed_text(asset: NormalizedStockAsset) -> str:
    """Build the text to embed for one asset: description + tags.

    Tags are appended as plain space-joined tokens so the embedding model
    sees them as additional context terms, not a separate structured field
    (bge-large-en-v1.5 is a plain text encoder).
    """
    parts = [asset.description or ""]
    if asset.tags:
        parts.append(" ".join(asset.tags))
    return " ".join(p for p in parts if p).strip()


async def index_assets(
    pool: asyncpg.Pool,
    assets: list[NormalizedStockAsset],
    embedder,
    r2_cache_keys: dict[str, str] | None = None,
) -> int:
    """Embed and upsert a list of normalized stock assets into Postgres.

    Idempotent: an asset already present for `(source, external_id)` is
    re-embedded and updated in place (same row `id` reused) rather than
    duplicated — safe to re-run the same Pexels search later (rule 9's
    "all Modal functions idempotent" principle, applied here even though
    this subsystem is not itself a Modal function).

    Args:
        pool: asyncpg connection pool.
        assets: Normalized assets from `PexelsClient.search()` (or any
            future source client producing the same shape).
        embedder: A loaded `models.embed_model.LocalEmbedder` instance
            (caller's responsibility to call `.load()` first — this
            function never loads the model itself, matching
            `LocalEmbedder`'s existing "load once at container startup"
            contract used elsewhere in the pipeline).
        r2_cache_keys: Optional mapping of `external_id -> r2_cache_key`
            for assets already download-cached to R2 by a caller (e.g.
            `knowledge_base/r2/uploader.py`). Assets not present in this
            mapping get `r2_cache_key=None` — caching is not this function's
            job, only recording the key if one is supplied.

    Returns:
        The number of `stock_assets` rows upserted.
    """
    if not assets:
        logger.debug("index_assets: nothing to index.")
        return 0

    r2_cache_keys = r2_cache_keys or {}

    # Resolve existing row ids for idempotent re-runs (id reused on update,
    # not regenerated, so a re-embed doesn't fork the row).
    records: list[StockAssetRecord] = []
    texts: list[str] = []
    for asset in assets:
        existing = await get_stock_asset_by_source_external_id(
            pool, asset.source, asset.external_id
        )
        row_id = existing.id if existing is not None else gen_id()
        records.append(
            StockAssetRecord(
                id=row_id,
                source=asset.source,
                license_type=asset.license_type,
                external_id=asset.external_id,
                description=asset.description,
                tags=asset.tags,
                embedding=None,  # filled in below
                r2_cache_key=r2_cache_keys.get(asset.external_id),
            )
        )
        texts.append(_embed_text(asset))

    embeddings = embedder.embed(texts)
    for record, embedding in zip(records, embeddings):
        record.embedding = embedding

    total = 0
    for batch in _batch(records, _BATCH_SIZE):
        await bulk_upsert_stock_assets(pool, batch)
        total += len(batch)
        logger.debug("index_assets: upserted batch of %d → total %d", len(batch), total)

    logger.info("index_assets: %d stock_assets rows upserted.", total)
    return total
