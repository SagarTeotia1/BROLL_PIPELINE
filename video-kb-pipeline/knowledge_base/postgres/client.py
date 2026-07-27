from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

try:
    from pgvector.asyncpg import register_vector
except ImportError as _pgvector_err:  # noqa: F841
    raise ImportError(
        "pgvector package is required but not installed. "
        "Install it with: pip install pgvector"
    ) from _pgvector_err

from shared.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pool singleton
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None
# Lock is created lazily inside get_pool() to avoid creating an asyncio
# primitive outside of a running event loop (required for Python < 3.10).
_pool_lock: asyncio.Lock | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Run once per new connection: ensure pgvector extension exists, then register codec."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)


async def get_pool() -> asyncpg.Pool:
    """Return the module-level connection pool, creating it on first call.

    The pool is configured to:
    - Maintain at least 2 and at most 10 connections (reasonable defaults).
    - Register the pgvector codec for every new connection via ``init``.

    Subsequent calls return the cached pool without acquiring the lock on the
    fast path.
    """
    global _pool, _pool_lock

    # Fast path — pool already exists.
    if _pool is not None:
        return _pool

    # Lazily create the lock inside the running event loop to avoid the
    # DeprecationWarning / RuntimeError on Python < 3.10 when an asyncio
    # primitive is created outside of a running loop.
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()

    # Slow path — create the pool under a lock to avoid double-init.
    async with _pool_lock:
        if _pool is not None:
            return _pool

        logger.info("Creating asyncpg connection pool → %s", settings.asyncpg_url)
        _pool = await asyncpg.create_pool(
            dsn=settings.asyncpg_url,
            min_size=2,
            max_size=10,
            init=_init_connection,
            # Recycle idle connections every 90 s so the server never kills
            # them while ASHFS / Whisper run (which can take 5–10 min).
            max_inactive_connection_lifetime=90.0,
            # TCP keepalives: ping the server every 60 s so NAT/proxies
            # don't drop the connection silently during long GPU stages.
            server_settings={
                "tcp_keepalives_idle": "60",
                "tcp_keepalives_interval": "10",
                "tcp_keepalives_count": "5",
            },
        )
        logger.info("Connection pool created.")

    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool.

    Safe to call even if the pool was never created.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Connection pool closed.")


@asynccontextmanager
async def db_pool() -> AsyncIterator[asyncpg.Pool]:
    """Async context manager that yields the connection pool.

    Creates the pool on entry if it does not yet exist.  The pool is
    **not** closed on exit — it is a singleton managed at application
    lifetime.  Use :func:`close_pool` at shutdown instead.

    Example::

        async with db_pool() as pool:
            row = await pool.fetchrow("SELECT 1")
    """
    pool = await get_pool()
    try:
        yield pool
    finally:
        # Intentionally not closing here; the pool lives for the application
        # lifetime and is closed via close_pool() at shutdown.
        pass
