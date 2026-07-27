from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from shared.config import settings

logger = logging.getLogger(__name__)

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not installed")

_driver: "AsyncDriver | None" = None


async def get_driver() -> "AsyncDriver":
    global _driver
    if _driver is None:
        if not NEO4J_AVAILABLE:
            raise RuntimeError("neo4j package not installed. Run: pip install neo4j>=5.0")
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        logger.info("Neo4j driver connected to %s", settings.NEO4J_URI)
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


@asynccontextmanager
async def neo4j_session(database: str | None = None):
    driver = await get_driver()
    db = database or settings.NEO4J_DATABASE
    async with driver.session(database=db) as session:
        yield session


async def run_cypher(query: str, params: dict | None = None, database: str | None = None) -> list[dict]:
    """Run a Cypher query and return list of record dicts."""
    async with neo4j_session(database) as session:
        result = await session.run(query, params or {})
        records = await result.data()
        return records
