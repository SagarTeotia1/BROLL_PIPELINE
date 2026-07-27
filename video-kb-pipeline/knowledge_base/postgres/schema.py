from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

# Path to the SQL migration file relative to this module.
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_INITIAL_SQL = _MIGRATIONS_DIR / "001_initial.sql"


def _has_real_sql(stmt: str) -> bool:
    """Return True if stmt has any non-comment, non-whitespace content."""
    for line in stmt.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return True
    return False


def _split_sql(sql: str) -> list[str]:
    """Split SQL on semicolons, skipping those inside $$ blocks or -- comments.

    Handles:
    - DO $$ BEGIN ... END $$; blocks (semicolons inside are skipped)
    - -- line comments that contain semicolons (skipped)
    - comment-only fragments are dropped (asyncpg rejects empty queries)
    """
    statements: list[str] = []
    current: list[str] = []
    in_dollar_quote = False
    in_line_comment = False

    i = 0
    while i < len(sql):
        if in_line_comment:
            current.append(sql[i])
            if sql[i] == "\n":
                in_line_comment = False
            i += 1
            continue

        if not in_dollar_quote and sql[i:i+2] == "--":
            in_line_comment = True
            current.append("--")
            i += 2
            continue

        if sql[i:i+2] == "$$":
            in_dollar_quote = not in_dollar_quote
            current.append("$$")
            i += 2
            continue

        if sql[i] == ";" and not in_dollar_quote:
            stmt = "".join(current).strip()
            if stmt and _has_real_sql(stmt):
                statements.append(stmt)
            current = []
        else:
            current.append(sql[i])
        i += 1

    stmt = "".join(current).strip()
    if stmt and _has_real_sql(stmt):
        statements.append(stmt)

    return statements


async def _run_one_migration(conn, sql_path: Path) -> None:
    """Execute all statements in a single migration file, skipping already-applied DDL."""
    sql_text = sql_path.read_text(encoding="utf-8")
    statements = _split_sql(sql_text)
    for stmt in statements:
        try:
            await conn.execute(stmt)
        except (
            asyncpg.exceptions.DuplicateTableError,
            asyncpg.exceptions.DuplicateObjectError,   # constraint already exists
            asyncpg.exceptions.UniqueViolationError,
            asyncpg.exceptions.InvalidColumnReferenceError,
        ):
            logger.debug("Already exists — skipping: %.60s…", stmt)
        except Exception as exc:
            logger.error("Migration statement failed:\n%s\nError: %s", stmt, exc)
            raise


async def run_migrations(pool: asyncpg.Pool) -> None:
    """Run ALL numbered migration files in ``migrations/`` in order.

    Every migration is idempotent — re-running the same file is safe.
    New migration files (003_, 004_, …) are picked up automatically.
    """
    migration_files = sorted(_MIGRATIONS_DIR.glob("[0-9]*.sql"))
    async with pool.acquire() as conn:
        for mf in migration_files:
            logger.info("Applying migration: %s", mf.name)
            await _run_one_migration(conn, mf)
    logger.info("All %d migration(s) applied.", len(migration_files))
