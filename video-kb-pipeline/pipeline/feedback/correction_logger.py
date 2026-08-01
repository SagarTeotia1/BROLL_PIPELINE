"""Correction feedback loop (Addendum 2, item 2).

`scene_overrides`/`storyline_overrides` (migration 005) let a human-caught
correction patch a `final` scene/storyline without a full L4 re-run.
`edit_plan_revisions` (migration 006) already records L5 plan diffs. Neither
wrote a durable, structured record of *what changed* until this module: every
override write and every plan revision write goes through the functions
here so that a matching `correction_events` row (migration 014) is always
written alongside it — the actual corrections dataset (rule 25: one function,
not duplicated logic per call site, so the dataset can't silently drift out
of sync with the override tables it mirrors).

`correction_source` is one of: 'client' | 'internal_editor' | 'qa_agent_flag'.
"""

from __future__ import annotations

import logging

import asyncpg

from knowledge_base.postgres.queries import (
    get_scene_by_id,
    get_storyline_by_id,
    insert_correction_event,
    insert_scene_override,
    insert_storyline_override,
)
from shared.types import (
    CorrectionEventRecord,
    SceneOverrideRecord,
    StorylineOverrideRecord,
)
from shared.utils import gen_id

logger = logging.getLogger(__name__)


async def log_correction(
    pool: asyncpg.Pool,
    *,
    video_id: str,
    level: int,
    entity_type: str,
    entity_id: str,
    field: str,
    original_value,
    corrected_value,
    correction_source: str,
    reason: str | None = None,
) -> str:
    """Write one `correction_events` row directly. Use this for entities that
    have no dedicated override table of their own (e.g. `edit_plan`,
    `speaker_turn`, `qa_report`). For `scene`/`storyline` corrections, prefer
    `log_scene_correction`/`log_storyline_correction` below — they also write
    the corresponding override row that L4/L5 actually read at plan time."""
    event = CorrectionEventRecord(
        id=gen_id(),
        video_id=video_id,
        level=level,
        entity_type=entity_type,
        entity_id=entity_id,
        field=field,
        original_value=original_value,
        corrected_value=corrected_value,
        correction_source=correction_source,
        reason=reason,
    )
    event_id = await insert_correction_event(pool, event)
    logger.info(
        "log_correction: video_id=%s level=%d entity_type=%s entity_id=%s field=%s -> correction_events id=%s",
        video_id, level, entity_type, entity_id, field, event_id,
    )
    return event_id


async def log_scene_correction(
    pool: asyncpg.Pool,
    *,
    video_id: str,
    scene_id: str,
    field: str,
    new_value,
    correction_source: str,
    reason: str | None = None,
    created_by: str | None = None,
) -> str:
    """Apply a correction to a `scenes` field: writes the `scene_overrides`
    row L4/L5 actually read, and a matching `correction_events` row capturing
    what the field's value was before the override (rule 25). Returns the
    `scene_overrides` id.

    Raises `ValueError` if `scene_id` doesn't exist or `field` isn't a
    present attribute on `SceneRecord` — never silently logs a correction
    against a value it couldn't actually read (would make original_value a
    guess, not a fact).
    """
    scene = await get_scene_by_id(pool, scene_id)
    if scene is None:
        raise ValueError(f"log_scene_correction: no scenes row for id={scene_id}")
    if not hasattr(scene, field):
        raise ValueError(f"log_scene_correction: SceneRecord has no field '{field}'")
    original_value = getattr(scene, field)

    override = SceneOverrideRecord(
        id=gen_id(),
        scene_id=scene_id,
        field=field,
        new_value=new_value,
        reason=reason,
        created_by=created_by,
    )
    override_id = await insert_scene_override(pool, override)

    await log_correction(
        pool,
        video_id=video_id,
        level=4,
        entity_type="scene",
        entity_id=scene_id,
        field=field,
        original_value=original_value,
        corrected_value=new_value,
        correction_source=correction_source,
        reason=reason,
    )
    return override_id


async def log_storyline_correction(
    pool: asyncpg.Pool,
    *,
    video_id: str,
    storyline_id: str,
    field: str,
    new_value,
    correction_source: str,
    reason: str | None = None,
    created_by: str | None = None,
) -> str:
    """Same contract as `log_scene_correction`, for `storylines`/`storyline_overrides`."""
    storyline = await get_storyline_by_id(pool, storyline_id)
    if storyline is None:
        raise ValueError(f"log_storyline_correction: no storylines row for id={storyline_id}")
    if not hasattr(storyline, field):
        raise ValueError(f"log_storyline_correction: StorylineRecord has no field '{field}'")
    original_value = getattr(storyline, field)

    override = StorylineOverrideRecord(
        id=gen_id(),
        storyline_id=storyline_id,
        field=field,
        new_value=new_value,
        reason=reason,
        created_by=created_by,
    )
    override_id = await insert_storyline_override(pool, override)

    await log_correction(
        pool,
        video_id=video_id,
        level=4,
        entity_type="storyline",
        entity_id=storyline_id,
        field=field,
        original_value=original_value,
        corrected_value=new_value,
        correction_source=correction_source,
        reason=reason,
    )
    return override_id


async def log_edit_plan_revision_correction(
    pool: asyncpg.Pool,
    *,
    video_id: str,
    edit_plan_id: str,
    original_operations: list[dict],
    diff_operations: list[dict],
    correction_source: str = "client",
    reason: str | None = None,
) -> str:
    """Log an `edit_plan_revisions` diff (already written by
    `pipeline/level5/planner_runner.py::apply_revision`) as a
    `correction_events` row. `edit_plan_revisions` itself already stores
    `diff_operations` (rule 21: diffs, not regenerates) — this just mirrors
    that into the corrections dataset with `original_operations` attached so
    the before/after is reconstructable without joining back through
    `edit_plans` version history."""
    return await log_correction(
        pool,
        video_id=video_id,
        level=5,
        entity_type="edit_plan",
        entity_id=edit_plan_id,
        field="operations",
        original_value=original_operations,
        corrected_value=diff_operations,
        correction_source=correction_source,
        reason=reason,
    )
