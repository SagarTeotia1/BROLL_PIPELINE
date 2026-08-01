"""Level-6 Compositing Agent — thin dispatcher (A3 + A5-decision).

Implements CLAUDE.md "PIPELINE ADDENDUM — Motion Graphics/Compositing +
Hardening" -> Part A's "Updated L6 diagram": the Compositing Agent is one
box in L6's parallel stage, alongside Color Grading and Caption/Text
Overlay — "background selection + emphasis selection (judgment)". This
module is the single call site `pipeline/level6/updater.py::run_level6`
imports, mirroring how it already imports `run_color_grading` from
`color_grading_runner.py` and `run_caption_overlay` from
`caption_overlay.py` — one function per agent, each agent writing its own
rows internally.

Two submodules, both real work happening elsewhere:
  - `background_selector.py::run_background_selection` — A3, scene-level
    (keyed to `scenes.id`, not `edit_plan_id`/`cut_list_item_id` — see that
    module's docstring for why).
  - `emphasis_selector.py::run_emphasis_selection` — A5 decision half,
    cut-order-level (keyed to `cut_list_items.id`, requires the Editing
    Director to have already run — same precondition as color grading).

This module does not itself call Groq, touch the database, or make any
judgment call — it only resolves *video_id* from the edit plan (background
selection is scene/video-scoped, not edit-plan-scoped) and sequences the
two calls, matching the "narrow and mostly deterministic... LLM judgment
used only where a judgment call genuinely can't be reduced to a formula"
design principle applied to orchestration code itself.
"""
from __future__ import annotations

import logging

import asyncpg

from knowledge_base.postgres.queries import get_edit_plan
from pipeline.level6.background_selector import run_background_selection
from pipeline.level6.emphasis_selector import run_emphasis_selection
from shared.types import BackgroundAssignmentRecord, EmphasisEffectRecord

logger = logging.getLogger(__name__)


async def run_compositing_agent(
    pool: asyncpg.Pool,
    edit_plan_id: str,
    license_type: str | None = None,
) -> dict:
    """Run both Compositing Agent submodules for *edit_plan_id*'s video.

    Requires the Editing Director to have already written `cut_list_items`
    for this edit plan (emphasis selection depends on it; background
    selection does not, but there is no reason to call this dispatcher
    before the cut list exists — `pipeline/level6/updater.py::run_level6`
    always sequences it that way, same precondition as color grading and
    captions).

    Returns:
        {
          "ok": True,
          "video_id": ...,
          "background_assignments": list[BackgroundAssignmentRecord],
          "emphasis_effects": list[EmphasisEffectRecord],
        }
        or {"ok": False, "reason": "..."} if the edit_plan doesn't exist. Note
        `ok: True` does not guarantee non-empty lists — either submodule can
        fail independently and is caught non-fatally (logged, empty list
        returned for that one), since A3/A5 are optional per-video features,
        not something that should ever block a render.
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        return {"ok": False, "reason": f"edit_plan_id={edit_plan_id} not found"}

    # Real-video finding, first live L6 run: run_level6 already checks
    # compositing_result.get("ok") and treats a failure here as non-fatal
    # ("never blocks render" — see updater.py) — but that contract was never
    # actually implemented here, so a real exception (missing
    # sentence-transformers dependency for background_selector.py's
    # embedding step) propagated straight through asyncio.gather and crashed
    # the ENTIRE L6 run, not just this optional compositing step. A3/A5 are
    # both genuinely optional (green-screen/background-swap, zoom/highlight
    # emphasis) — neither should be able to take down color grading,
    # captions, or the render itself. Catching broad Exception here (not a
    # narrower type) is deliberate: this whole submodule is best-effort by
    # design, same as L4's Neo4j/embedding propagation steps.
    background_assignments: list[BackgroundAssignmentRecord] = []
    emphasis_effects: list[EmphasisEffectRecord] = []
    try:
        background_assignments = await run_background_selection(
            pool, plan.video_id, license_type=license_type
        )
    except Exception as exc:
        logger.warning(
            "[L6] Compositing Agent: background selection failed (non-fatal, "
            "A3 skipped for this run): %s", exc,
        )
    try:
        emphasis_effects = await run_emphasis_selection(pool, edit_plan_id)
    except Exception as exc:
        logger.warning(
            "[L6] Compositing Agent: emphasis selection failed (non-fatal, "
            "A5 decision skipped for this run): %s", exc,
        )

    logger.info(
        "[L6] Compositing Agent complete — edit_plan_id=%s video_id=%s "
        "background_assignments=%d emphasis_effects=%d",
        edit_plan_id, plan.video_id, len(background_assignments), len(emphasis_effects),
    )

    return {
        "ok": True,
        "video_id": plan.video_id,
        "background_assignments": background_assignments,
        "emphasis_effects": emphasis_effects,
    }
