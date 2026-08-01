"""Level-6 orchestration wrapper — mirrors `pipeline/level4/updater.py`'s
sequencing style (each stage writes its own rows internally; this module
only sequences the calls, reconciles each agent's per-clip output into one
merged `extra_filters` dict, and drives the final render + XML export).

Per CLAUDE.md "LEVEL 6 — SPECIALIZED ACTION AGENTS" -> "Design principle":
color grading and captioning don't depend on each other and could run
concurrently, but both need `cut_list_items` to exist first (color keys
its `sequence_color_adjustments` rows to `cut_list_item.id`; captions key
their filters to the SELECT_CLIP op they attach to, which this module
remaps to the same `cut_list_item.id` space) — so the Editing Director's
cut list always runs first, then color + captions, then the merged render.

Ownership boundaries (do not edit these from this module):
  - Engineer A owns `pipeline/level6/editing_director.py`:
    `snap_cut_points`, `write_cut_list`, `export_xml`, `render_direct`, and
    (as of the PIPELINE ADDENDUM A4/A5 mechanics phase) `materialize_layer_
    composite_ops` + `get_compositing_render_extras` — the "later, separate
    phase" this docstring used to say wasn't built yet; it now is.
  - Engineer B owns `pipeline/level6/color_grading_runner.py`:
    `run_color_grading`, `build_ffmpeg_color_filters`.
  - This module owns `run_caption_overlay` (`caption_overlay.py`) and the
    audio filter builders (`audio_sync.py`), both pure/self-contained.
  - The Compositing Agent (`compositing_agent.py::run_compositing_agent`,
    dispatching `background_selector.py`/`emphasis_selector.py`) writes its
    own `background_assignments`/`emphasis_effects` rows internally, same
    "agent writes its own rows" pattern as color grading/captions. Per
    CLAUDE.md's updated L6 diagram ("PIPELINE ADDENDUM" -> A3/A5), it does
    not depend on color/audio/caption output, so it runs concurrently with
    them (see the `asyncio.gather` below) rather than sequentially after.
    Its two outputs feed `editing_director.py`'s A4/A5 mechanics phase
    (step 3b below) — that phase must run AFTER this `asyncio.gather`
    completes (it reads `background_assignments`/`emphasis_effects`, which
    only exist once the Compositing Agent has written them).
  - `pipeline/level6/qa_agent.py::run_qa_agent` (PIPELINE ADDENDUM 2, item
    1) is the LAST step, run only after the render + everything above has
    finished — it reads the rendered output file plus this module's own
    already-computed `caption_results`/`color_adjustments` (never re-runs
    those agents' LLM calls) and writes its own `qa_reports` row.

Audio: `loudnorm` is a sequence-level filter (belongs on the whole
assembled audio track once, not per clip), so it does NOT go through the
per-clip `extra_filters` dict — it's passed as `render_direct`'s separate
`final_audio_filter` param, applied post-concat on the `[outa]` label
(`editing_director.py::render_direct` gained this hook specifically for
this). `compute_ducking_filter` still returns `None` unconditionally —
this pipeline's schema has no music-bed/secondary-audio-track concept for
`sidechaincompress` to duck against yet (see `audio_sync.py`'s
docstring) — `audio_ducking_note` in `run_level6`'s result surfaces that
gap when a plan requests it, rather than silently dropping the request.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import asyncpg

from knowledge_base.postgres.queries import (
    delete_cut_list_items_for_edit_plan,
    get_edit_plan,
    get_sequence_color_adjustments_for_edit_plan,
)
from pipeline.level6.audio_sync import (
    apply_multicam_sync,
    compute_ducking_filter,
    compute_loudnorm_filter,
)
from pipeline.level6.caption_overlay import run_caption_overlay
from pipeline.level6.compositing_agent import run_compositing_agent
from pipeline.level6.editing_director import (
    export_xml,
    get_compositing_render_extras,
    materialize_layer_composite_ops,
    render_direct,
    snap_cut_points,
    write_cut_list,
)
from pipeline.level6.qa_agent import run_qa_agent

logger = logging.getLogger(__name__)

# color_grading_runner.py is Engineer B's file — imported here because
# run_level6 is "the ONE place that has to actually import and successfully
# call into all three engineers' modules" (per the task brief). As of
# writing this module, `pipeline/level6/color_grading_runner.py` does not
# exist yet in this checkout — this import will raise ModuleNotFoundError
# until Engineer B lands it. That is intentional: run_level6 must not
# silently stub around a genuinely missing dependency (see this module's
# own docstring / the task's instruction not to fake a missing engineer's
# work). Once the file exists with the documented signatures
# (`run_color_grading(pool, edit_plan_id)`,
# `build_ffmpeg_color_filters(base_parameters, sequence_delta) -> str`),
# this import resolves with no further change needed here.
from pipeline.level6.color_grading_runner import (  # type: ignore[import]
    build_ffmpeg_color_filters,
    run_color_grading,
)


def _merge_filter(existing: str | None, addition: str) -> str:
    """Chain two per-clip video filter strings for the same
    `cut_list_item.id` into one comma-joined `extra_filters` value —
    ffmpeg filter chains compose left-to-right with `,` inside a single
    labeled filter segment."""
    return f"{existing},{addition}" if existing else addition


async def run_level6(
    pool: asyncpg.Pool,
    edit_plan_id: str,
    source_video_path: str,
    output_path: str,
) -> dict:
    """Run the full Level-6 action-agent pass for a finalized `edit_plan`:

      1. Editing Director — snap cut points, write `cut_list_items`
         (Engineer A: `snap_cut_points` + `write_cut_list`).
      2. Color Grading Agent, Caption/Text Overlay Agent, and the
         Compositing Agent all run CONCURRENTLY (none of the three depends
         on another's output, only on step 1's `cut_list_items` already
         existing — per CLAUDE.md's updated L6 diagram):
           - Color Grading — sequence-aware per-clip color deltas, written
             to `sequence_color_adjustments` (Engineer B: `run_color_grading`).
           - Caption/Text Overlay — style + FFmpeg filter per
             `TEXT_OVERLAY_REQUEST` op (this module: `run_caption_overlay`).
           - Compositing Agent — background selection (A3) + emphasis
             selection (A5 decision half), written to `background_
             assignments`/`emphasis_effects` (`compositing_agent.py::
             run_compositing_agent`).
      3b. Editing Director A4/A5 MECHANICS — materializes `layer_composites`
          FROM the Compositing Agent's `background_assignments` (`materialize_
          layer_composite_ops`), then pulls A4+A5's actual FFmpeg output
          together (`get_compositing_render_extras`): zoom/highlight merge
          into the same per-clip filter dict as color/captions; background-
          swap's two extra media inputs go into `render_direct`'s dedicated
          `extra_inputs`/`composite_filters` params (see that function's
          docstring for why background-swap can't reuse the plain
          `extra_filters` contract).
      4. Reconcile every agent's per-clip output into one
         `{cut_list_item.id: ffmpeg_filter_string}` dict (color + caption +
         A5 zoom/highlight — see module docstring for why audio is excluded
         from this dict) and render via Engineer A's `render_direct`
         (also passing A4's `extra_inputs`/`composite_filters`).
      5. Export the FCPXML/EDL artifact via Engineer A's `export_xml`.
      6. QA/Validation Agent (PIPELINE ADDENDUM 2, item 1,
         `pipeline/level6/qa_agent.py::run_qa_agent`) — the LAST L6 step,
         run only after the render + every other agent above has finished.
         Deterministic checks (blackdetect/silencedetect/loudness/caption-
         drift/color-clipping) plus one Groq intent-match pass, aggregated
         into one `qa_reports` row. QA never mutates `edit_plans`/
         `cut_list_items` and never triggers a re-render (rule 24 — QA
         reports, it doesn't edit) — a `fail`/`warn` status is surfaced as
         DATA in this function's own return value; `run_level6` does not
         raise or change its own `ok` flag because of it (whatever
         delivers the video to the client is what actually honors the
         gate semantics).

    Returns:
        {
          "ok": True,
          "edit_plan_id": ...,
          "cut_list_item_count": int,
          "color_adjustment_count": int,
          "caption_count": int,
          "output_path": ...,          # rendered MP4
          "xml_path": ...,             # FCPXML export
          "audio_normalization_filter": str,  # applied post-concat via render_direct's final_audio_filter
          "audio_ducking_note": str | None,   # non-None only if the plan requested ducking
          "background_assignment_count": int,  # A3 — scene-level, informational
          "emphasis_effect_count": int,        # A5 decision half — informational
          "layer_composite_count": int,        # A4 — materialized layer_composites rows
          "zoom_highlight_applied_count": int, # A5 mechanics — clips with a zoom/highlight filter applied
          "qa_status": str | None,             # pass | warn | fail | None if QA itself couldn't run
          "qa_report_id": str | None,
        }
        or {"ok": False, "reason": "..."} if the edit_plan doesn't exist
        or has no SELECT_CLIP ops to cut.
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        return {"ok": False, "reason": f"edit_plan_id={edit_plan_id} not found"}

    logger.info("[L6] run_level6 start — edit_plan_id=%s video_id=%s", edit_plan_id, plan.video_id)

    # ------------------------------------------------------------------
    # 1. Editing Director — cut list (must exist before color/captions,
    #    both of which key their output to cut_list_item.id / the
    #    SELECT_CLIP op it was built from).
    # ------------------------------------------------------------------
    cut_items = await snap_cut_points(pool, edit_plan_id)
    if not cut_items:
        return {"ok": False, "reason": "snap_cut_points produced no cut_list_items (no usable SELECT_CLIP ops)"}
    # Multicam sync is a documented no-op today (CLAUDE.md — nothing in the
    # schema carries multi-camera source info yet); called here anyway so
    # the pipeline has the correct call site wired in once that lands.
    cut_items = apply_multicam_sync(cut_items)
    # Delete-then-insert, not rely on ON CONFLICT (id) alone: cut_list_items
    # ids are freshly generated every run, so a re-run of L6 for the same
    # edit_plan_id (e.g. retrying after an earlier ffmpeg failure) never
    # matches a prior run's rows and just accumulates on top of them —
    # sequence_color_adjustments/emphasis_effects/layer_composites cascade
    # from cut_list_items (ON DELETE CASCADE, migrations 007/010/011) so
    # this one delete clears all of L6's output tables for a clean rewrite.
    # See delete_cut_list_items_for_edit_plan()'s docstring for the real
    # ffmpeg "Cannot allocate memory" crash this caused before the fix.
    deleted = await delete_cut_list_items_for_edit_plan(pool, edit_plan_id)
    if deleted:
        logger.info(
            "[L6] cleared %d cut_list_item(s) from a prior run before rewriting", deleted,
        )
    await write_cut_list(pool, edit_plan_id, cut_items)
    op_id_to_cut_item_id = {item.op_id: item.id for item in cut_items}
    logger.info("[L6] Editing Director complete — %d cut_list_items written", len(cut_items))

    # ------------------------------------------------------------------
    # 2. Color Grading, Caption/Text Overlay, and Compositing Agents — all
    #    three key off `cut_list_items` (already written above) and don't
    #    depend on each other's output, so they run concurrently per
    #    CLAUDE.md's updated L6 diagram (each writes its own rows
    #    internally, same pattern as L4's agents).
    # ------------------------------------------------------------------
    _, caption_results, compositing_result = await asyncio.gather(
        run_color_grading(pool, edit_plan_id),
        run_caption_overlay(pool, edit_plan_id),
        run_compositing_agent(pool, edit_plan_id),
    )
    color_adjustments = await get_sequence_color_adjustments_for_edit_plan(pool, edit_plan_id)
    logger.info("[L6] Color Grading Agent complete — %d sequence_color_adjustments rows", len(color_adjustments))
    logger.info("[L6] Caption Agent complete — %d caption filter(s) built", len(caption_results))
    if not compositing_result.get("ok"):
        logger.warning(
            "[L6] Compositing Agent did not complete cleanly — reason=%r "
            "(non-fatal, matches L6's 'never blocks render' precedent)",
            compositing_result.get("reason"),
        )
    background_assignments = compositing_result.get("background_assignments") or []
    emphasis_effects = compositing_result.get("emphasis_effects") or []
    logger.info(
        "[L6] Compositing Agent complete — %d background_assignments, %d emphasis_effects",
        len(background_assignments), len(emphasis_effects),
    )

    # ------------------------------------------------------------------
    # 4. Reconcile into one extra_filters dict keyed by cut_list_item.id —
    #    color rows are already keyed that way; captions are keyed by the
    #    SELECT_CLIP op_id they attach to and need remapping.
    # ------------------------------------------------------------------
    merged_filters: dict[str, str] = {}
    for adj in color_adjustments:
        filt = build_ffmpeg_color_filters(adj.base_parameters, adj.sequence_delta)
        if filt:
            merged_filters[adj.cut_list_item_id] = _merge_filter(
                merged_filters.get(adj.cut_list_item_id), filt
            )

    n_captions_applied = 0
    for cap in caption_results:
        target_op_id = cap.get("target_op_id")
        cut_item_id = op_id_to_cut_item_id.get(target_op_id) if target_op_id else None
        if cut_item_id is None:
            logger.warning(
                "[L6] caption op_id=%s has no resolvable cut_list_item (target_op_id=%r not "
                "in this edit plan's cut list) — dropping this caption's filter, not guessing.",
                cap.get("op_id"), target_op_id,
            )
            continue
        merged_filters[cut_item_id] = _merge_filter(merged_filters.get(cut_item_id), cap["filter"])
        n_captions_applied += 1

    # ------------------------------------------------------------------
    # 3b. A4/A5 mechanics — Editing Director's own extension of the L6
    #     diagram (see CLAUDE.md "PIPELINE ADDENDUM" -> A4/A5): the
    #     Compositing Agent (just run above) only wrote WHAT/decisions
    #     (`background_assignments`, `emphasis_effects`); this materializes
    #     A4's `layer_composites` mechanics record FROM those decisions
    #     (must run after `run_compositing_agent` — needs `background_
    #     assignments` to exist — and after the cut list — needs `cut_list_
    #     items` to exist, both already true at this point), then pulls
    #     A4+A5's actual FFmpeg output together via `get_compositing_render_
    #     extras`. Zoom/highlight (A5 mechanics) need no new ffmpeg input,
    #     so they merge into the SAME `merged_filters` dict as color/
    #     captions above; background-swap (A4) needs two extra `-i` sources,
    #     kept in `composite_filters`/`extra_inputs` — `render_direct`'s
    #     dedicated extension point for that (see its docstring).
    # ------------------------------------------------------------------
    layer_composites = await materialize_layer_composite_ops(pool, edit_plan_id)
    logger.info("[L6] Editing Director A4 materialization complete — %d layer_composites rows", len(layer_composites))

    compositing_extra_filters, composite_filters, extra_inputs = await get_compositing_render_extras(
        pool, edit_plan_id
    )
    n_zoom_highlight_applied = 0
    for cut_item_id, filt in compositing_extra_filters.items():
        merged_filters[cut_item_id] = _merge_filter(merged_filters.get(cut_item_id), filt)
        n_zoom_highlight_applied += 1

    # ------------------------------------------------------------------
    # Audio — loudnorm is a sequence-level filter (applies to the whole
    # assembled track, not per-clip) so it hooks onto render_direct's
    # `final_audio_filter` param (applied post-concat, on [outa]) rather
    # than the per-clip video-only `extra_filters` dict above.
    # ------------------------------------------------------------------
    audio_filter = compute_loudnorm_filter(plan.platform)
    ducking_filter = compute_ducking_filter(plan.operations)
    audio_ducking_note = None
    if any(op.get("type") == "AUDIO_DUCK_REQUEST" for op in plan.operations):
        audio_ducking_note = (
            "edit_plan requested AUDIO_DUCK_REQUEST but this pipeline's schema has no "
            "music-bed/secondary-audio-track concept to duck against yet — see "
            "pipeline/level6/audio_sync.py::compute_ducking_filter docstring for the gap."
        )
    if ducking_filter is not None:
        # Not reachable today (compute_ducking_filter always returns None),
        # but kept symmetrical in case that changes without this file being touched.
        merged_filters_note = ducking_filter  # noqa: F841 — intentionally unused, see gap note
    logger.info(
        "[L6] Reconciled extra_filters for %d/%d cut_list_items (color=%d, captions=%d/%d applied); "
        "audio_normalization_filter=%r applied post-concat.",
        len(merged_filters), len(cut_items), len(color_adjustments), n_captions_applied, len(caption_results),
        audio_filter,
    )

    # ------------------------------------------------------------------
    # 5. Render + XML export.
    # ------------------------------------------------------------------
    rendered_path = await render_direct(
        pool, edit_plan_id, source_video_path, output_path,
        extra_filters=merged_filters, final_audio_filter=audio_filter,
        extra_inputs=extra_inputs, composite_filters=composite_filters,
    )
    xml_output_path = str(Path(output_path).with_suffix(".fcpxml"))
    xml_path = await export_xml(pool, edit_plan_id, xml_output_path)

    logger.info("[L6] run_level6 complete — edit_plan_id=%s output_path=%s xml_path=%s", edit_plan_id, rendered_path, xml_path)

    # ------------------------------------------------------------------
    # 6. QA/Validation Agent — LAST step, after render + every other L6
    #    agent above has finished. Never blocks/raises on its own findings
    #    (rule 24) — a QA pass that itself errors out is logged and
    #    surfaced as qa_status=None, not allowed to fail run_level6's own
    #    "ok" result (the render already succeeded; QA is a report on top
    #    of it, not a precondition for it having happened).
    # ------------------------------------------------------------------
    qa_status: str | None = None
    qa_report_id: str | None = None
    try:
        qa_report = await run_qa_agent(
            pool, edit_plan_id, rendered_path,
            caption_results=caption_results,
            color_adjustments=color_adjustments,
        )
        if qa_report is not None:
            qa_status = qa_report.status
            qa_report_id = qa_report.id
            logger.info("[L6] QA Agent complete — edit_plan_id=%s qa_status=%s", edit_plan_id, qa_status)
        else:
            logger.warning("[L6] QA Agent did not produce a report for edit_plan_id=%s", edit_plan_id)
    except Exception as exc:  # noqa: BLE001 — QA failing to run must never take down a successful render
        logger.error("[L6] QA Agent raised for edit_plan_id=%s: %s", edit_plan_id, exc)

    return {
        "ok": True,
        "edit_plan_id": edit_plan_id,
        "cut_list_item_count": len(cut_items),
        "color_adjustment_count": len(color_adjustments),
        "caption_count": n_captions_applied,
        "output_path": rendered_path,
        "xml_path": xml_path,
        "audio_normalization_filter": audio_filter,
        "audio_ducking_note": audio_ducking_note,
        "background_assignment_count": len(background_assignments),
        "emphasis_effect_count": len(emphasis_effects),
        "layer_composite_count": len(layer_composites),
        "zoom_highlight_applied_count": n_zoom_highlight_applied,
        "qa_status": qa_status,
        "qa_report_id": qa_report_id,
    }
