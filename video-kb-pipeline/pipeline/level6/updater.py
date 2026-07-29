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
    `snap_cut_points`, `write_cut_list`, `export_xml`, `render_direct`.
  - Engineer B owns `pipeline/level6/color_grading_runner.py`:
    `run_color_grading`, `build_ffmpeg_color_filters`.
  - This module owns `run_caption_overlay` (`caption_overlay.py`) and the
    audio filter builders (`audio_sync.py`), both pure/self-contained.

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

import logging
from pathlib import Path

import asyncpg

from knowledge_base.postgres.queries import (
    get_edit_plan,
    get_sequence_color_adjustments_for_edit_plan,
)
from pipeline.level6.audio_sync import (
    apply_multicam_sync,
    compute_ducking_filter,
    compute_loudnorm_filter,
)
from pipeline.level6.caption_overlay import run_caption_overlay
from pipeline.level6.editing_director import (
    export_xml,
    render_direct,
    snap_cut_points,
    write_cut_list,
)

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
      2. Color Grading Agent — sequence-aware per-clip color deltas,
         written to `sequence_color_adjustments` (Engineer B:
         `run_color_grading`).
      3. Caption/Text Overlay Agent — style + FFmpeg filter per
         `TEXT_OVERLAY_REQUEST` op (this module: `run_caption_overlay`).
      4. Reconcile every agent's per-clip output into one
         `{cut_list_item.id: ffmpeg_filter_string}` dict (color + caption
         only — see module docstring for why audio is excluded from this
         dict) and render via Engineer A's `render_direct`.
      5. Export the FCPXML/EDL artifact via Engineer A's `export_xml`.

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
    await write_cut_list(pool, edit_plan_id, cut_items)
    op_id_to_cut_item_id = {item.op_id: item.id for item in cut_items}
    logger.info("[L6] Editing Director complete — %d cut_list_items written", len(cut_items))

    # ------------------------------------------------------------------
    # 2. Color Grading Agent — sequence-aware deltas (Engineer B writes
    #    its own rows internally, same pattern as L4's agents).
    # ------------------------------------------------------------------
    await run_color_grading(pool, edit_plan_id)
    color_adjustments = await get_sequence_color_adjustments_for_edit_plan(pool, edit_plan_id)
    logger.info("[L6] Color Grading Agent complete — %d sequence_color_adjustments rows", len(color_adjustments))

    # ------------------------------------------------------------------
    # 3. Caption/Text Overlay Agent.
    # ------------------------------------------------------------------
    caption_results = await run_caption_overlay(pool, edit_plan_id)
    logger.info("[L6] Caption Agent complete — %d caption filter(s) built", len(caption_results))

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
    )
    xml_output_path = str(Path(output_path).with_suffix(".fcpxml"))
    xml_path = await export_xml(pool, edit_plan_id, xml_output_path)

    logger.info("[L6] run_level6 complete — edit_plan_id=%s output_path=%s xml_path=%s", edit_plan_id, rendered_path, xml_path)

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
    }
