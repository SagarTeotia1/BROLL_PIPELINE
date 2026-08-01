"""Level-6 Editing Director.

Turns a finalized `edit_plan`'s `SELECT_CLIP` operations into an actual cut
list (`cut_list_items`), then exports that cut list either as a portable
FCPXML project (primary output — editable in Resolve/Premiere) or as a
directly rendered MP4 via a single FFmpeg `filter_complex` pass (secondary
output — quick preview).

Per CLAUDE.md "LEVEL 6 — SPECIALIZED ACTION AGENTS" / "Editing Director":
the only judgment call here is exact cut-point snapping to a natural
pause/breath point using existing word-level transcript timestamps — a
small deterministic search over an existing signal, never an LLM call.
Everything else (XML writing, FFmpeg command building) is a pure function
of `operations` / `cut_list_items`.

PIPELINE ADDENDUM A4/A5 mechanics (added alongside the above — pure
execution, no new judgment, per CLAUDE.md "PIPELINE ADDENDUM —
Motion Graphics/Compositing + Hardening"):

  - A4 (`materialize_layer_composite_ops` / `build_layer_composite_filter_
    template` / `build_pip_overlay_filter_template`) — turns
    `background_assignments` (A3's decision, scene-level) + `shot_mattes`
    (L2's matting output) into an actual FFmpeg alpha-composite. This is
    the one branch that needs ADDITIONAL ffmpeg `-i` inputs beyond the
    source video (the matte clip + the stock background clip), so
    `render_direct` gained two new params (`extra_inputs`,
    `composite_filters`) purely to carry that — see `render_direct`'s
    docstring for exactly how they compose with the pre-existing
    `extra_filters`/`final_audio_filter` extension points.
  - A5 mechanics (`build_zoom_emphasis_filter` / `build_highlight_callout_
    filter`) — turns `emphasis_effects` (A5's decision half, already
    written by `emphasis_selector.py`) into a `crop`+`scale` Ken-Burns move
    or a `drawbox`/`drawtext` callout. Neither needs a new ffmpeg input, so
    both plug directly into the EXISTING `extra_filters: dict[cut_list_
    item_id, str]` contract the color/caption agents already use (see
    `get_compositing_render_extras`, which is the single call site
    `pipeline/level6/updater.py` uses to pull all of A4+A5's render-time
    output together).

Every parameter these branches use comes from a `background_assignments` /
`emphasis_effects` / `shot_mattes` / `layer_composites` row already written
by an upstream agent, or from the op/clip's own already-known fields
(`cut_list_items` timing, `videos.width/height`) — never invented here
(rule 12/18/20).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import asyncpg

from knowledge_base.postgres.queries import (
    bulk_insert_cut_list_items,
    bulk_upsert_layer_composites,
    get_background_assignments_for_video,
    get_cut_list_items_for_edit_plan,
    get_edit_plan,
    get_emphasis_effects_for_edit_plan,
    get_layer_composites_for_edit_plan,
    get_shot_mattes_for_shots,
    get_shots_for_video,
    get_stock_asset_by_id,
    get_transcript_segments_for_video,
    get_video,
)
from knowledge_base.r2.uploader import get_presigned_url
from shared.types import (
    CutListItemRecord,
    EditOperationItem,
    EmphasisEffectRecord,
    HighlightCalloutMechanics,
    LayerCompositeRecord,
    ShotRecord,
    VideoMeta,
    ZoomEmphasisMechanics,
)
from shared.utils import gen_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cut-point snapping (the one deterministic-judgment piece)
# ---------------------------------------------------------------------------


def _flatten_words(segments) -> list[tuple[float, float]]:
    """Flatten every word's (start, end) across all transcript segments for a
    video into one time-sorted list.

    `transcript_segments.words` is a JSONB array of
    ``{"word": str, "start": float, "end": float, "probability": float}``
    (verified against a live row — video_id
    ``ca9b60d8-3e7f-480b-b183-9a9b27dc7d01``, e.g.
    ``{"end": 2.76, "word": " ...", "start": 2.28, "probability": 0.86}``).
    Timestamps are absolute (video-relative), not segment-relative, so
    flattening across segments and sorting by `start` is safe.
    """
    words: list[tuple[float, float]] = []
    for seg in segments:
        for w in seg.words or []:
            try:
                start = float(w["start"])
                end = float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end >= start:
                words.append((start, end))
    words.sort(key=lambda t: t[0])
    return words


def _snap_boundary(words: list[tuple[float, float]], boundary: float, window: float = 0.5) -> float:
    """Snap *boundary* to the midpoint of the largest word-gap (silence)
    within ``±window`` seconds of it, or return *boundary* unchanged if no
    gap falls in that window.

    A "gap" is the silence between one word's `end` and the following
    word's `start`. We only consider gaps whose interval
    ``[prev_end, next_start]`` overlaps ``[boundary - window, boundary +
    window]`` — i.e. the gap is *near* the approximate boundary the
    planner picked, not just anywhere in the transcript.
    """
    if not words or len(words) < 2:
        return boundary

    lo = boundary - window
    hi = boundary + window

    best_gap_size = 0.0
    best_mid: float | None = None

    for (prev_start, prev_end), (next_start, next_end) in zip(words, words[1:]):
        gap_start, gap_end = prev_end, next_start
        if gap_end <= gap_start:
            continue  # no silence between these words (overlapping/adjacent)
        # Does this gap interval overlap the search window around boundary?
        if gap_end < lo or gap_start > hi:
            continue
        # Clamp to the portion of the gap that actually falls inside the
        # search window. Without this, a wide silence that merely *touches*
        # the window (e.g. a long pause spanning several seconds) wins on
        # raw size and hands back its own global midpoint — which can sit
        # well outside [lo, hi] and far from `boundary`. Every boundary
        # whose window happens to overlap that same wide gap then collapses
        # onto that one identical, boundary-independent midpoint.
        window_start = max(gap_start, lo)
        window_end = min(gap_end, hi)
        if window_end <= window_start:
            continue
        gap_size = window_end - window_start
        if gap_size > best_gap_size:
            best_gap_size = gap_size
            best_mid = (window_start + window_end) / 2.0

    if best_mid is None:
        return boundary
    return best_mid


async def snap_cut_points(pool: asyncpg.Pool, edit_plan_id: str) -> list[CutListItemRecord]:
    """Build one `CutListItemRecord` per `SELECT_CLIP` op in the finalized
    `edit_plan`, snapping each op's approximate `start_time`/`end_time` to
    the nearest natural pause point in the word-level transcript.

    Does not write to the database — see `write_cut_list`.
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        raise ValueError(f"No edit_plan found for id={edit_plan_id}")

    segments = await get_transcript_segments_for_video(pool, plan.video_id)
    words = _flatten_words(segments)

    items: list[CutListItemRecord] = []
    for raw_op in plan.operations:
        if raw_op.get("type") != "SELECT_CLIP":
            continue
        try:
            op = EditOperationItem.model_validate(raw_op)
        except Exception as exc:
            logger.warning(
                "Skipping malformed SELECT_CLIP op in edit_plan %s: %s (%s)",
                edit_plan_id, raw_op.get("op_id"), exc,
            )
            continue
        if op.start_time is None or op.end_time is None:
            logger.warning(
                "SELECT_CLIP op %s in edit_plan %s has no start_time/end_time — "
                "L5 output is underspecified, skipping (rule 18: L6 never guesses).",
                op.op_id, edit_plan_id,
            )
            continue

        snapped_start = _snap_boundary(words, op.start_time)
        snapped_end = _snap_boundary(words, op.end_time)
        if snapped_end <= snapped_start:
            # Snapping collapsed the clip (pathological — very short clip or
            # word-dense boundary region). Fall back to the plan's original,
            # unsnapped boundaries rather than emit a zero/negative-length cut.
            logger.warning(
                "Snapping collapsed SELECT_CLIP op %s (%.3f -> %.3f); "
                "falling back to unsnapped plan boundaries.",
                op.op_id, snapped_start, snapped_end,
            )
            snapped_start, snapped_end = op.start_time, op.end_time

        items.append(
            CutListItemRecord(
                id=gen_id(),
                edit_plan_id=edit_plan_id,
                op_id=op.op_id,
                sequence_index=op.sequence_index,
                source_start=snapped_start,
                source_end=snapped_end,
                audio_lead_ms=0,
                video_lead_ms=0,
                transition="cut",
            )
        )

    items.sort(key=lambda i: i.sequence_index)
    return items


async def write_cut_list(
    pool: asyncpg.Pool, edit_plan_id: str, items: list[CutListItemRecord]
) -> None:
    """Bulk-insert *items* into `cut_list_items`."""
    for item in items:
        if item.edit_plan_id != edit_plan_id:
            raise ValueError(
                f"CutListItemRecord.edit_plan_id={item.edit_plan_id!r} does not match "
                f"edit_plan_id={edit_plan_id!r} passed to write_cut_list"
            )
    await bulk_insert_cut_list_items(pool, items)


# ---------------------------------------------------------------------------
# A4/A5 mechanics — shared helpers
# ---------------------------------------------------------------------------


def _shot_for_window(shots: list[ShotRecord], start: float, end: float) -> ShotRecord | None:
    """Map a cut-order clip's `[start, end]` window back to the single
    source-order `shots` row it falls inside (by window midpoint) — the one
    place A4's mechanics genuinely needs a source-order lookup (matting is
    per-shot, per CLAUDE.md A1), never used for anything cut-order-sensitive
    (rule 19: sequence-aware decisions — color/zoom neighbor deltas — must
    still key off `cut_list_items`/`sequence_index`, not this).

    Prefers a shot whose time range actually contains the midpoint; falls
    back to the nearest shot by `start_time` distance if the midpoint lands
    in a gap (e.g. cut-point snapping moved the boundary slightly outside
    the shot table's own boundaries) — same graceful-fallback shape as
    `emphasis_selector.py::_nearest_frame`.
    """
    midpoint = (start + end) / 2.0
    containing = [
        s for s in shots
        if s.start_time is not None and s.end_time is not None
        and s.start_time <= midpoint <= s.end_time
    ]
    if containing:
        return containing[0]
    with_time = [s for s in shots if s.start_time is not None]
    if not with_time:
        return None
    return min(with_time, key=lambda s: abs(s.start_time - midpoint))


def _chain_filter(existing: str | None, addition: str) -> str:
    """Chain two per-clip video filter strings for the same
    `cut_list_item.id` — identical semantics to
    `pipeline/level6/updater.py::_merge_filter`, duplicated locally to avoid
    an editing_director -> updater import cycle (updater already imports
    from this module)."""
    return f"{existing},{addition}" if existing else addition


def _clamp_rect(rect: dict, width: float, height: float) -> dict:
    """Clamp a `{x, y, w, h}` pixel rect to the frame bounds so a
    downstream-derived crop/highlight region is always well-formed even if
    the source `target_bbox` (from `face_appearances`, upstream of
    `emphasis_effects.parameters`) sits slightly outside frame bounds
    (detector jitter at the frame edge)."""
    x = max(0.0, min(float(rect.get("x", 0.0)), width))
    y = max(0.0, min(float(rect.get("y", 0.0)), height))
    w = max(1.0, min(float(rect.get("w", 1.0)), max(1.0, width - x)))
    h = max(1.0, min(float(rect.get("h", 1.0)), max(1.0, height - y)))
    return {"x": x, "y": y, "w": w, "h": h}


# ---------------------------------------------------------------------------
# A5 mechanics — ZOOM_EMPHASIS (Ken-Burns crop+scale) / HIGHLIGHT_CALLOUT
# (drawbox/drawtext), driven entirely by `emphasis_effects` rows already
# written by `pipeline/level6/emphasis_selector.py` (decision half — WHAT/
# WHOM to emphasize, never HOW). Neither needs an extra ffmpeg input, so
# both produce a plain filter-chain string that plugs directly into the
# pre-existing `extra_filters` contract (see `get_compositing_render_extras`).
# ---------------------------------------------------------------------------

# How much padding (as a multiple of the target face bbox) the zoomed-in
# `end_rect` keeps around the target — a tight face-fill crop with a little
# headroom, not a pixel-exact crop to the detector's raw bbox.
_ZOOM_END_PADDING = 1.6


def _build_zoom_emphasis_mechanics(
    effect: EmphasisEffectRecord,
    item: CutListItemRecord,
    video_width: float,
    video_height: float,
) -> ZoomEmphasisMechanics | None:
    """Derive `start_rect`/`end_rect`/`easing`/`duration` for one `zoom`
    `emphasis_effects` row — deterministic, no judgment: `start_rect` is
    always the full frame (established wide shot), `end_rect` is a padded
    crop centered on `effect.parameters['target_bbox']` (the face bbox the
    Compositing Agent already grounded its decision in — never re-picked or
    guessed here), `duration` is this clip's own cut-order length.

    Returns None (never invents a rect) when `target_bbox` is missing or
    malformed — rule 12/18.
    """
    bbox = effect.parameters.get("target_bbox")
    if not isinstance(bbox, dict) or not all(k in bbox for k in ("x", "y", "w", "h")):
        logger.warning(
            "editing_director: emphasis_effects row for cut_list_item=%s "
            "(effect_type=zoom) has no usable target_bbox — skipping zoom "
            "mechanics, nothing real to crop toward (rule 12/18).",
            item.id,
        )
        return None

    cx = float(bbox["x"]) + float(bbox["w"]) / 2.0
    cy = float(bbox["y"]) + float(bbox["h"]) / 2.0
    end_w = max(1.0, float(bbox["w"]) * _ZOOM_END_PADDING)
    end_h = max(1.0, float(bbox["h"]) * _ZOOM_END_PADDING)
    end_rect = _clamp_rect(
        {"x": cx - end_w / 2.0, "y": cy - end_h / 2.0, "w": end_w, "h": end_h},
        video_width, video_height,
    )
    start_rect = {"x": 0.0, "y": 0.0, "w": float(video_width), "h": float(video_height)}
    duration = max(0.0, item.source_end - item.source_start)

    return ZoomEmphasisMechanics(
        cut_list_item_id=item.id,
        start_rect=start_rect,
        end_rect=end_rect,
        easing="ease_in_out",
        duration=duration,
    )


def _ease_progress_expr(t_var: str, duration: float) -> str:
    """ffmpeg-expression smoothstep progress (`3p^2 - 2p^3`, `p = t/dur`),
    written without `pow()`/commas to sidestep filtergraph comma-escaping
    entirely (a comma inside a filter option value must otherwise be
    backslash-escaped or it's read as ending the filter chain)."""
    p = f"({t_var}/{duration})"
    return f"(3*{p}*{p}-2*{p}*{p}*{p})"


def build_zoom_emphasis_filter(
    mech: ZoomEmphasisMechanics, video_width: float, video_height: float
) -> str:
    """Build the `crop`+`scale` Ken-Burns filter chain for one clip's own
    LOCAL timeline (`t` starts at 0 at this clip's own trimmed start,
    upstream `setpts=PTS-STARTPTS` already guarantees that — cut-order
    local time, never the source video's absolute timestamp, rule 19).

    Deviates from CLAUDE.md A5's literal `zoompan` filter name: `zoompan`'s
    duration-in-frames/looping model is built for animating a still image
    (or looping a source), not driving motion across a live video clip's
    own native timeline. `crop` accepts `t`-based expressions directly and
    composes into this filter graph as a single stage — the standard idiom
    for real Ken-Burns motion on an actual video clip. The EFFECT (smooth
    interpolated crop from a wide `start_rect` to a tight `end_rect`) is
    exactly what CLAUDE.md specifies; only the specific FFmpeg filter
    primitive differs, flagged here explicitly rather than silently.
    """
    ease = _ease_progress_expr("t", max(mech.duration, 1e-3))
    x0, y0 = mech.start_rect["x"], mech.start_rect["y"]
    w0, h0 = mech.start_rect["w"], mech.start_rect["h"]
    x1, y1 = mech.end_rect["x"], mech.end_rect["y"]
    w1, h1 = mech.end_rect["w"], mech.end_rect["h"]

    def lerp(a: float, b: float) -> str:
        return f"({a}+({b}-({a}))*{ease})"

    crop = (
        f"crop=w='{lerp(w0, w1)}':h='{lerp(h0, h1)}':"
        f"x='{lerp(x0, x1)}':y='{lerp(y0, y1)}'"
    )
    return f"{crop},scale={int(video_width)}:{int(video_height)}"


def _build_highlight_callout_mechanics(
    effect: EmphasisEffectRecord, item: CutListItemRecord
) -> HighlightCalloutMechanics | None:
    """Derive `shape`/`target_bbox`/`start_time`/`duration` for one
    `highlight` `emphasis_effects` row. `emphasis_effects.parameters` is
    decision-level only (WHAT/WHOM, never a drawing shape — see
    `emphasis_selector.py`'s module docstring), so `shape` is the one
    deterministic default this mechanics phase picks: `'circle'`
    (a bounding-region highlight) fits a face-bbox target better than an
    underline/arrow, which are more naturally suited to a text/object
    target this pipeline doesn't currently emit `emphasis_effects` for.
    Visible for the clip's own full duration — no finer-grained timing
    signal exists in `emphasis_effects.parameters` to do better than that.
    """
    bbox = effect.parameters.get("target_bbox")
    if not isinstance(bbox, dict) or not all(k in bbox for k in ("x", "y", "w", "h")):
        logger.warning(
            "editing_director: emphasis_effects row for cut_list_item=%s "
            "(effect_type=highlight) has no usable target_bbox — skipping "
            "highlight mechanics, nothing real to draw around (rule 12/18).",
            item.id,
        )
        return None
    return HighlightCalloutMechanics(
        cut_list_item_id=item.id,
        shape="circle",
        target_bbox={k: float(bbox[k]) for k in ("x", "y", "w", "h")},
        start_time=0.0,
        duration=max(0.0, item.source_end - item.source_start),
    )


def build_highlight_callout_filter(mech: HighlightCalloutMechanics) -> str:
    """Build the overlay-shape filter chain for one `HIGHLIGHT_CALLOUT`.

    `shape` vocabulary per CLAUDE.md A5 (`circle | arrow | underline`).
    FFmpeg has no native circle/arrow primitive without an external asset
    (SVG-to-PNG-to-overlay would add a new dependency, ruled out by the
    task brief's "don't add a new heavyweight dependency"):
      - `underline`: a thin `drawbox` bar under the bbox — exact.
      - `circle`: approximated with a bounding-rectangle `drawbox` around
        the bbox — a documented limitation (not a true circle), same
        "flag the gap honestly" precedent as this module's FCPXML-untested-
        import note.
      - `arrow`: approximated with a `drawtext` arrow glyph placed just
        outside the bbox, pointing at it — also an approximation.
    `enable='between(t,start,end)'` mirrors `caption_overlay.py::
    _drawtext_clause`'s exact convention (single-quoted, plain commas —
    verified safe inside a single-quoted filter option value in this
    codebase's already-shipped caption filters).
    """
    bbox = mech.target_bbox
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    start, end = mech.start_time, mech.start_time + mech.duration
    enable = f"between(t,{start:.3f},{end:.3f})"

    if mech.shape == "underline":
        bar_h = max(4.0, h * 0.06)
        return (
            f"drawbox=x={x:.1f}:y={y + h:.1f}:w={w:.1f}:h={bar_h:.1f}:"
            f"color=yellow@0.9:t=fill:enable='{enable}'"
        )
    if mech.shape == "arrow":
        fontsize = max(24, int(h * 0.5))
        arrow_x = max(0.0, x - h * 0.6)
        return (
            f"drawtext=text='→':fontsize={fontsize}:fontcolor=yellow:"
            f"x={arrow_x:.1f}:y={y:.1f}:enable='{enable}'"
        )
    # default / 'circle'
    return (
        f"drawbox=x={x:.1f}:y={y:.1f}:w={w:.1f}:h={h:.1f}:"
        f"color=yellow@0.9:t=4:enable='{enable}'"
    )


# ---------------------------------------------------------------------------
# A4 mechanics — LAYER_COMPOSITE (background_swap / pip / overlay), driven
# by `background_assignments` (A3's decision, via the clip's own SELECT_CLIP
# op's `scene_id`) + `shot_mattes` (L2's matting output). Background-swap
# needs two ADDITIONAL ffmpeg inputs beyond the source video (the matte clip
# + the stock background clip) — that's the one branch `render_direct`
# needed new plumbing for (`extra_inputs`/`composite_filters`).
# ---------------------------------------------------------------------------


async def materialize_layer_composite_ops(
    pool: asyncpg.Pool, edit_plan_id: str
) -> list[LayerCompositeRecord]:
    """Auto-materialize `layer_composites` rows for a finalized edit_plan's
    `cut_list_items`, FROM `background_assignments` — L5 never decides
    compositing (it only knows `storylines`/`scenes`), and the Compositing
    Agent's background pick happens at the SCENE level, decided
    independently of any particular edit_plan/cut order. This is the one
    place that resolves "which scene does this cut-order clip belong to"
    (via the clip's own `SELECT_CLIP` op's `scene_id` in
    `edit_plan.operations`) and pulls that scene's background pick + the
    matching shot's alpha matte into a `cut_list_item`-keyed mechanics
    record, ready for `get_compositing_render_extras` to turn into an
    actual FFmpeg composite at render time.

    Only emits `layer_type='background_swap'` rows — the only compositing
    decision any agent in this pipeline currently writes
    (`background_selector.py`). `pip`/`overlay` mechanics are implemented
    (`build_pip_overlay_filter_template`) for the day an agent starts
    writing that decision, but nothing materializes them automatically yet
    — flagged here rather than silently faked.

    Idempotent re-run: `bulk_upsert_layer_composites` upserts on
    `(cut_list_item_id, layer_type)` (rule 9/15).
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        raise ValueError(f"No edit_plan found for id={edit_plan_id}")

    cut_items = await get_cut_list_items_for_edit_plan(pool, edit_plan_id)
    if not cut_items:
        logger.warning(
            "materialize_layer_composite_ops: edit_plan_id=%s has no "
            "cut_list_items — has the Editing Director's cut list run yet?",
            edit_plan_id,
        )
        return []

    select_clip_ops_by_id = {
        op.get("op_id"): op for op in plan.operations if op.get("type") == "SELECT_CLIP"
    }
    background_assignments = await get_background_assignments_for_video(pool, plan.video_id)
    bg_by_scene_id = {b.scene_id: b for b in background_assignments}

    shots = await get_shots_for_video(pool, plan.video_id)
    mattes_by_shot = await get_shot_mattes_for_shots(pool, [s.id for s in shots])

    records: list[LayerCompositeRecord] = []
    n_no_scene = n_no_bg = n_no_matte = 0
    for item in cut_items:
        op = select_clip_ops_by_id.get(item.op_id)
        scene_id = op.get("scene_id") if op else None
        if not scene_id:
            n_no_scene += 1
            continue
        bg = bg_by_scene_id.get(scene_id)
        if bg is None:
            n_no_bg += 1
            continue
        shot = _shot_for_window(shots, item.source_start, item.source_end)
        matte = mattes_by_shot.get(shot.id) if shot else None
        if matte is None:
            n_no_matte += 1
            logger.warning(
                "materialize_layer_composite_ops: cut_list_item=%s (scene=%s) "
                "has a background_assignments pick but no shot_mattes row for "
                "its shot — background_swap needs both A1 (matting) and A3 "
                "(background pick) to have run; skipping, not faking a matte.",
                item.id, scene_id,
            )
            continue

        records.append(
            LayerCompositeRecord(
                id=gen_id(),
                cut_list_item_id=item.id,
                layer_type="background_swap",
                source_ref=bg.id,
                position=None,   # full-frame swap — no explicit placement rect
                opacity=1.0,
                z_index=0,
            )
        )

    if records:
        await bulk_upsert_layer_composites(pool, records)

    logger.info(
        "materialize_layer_composite_ops: edit_plan_id=%s materialized %d/%d "
        "background_swap layer_composites (%d no scene_id, %d no background "
        "pick for their scene, %d no matte for their shot)",
        edit_plan_id, len(records), len(cut_items), n_no_scene, n_no_bg, n_no_matte,
    )
    return records


def build_layer_composite_filter_template(
    layer: LayerCompositeRecord, video_width: int, video_height: int
) -> str:
    """Return a filter_complex TEMPLATE for one `background_swap`
    LAYER_COMPOSITE op: alpha-composites the clip's own footage (masked by
    its shot's alpha matte) over the assigned stock background asset.

    Contains the placeholders `{V}` (this clip's current running video
    label), `{OUT}` (the label `render_direct` assigns this stage's
    output), and `{IN_matte}`/`{IN_bg}` (the matte/background ffmpeg input
    labels) — `get_compositing_render_extras` renames the latter two to
    this clip's own unique input keys before handing the template to
    `render_direct`, which is the only place that resolves `{IN_<key>}` to
    an actual `"<global_ffmpeg_input_index>:v"` label (index assignment is
    a render-step concern, not this function's — mirrors the same
    "template vs. resolved label" boundary `render_direct`'s pre-existing
    `extra_filters` docstring already draws for per-clip color/caption
    filters).
    """
    w, h = int(video_width), int(video_height)
    opacity = max(0.0, min(1.0, layer.opacity))
    return (
        f"[{{V}}]format=yuva420p[fg_rgb];"
        f"[{{IN_matte}}]format=gray,scale={w}:{h}[alpha];"
        f"[fg_rgb][alpha]alphamerge,colorchannelmixer=aa={opacity}[fg];"
        f"[{{IN_bg}}]scale={w}:{h},setsar=1,format=yuv420p[bgscaled];"
        f"[bgscaled][fg]overlay=0:0:format=auto[{{OUT}}]"
    )


def build_pip_overlay_filter_template(
    layer: LayerCompositeRecord,
    other_item: CutListItemRecord,
    video_width: int,
    video_height: int,
) -> str:
    """Return a filter_complex TEMPLATE for a `pip`/`overlay` LAYER_COMPOSITE
    op whose `source_ref` is ANOTHER `cut_list_item_id` in the same edit
    plan. Unlike background-swap, this needs no extra ffmpeg `-i` — it
    re-trims the SAME main source input (`[0:v]`) at `other_item`'s own
    `source_start`/`source_end`, so only `{V}`/`{OUT}` are left as
    placeholders for `render_direct` to resolve.

    Not called by anything today — no agent in this pipeline currently
    writes a `pip`/`overlay` `layer_composites` row (see
    `materialize_layer_composite_ops`'s docstring) — implemented so the
    mechanics exist the moment one does, per the same "mechanics vs.
    decision" split as everywhere else in A4/A5.
    """
    pos = layer.position or {}
    x = float(pos.get("x", 0.0))
    y = float(pos.get("y", 0.0))
    pw = max(1, int(pos.get("w", video_width * 0.3)))
    ph = max(1, int(pos.get("h", video_height * 0.3)))
    opacity = max(0.0, min(1.0, layer.opacity))
    # FFmpeg filtergraph link labels only reliably accept
    # alnum/underscore — `cut_list_item.id` is a UUID4 string (hyphens at
    # fixed positions), so strip anything non-alnum defensively rather than
    # relying on "the first 8 chars of a UUID4 never contain a hyphen"
    # staying true if the id generator ever changes (`shared/utils.py::gen_id`).
    safe_id = re.sub(r"[^0-9A-Za-z]", "", other_item.id)[:12] or "0"
    pip_label = f"pip_src_{safe_id}"
    return (
        f"[0:v]trim=start={other_item.source_start}:end={other_item.source_end},"
        f"setpts=PTS-STARTPTS,scale={pw}:{ph},setsar=1,"
        f"format=yuva420p,colorchannelmixer=aa={opacity}[{pip_label}];"
        f"[{{V}}][{pip_label}]overlay={int(x)}:{int(y)}:format=auto[{{OUT}}]"
    )


async def get_compositing_render_extras(
    pool: asyncpg.Pool, edit_plan_id: str
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Build `render_direct`'s three A4/A5 render-time inputs from what's
    already been decided/materialized: A5's `emphasis_effects` (read
    directly — no separate mechanics table, see `shared/types.py::
    ZoomEmphasisMechanics`'s docstring) and A4's `layer_composites` (read
    directly — call `materialize_layer_composite_ops` first if it hasn't
    been populated for this edit_plan yet; this function only READS, same
    read/write split as every other L6 stage).

    Returns `(extra_filters, composite_filters, extra_inputs)`:
      - `extra_filters`: `{cut_list_item_id: filter_chain_str}` — zoom/
        highlight results, meant to be merged into the SAME dict
        `pipeline/level6/updater.py` already builds for color/captions
        (via its `_merge_filter`), since neither needs a new ffmpeg input.
      - `composite_filters` / `extra_inputs`: `render_direct`'s new
        params — background-swap's two extra media inputs (matte + stock
        background), see `render_direct`'s docstring.

    Any clip this function touches (either dict) FORCES `render_direct`
    onto the filter_complex re-encode path — compositing/zoom/highlight can
    never take the stream-copy fast path (`render_direct` already checks
    `composite_filters`/`extra_filters` truthiness for this; nothing extra
    needed here).
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        raise ValueError(f"No edit_plan found for id={edit_plan_id}")
    video = await get_video(pool, plan.video_id)
    if video is None:
        raise ValueError(f"No video found for id={plan.video_id}")
    width = video.width or 1920
    height = video.height or 1080

    cut_items = await get_cut_list_items_for_edit_plan(pool, edit_plan_id)
    cut_items_by_id = {c.id: c for c in cut_items}

    extra_filters: dict[str, str] = {}
    composite_filters: dict[str, list[str]] = {}
    extra_inputs: dict[str, str] = {}

    # ---- A5 mechanics: zoom / highlight — no extra ffmpeg input needed ----
    emphasis_effects = await get_emphasis_effects_for_edit_plan(pool, edit_plan_id)
    for effect in emphasis_effects:
        item = cut_items_by_id.get(effect.cut_list_item_id)
        if item is None:
            continue
        if effect.effect_type == "zoom":
            mech = _build_zoom_emphasis_mechanics(effect, item, width, height)
            if mech is None:
                continue
            filt = build_zoom_emphasis_filter(mech, width, height)
            extra_filters[item.id] = _chain_filter(extra_filters.get(item.id), filt)
        elif effect.effect_type == "highlight":
            hl_mech = _build_highlight_callout_mechanics(effect, item)
            if hl_mech is None:
                continue
            filt = build_highlight_callout_filter(hl_mech)
            extra_filters[item.id] = _chain_filter(extra_filters.get(item.id), filt)

    # ---- A4 mechanics: layer composites — background_swap needs 2 extra -i ----
    layers = await get_layer_composites_for_edit_plan(pool, edit_plan_id)
    if layers:
        shots = await get_shots_for_video(pool, plan.video_id)
        mattes_by_shot = await get_shot_mattes_for_shots(pool, [s.id for s in shots])
        bg_assignments = await get_background_assignments_for_video(pool, plan.video_id)
        bg_by_id = {b.id: b for b in bg_assignments}

        for layer in layers:
            item = cut_items_by_id.get(layer.cut_list_item_id)
            if item is None:
                continue

            if layer.layer_type == "background_swap":
                shot = _shot_for_window(shots, item.source_start, item.source_end)
                matte = mattes_by_shot.get(shot.id) if shot else None
                bg = bg_by_id.get(layer.source_ref) if layer.source_ref else None
                asset = await get_stock_asset_by_id(pool, bg.asset_id) if bg else None
                if matte is None or bg is None or asset is None or not asset.r2_cache_key:
                    logger.warning(
                        "get_compositing_render_extras: cut_list_item=%s "
                        "background_swap layer_composites row could not be "
                        "resolved at render time (matte=%s bg=%s asset=%s) — "
                        "skipping this clip's compositing, it renders with its "
                        "own original background instead.",
                        item.id, matte is not None, bg is not None, asset is not None,
                    )
                    continue

                matte_key = f"matte_{item.id}"
                bg_key = f"bg_{item.id}"
                extra_inputs[matte_key] = get_presigned_url(matte.r2_key)
                extra_inputs[bg_key] = get_presigned_url(asset.r2_cache_key)

                template = build_layer_composite_filter_template(layer, width, height)
                template = (
                    template.replace("{IN_matte}", f"{{IN_{matte_key}}}")
                    .replace("{IN_bg}", f"{{IN_{bg_key}}}")
                )
                composite_filters.setdefault(item.id, []).append(template)

            elif layer.layer_type in ("pip", "overlay") and layer.source_ref:
                other_item = cut_items_by_id.get(layer.source_ref)
                if other_item is None:
                    logger.warning(
                        "get_compositing_render_extras: cut_list_item=%s %s "
                        "layer_composites row references source_ref=%s which "
                        "is not a cut_list_item in this edit_plan — skipping "
                        "(rule 12: never invent a source).",
                        item.id, layer.layer_type, layer.source_ref,
                    )
                    continue
                template = build_pip_overlay_filter_template(layer, other_item, width, height)
                composite_filters.setdefault(item.id, []).append(template)

    logger.info(
        "get_compositing_render_extras: edit_plan_id=%s built extra_filters=%d "
        "(zoom/highlight) composite_filters=%d (layer composites) extra_inputs=%d",
        edit_plan_id, len(extra_filters), len(composite_filters), len(extra_inputs),
    )
    return extra_filters, composite_filters, extra_inputs


# ---------------------------------------------------------------------------
# FCPXML export (primary output — Resolve/Premiere importable)
# ---------------------------------------------------------------------------


def _rational_time(seconds: float, fps: float) -> str:
    """Format *seconds* as an FCPXML rational time string, frame-aligned to
    *fps* (FCPXML's `N/Ds` convention — e.g. ``"125/25s"``).

    All FCPXML `offset`/`start`/`duration` values must be exact multiples
    of the edit rate. We round to the nearest whole frame at *fps* and
    express it as ``frames/den`` where ``den`` is the integer frame rate.

    Note: this assumes an integer-ish fps (e.g. 25, 30, 24, 60), which
    matches every `videos.fps` value observed in this pipeline so far
    (verified 25.0 on a live row). True NTSC drop-frame rates (23.976,
    29.97, 59.94) need the ``N * 1000 / 1001`` timebase convention
    (e.g. ``30000/1001s``) instead of a plain integer denominator — not
    implemented here; flagged as a known gap if a non-integer-fps source
    is ever fed into this pipeline.
    """
    den = max(1, round(fps)) if fps else 25
    frames = round(seconds * den)
    return f"{frames}/{den}s"


def _asset_src_uri(video: VideoMeta) -> str:
    """Resolve the source media reference for the FCPXML `<asset>` `src`.

    Prefers `r2_key` (an object key, not a URI — turned into a `file://`
    style reference the NLE operator resolves against their local copy)
    over `path`. `videos.path` observed in the live DB is itself already a
    full `https://` URL (e.g. R2 public URL) — FCPXML strictly expects a
    `file://` URI for local media, but many NLEs (Resolve, Premiere) will
    at least surface an `http(s)://` `src` as a relinkable reference rather
    than failing import outright. This is passed through as-is; exact
    behavior depends on the importing application.
    """
    if video.r2_key:
        return f"file:///{video.r2_key.lstrip('/')}"
    if video.path.startswith(("http://", "https://", "file://")):
        return video.path
    normalized = video.path.replace("\\", "/")
    return f"file:///{normalized.lstrip('/')}"


async def export_xml(pool: asyncpg.Pool, edit_plan_id: str, output_path: str) -> str:
    """Build a minimal, valid FCPXML 1.9 project from `cut_list_items` and
    write it to *output_path*. Returns *output_path*.

    Structure: one `<asset>` (the source video) in `<resources>`, one
    `<sequence>`/`<spine>` with one `<asset-clip>` per cut_list_item, laid
    out back-to-back (each clip's timeline `offset` is the running sum of
    prior clip durations) referencing the source asset's `start`/`duration`
    in-point.

    FCPXML-schema confidence: best-effort, based on the documented FCPXML
    1.9+ element/attribute names (`fcpxml`, `resources`, `format`, `asset`,
    `library`, `event`, `project`, `sequence`, `spine`, `asset-clip`, and
    the rational `N/Ds` time convention). Not verified against an actual
    Resolve or Premiere import in this environment (no NLE available to
    test against) — treat as needing a real-import review before relying
    on it in production.
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        raise ValueError(f"No edit_plan found for id={edit_plan_id}")
    video = await get_video(pool, plan.video_id)
    if video is None:
        raise ValueError(f"No video found for id={plan.video_id}")
    items = await get_cut_list_items_for_edit_plan(pool, edit_plan_id)
    if not items:
        raise ValueError(f"No cut_list_items found for edit_plan_id={edit_plan_id}")

    fps = video.fps or 25.0
    width = video.width or 1920
    height = video.height or 1080
    src_uri = _asset_src_uri(video)
    total_duration = sum(max(0.0, item.source_end - item.source_start) for item in items)

    fcpxml = ET.Element("fcpxml", version="1.9")

    resources = ET.SubElement(fcpxml, "resources")
    fmt = ET.SubElement(
        resources,
        "format",
        id="r1",
        name=f"FFVideoFormat{height}p{round(fps)}",
        frameDuration=f"1/{max(1, round(fps))}s",
        width=str(width),
        height=str(height),
    )
    asset = ET.SubElement(
        resources,
        "asset",
        id="r2",
        name=Path(video.path).stem or "source",
        src=src_uri,
        start="0/1s",
        duration=_rational_time(video.duration_s or total_duration, fps),
        hasVideo="1",
        hasAudio="1",
        format="r1",
    )
    del fmt, asset  # constructed for their side effect (attached to tree)

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="Video KB Pipeline")
    project = ET.SubElement(event, "project", name=f"edit_plan_{edit_plan_id}")
    sequence = ET.SubElement(
        project,
        "sequence",
        format="r1",
        duration=_rational_time(total_duration, fps),
        tcStart="0/1s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence, "spine")

    offset = 0.0
    for item in items:
        clip_start = item.source_start
        clip_duration = max(0.0, item.source_end - item.source_start)
        ET.SubElement(
            spine,
            "asset-clip",
            name=f"cut_{item.sequence_index}_{item.op_id}",
            ref="r2",
            offset=_rational_time(offset, fps),
            start=_rational_time(clip_start, fps),
            duration=_rational_time(clip_duration, fps),
        )
        offset += clip_duration

    # Pretty-print via minidom (ElementTree has no built-in indent on all
    # supported Python versions) and prepend the required DOCTYPE.
    rough = ET.tostring(fcpxml, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    # minidom's toprettyxml emits its own `<?xml ...?>` line; replace it and
    # add the FCPXML DOCTYPE declaration required by importing applications.
    body = pretty.split("\n", 1)[1] if pretty.startswith("<?xml") else pretty
    full = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n' + body

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(full, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Direct FFmpeg render (secondary output — quick preview)
# ---------------------------------------------------------------------------


def _nearest_preceding_keyframe_time(
    video_path: str, timestamp: float, search_back: float = 6.0, timeout: float = 30.0
) -> float | None:
    """Return the timestamp of the nearest I-frame at or before *timestamp*,
    scanning only a small window (`[timestamp - search_back, timestamp +
    0.1]`) via ffprobe's `-read_intervals` so this stays fast on long
    videos. Returns None if ffprobe fails, times out, or finds nothing
    (e.g. unreachable remote source) — callers must treat None as "assume
    not aligned, take the safe re-encode path".
    """
    start = max(0.0, timestamp - search_back)
    end = timestamp + 0.1
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-read_intervals", f"{start}%{end}",
        "-show_entries", "frame=pts_time,pict_type",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        logger.warning("ffprobe keyframe scan failed for %s @ %.3f: %s", video_path, timestamp, exc)
        return None
    if proc.returncode != 0:
        logger.warning("ffprobe keyframe scan non-zero exit for %s @ %.3f: %s", video_path, timestamp, proc.stderr[-500:])
        return None

    best: float | None = None
    for line in proc.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        t_str, frame_type = parts
        if frame_type.strip() != "I":
            continue
        try:
            t = float(t_str)
        except ValueError:
            continue
        if t <= timestamp and (best is None or t > best):
            best = t
    return best


def _is_keyframe_aligned(video_path: str, timestamp: float, tol: float = 0.04) -> bool:
    """Whether *timestamp* lands on (or within `tol` seconds of) an
    existing I-frame — the condition under which a stream-copy cut at this
    boundary is frame-accurate without re-encoding."""
    nearest = _nearest_preceding_keyframe_time(video_path, timestamp)
    return nearest is not None and abs(nearest - timestamp) <= tol


def _write_concat_list(items: list[CutListItemRecord], source_video_path: str) -> str:
    """Write an ffmpeg concat-demuxer list file (one `file`/`inpoint`/
    `outpoint` triple per cut_list_item) and return its path. Used only on
    the all-keyframe-aligned, no-extra-filters fast path."""
    fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="cutlist_")
    escaped = source_video_path.replace("'", r"'\''")
    with open(fd, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"file '{escaped}'\n")
            f.write(f"inpoint {item.source_start}\n")
            f.write(f"outpoint {item.source_end}\n")
    return list_path


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr (tail): {proc.stderr[-2000:]}"
        )


def _extract_clip(
    source_video_path: str,
    item: "CutListItemRecord",
    out_path: str,
    extra_filter: str | None,
    composite_templates: list[str],
    extra_inputs: dict[str, str],
) -> None:
    """Extract+filter ONE `cut_list_item` to *out_path* via its own
    independent ffmpeg process — its own input-level seek, its own decode,
    isolated from every other clip.

    Real-video finding, first live L6 run: the old approach ran ONE
    ffmpeg invocation with every clip's `trim`/`atrim` sharing a SINGLE
    decode pass of `source_video_path`, then `concat`ed them in the plan's
    output order. That's fine when clips are in chronological source
    order, but a real edit reorders clips relative to source time (e.g.
    use the clip at 131s first, then a clip spanning 0-131s second) —
    `concat` must consume frames in plan order, but the shared decode only
    produces them in source order, so ffmpeg has to buffer entire
    out-of-order segments in memory waiting for `concat` to get to them.
    Confirmed via direct reproduction: a 2-branch case (a tiny clip placed
    before a 131s clip) logged "1000 buffers queued... something may be
    wrong" and took 2m38s for what should be instant; the real 7-clip plan
    crashed outright with `Cannot allocate memory` (not a real system OOM —
    ~7GB was free). Per-clip independent extraction sidesteps this
    entirely: each clip gets its own bounded decode, nothing is ever
    buffered waiting on another clip's position in a shared stream.

    Always re-encodes (never stream-copy) — this per-clip pass is what
    guarantees every temp clip shares the same codec/pixel format/audio
    params, which is what makes the final join losslessly copyable
    regardless of whether any individual clip actually needed filtering.

    *composite_templates* are resolved with LOCAL input indices scoped to
    just this one ffmpeg process (only the `extra_inputs` entries this
    clip's own templates actually reference get opened here — not every
    entry in the caller's global `extra_inputs` dict), since each clip now
    runs in total isolation from every other clip's inputs.
    """
    used_keys: list[str] = []
    for tmpl in composite_templates:
        for key in extra_inputs:
            if f"{{IN_{key}}}" in tmpl and key not in used_keys:
                used_keys.append(key)

    input_args: list[str] = ["-i", source_video_path]
    local_input_key_to_label: dict[str, str] = {}
    for i, key in enumerate(used_keys, start=1):
        input_args += ["-i", extra_inputs[key]]
        local_input_key_to_label[key] = f"{i}:v"

    v_label = "v0"
    a_label = "a0"
    filter_parts = [
        f"[0:v]trim=start={item.source_start}:end={item.source_end},"
        f"setpts=PTS-STARTPTS[{v_label}]",
        f"[0:a]atrim=start={item.source_start}:end={item.source_end},"
        f"asetpts=PTS-STARTPTS[{a_label}]",
    ]
    if extra_filter:
        filtered_v_label = f"{v_label}f"
        filter_parts.append(f"[{v_label}]{extra_filter}[{filtered_v_label}]")
        v_label = filtered_v_label

    for stage_idx, template in enumerate(composite_templates):
        out_label = f"{v_label}c{stage_idx}"
        format_kwargs = {"V": v_label, "OUT": out_label}
        format_kwargs.update(
            {f"IN_{key}": label for key, label in local_input_key_to_label.items()}
        )
        try:
            rendered = template.format(**format_kwargs)
        except KeyError as exc:
            raise ValueError(
                f"render_direct: composite_filters template for cut_list_item="
                f"{item.id} references placeholder {exc} not present in "
                f"extra_inputs — every {{IN_<key>}} a template uses must have "
                f"a matching extra_inputs entry."
            ) from exc
        filter_parts.append(rendered)
        v_label = out_label

    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", f"[{v_label}]", "-map", f"[{a_label}]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    _run_ffmpeg(cmd)


async def render_direct(
    pool: asyncpg.Pool,
    edit_plan_id: str,
    source_video_path: str,
    output_path: str,
    extra_filters: dict[str, str] | None = None,
    final_audio_filter: str | None = None,
    extra_inputs: dict[str, str] | None = None,
    composite_filters: dict[str, list[str]] | None = None,
) -> str:
    """Render `cut_list_items` for *edit_plan_id* directly to *output_path*
    with a single ffmpeg invocation.

    Two internal paths, chosen automatically:

    1. **Fast path** (`-c copy`): only taken when every cut's `source_start`
       is keyframe-aligned (checked per-clip via ffprobe,
       `_is_keyframe_aligned`) *and* no `extra_filters`/`composite_filters`
       were supplied. Uses the ffmpeg concat demuxer (`-f concat -c copy`)
       — lossless, no re-encode, one ffmpeg invocation, no intermediate
       files besides the tiny concat list.
    2. **filter_complex path**: taken whenever any cut needs a mid-GOP
       frame-accurate boundary or has per-clip filters (`extra_filters`
       and/or `composite_filters`). Single main input, per-clip
       `trim`/`atrim` + `setpts`/`asetpts`, then `concat` — one ffmpeg
       invocation, no intermediate files at all.

    `extra_filters` is the extension point for per-clip VIDEO adjustments:
    `{cut_list_item.id: "<ffmpeg filter chain string>"}`, applied to that
    clip's trimmed video stream before the final concat (e.g. Engineer B's
    `sequence_color_adjustments` translated into `eq=...,colorbalance=...`
    via `build_ffmpeg_color_filters`, or A5's zoom/highlight mechanics via
    `build_zoom_emphasis_filter`/`build_highlight_callout_filter` — see
    `get_compositing_render_extras`). It only ever wraps the `v{idx}`
    label — there is no per-clip audio hook, because the one audio
    operation this pipeline has (loudnorm normalization, from
    `audio_sync.py`) is a SEQUENCE-level operation by nature — it
    normalizes across the whole assembled track, not per-clip — so it
    belongs after concat, not per-clip. Use `final_audio_filter` for that
    instead (see below).

    `final_audio_filter` is a single ffmpeg filter chain string applied to
    the `[outa]` label AFTER concat (e.g. `audio_sync.compute_loudnorm_filter(...)`'s
    result) — sequence-level audio, not per-clip.

    `extra_inputs`/`composite_filters` are A4's (`LAYER_COMPOSITE`)
    extension point, added alongside `extra_filters` rather than folded
    into it because background-swap compositing needs ADDITIONAL ffmpeg
    `-i` sources (the shot's alpha matte + the stock background clip) that
    a plain `{cut_list_item_id: filter_chain_str}` value has no way to
    express — a filter chain string can reference other LABELS, but it
    can't tell this function "and also open this other file as a new input":
      - `extra_inputs`: `{input_key: url_or_path}` — every additional
        media source any clip's `composite_filters` templates need,
        keyed by an arbitrary caller-chosen string (`get_compositing_
        render_extras` uses `f"matte_{cut_list_item_id}"` /
        `f"bg_{cut_list_item_id}"` so keys never collide across clips).
        Each gets its own `-i` arg, assigned ffmpeg input index
        `1, 2, 3, ...` in `extra_inputs`' iteration order (main
        `source_video_path` is always index `0`).
      - `composite_filters`: `{cut_list_item_id: [filter_template, ...]}`
        — one or more filter_complex FRAGMENT strings (each may itself
        contain multiple `;`-separated statements, e.g. background-swap's
        alpha-merge chain) applied to that clip IN ORDER, after any
        `extra_filters` entry for the same clip. Each template must use
        `{V}` for the clip's current running video label and `{OUT}` for
        the label this function assigns as that stage's output (the next
        stage, or the final concat, then reads `{OUT}`'s resolved label as
        its own `{V}`) — mirrors `extra_filters`' existing "wraps `v{idx}`"
        contract, just multi-stage. A template may also reference
        `{IN_<input_key>}` for any key present in `extra_inputs`, resolved
        here to that input's assigned `"<index>:v"` label — never resolved
        by the caller (index assignment is this function's job, same
        "template vs. resolved label" split `build_layer_composite_filter_
        template`'s docstring already documents).

    Any clip with an `extra_filters` and/or `composite_filters` entry (or
    any `final_audio_filter`) forces the filter_complex re-encode path —
    compositing/zoom/highlight, like color/captions before it, can never
    take the stream-copy fast path.

    This function performs cut-only assembly plus whatever `extra_filters`,
    `composite_filters`/`extra_inputs` (video, per-clip) and
    `final_audio_filter` (audio, sequence-level) the caller supplies —
    nothing is applied unless explicitly passed in.
    """
    items = await get_cut_list_items_for_edit_plan(pool, edit_plan_id)
    if not items:
        raise ValueError(f"No cut_list_items found for edit_plan_id={edit_plan_id}")
    extra_filters = extra_filters or {}
    extra_inputs = extra_inputs or {}
    composite_filters = composite_filters or {}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_aligned = all(
        _is_keyframe_aligned(source_video_path, item.source_start) for item in items
    )

    if all_aligned and not extra_filters and not final_audio_filter and not composite_filters:
        logger.info(
            "render_direct: all %d cuts keyframe-aligned, no extra_filters/"
            "composite_filters — using stream-copy concat-demuxer fast path.",
            len(items),
        )
        list_path = _write_concat_list(items, source_video_path)
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                output_path,
            ]
            _run_ffmpeg(cmd)
        finally:
            try:
                Path(list_path).unlink(missing_ok=True)
            except Exception:
                pass
        return output_path

    logger.info(
        "render_direct: %d cuts, at least one mid-GOP or filtered "
        "(extra_filters=%d composite_filters=%d extra_inputs=%d) — "
        "extracting each clip independently, then joining (see _extract_clip "
        "docstring for why this replaced a single shared-decode filter_complex).",
        len(items), len(extra_filters), len(composite_filters), len(extra_inputs),
    )

    tmp_dir = tempfile.mkdtemp(prefix="render_clips_")
    try:
        clip_paths: list[str] = []
        for idx, item in enumerate(items):
            clip_path = str(Path(tmp_dir) / f"clip_{idx:04d}.mp4")
            _extract_clip(
                source_video_path,
                item,
                clip_path,
                extra_filters.get(item.id),
                composite_filters.get(item.id, []),
                extra_inputs,
            )
            clip_paths.append(clip_path)

        if not final_audio_filter:
            # Every temp clip already shares codec/pixel-format/audio params
            # (see _extract_clip) — a lossless, cheap concat-demuxer join.
            list_path = str(Path(tmp_dir) / "concat_list.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in clip_paths:
                    escaped = p.replace("'", r"'\''")
                    f.write(f"file '{escaped}'\n")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                output_path,
            ]
            _run_ffmpeg(cmd)
        else:
            # final_audio_filter (e.g. audio_sync.py's loudnorm) is a
            # SEQUENCE-LEVEL filter — needs a re-encode pass regardless, so
            # join via filter_complex over the N independent temp clips
            # (each its own `-i`, already-decoded-once-each small files) —
            # NOT the old single-shared-source-decode approach this whole
            # fix replaced. N independent short inputs don't have the
            # out-of-order buffering problem a single long shared decode
            # does, so this stage is safe even though it re-encodes.
            input_args: list[str] = []
            concat_inputs: list[str] = []
            for i, p in enumerate(clip_paths):
                input_args += ["-i", p]
                concat_inputs.append(f"[{i}:v][{i}:a]")
            n = len(clip_paths)
            filter_complex = (
                f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa];"
                f"[outa]{final_audio_filter}[outa_final]"
            )
            cmd = [
                "ffmpeg", "-y",
                *input_args,
                "-filter_complex", filter_complex,
                "-map", "[outv]", "-map", "[outa_final]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ]
            _run_ffmpeg(cmd)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return output_path
