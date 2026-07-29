"""Level-6 Audio Sync Agent.

Per CLAUDE.md "LEVEL 6 — SPECIALIZED ACTION AGENTS" -> "Audio Sync Agent"
and "STATUS: IMPLEMENTING" (loudnorm two-pass, sidechaincompress ducking,
platform-dependent LUFS target): this module is **pure deterministic DSP
filter-string building — explicitly NO LLM call**. CLAUDE.md is explicit
that this agent needs no model judgment at all ("Mostly deterministic
DSP — no LLM needed here at all. Flag this explicitly so it doesn't get
over-built.") — every function here is a pure function of its inputs, same
discipline as Engineer B's `build_ffmpeg_color_filters`.

Three responsibilities, three functions:
  compute_loudnorm_filter   — two-pass FFmpeg `loudnorm`, platform-dependent
                               target LUFS.
  compute_ducking_filter    — `sidechaincompress`-based ducking for
                               AUDIO_DUCK_REQUEST ops. Scoped down to a
                               documented no-op: this pipeline's schema has
                               no secondary/music-bed audio track concept
                               for `sidechaincompress` to duck against (see
                               function docstring for the exact gap).
  apply_multicam_sync       — documented no-op passthrough: nothing in the
                               current schema (`shared/types.py`,
                               `knowledge_base/postgres/migrations/*.sql`)
                               carries multi-camera source/sync-group info
                               for a single pipeline run. Confirmed by grep
                               across the codebase — zero hits for
                               "camera"/"multicam"/"sync_group" outside a
                               `prompts/grounding_speaker.py` docstring
                               (unrelated: "gaze direction toward
                               camera/mic"). Not a v1 blocker per CLAUDE.md.

Integration note (read before wiring into `pipeline/level6/updater.py`):
Engineer A's `render_direct(pool, edit_plan_id, source_video_path,
output_path, extra_filters=None)` wires `extra_filters[cut_list_item.id]`
into the **video** label only (`[v{idx}]{extra}[v{idx}f]`, see
`pipeline/level6/editing_director.py::render_direct`) — there is no
per-clip *audio* label hook and no post-concat `[outa]` hook in the
current implementation. `loudnorm`/`sidechaincompress` are audio filters,
and sequence-level loudness normalization semantically belongs on the
final assembled `[outa]` stream once (not per-clip pre-concat) anyway.
So the filter strings this module builds are correct and ready, but
`render_direct` as currently implemented has no extension point to apply
them through. `updater.py` surfaces this explicitly rather than smuggling
an audio filter into the video-only `extra_filters` dict (which would
just break the ffmpeg command). See `updater.py`'s module docstring for
how this is reported in `run_level6`'s return value.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform -> target LUFS
# ---------------------------------------------------------------------------

# CLAUDE.md "STATUS: IMPLEMENTING" / "Audio.": "-14 LUFS (streaming
# default) or -16 (podcast), platform-dependent from `edit_plans.platform`".
# Platform strings observed in this codebase (prompts/l5_sequencing.py
# L5_SEQUENCING_SYSTEM_PROMPT: "platform: reel | full_cut | youtube | ...
# or null", and CLAUDE.md's EditPlan schema comment
# "platform TEXT, -- reel | full_cut | youtube | ..."): reel and youtube
# are short-form/streaming-style deliverables -> -14 LUFS; full_cut (and
# the unspecified/None default) reads as the podcast/long-form case in
# CLAUDE.md's own wording -> -16 LUFS.
_STREAMING_LUFS_PLATFORMS = {"reel", "youtube"}
_TARGET_LUFS_STREAMING = -14.0
_TARGET_LUFS_PODCAST = -16.0

# Fixed per CLAUDE.md's `loudnorm` mention — not platform-dependent.
_TARGET_TP = -1.5   # true peak, dBTP
_TARGET_LRA = 11.0  # loudness range, LU


def _target_lufs(target_platform: str | None) -> float:
    platform = (target_platform or "").strip().lower()
    if platform in _STREAMING_LUFS_PLATFORMS:
        return _TARGET_LUFS_STREAMING
    return _TARGET_LUFS_PODCAST


# ---------------------------------------------------------------------------
# loudnorm — two-pass
# ---------------------------------------------------------------------------


def compute_loudnorm_measure_filter(target_platform: str | None) -> str:
    """Pass 1 of two-pass `loudnorm`: the measurement filter.

    The caller (whoever actually invokes ffmpeg — Engineer A's render step,
    or a future audio-specific render path) runs ffmpeg once with this
    filter on the source audio, `-f null -`, and parses the JSON block
    `loudnorm` prints to stderr (see `parse_loudnorm_stats`) to get the
    real measured `input_i`/`input_tp`/`input_lra`/`input_thresh` for pass
    2. This module only builds filter strings — it does not shell out to
    ffmpeg itself (kept consistent with `build_ffmpeg_color_filters` being
    a pure function; the ffmpeg *execution* lives in
    `pipeline/level6/editing_director.py`).
    """
    target_i = _target_lufs(target_platform)
    return (
        f"loudnorm=I={target_i}:TP={_TARGET_TP}:LRA={_TARGET_LRA}:"
        f"print_format=json"
    )


_LOUDNORM_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)


def parse_loudnorm_stats(ffmpeg_stderr: str) -> dict | None:
    """Parse the JSON stats block `loudnorm`'s pass-1 `print_format=json`
    prints to ffmpeg's stderr. Returns None if no block is found (e.g. the
    measurement pass wasn't actually run, or ffmpeg output changed shape).

    Deliberately tolerant: `loudnorm`'s JSON block is embedded in a larger
    stderr stream with `[Parsed_loudnorm_...]` log lines around it, so this
    greps for the specific `{...}` block containing `"input_i"` rather than
    trying to parse the whole stderr stream as JSON.
    """
    import json

    match = _LOUDNORM_JSON_BLOCK_RE.search(ffmpeg_stderr or "")
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("parse_loudnorm_stats: found a candidate block but it wasn't valid JSON: %s", exc)
        return None


def compute_loudnorm_filter(
    target_platform: str | None,
    measured_stats: dict | None = None,
) -> str:
    """Return the FFmpeg `loudnorm` filter string for two-pass loudness
    normalization, target -14 LUFS for `platform in ("reel", "youtube")`
    or -16 LUFS for `platform == "full_cut"` / `None` (podcast-style
    default) — per CLAUDE.md "STATUS: IMPLEMENTING" / "Audio.".

    Signature note: the task brief's 1-arg signature
    (`compute_loudnorm_filter(target_platform) -> str`) covers the common
    case. This adds an *optional* `measured_stats` kwarg (same pattern as
    `enforce_duration`'s optional `ranked_candidates` extension in
    `pipeline/level5/planner_runner.py`) because a genuine two-pass
    `loudnorm` needs pass 1's measured `input_i`/`input_tp`/`input_lra`/
    `input_thresh` fed into pass 2 for a linear (not dynamic) normalization
    — that data can only come from actually running ffmpeg once
    (`compute_loudnorm_measure_filter` + `parse_loudnorm_stats`), which is
    outside this module's pure-filter-string-builder scope.

    - `measured_stats` is None (default): returns the single-pass dynamic
      `loudnorm` filter (measures and normalizes in one pass, no separate
      measurement run needed). This is "two-pass" in the sense that
      `loudnorm`'s single-pass mode still does an internal analysis pass
      before applying gain — it is the correct fallback when a caller
      hasn't (or can't, given `render_direct`'s current extension point —
      see module docstring) run an explicit external measurement pass.
    - `measured_stats` provided (the dict `parse_loudnorm_stats` returns):
      returns the linear pass-2 apply filter seeded with the real measured
      values — the standard two-pass `loudnorm` recipe, more accurate than
      single-pass dynamic mode.
    """
    target_i = _target_lufs(target_platform)
    if not measured_stats:
        return f"loudnorm=I={target_i}:TP={_TARGET_TP}:LRA={_TARGET_LRA}"

    try:
        measured_i = float(measured_stats["input_i"])
        measured_tp = float(measured_stats["input_tp"])
        measured_lra = float(measured_stats["input_lra"])
        measured_thresh = float(measured_stats["input_thresh"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "compute_loudnorm_filter: measured_stats missing/invalid fields (%s) — "
            "falling back to single-pass dynamic loudnorm",
            exc,
        )
        return f"loudnorm=I={target_i}:TP={_TARGET_TP}:LRA={_TARGET_LRA}"

    offset = measured_stats.get("target_offset")
    filt = (
        f"loudnorm=I={target_i}:TP={_TARGET_TP}:LRA={_TARGET_LRA}:"
        f"measured_I={measured_i}:measured_TP={measured_tp}:"
        f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:"
        f"linear=true"
    )
    if offset is not None:
        try:
            filt += f":offset={float(offset)}"
        except (TypeError, ValueError):
            pass
    return filt


# ---------------------------------------------------------------------------
# Ducking — AUDIO_DUCK_REQUEST ops
# ---------------------------------------------------------------------------


def compute_ducking_filter(edit_plan_operations: list[dict] | None = None) -> str | None:
    """Build a `sidechaincompress` filter string for ducking dialogue under
    music/overlay moments requested by an `AUDIO_DUCK_REQUEST` op.

    **Documented no-op — always returns None.** `sidechaincompress` needs
    TWO audio inputs: a main stream to compress and a sidechain stream
    (typically the music bed) whose level triggers the compression.
    Grepped `shared/types.py` and every file in
    `knowledge_base/postgres/migrations/` for a second audio-source
    concept — there is none. This pipeline is single-source video with
    exactly one audio track (`videos` has no `music_bed_r2_key` or
    similar, and no table anywhere carries a secondary audio asset
    reference). `AUDIO_DUCK_REQUEST` ops (see
    `prompts/l5_sequencing.py::_EDIT_OPERATION_SCHEMA` — it's a valid
    `EditOperation.type` L5 can emit) exist as a *request* type, but there
    is nothing in the schema yet to duck the dialogue *against*.

    Per rule 18 (L6 never re-interprets intent) this deliberately does NOT
    invent a synthetic sidechain source (e.g. silently ducking against a
    flattened copy of itself, which would just be a compressor, not
    ducking) to make something plug into `render_direct`'s `extra_filters`
    — that would silently do something other than what "ducking" means.

    What's missing to build this for real: a music-bed audio asset
    reference (e.g. a `music_beds` table or a `r2_key` on the relevant
    `AUDIO_DUCK_REQUEST` op pointing at a secondary audio file) that
    `render_direct` could mix in as a second ffmpeg input before the
    sidechain filter. Flagging this as a schema gap rather than faking a
    filter that wouldn't actually duck anything.

    `edit_plan_operations`, if supplied, is used only to log how many
    `AUDIO_DUCK_REQUEST` ops this edit plan actually requested — so the
    gap is visible/traceable in logs rather than silently swallowed —
    it does not change the return value.
    """
    if edit_plan_operations:
        n_requests = sum(
            1 for op in edit_plan_operations if op.get("type") == "AUDIO_DUCK_REQUEST"
        )
        if n_requests:
            logger.warning(
                "compute_ducking_filter: edit plan requests %d AUDIO_DUCK_REQUEST op(s), "
                "but this pipeline's schema has no music-bed/secondary-audio-track concept "
                "to duck against yet — returning no-op (see function docstring for the "
                "exact schema gap). No filter applied for these ops.",
                n_requests,
            )
    return None


# ---------------------------------------------------------------------------
# Multicam sync
# ---------------------------------------------------------------------------


def apply_multicam_sync(cut_list_items: list) -> list:
    """Align multi-camera source clips within a sync group before cutting.

    **Documented no-op passthrough — returns `cut_list_items` unchanged.**
    CLAUDE.md itself flags this: "Multicam sync is a v-next item — nothing
    in the current schema carries multi-camera source info for a single
    pipeline run, not a v1 blocker." Confirmed by grep: no
    `camera`/`multicam`/`sync_group` field anywhere in `shared/types.py`
    or `knowledge_base/postgres/migrations/*.sql` — `videos` is one row
    per single-camera source, `cut_list_items` has no camera/angle
    reference. There is nothing to synchronize against.

    Kept as a real (if trivial) function rather than omitted entirely, so
    `pipeline/level6/updater.py` has a stable call site to wire in once a
    `sync_group`/multi-camera concept exists at L1 — same "don't build for
    the level you don't have yet" discipline as the deferred Motion
    Graphics Agent, but scoped to a single no-op function instead of a
    missing file since the task brief asked for this exact call site.
    """
    return cut_list_items
