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
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import asyncpg

from knowledge_base.postgres.queries import (
    bulk_insert_cut_list_items,
    get_cut_list_items_for_edit_plan,
    get_edit_plan,
    get_transcript_segments_for_video,
    get_video,
)
from shared.types import CutListItemRecord, EditOperationItem, VideoMeta
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
        gap_size = gap_end - gap_start
        if gap_size > best_gap_size:
            best_gap_size = gap_size
            best_mid = (gap_start + gap_end) / 2.0

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


async def render_direct(
    pool: asyncpg.Pool,
    edit_plan_id: str,
    source_video_path: str,
    output_path: str,
    extra_filters: dict[str, str] | None = None,
    final_audio_filter: str | None = None,
) -> str:
    """Render `cut_list_items` for *edit_plan_id* directly to *output_path*
    with a single ffmpeg invocation.

    Two internal paths, chosen automatically:

    1. **Fast path** (`-c copy`): only taken when every cut's `source_start`
       is keyframe-aligned (checked per-clip via ffprobe,
       `_is_keyframe_aligned`) *and* no `extra_filters` were supplied. Uses
       the ffmpeg concat demuxer (`-f concat -c copy`) — lossless, no
       re-encode, one ffmpeg invocation, no intermediate files besides the
       tiny concat list.
    2. **filter_complex path**: taken whenever any cut needs a mid-GOP
       frame-accurate boundary or has per-clip filters. Single input,
       per-clip `trim`/`atrim` + `setpts`/`asetpts`, then `concat`
       — one ffmpeg invocation, no intermediate files at all.

    `extra_filters` is the extension point for per-clip VIDEO adjustments:
    `{cut_list_item.id: "<ffmpeg filter chain string>"}`, applied to that
    clip's trimmed video stream before the final concat (e.g. Engineer B's
    `sequence_color_adjustments` translated into `eq=...,colorbalance=...`
    via `build_ffmpeg_color_filters`). It only ever wraps the `v{idx}`
    label — there is no per-clip audio hook, because the one audio
    operation this pipeline has (loudnorm normalization, from
    `audio_sync.py`) is a SEQUENCE-level operation by nature — it
    normalizes across the whole assembled track, not per-clip — so it
    belongs after concat, not per-clip. Use `final_audio_filter` for that
    instead (see below).

    `final_audio_filter` is a single ffmpeg filter chain string applied to
    the `[outa]` label AFTER concat (e.g. `audio_sync.compute_loudnorm_filter(...)`'s
    result) — sequence-level audio, not per-clip.

    This function performs cut-only assembly plus whatever `extra_filters`
    (video, per-clip) and `final_audio_filter` (audio, sequence-level) the
    caller supplies — nothing is applied unless explicitly passed in.
    """
    items = await get_cut_list_items_for_edit_plan(pool, edit_plan_id)
    if not items:
        raise ValueError(f"No cut_list_items found for edit_plan_id={edit_plan_id}")
    extra_filters = extra_filters or {}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_aligned = all(
        _is_keyframe_aligned(source_video_path, item.source_start) for item in items
    )

    if all_aligned and not extra_filters and not final_audio_filter:
        logger.info(
            "render_direct: all %d cuts keyframe-aligned, no extra_filters — "
            "using stream-copy concat-demuxer fast path.",
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
        "render_direct: %d cuts, at least one mid-GOP or filtered — "
        "using single filter_complex re-encode pass.",
        len(items),
    )
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for idx, item in enumerate(items):
        v_label = f"v{idx}"
        a_label = f"a{idx}"
        filter_parts.append(
            f"[0:v]trim=start={item.source_start}:end={item.source_end},"
            f"setpts=PTS-STARTPTS[{v_label}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={item.source_start}:end={item.source_end},"
            f"asetpts=PTS-STARTPTS[{a_label}]"
        )
        extra = extra_filters.get(item.id)
        if extra:
            filtered_v_label = f"{v_label}f"
            filter_parts.append(f"[{v_label}]{extra}[{filtered_v_label}]")
            v_label = filtered_v_label
        concat_inputs.append(f"[{v_label}][{a_label}]")

    n = len(items)
    filter_parts.append(f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]")

    # final_audio_filter is a SEQUENCE-LEVEL filter (e.g. audio_sync.py's
    # loudnorm) — it applies to the whole assembled audio track after
    # concat, not per-clip, so it hooks onto [outa] here rather than into
    # extra_filters (which is per-clip, pre-concat, video-label-only —
    # see the docstring above; audio never had a per-clip hook and didn't
    # need one, since loudnorm/ducking are sequence-level operations).
    out_audio_label = "outa"
    if final_audio_filter:
        out_audio_label = "outa_final"
        filter_parts.append(f"[outa]{final_audio_filter}[{out_audio_label}]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        "-i", source_video_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", f"[{out_audio_label}]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    _run_ffmpeg(cmd)
    return output_path
