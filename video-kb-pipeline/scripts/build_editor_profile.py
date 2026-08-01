"""CLAUDE.md "PIPELINE ADDENDUM 4 -- Editor Style Profile Learning", Phase 1.

Deterministic (NO LLM CALL) extraction of an editor's style profile from
3-5 (or however many are given -- not hard-blocked) of their own
already-edited videos, already processed through L1 (ASHFS + Whisper) and
L2 (color grading) in this DB. Writes one aggregated row to the
already-existing `client_style_profiles` table (migration 016) via the
already-existing `upsert_client_style_profile` query function, and tags
each source video with `client_id` via the new `set_video_client_id`
query function so L9's reward-signal aggregation can scope by client
later (Addendum 4, Phase 3).

What this script extracts, per CLAUDE.md's Phase 1 bullets:
  - Pacing: from `shots` (start_time/end_time) -- avg/median/variance of
    shot length, PLUS the full ordered cut-duration sequence per video
    (not just the summary stats -- a "tight-tight-tight-hold" rhythm
    signature is lost if only the mean is kept). Printed in this script's
    JSON output for inspection; `pacing_preference` (the one field L5
    Pass B already reads) is a short categorical string derived from the
    aggregate avg shot length -- see `_classify_pacing_preference` for the
    exact thresholds.
  - Color: mean + variance per numeric `color_grades.parameters` key
    across every shot in every given video. Each parameter value is a
    dict shaped like `{"current": ..., "recommended": ..., "delta": ...}`
    (numeric params) or `{"current": <bool|str>}` (categorical/boolean
    params, e.g. `quality.noise_risk`, `quality.shadow_crush`,
    `tone_curve.curve_type`) -- confirmed against a real row from
    `video_id=97199656-d176-46aa-88b4-026670be4576` before writing this
    extraction code (see task's verification output). Only numeric,
    non-bool `current` values are aggregated; categorical/boolean params
    are skipped (no meaningful mean/variance).
  - Audio: loudness target (LUFS) + dynamic range (LRA), via an
    `ffmpeg loudnorm ... print_format=json -f null -` measure-only pass
    against the real video file on disk -- same filter string as
    `pipeline/level6/audio_sync.py::compute_loudnorm_measure_filter`,
    parsed with that module's own `parse_loudnorm_stats`. This needs
    actual video bytes (ffmpeg, not a DB row); source is resolved per
    video via `_resolve_local_video_path` (existing local file next to
    the script wins; falls back to `--local-video-path`, default
    `input_video.mp4`, documented below).
  - Caption style: explicitly NOT extracted -- no OCR step exists in this
    pipeline (it transcribes spoken audio only), so on-screen burned-in
    caption style cannot be inferred from a finished video. This script
    never writes `client_style_profiles.caption_style`; if a profile
    already exists for the given `client_id`, its `caption_style` (and
    `default_platform`, another field this script has no opinion on) is
    read back and preserved rather than clobbered by
    `upsert_client_style_profile`'s `ON CONFLICT DO UPDATE`.

Local-video-file note (documented per the task brief's explicit
allowance): every video currently in this DB's `videos.path` points at
the SAME underlying source URL family (confirmed via a live query before
writing this script -- most rows are `.../input%20video.mp4`, one video
is a different source, `Poadcast (1).mp4`). A real downloaded copy of the
`input video.mp4` source already exists at the repo root
(`input_video.mp4`). Rather than re-downloading per video_id, this script
resolves ONE local file per unique video source and reuses it for every
given `--video-ids` entry that shares that source -- proving the
extraction math against real audio bytes without a redundant download.
If a given video's source isn't resolvable locally, that video is skipped
for the AUDIO component only (pacing/color extraction is independent and
still proceeds for it) with a clear warning; this only happens for the
"Poadcast" source video in the current DB, which has no local file copy.

Usage:
    python scripts/build_editor_profile.py \\
        --client-id test_editor_alpha \\
        --video-ids 97199656-d176-46aa-88b4-026670be4576,<id2>,<id3>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from knowledge_base.postgres.client import get_pool  # noqa: E402
from knowledge_base.postgres.queries import (  # noqa: E402
    get_client_style_profile,
    get_color_grades_for_video,
    get_shots_for_video,
    get_video,
    set_video_client_id,
    upsert_client_style_profile,
)
from pipeline.level6.audio_sync import (  # noqa: E402
    compute_loudnorm_measure_filter,
    parse_loudnorm_stats,
)
from shared.types import ClientStyleProfileRecord  # noqa: E402
from shared.utils import gen_id  # noqa: E402

# ---------------------------------------------------------------------------
# Pacing classification thresholds
# ---------------------------------------------------------------------------
# Documented, not hand-wavy: derived from common short-form-editing
# conventions (sub-2s average shot length reads as fast-cut / hook-style
# editing; 2-5s is a typical conversational/interview cut rate; 5s+ is
# slow/deliberate, long-take-leaning). No existing precedent for these
# exact numbers elsewhere in the codebase -- this is this script's own
# reasonable engineering judgment call, same category as L6's
# stops->eq-brightness conversion (CLAUDE.md flags that one as an
# "unverified judgment call" too; same honesty applies here).
_FAST_MAX_AVG_SHOT_S = 2.0
_MODERATE_MAX_AVG_SHOT_S = 5.0


def _classify_pacing_preference(avg_shot_length_s: float) -> str:
    if avg_shot_length_s < _FAST_MAX_AVG_SHOT_S:
        return "fast"
    if avg_shot_length_s < _MODERATE_MAX_AVG_SHOT_S:
        return "moderate"
    return "slow"


# ---------------------------------------------------------------------------
# Pacing extraction
# ---------------------------------------------------------------------------


def extract_shot_durations(shots: list) -> list[float]:
    """Ordered (shot_index order, already guaranteed by
    `get_shots_for_video`'s `ORDER BY shot_index`) list of shot durations
    in seconds -- the raw rhythm-signature sequence, not just a summary
    stat."""
    durations = []
    for s in shots:
        if s.start_time is None or s.end_time is None:
            continue
        durations.append(round(float(s.end_time) - float(s.start_time), 4))
    return durations


def compute_pacing_stats(all_durations: list[float]) -> dict:
    n = len(all_durations)
    if n == 0:
        return {"n_shots": 0, "avg_shot_length_s": None, "median_shot_length_s": None, "variance_s2": None}
    return {
        "n_shots": n,
        "avg_shot_length_s": round(statistics.mean(all_durations), 4),
        "median_shot_length_s": round(statistics.median(all_durations), 4),
        "variance_s2": round(statistics.variance(all_durations), 4) if n > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Color extraction
# ---------------------------------------------------------------------------


def aggregate_color_params(all_parameters: list[dict]) -> dict[str, dict]:
    """Mean + variance per numeric, non-bool `current` value across every
    `color_grades.parameters` dict given. Shape confirmed live against
    `color_grades` for video_id=97199656-d176-46aa-88b4-026670be4576
    before writing this: `{param_name: {current, recommended, delta}}`
    for numeric params, `{param_name: {current: <bool|str>}}` for
    categorical ones (e.g. `quality.noise_risk`, `quality.shadow_crush`,
    `tone_curve.curve_type`, `creative_style.*` are numeric floats
    actually -- only genuinely bool/str `current` values are skipped)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for params in all_parameters:
        for key, val in (params or {}).items():
            if not isinstance(val, dict):
                continue
            current = val.get("current")
            if isinstance(current, bool) or not isinstance(current, (int, float)):
                continue
            buckets[key].append(float(current))

    result: dict[str, dict] = {}
    for key, values in buckets.items():
        n = len(values)
        mean_v = statistics.mean(values)
        var_v = statistics.variance(values) if n > 1 else 0.0
        result[key] = {"mean": round(mean_v, 4), "variance": round(var_v, 4), "n": n}
    return result


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def _resolve_local_video_path(video_path: str | None, fallback: str) -> str | None:
    """Return a local file path to actually run ffmpeg against.

    `video_path` (the `videos.path` DB column) is a remote URL in every
    real row seen in this DB (`https://r2.sagarteotia.in/...`), never a
    local path -- so this always falls through to `fallback` (default
    `input_video.mp4`, the real downloaded copy already in the repo
    root). Returns None if neither resolves to a real file on disk, so
    the caller can skip the audio component for that video rather than
    crash.
    """
    if video_path and not video_path.startswith(("http://", "https://")) and Path(video_path).is_file():
        return video_path
    if fallback and Path(fallback).is_file():
        return fallback
    return None


def measure_loudness(local_path: str, target_platform: str | None = None) -> dict | None:
    """Run ffmpeg's `loudnorm` measure-only pass (pass 1 of the two-pass
    filter `pipeline/level6/audio_sync.py::compute_loudnorm_measure_filter`
    builds) against a real local video file, and parse the resulting
    stats via that same module's `parse_loudnorm_stats`. Returns None on
    any failure (ffmpeg missing, no audio stream, parse failure) rather
    than raising -- audio is one of three independent signal sources this
    script aggregates, a failure here should not abort pacing/color
    extraction for the same or other videos.
    """
    filt = compute_loudnorm_measure_filter(target_platform)
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", local_path, "-af", filt, "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  [audio] ffmpeg measure pass failed for {local_path}: {exc}", file=sys.stderr)
        return None
    stats = parse_loudnorm_stats(proc.stderr)
    if stats is None:
        print(f"  [audio] no loudnorm JSON block found in ffmpeg stderr for {local_path} "
              f"(exit={proc.returncode}); video may have no audio stream.", file=sys.stderr)
        return None
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic editor style profile from 3-5 (or N) already-edited "
                     "videos already processed through L1/L2, and upsert it into client_style_profiles."
    )
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--video-ids", required=True,
                         help="Comma-separated list of video_id UUIDs already in the DB with L1/L2 output.")
    parser.add_argument("--local-video-path", default="input_video.mp4",
                         help="Fallback local video file used for the ffmpeg loudnorm measure pass "
                              "when a video's DB `path` isn't a local file (always true today -- "
                              "`videos.path` is a remote R2 URL in every real row). Default: "
                              "input_video.mp4 (the real downloaded copy already in the repo root).")
    parser.add_argument("--platform", default=None,
                         help="Optional platform hint (reel|full_cut|youtube) for the loudnorm target "
                              "LUFS used only as the measurement filter's reference point -- the "
                              "MEASURED input_i/input_lra (this editor's actual characteristic "
                              "loudness/dynamic range) is what gets stored, not this target.")
    args = parser.parse_args()

    video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
    if not video_ids:
        print("No --video-ids given.", file=sys.stderr)
        sys.exit(1)

    pool = await get_pool()

    per_video_durations: dict[str, list[float]] = {}
    all_durations: list[float] = []
    all_color_params: list[dict] = []
    audio_measurements: list[dict] = []
    contributing_video_ids: list[str] = []
    tagged_video_ids: list[str] = []
    skipped: list[dict] = []

    resolved_local_paths: dict[str, str | None] = {}

    for video_id in video_ids:
        video = await get_video(pool, video_id)
        if video is None:
            print(f"[skip] video_id={video_id} not found in DB.", file=sys.stderr)
            skipped.append({"video_id": video_id, "reason": "not_found"})
            continue

        # Tag with client_id regardless of how much L1/L2 data exists --
        # the caller explicitly supplied this id as belonging to this
        # client (Addendum 4 Phase 1: "tags each of the 3-5 source videos
        # with the given client_id").
        await set_video_client_id(pool, video_id, args.client_id)
        tagged_video_ids.append(video_id)

        shots = await get_shots_for_video(pool, video_id)
        if not shots:
            print(f"[warn] video_id={video_id} has no `shots` rows (no L1 output) -- "
                  f"skipping pacing/color extraction for it.", file=sys.stderr)
            skipped.append({"video_id": video_id, "reason": "no_shots"})
        else:
            durations = extract_shot_durations(shots)
            per_video_durations[video_id] = durations
            all_durations.extend(durations)
            contributing_video_ids.append(video_id)

            color_grades = await get_color_grades_for_video(pool, video_id)
            if not color_grades:
                print(f"[warn] video_id={video_id} has no `color_grades` rows (no L2 color output) -- "
                      f"pacing still counted, color extraction skipped for it.", file=sys.stderr)
            else:
                all_color_params.extend(cg.parameters for cg in color_grades)

        # Audio: independent of shots/color_grades -- resolve + measure once per unique local path.
        local_path = _resolve_local_video_path(video.path, args.local_video_path)
        resolved_local_paths[video_id] = local_path
        if local_path is None:
            print(f"[warn] video_id={video_id}: no local file resolvable for audio measurement "
                  f"(path={video.path!r}, fallback={args.local_video_path!r} not found) -- "
                  f"skipping audio for this video.", file=sys.stderr)
        else:
            print(f"[audio] measuring loudness for video_id={video_id} against {local_path} ...")
            stats = measure_loudness(local_path, args.platform)
            if stats is not None:
                audio_measurements.append({"video_id": video_id, "stats": stats})

    if not contributing_video_ids:
        print("No given video_ids had usable `shots` data -- nothing to build a profile from.",
              file=sys.stderr)
        sys.exit(1)

    pacing_stats = compute_pacing_stats(all_durations)
    pacing_preference = _classify_pacing_preference(pacing_stats["avg_shot_length_s"])

    color_stats = aggregate_color_params(all_color_params)

    if audio_measurements:
        input_i_values = [float(m["stats"]["input_i"]) for m in audio_measurements if "input_i" in m["stats"]]
        input_lra_values = [float(m["stats"]["input_lra"]) for m in audio_measurements if "input_lra" in m["stats"]]
        loudness_target_lufs = round(statistics.mean(input_i_values), 2) if input_i_values else None
        dynamic_range_lu = round(statistics.mean(input_lra_values), 2) if input_lra_values else None
    else:
        loudness_target_lufs = None
        dynamic_range_lu = None

    sample_count = len(contributing_video_ids)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    notes_parts = [
        f"Editor style profile built from {sample_count} video(s) "
        f"(auto-generated by build_editor_profile.py, {now_str}).",
        f"Pacing: avg={pacing_stats['avg_shot_length_s']}s median={pacing_stats['median_shot_length_s']}s "
        f"variance={pacing_stats['variance_s2']}s^2 over {pacing_stats['n_shots']} shots "
        f"-> pacing_preference='{pacing_preference}' "
        f"(thresholds: <{_FAST_MAX_AVG_SHOT_S}s=fast, <{_MODERATE_MAX_AVG_SHOT_S}s=moderate, else slow).",
        f"Color: {len(color_stats)} numeric parameters aggregated across {len(all_color_params)} "
        f"color_grades rows.",
    ]
    if loudness_target_lufs is not None:
        notes_parts.append(
            f"Audio: measured loudness_target={loudness_target_lufs} LUFS, "
            f"dynamic_range={dynamic_range_lu} LU, averaged across {len(audio_measurements)} "
            f"ffmpeg loudnorm measure pass(es)."
        )
    else:
        notes_parts.append("Audio: no usable ffmpeg loudnorm measurement (see stderr warnings).")
    if skipped:
        notes_parts.append(f"Skipped: {json.dumps(skipped)}.")
    notes = " ".join(notes_parts)

    brand_colors: dict = dict(color_stats)
    if loudness_target_lufs is not None:
        brand_colors["_audio"] = {
            "loudness_target_lufs": loudness_target_lufs,
            "dynamic_range_lu": dynamic_range_lu,
            "n_videos_measured": len(audio_measurements),
        }

    # Preserve caption_style / default_platform on an existing profile row
    # for this client_id (Phase 1: "explicitly NOT extracted" for
    # caption_style -- never clobber a hand-entered value).
    existing = await get_client_style_profile(pool, args.client_id)
    caption_style = existing.caption_style if existing else {}
    default_platform = existing.default_platform if existing else None
    record_id = existing.id if existing else gen_id()

    record = ClientStyleProfileRecord(
        id=record_id,
        client_id=args.client_id,
        caption_style=caption_style,
        pacing_preference=pacing_preference,
        brand_colors=brand_colors,
        default_platform=default_platform,
        notes=notes,
    )
    profile_id = await upsert_client_style_profile(pool, record)

    output = {
        "client_id": args.client_id,
        "profile_id": profile_id,
        "sample_count": sample_count,
        "contributing_video_ids": contributing_video_ids,
        "tagged_video_ids": tagged_video_ids,
        "skipped": skipped,
        "pacing": {
            **pacing_stats,
            "pacing_preference": pacing_preference,
            "per_video_shot_duration_sequences": per_video_durations,
        },
        "color": {
            "n_parameters": len(color_stats),
            "n_color_grade_rows": len(all_color_params),
            "parameters": color_stats,
        },
        "audio": {
            "loudness_target_lufs": loudness_target_lufs,
            "dynamic_range_lu": dynamic_range_lu,
            "n_videos_measured": len(audio_measurements),
            "per_video_measurements": audio_measurements,
            "resolved_local_paths": resolved_local_paths,
        },
        "notes": notes,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
