"""Transcript-to-chunk aligner for the Level 1 video understanding pipeline.

Assigns each :class:`~shared.types.TranscriptSegment` to the
:class:`~shared.types.ChunkRecord` whose time window it overlaps most, and
provides a helper to retrieve the transcript text for a given
:class:`~shared.types.ShotRecord`.
"""
from __future__ import annotations

import logging

from shared.types import ChunkRecord, ShotRecord, TranscriptSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _overlap(
    seg_start: float,
    seg_end: float,
    chunk_start: float,
    chunk_end: float,
) -> float:
    """Return the overlap duration (in seconds) between two time intervals.

    Both intervals are treated as closed: ``[seg_start, seg_end]`` and
    ``[chunk_start, chunk_end]``.  Returns ``0.0`` when the intervals do not
    overlap.
    """
    lo = max(seg_start, chunk_start)
    hi = min(seg_end, chunk_end)
    return max(0.0, hi - lo)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_transcript_to_chunks(
    segments: list[TranscriptSegment],
    chunks: list[ChunkRecord],
) -> list[TranscriptSegment]:
    """Assign each transcript segment to the chunk it overlaps most.

    For every :class:`~shared.types.TranscriptSegment`, the function finds the
    :class:`~shared.types.ChunkRecord` with the maximum temporal overlap
    (measured in seconds) and sets ``segment.chunk_id`` to that chunk's ``id``.

    Segments that do not overlap any chunk at all retain ``chunk_id = None``.

    The function mutates the ``chunk_id`` attribute on each segment in-place
    and also returns the updated list for convenience.

    Parameters
    ----------
    segments:
        All transcript segments for a video, as returned by
        :func:`pipeline.level1.whisper_runner.run_whisper`.
    chunks:
        All chunk records for the same video, as returned by
        :func:`pipeline.level1.ashfs_runner.parse_ashfs_manifest`.

    Returns
    -------
    list[TranscriptSegment]
        The same list passed in, with ``chunk_id`` fields updated.
    """
    if not chunks:
        logger.warning("align_transcript_to_chunks: no chunks provided — all chunk_ids remain None.")
        return segments

    # Pre-compute chunk time boundaries once.  Chunks whose start_time or
    # end_time is None are skipped (they cannot match any segment).
    valid_chunks: list[tuple[ChunkRecord, float, float]] = []
    for chunk in chunks:
        if chunk.start_time is not None and chunk.end_time is not None:
            valid_chunks.append((chunk, float(chunk.start_time), float(chunk.end_time)))

    if not valid_chunks:
        logger.warning(
            "align_transcript_to_chunks: all %d chunks lack time boundaries — "
            "no segments will be assigned.",
            len(chunks),
        )
        return segments

    unassigned = 0
    for seg in segments:
        if seg.start_time is None or seg.end_time is None:
            unassigned += 1
            continue
        seg_start = seg.start_time
        seg_end = seg.end_time

        best_chunk_id: str | None = None
        best_overlap: float = 0.0

        for chunk, c_start, c_end in valid_chunks:
            ov = _overlap(seg_start, seg_end, c_start, c_end)
            if ov > best_overlap:
                best_overlap = ov
                best_chunk_id = chunk.id

        seg.chunk_id = best_chunk_id
        if best_chunk_id is None:
            unassigned += 1

    if unassigned:
        logger.debug(
            "align_transcript_to_chunks: %d/%d segments could not be assigned to any chunk.",
            unassigned,
            len(segments),
        )

    logger.info(
        "align_transcript_to_chunks: assigned %d/%d segments to chunks.",
        len(segments) - unassigned,
        len(segments),
    )
    return segments


def get_transcript_for_shot(
    segments: list[TranscriptSegment],
    shot: ShotRecord,
    window_before: float = 1.0,
    window_after: float = 1.0,
) -> str:
    """Return joined transcript text covering the time range of *shot*.

    Collects all segments whose time range overlaps the extended window
    ``[shot.start_time - window_before, shot.end_time + window_after]`` and
    joins their ``text`` fields with a single space.

    Parameters
    ----------
    segments:
        All transcript segments for the video (or at least the relevant chunk).
        The list need not be pre-filtered.
    shot:
        The shot whose transcript coverage is requested.
    window_before:
        Seconds of padding to add before ``shot.start_time`` (default 1.0).
    window_after:
        Seconds of padding to add after ``shot.end_time`` (default 1.0).

    Returns
    -------
    str
        Space-joined text of all overlapping segments.  Returns an empty string
        when the shot has no time boundaries or no segments overlap the window.
    """
    if shot.start_time is None or shot.end_time is None:
        logger.debug(
            "get_transcript_for_shot: shot_id=%s has no time boundaries — returning empty string.",
            shot.id,
        )
        return ""

    window_start = shot.start_time - window_before
    window_end = shot.end_time + window_after

    matching: list[str] = [
        seg.text
        for seg in segments
        if seg.start_time is not None and seg.end_time is not None
        and _overlap(seg.start_time, seg.end_time, window_start, window_end) > 0.0
    ]

    return " ".join(matching)
