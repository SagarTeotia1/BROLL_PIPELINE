"""Pipeline orchestrator.

Pure Python pipeline coordinator — NOT a Modal function.  Drives the full
L1 → L2 → L3 pipeline locally (or can be imported by Modal functions).

Each level is wrapped in try/except: failures are recorded in the DB and
reflected in PipelineResult, but never crash the whole run — partial results
are always more valuable than nothing.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    video_id: str
    level1_ok: bool = False
    level2_ok: bool = False
    level3_ok: bool = False
    error: str | None = None
    keyframe_count: int = 0
    fact_count: int = 0
    person_count: int = 0


# ---------------------------------------------------------------------------
# Sync helper: ASHFS + R2 upload
# ---------------------------------------------------------------------------


def _run_ashfs_and_save(
    video_path: str,
    video_id: str,
    tmp_dir: str,
    skip_r2: bool,
) -> tuple:
    """Run ASHFS and upload keyframe PNGs to R2.  Returns (chunks, shots, keyframes).

    DB writes are intentionally NOT performed here — this function runs in a
    thread-pool worker via ``asyncio.to_thread`` and must not create a new event
    loop (which would conflict with asyncpg's pool-singleton that is bound to the
    main event loop).  The caller writes the returned records to the DB after
    ``await asyncio.gather(...)`` completes.
    """
    from pipeline.level1.ashfs_runner import run_ashfs, parse_ashfs_manifest
    from knowledge_base.r2.uploader import upload_frame as r2_upload_frame
    from shared.utils import frame_to_png_bytes

    logger.info("[L1-ASHFS] Starting ASHFS for video_id=%s", video_id)
    manifest = run_ashfs(video_path, video_id, tmp_dir)
    chunks, shots, keyframes = parse_ashfs_manifest(manifest, video_id)
    logger.info(
        "[L1-ASHFS] ASHFS complete: %d chunks, %d shots, %d keyframes",
        len(chunks), len(shots), len(keyframes),
    )

    # Upload keyframe PNGs to R2 in parallel (8 concurrent upload threads)
    if not skip_r2 and keyframes:
        cap = cv2.VideoCapture(video_path)
        frame_payloads: list[tuple] = []  # (kf, png_bytes)
        for kf in keyframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, kf.frame_index)
            ret, frame = cap.read()
            if ret:
                frame_payloads.append((kf, frame_to_png_bytes(frame)))
        cap.release()

        _r2_ok = True
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_kf = {
                executor.submit(r2_upload_frame, png_bytes, video_id, kf.frame_index): kf
                for kf, png_bytes in frame_payloads
                if _r2_ok
            }
            for future in as_completed(future_to_kf):
                kf = future_to_kf[future]
                try:
                    kf.r2_key = future.result()
                except Exception as exc:
                    exc_str = str(exc)
                    if any(k in exc_str for k in ("AccessDenied", "NoSuchBucket", "InvalidAccessKeyId")):
                        logger.warning(
                            "[L1-ASHFS] R2 permission error — aborting remaining uploads. "
                            "Error: %s", exc_str
                        )
                        _r2_ok = False
                    else:
                        logger.warning(
                            "[L1-ASHFS] R2 upload failed for frame %d: %s", kf.frame_index, exc
                        )

    logger.info("[L1-ASHFS] R2 uploads complete for video_id=%s", video_id)
    return chunks, shots, keyframes


# ---------------------------------------------------------------------------
# Sync helper: Whisper
# ---------------------------------------------------------------------------


def _run_whisper_and_save(
    video_path: str,
    video_id: str,
    tmp_dir: str,
) -> list:
    """Transcribe video audio and return TranscriptSegments.

    DB writes are intentionally NOT performed here — this function runs in a
    thread-pool worker.  ``bulk_insert_transcript_segments`` is called by the
    orchestrator after alignment assigns chunk_ids, running in the async context
    with the main event loop's connection pool.
    """
    from pipeline.level1.whisper_runner import run_whisper

    logger.info("[L1-Whisper] Starting transcription for video_id=%s", video_id)
    segments = run_whisper(video_path, video_id, tmp_dir)
    logger.info(
        "[L1-Whisper] Transcription complete: %d segments for video_id=%s",
        len(segments), video_id,
    )
    return segments


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


async def run_pipeline(
    video_path: str,
    video_id: str | None = None,
    cast_db_path: str | None = None,
    skip_r2_upload: bool = False,
) -> PipelineResult:
    """Run the full L1 → L2 → L3 pipeline.

    Args:
        video_path:      Local path to the video file OR an R2 key starting
                         with "videos/".  When an R2 key is provided the file
                         is downloaded to a temporary path before processing.
        video_id:        Stable identifier for the video.  A new nanoid is
                         generated when None is passed.
        cast_db_path:    Path to a cast SQLite DB for face recognition.
                         Pass None to treat every person as Unknown.
        skip_r2_upload:  When True, keyframe PNGs are NOT uploaded to R2.
                         Useful for local development without R2 credentials.

    Returns:
        PipelineResult with per-level success flags and aggregate counts.
    """
    from shared.utils import gen_id
    from knowledge_base.postgres.client import get_pool
    from knowledge_base.postgres.schema import run_migrations
    from knowledge_base.postgres.queries import (
        upsert_video,
        upsert_job,
        update_job_status,
        bulk_insert_transcript_segments,
        get_shots_for_video,
        get_chunks_for_video,
        get_keyframes_for_video,
        get_persons_for_video,
        get_transcript_segments_for_video,
        get_face_appearances_for_video_sampled,
        get_existing_frame_analyses,
    )
    from shared.types import VideoMeta, ProcessingJob, LevelStatus

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    pool = await get_pool()
    await run_migrations(pool)

    if video_id is None:
        video_id = gen_id()

    logger.info("Pipeline starting for video_id=%s path=%s", video_id, video_path)

    # Download from R2 if video_path is an R2 key
    tmp_video_file = None
    local_video_path = video_path
    if video_path.startswith("videos/"):
        from knowledge_base.r2.uploader import download_bytes
        logger.info("Downloading video from R2 key: %s", video_path)
        video_bytes = download_bytes(video_path)
        tmp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_video_file.write(video_bytes)
        tmp_video_file.close()
        local_video_path = tmp_video_file.name
        logger.info("R2 download complete → %s (%d bytes)", local_video_path, len(video_bytes))

    # Probe video metadata
    import os as _os
    if not _os.path.exists(local_video_path):
        raise FileNotFoundError(
            f"Video file not found: {local_video_path!r}. "
            "Provide a valid local path or an R2 key starting with 'videos/'."
        )
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open video: {local_video_path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / fps if fps > 0 else 0.0
    cap.release()

    logger.info(
        "Video probed: fps=%.2f frames=%d size=%dx%d duration=%.1fs",
        fps, frame_count, width, height, duration_s,
    )

    await upsert_video(
        pool,
        VideoMeta(
            id=video_id,
            path=video_path,
            r2_key=video_path if video_path.startswith("videos/") else None,
            duration_s=duration_s,
            fps=fps,
            width=width,
            height=height,
        ),
    )

    result = PipelineResult(video_id=video_id)

    # ------------------------------------------------------------------
    # Level 1: ASHFS (keyframe sampling) + Whisper (transcription)
    # Runs both in parallel via asyncio.to_thread.
    # ------------------------------------------------------------------
    await upsert_job(
        pool,
        ProcessingJob(
            id=gen_id(),
            video_id=video_id,
            level=1,
            status=LevelStatus.RUNNING,
            error_msg=None,
            meta={},
        ),
    )

    chunks, shots, keyframes, segments = [], [], [], []

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info("[L1] Launching ASHFS + Whisper in parallel")

            ashfs_task = asyncio.create_task(
                asyncio.to_thread(
                    _run_ashfs_and_save,
                    local_video_path,
                    video_id,
                    tmp_dir,
                    skip_r2_upload,
                )
            )
            whisper_task = asyncio.create_task(
                asyncio.to_thread(
                    _run_whisper_and_save,
                    local_video_path,
                    video_id,
                    tmp_dir,
                )
            )

            l1_results = await asyncio.gather(
                ashfs_task, whisper_task, return_exceptions=True
            )

        ashfs_result = l1_results[0]
        whisper_result = l1_results[1]

        if isinstance(ashfs_result, Exception):
            raise RuntimeError(f"ASHFS failed: {ashfs_result}") from ashfs_result
        if isinstance(whisper_result, Exception):
            logger.error("[L1] Whisper failed (non-fatal): %s", whisper_result)
            segments = []
        else:
            segments = whisper_result

        chunks, shots, keyframes = ashfs_result
        result.keyframe_count = len(keyframes)

        # Write L1 records to DB in the async context (safe: uses the main
        # event loop's asyncpg pool, no nested-loop or cross-loop issues).
        from knowledge_base.postgres.queries import (  # noqa: F811
            bulk_insert_chunks,
            bulk_insert_shots,
            bulk_insert_keyframes,
        )

        await bulk_insert_chunks(pool, chunks)
        await bulk_insert_shots(pool, shots)
        await bulk_insert_keyframes(pool, keyframes)
        logger.info("[L1] DB writes complete: chunks=%d shots=%d keyframes=%d",
                    len(chunks), len(shots), len(keyframes))

        # Align transcript to chunks, then write to DB
        from pipeline.level1.aligner import align_transcript_to_chunks

        segments = align_transcript_to_chunks(segments, chunks)
        if segments:
            await bulk_insert_transcript_segments(pool, segments)
            logger.info("[L1] Inserted %d transcript segments", len(segments))

        await update_job_status(pool, video_id, 1, LevelStatus.DONE)
        result.level1_ok = True
        logger.info(
            "[L1] Complete: chunks=%d shots=%d keyframes=%d segments=%d",
            len(chunks), len(shots), len(keyframes), len(segments),
        )

    except Exception as exc:
        err_str = str(exc)
        logger.error("[L1] FAILED: %s", err_str, exc_info=True)
        await update_job_status(pool, video_id, 1, LevelStatus.FAILED, err_str)
        result.error = err_str
        # Cannot continue to L2/L3 without L1 outputs
        _cleanup_tmp(tmp_video_file)
        return result

    # ------------------------------------------------------------------
    # Level 2: Face analysis (ArcFace + HSEmotion) + Color grading (OpenCV)
    # Sequential within level, run after L1 completes.
    # ------------------------------------------------------------------
    await upsert_job(
        pool,
        ProcessingJob(
            id=gen_id(),
            video_id=video_id,
            level=2,
            status=LevelStatus.RUNNING,
            error_msg=None,
            meta={},
        ),
    )

    try:
        # Load shots from DB (canonical source after L1)
        shots = await get_shots_for_video(pool, video_id)

        logger.info("[L2] Starting face analysis + color grading in parallel")
        face_task = asyncio.create_task(
            asyncio.to_thread(_face_analysis_wrapper, local_video_path, video_id, cast_db_path)
        )
        color_task = asyncio.create_task(
            asyncio.to_thread(_color_grading_wrapper, local_video_path, video_id, shots)
        )
        face_result, color_grades = await asyncio.gather(face_task, color_task)

        from pipeline.level2.updater import write_level2_to_kb

        await write_level2_to_kb(
            pool,
            video_id,
            face_result["persons"],
            face_result["appearances"],
            face_result["timeline_events"],
            color_grades,
        )

        result.person_count = len(face_result["persons"])
        await update_job_status(pool, video_id, 2, LevelStatus.DONE)
        result.level2_ok = True
        logger.info(
            "[L2] Complete: persons=%d appearances=%d timeline_events=%d color_grades=%d",
            len(face_result["persons"]),
            len(face_result["appearances"]),
            len(face_result["timeline_events"]),
            len(color_grades),
        )

    except Exception as exc:
        err_str = str(exc)
        logger.error("[L2] FAILED (non-fatal, continuing to L3): %s", err_str, exc_info=True)
        await update_job_status(pool, video_id, 2, LevelStatus.FAILED, err_str)
        # L2 failure is non-fatal — L3 proceeds with empty person/appearance data

    # ------------------------------------------------------------------
    # Level 3: Qwen3-VL 8B via vLLM → knowledge graph → PostgreSQL + Pinecone
    # ------------------------------------------------------------------
    await upsert_job(
        pool,
        ProcessingJob(
            id=gen_id(),
            video_id=video_id,
            level=3,
            status=LevelStatus.RUNNING,
            error_msg=None,
            meta={},
        ),
    )

    try:
        logger.info("[L3] Loading KB data for context building")
        keyframes = await get_keyframes_for_video(pool, video_id)
        shots = await get_shots_for_video(pool, video_id)
        transcript_segments = await get_transcript_segments_for_video(pool, video_id)
        persons = await get_persons_for_video(pool, video_id)
        appearances = await get_face_appearances_for_video_sampled(pool, video_id)

        logger.info(
            "[L3] Context data loaded: keyframes=%d shots=%d segments=%d persons=%d appearances=%d",
            len(keyframes), len(shots), len(transcript_segments),
            len(persons), len(appearances),
        )

        from pipeline.level3.context_builder import build_all_frame_contexts
        from pipeline.level3.qwen_runner import run_qwen_analysis
        from pipeline.level3.kb_finalizer import finalize_level3_kb
        from models.qwen_vllm import QwenVLLM

        contexts = build_all_frame_contexts(
            keyframes, transcript_segments, appearances, persons
        )
        logger.info("[L3] Built %d frame contexts", len(contexts))

        # Skip frames already analysed (idempotent re-runs)
        keyframe_ids = [ctx.keyframe.id for ctx in contexts]
        existing = await get_existing_frame_analyses(pool, keyframe_ids)
        if existing:
            logger.info("[L3] Skipping %d already-analysed frames", len(existing))
            contexts = [ctx for ctx in contexts if ctx.keyframe.id not in existing]

        qwen = QwenVLLM.get()
        qwen.load()

        frame_results = await run_qwen_analysis(contexts)
        # Merge in existing results for full finalize
        for kf_id, output in existing.items():
            kf = next((ctx.keyframe for ctx in contexts if ctx.keyframe.id == kf_id), None)
            if kf is None:
                kf = next((kf for kf in keyframes if kf.id == kf_id), None)
            if kf is not None:
                frame_results.append((kf, output))
        logger.info("[L3] Qwen analysis complete: %d results (+ %d cached)", len(frame_results), len(existing))

        await finalize_level3_kb(pool, video_id, frame_results, persons, shots)

        valid_count = sum(1 for _, out in frame_results if out is not None)
        result.fact_count = valid_count  # approximate: facts ≈ valid frames * ~6

        await update_job_status(pool, video_id, 3, LevelStatus.DONE)
        result.level3_ok = True
        logger.info("[L3] Complete: %d/%d frames analysed", valid_count, len(frame_results))

    except Exception as exc:
        err_str = str(exc)
        logger.error("[L3] FAILED: %s", err_str, exc_info=True)
        await update_job_status(pool, video_id, 3, LevelStatus.FAILED, err_str)
        result.error = err_str

    _cleanup_tmp(tmp_video_file)

    logger.info(
        "Pipeline complete for video_id=%s — L1=%s L2=%s L3=%s",
        video_id, result.level1_ok, result.level2_ok, result.level3_ok,
    )
    return result


# ---------------------------------------------------------------------------
# Private wrappers called via asyncio.to_thread
# ---------------------------------------------------------------------------


def _face_analysis_wrapper(
    video_path: str,
    video_id: str,
    cast_db_path: str | None,
) -> dict:
    """Thin synchronous wrapper around run_face_analysis for to_thread usage."""
    from pipeline.level2.face_runner import run_face_analysis

    return run_face_analysis(video_path, video_id, cast_db_path)


def _color_grading_wrapper(
    video_path: str,
    video_id: str,
    shots: list,
) -> list:
    """Thin synchronous wrapper around run_color_grading for to_thread usage."""
    from pipeline.level2.color_runner import run_color_grading

    return run_color_grading(video_path, video_id, shots)


def _cleanup_tmp(tmp_file) -> None:
    """Remove a temporary file created by NamedTemporaryFile(delete=False)."""
    if tmp_file is None:
        return
    import os

    try:
        os.unlink(tmp_file.name)
        logger.debug("Removed tmp file: %s", tmp_file.name)
    except OSError as exc:
        logger.warning("Could not remove tmp file %s: %s", tmp_file.name, exc)
