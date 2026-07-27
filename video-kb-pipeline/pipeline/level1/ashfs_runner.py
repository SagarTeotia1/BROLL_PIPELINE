"""ASHFS pipeline wrapper for the Level 1 video understanding pipeline.

Provides :func:`run_ashfs` which invokes the ASH-FS sampler pipeline and
returns a normalised manifest dict, and :func:`parse_ashfs_manifest` which
converts that manifest into structured :class:`~shared.types.ChunkRecord`,
:class:`~shared.types.ShotRecord`, and :class:`~shared.types.KeyframeRecord`
instances.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from shared.types import ChunkRecord, KeyframeRecord, ShotRecord
from shared.utils import gen_id

logger = logging.getLogger(__name__)

# Number of shots to group into a single chunk when the manifest does not
# provide an explicit grouping structure.
_DEFAULT_SHOTS_PER_CHUNK = 5  # matches ASHFS config.yaml hierarchy.chunk_size_shots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ashfs(video_path: str, video_id: str, output_dir: str) -> dict:
    """Run the ASH-FS keyframe sampling pipeline on *video_path*.

    Adds the sibling ``ashfs`` directory to ``sys.path`` at call time (not at
    module import time) so that the rest of the pipeline can import it without
    polluting the module namespace prematurely.

    Parameters
    ----------
    video_path:
        Absolute path to the video file to process.
    video_id:
        Caller-assigned identifier for this video (used for logging only;
        the manifest itself does not carry it).
    output_dir:
        Directory where any ASHFS output artefacts should be written.  The
        directory is created if it does not already exist.

    Returns
    -------
    dict
        The raw manifest dict produced by :class:`ashfs.sampler_pipeline.SamplerPipeline`.
        Schema::

            {
                "video_path": str,
                "total_shots": int,
                "total_candidate_frames": int,
                "total_selected_frames": int,
                "frames_per_shot_distribution": list[int],
                "budget_stats": dict,
                "compression_stats": dict,
                "novelty_stats": dict,
                "selected_frame_indices": list[int],
                "frame_roles": list[str],
                "shot_boundaries": list[{
                    "start_frame": int,
                    "end_frame": int,
                    "duration_s": float,
                    "fps": float,
                }],
                "timing": dict[str, float],
                # optional keys present on fallback:
                "fallback": bool,
                "fallback_error": str,
            }

    Raises
    ------
    RuntimeError
        When ASHFS fails in a way that cannot be recovered by its internal
        fallback mechanism (e.g. config file missing, import error).
    """
    import sys

    # Resolve the root of "New folder (3)" — four levels up from this file:
    #   this file  → pipeline/level1/ashfs_runner.py
    #   up 1       → pipeline/level1/
    #   up 2       → pipeline/
    #   up 3       → video-kb-pipeline/   (project root)
    #   up 4       → New folder (3)/      (sibling root)
    ROOT = Path(__file__).parent.parent.parent.parent

    # Modal containers mount the sibling at /app_modules/ashfs; local dev uses
    # the relative 4-levels-up path.  Try Modal path first.
    _modal_ashfs = Path("/app_modules/ashfs")
    ashfs_root = _modal_ashfs if _modal_ashfs.exists() else ROOT / "ashfs"
    config_path = ashfs_root / "config.yaml"

    # Validate prerequisites before touching sys.path.
    if not ashfs_root.exists():
        raise RuntimeError(
            f"ASHFS root directory not found at expected path: {ashfs_root}. "
            "Ensure the sibling 'ashfs' module is present."
        )
    if not config_path.exists():
        raise RuntimeError(
            f"ASHFS config.yaml not found at: {config_path}. "
            "Cannot run pipeline without configuration."
        )

    # Insert the ashfs directory so `from ashfs.sampler_pipeline import ...`
    # resolves correctly.  Insert at position 0 to take precedence over any
    # installed package with the same name.
    ashfs_root_str = str(ashfs_root)
    if ashfs_root_str not in sys.path:
        sys.path.insert(0, ashfs_root_str)
        logger.debug("Inserted '%s' into sys.path.", ashfs_root_str)

    try:
        from ashfs.sampler_pipeline import SamplerPipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import ASH-FS SamplerPipeline from {ashfs_root}: {exc}"
        ) from exc

    # Ensure output directory exists.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running ASHFS pipeline on video_id=%s path=%s config=%s",
        video_id,
        video_path,
        config_path,
    )

    try:
        import copy as _copy
        import os as _os
        import tempfile as _tempfile
        import yaml as _yaml

        # Load base config and apply L40s GPU overrides before constructing the pipeline.
        # SamplerPipeline accepts a YAML file path, so we write a patched copy to a
        # temporary file rather than mutating the shared config.yaml on disk.
        with open(str(config_path)) as _f:
            _raw_config = _yaml.safe_load(_f)

        # L40s has 48 GB VRAM vs 24 GB for L4 — double the DINOv2 batch size.
        _raw_config["embeddings"]["dinov2_batch_size"] = 1024  # was 512 (L4 24GB)

        with _tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as _tmp:
            _yaml.dump(_raw_config, _tmp)
            _patched_config_path = _tmp.name

        logger.debug(
            "ASHFS: using patched config at %s (dinov2_batch_size=1024 for L40s)",
            _patched_config_path,
        )

        try:
            pipeline = SamplerPipeline(_patched_config_path)
            manifest: dict = pipeline.run(video_path)
        finally:
            try:
                _os.unlink(_patched_config_path)
            except OSError:
                pass

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"ASHFS SamplerPipeline.run() raised an unrecoverable error "
            f"for video_id={video_id}: {type(exc).__name__}: {exc}"
        ) from exc

    # Surface fallback situation as a warning but do not raise — the fallback
    # manifest is still usable (uniform sampling).
    if manifest.get("fallback"):
        logger.warning(
            "ASHFS fell back to uniform sampling for video_id=%s. "
            "Fallback error: %s",
            video_id,
            manifest.get("fallback_error", "unknown"),
        )

    logger.info(
        "ASHFS complete for video_id=%s: %d shots, %d candidate frames, "
        "%d selected keyframes.",
        video_id,
        manifest.get("total_shots", 0),
        manifest.get("total_candidate_frames", 0),
        manifest.get("total_selected_frames", 0),
    )

    return manifest


def parse_ashfs_manifest(
    manifest: dict,
    video_id: str,
) -> tuple[list[ChunkRecord], list[ShotRecord], list[KeyframeRecord]]:
    """Convert a raw ASHFS manifest into typed record dataclasses.

    Shots are grouped into chunks of :data:`_DEFAULT_SHOTS_PER_CHUNK` shots
    each (or fewer for the last chunk).  This matches the intent of the
    hierarchical diversity planner's ``chunk_size_shots`` setting.

    Parameters
    ----------
    manifest:
        Dict returned by :func:`run_ashfs`.
    video_id:
        Identifier of the video these records belong to.

    Returns
    -------
    tuple[list[ChunkRecord], list[ShotRecord], list[KeyframeRecord]]
        ``(chunks, shots, keyframes)`` — all IDs are freshly generated nanoids.
        Keyframes carry their DINOv2 embedding when the embedding is available
        from the selected-frame index (via the manifest's embedding data if
        present, otherwise ``None``).
    """
    shot_boundaries: list[dict] = manifest.get("shot_boundaries", [])
    selected_frame_indices: list[int] = manifest.get("selected_frame_indices", [])
    frame_roles: list[str] = manifest.get("frame_roles", [])
    frames_per_shot_distribution: list[int] = manifest.get("frames_per_shot_distribution", [])

    if not shot_boundaries or not selected_frame_indices:
        fallback_err = manifest.get("fallback_error", "unknown")
        raise RuntimeError(
            f"ASHFS returned empty manifest — shot_boundaries={len(shot_boundaries)} "
            f"selected_frame_indices={len(selected_frame_indices)} "
            f"fallback={manifest.get('fallback', False)} fallback_error={fallback_err!r}. "
            "Cannot build KB from empty output."
        )

    # Pad frame_roles to match selected_frame_indices length defensively.
    if len(frame_roles) < len(selected_frame_indices):
        frame_roles = frame_roles + ["unknown"] * (
            len(selected_frame_indices) - len(frame_roles)
        )

    # ----------------------------------------------------------------
    # Build per-shot frame index ranges.
    # ASHFS selected_frame_indices are global across the whole video,
    # sorted ascending.  We assign them to shots by using each shot's
    # [start_frame, end_frame] boundary.
    # ----------------------------------------------------------------
    # Build an index → role mapping for O(1) lookup.
    frame_role_map: dict[int, str] = {
        idx: frame_roles[pos]
        for pos, idx in enumerate(selected_frame_indices)
    }

    # ----------------------------------------------------------------
    # Step 1: compute FPS for timestamp derivation.
    # ----------------------------------------------------------------
    # The Shot dataclass stores fps; we pick it from the first shot with
    # a non-zero fps value, or fall back to 25 fps.
    fps: float = 25.0
    for sb in shot_boundaries:
        shot_fps = sb.get("fps", 0.0)
        if shot_fps and shot_fps > 0:
            fps = float(shot_fps)
            break

    # ----------------------------------------------------------------
    # Step 2: group shot boundary indices into chunks.
    # ----------------------------------------------------------------
    n_shots = len(shot_boundaries)
    shots_per_chunk = _DEFAULT_SHOTS_PER_CHUNK
    n_chunks = max(1, math.ceil(n_shots / shots_per_chunk)) if n_shots > 0 else 1

    chunks: list[ChunkRecord] = []
    shots: list[ShotRecord] = []
    keyframes: list[KeyframeRecord] = []

    # Pre-assign selected frame indices into per-shot buckets so we can build
    # KeyframeRecords correctly.
    shot_frame_buckets: list[list[int]] = []
    for sb in shot_boundaries:
        s_start = sb.get("start_frame", 0)
        s_end = sb.get("end_frame", s_start)
        bucket = [
            idx for idx in selected_frame_indices
            if s_start <= idx <= s_end
        ]
        shot_frame_buckets.append(bucket)

    global_shot_index = 0

    for chunk_index in range(n_chunks):
        shot_start = chunk_index * shots_per_chunk
        shot_end = min(shot_start + shots_per_chunk, n_shots)
        chunk_shots = shot_boundaries[shot_start:shot_end]

        # Determine chunk time/frame boundaries from member shots.
        if chunk_shots:
            chunk_start_frame: int | None = chunk_shots[0].get("start_frame")
            chunk_end_frame: int | None = chunk_shots[-1].get("end_frame")
            chunk_start_time: float | None = (
                chunk_start_frame / fps if chunk_start_frame is not None else None
            )
            chunk_end_time: float | None = (
                chunk_end_frame / fps if chunk_end_frame is not None else None
            )
        else:
            chunk_start_frame = None
            chunk_end_frame = None
            chunk_start_time = None
            chunk_end_time = None

        chunk_id = gen_id()
        chunk = ChunkRecord(
            id=chunk_id,
            video_id=video_id,
            chunk_index=chunk_index,
            start_frame=chunk_start_frame,
            end_frame=chunk_end_frame,
            start_time=chunk_start_time,
            end_time=chunk_end_time,
        )
        chunks.append(chunk)

        # ---- shots within this chunk --------------------------------
        for local_shot_idx, sb in enumerate(chunk_shots):
            shot_start_frame: int | None = sb.get("start_frame")
            shot_end_frame: int | None = sb.get("end_frame")
            duration_s: float = sb.get("duration_s", 0.0)

            shot_start_time: float | None = (
                shot_start_frame / fps if shot_start_frame is not None else None
            )
            shot_end_time: float | None = (
                shot_end_frame / fps if shot_end_frame is not None else None
            )

            shot_id = gen_id()
            shot = ShotRecord(
                id=shot_id,
                chunk_id=chunk_id,
                video_id=video_id,
                shot_index=global_shot_index,
                start_frame=shot_start_frame,
                end_frame=shot_end_frame,
                start_time=shot_start_time,
                end_time=shot_end_time,
                shot_type=None,     # ASHFS does not classify shot type
                complexity=None,    # complexity score lives in budget_stats only
            )
            shots.append(shot)

            # ---- keyframes within this shot -------------------------
            bucket = shot_frame_buckets[global_shot_index]
            for frame_index in bucket:
                timestamp_s = frame_index / fps
                selection_reason = frame_role_map.get(frame_index, "unknown")

                keyframe = KeyframeRecord(
                    id=gen_id(),
                    shot_id=shot_id,
                    video_id=video_id,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    r2_key=None,            # uploaded in a later stage
                    selection_reason=selection_reason,
                    dino_embedding=None,    # embeddings not serialised in manifest
                    siglip_embedding=None,
                )
                keyframes.append(keyframe)

            global_shot_index += 1

    logger.info(
        "parse_ashfs_manifest: video_id=%s → %d chunks, %d shots, %d keyframes.",
        video_id,
        len(chunks),
        len(shots),
        len(keyframes),
    )

    return chunks, shots, keyframes
