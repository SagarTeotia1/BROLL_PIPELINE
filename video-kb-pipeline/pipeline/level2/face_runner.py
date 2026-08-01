"""Level-2 face analysis runner.

Wraps the face_analysis-model pipeline headlessly and maps its output to the
shared PersonRecord / FaceAppearanceRecord / FaceTimelineEvent types used by
the rest of the knowledge-base pipeline.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Optional

from shared.types import FaceAppearanceRecord, FaceTimelineEvent, PersonRecord
from shared.utils import gen_id

logger = logging.getLogger(__name__)

# Standard ArcFace 5-point facial landmark template (112×112 output).
# These are the canonical destination keypoints used by InsightFace / ArcFace.
_ARCFACE_DST = None  # populated lazily to avoid importing numpy at module load


def _get_arcface_dst():
    global _ARCFACE_DST
    if _ARCFACE_DST is None:
        import numpy as np
        _ARCFACE_DST = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
    return _ARCFACE_DST


def _det_score(face) -> float:
    """Return detection confidence score regardless of attribute name."""
    import numpy as np
    for attr in ("det_score", "score", "confidence", "conf", "prob"):
        val = getattr(face, attr, None)
        if val is None:
            continue
        try:
            # numpy arrays with ndim>0 need item() to get a Python scalar
            if isinstance(val, np.ndarray):
                val = val.flat[0]
            return float(val)
        except Exception:
            continue
    return 0.0


def _det_kps(face):
    """Return 5-point keypoints regardless of attribute name, or None."""
    for attr in ("kps", "landmarks", "keypoints", "landmark", "landmark_3d_68",
                 "landmark_2d_106", "pts"):
        val = getattr(face, attr, None)
        if val is not None:
            return val
    return None


def _arcface_align(img_bgr, kps, size: int = 112):
    """Return a landmark-aligned face crop using the standard ArcFace template.

    Args:
        img_bgr: Full BGR image (numpy array).
        kps:     5×2 array of facial keypoints (left-eye, right-eye, nose,
                 left-mouth, right-mouth) in image coordinates.
        size:    Output square size (default 112 for ArcFace).

    Returns:
        Aligned BGR face crop of shape (size, size, 3), or None on failure.
    """
    import cv2
    import numpy as np

    try:
        src = np.array(kps, dtype=np.float32).reshape(5, 2)
        dst = _get_arcface_dst()
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return None
        return cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR)
    except Exception:
        return None


# Root of the monorepo — four levels up from this file:
#   video-kb-pipeline/pipeline/level2/face_runner.py
#   -> video-kb-pipeline/pipeline/level2/
#   -> video-kb-pipeline/pipeline/
#   -> video-kb-pipeline/
#   -> <monorepo root>/
_MONOREPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# Modal containers mount the sibling at /app_modules/face_analysis
_FACE_MODEL_ROOT = (
    Path("/app_modules/face_analysis")
    if Path("/app_modules/face_analysis").exists()
    else _MONOREPO_ROOT / "face_analysis-model"
)


def _load_face_modules():
    """Import face_analysis-model modules without polluting sys.path with a
    directory that also contains a ``pipeline/`` package.

    The face_analysis-model has its own ``pipeline/`` sub-package which would
    shadow our top-level ``pipeline/`` package if ``face_analysis-model/`` were
    placed on sys.path directly.  Instead we temporarily inject the directory,
    import every needed symbol into local variables, and then:
      1. remove the directory from sys.path again, and
      2. restore the original ``pipeline`` and ``pipeline.*`` entries in
         sys.modules so that subsequent ``from pipeline.X import Y`` calls
         continue to resolve against our own package.
    """
    face_model_root = str(_FACE_MODEL_ROOT)
    already_on_path = face_model_root in sys.path
    if not already_on_path:
        sys.path.insert(0, face_model_root)

    # Snapshot ``pipeline``, ``pipeline.*``, and ``models``, ``models.*`` from
    # sys.modules before importing face-model symbols.
    #
    # ``pipeline`` conflict: face_analysis-model has its own pipeline.batching
    # which shadows our top-level pipeline package.
    #
    # ``models`` conflict: when Qwen loads in the same process (single-GPU mode),
    # it caches ``models`` (pointing to /app/models/) in sys.modules.  Face
    # analysis needs ``models.model_zoo`` from /app_modules/face_analysis/models/.
    # Evicting the cached entry lets the face imports resolve correctly.
    def _snapshot_and_evict(*prefixes: str) -> dict[str, types.ModuleType | None]:
        snap: dict[str, types.ModuleType | None] = {}
        for k in list(sys.modules):
            if any(k == p or k.startswith(p + ".") for p in prefixes):
                snap[k] = sys.modules.pop(k, None)
        return snap

    combined_snapshot = _snapshot_and_evict("pipeline", "models")

    try:
        from configs.config import AppConfig  # type: ignore[import]
        worker_path = _FACE_MODEL_ROOT / "pipeline" / "worker.py"
        spec = importlib.util.spec_from_file_location(
            "face_analysis_model.pipeline.worker", worker_path
        )
        worker_mod: types.ModuleType = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules.setdefault("face_analysis_model.pipeline", types.ModuleType("face_analysis_model.pipeline"))
        sys.modules["face_analysis_model.pipeline.worker"] = worker_mod
        spec.loader.exec_module(worker_mod)  # type: ignore[union-attr]

        from recognition.cast_database import CastDatabase  # type: ignore[import]
        from recognition.registration import CastRegistrar  # type: ignore[import]
        from detector.scrfd import SCRFDDetector as _SCRFDDetector  # type: ignore[import]
        from recognition.arcface import ArcFaceEmbedder as _ArcFaceEmbedder  # type: ignore[import]

        AnalysisPipeline = worker_mod.AnalysisPipeline
        PipelineCallbacks = worker_mod.PipelineCallbacks
    finally:
        # Remove any pipeline.* / models.* the face model registered
        for key in list(sys.modules):
            if any(key == p or key.startswith(p + ".") for p in ("pipeline", "models")):
                del sys.modules[key]

        # Restore our own pipeline.* and models.* entries
        for key, original in combined_snapshot.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original

        # Remove face model root from sys.path
        if not already_on_path and face_model_root in sys.path:
            sys.path.remove(face_model_root)

    return AppConfig, AnalysisPipeline, PipelineCallbacks, CastDatabase, CastRegistrar, _SCRFDDetector, _ArcFaceEmbedder


def build_cast_db_from_cast_list(
    cast_list: list[dict],
    output_db_path: str,
) -> int:
    """Download reference images, embed each face with ArcFace, write CastDatabase.

    Uses the face_analysis-model's own CastRegistrar for detection → alignment →
    embedding, which is the correct API and avoids fragile custom code.

    Args:
        cast_list: List of dicts, each with:
            - ``name``   (str): Display name for this person.
            - ``images`` (list[str]): One or more image URLs or local file paths.
        output_db_path: Path to write the SQLite cast DB (e.g. /tmp/cast.db).

    Returns:
        Number of actors successfully registered.
    """
    import tempfile
    import urllib.request

    try:
        AppConfig, AnalysisPipeline, PipelineCallbacks, CastDatabase, CastRegistrar, SCRFDDetector, ArcFaceEmbedder = _load_face_modules()
    except Exception as exc:
        logger.error("Could not load face modules for cast DB build: %s", exc)
        return 0

    config = AppConfig.load()
    config.logging.to_file = False
    config.logging.profile = False
    config.gui.overlay_boxes = False

    # Force GPU — this always runs on Modal L40S.
    try:
        config.gpu.device = "cuda"
        config.gpu.fp16   = True
    except Exception:
        pass

    # Relax quality gates for registration — reference images are often not
    # perfectly frontal. Lower thresholds so the centroid is built from real
    # crops rather than the 0.7× weight salvage path.
    try:
        config.quality.min_det_score = 0.40
        config.quality.min_blur_var  = 15.0
        config.quality.max_yaw_ratio = 4.0
    except Exception:
        pass

    db_path = Path(output_db_path)
    pkl_path = db_path.with_suffix(".pkl")
    db = CastDatabase(db_path, pkl_path)

    # Build detector + embedder directly (no full AnalysisPipeline needed).
    try:
        import torch as _torch
        device = _torch.device("cuda")
        detector = SCRFDDetector(config, device, stream_handle=None)
        embedder = ArcFaceEmbedder(config, device, stream_handle=None)
    except Exception as exc:
        logger.error("Failed to initialise detector/embedder for cast DB: %s", exc)
        return 0

    registrar = CastRegistrar(config, detector, embedder, db)

    registered = 0
    for entry in cast_list:
        name: str = entry.get("name", "").strip()
        image_sources: list[str] = entry.get("images", [])
        if not name or not image_sources:
            logger.warning("Cast entry missing name or images — skipping: %s", entry)
            continue

        # Download URLs to temp files; pass temp paths to CastRegistrar.
        tmp_paths: list[str] = []
        tmp_files: list = []
        for src in image_sources:
            try:
                if src.startswith("http://") or src.startswith("https://"):
                    req = urllib.request.Request(
                        src,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; video-kb-pipeline/1.0)"},
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        img_bytes = resp.read()
                    suffix = Path(src.split("?")[0]).suffix or ".jpg"
                    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                    tmp.write(img_bytes)
                    tmp.flush()
                    tmp_files.append(tmp)
                    tmp_paths.append(tmp.name)
                else:
                    tmp_paths.append(src)
            except Exception as exc:
                logger.warning("Failed to fetch image for %s (%s): %s", name, src, exc)

        if not tmp_paths:
            logger.warning("No images fetched for cast member %s — skipping", name)
            continue

        try:
            result = registrar.register(name, tmp_paths)
            if result.ok:
                registered += 1
                logger.info(
                    "Registered cast member: %s (%d accepted, %d rejected)",
                    name, result.accepted, result.rejected,
                )
            else:
                logger.warning(
                    "No valid reference images for cast member %s (%d rejected): %s",
                    name, result.rejected,
                    "; ".join(o.reason for o in result.outcomes if not o.accepted),
                )
        except Exception as exc:
            logger.warning("CastRegistrar.register failed for %s: %s", name, exc)
        finally:
            for tmp in tmp_files:
                try:
                    import os as _os
                    _os.unlink(tmp.name)
                except OSError:
                    pass

    try:
        db.close()
    except Exception as exc:
        logger.warning("Failed to close cast DB: %s", exc)

    try:
        detector.close()
        embedder.close()
    except Exception:
        pass

    logger.info(
        "Cast DB built: %d/%d members registered → %s",
        registered, len(cast_list), output_db_path,
    )
    return registered


def run_face_analysis(
    video_path: str,
    video_id: str,
    cast_db_path: Optional[str] = None,
    keyframe_timestamps: list[float] | None = None,
    shots: list[dict] | None = None,
) -> dict:
    """Run the face_analysis-model pipeline against *video_path* headlessly.

    Args:
        video_path:           Absolute path to the video file to analyse.
        video_id:             Stable identifier for this video in the knowledge base.
        cast_db_path:         Optional path to a cast SQLite database.
        keyframe_timestamps:  Timestamps (seconds) of ASHFS keyframes.  When provided,
                              appearances are sampled at those timestamps instead of 1 fps
                              so that L3 Qwen cross-reference is a direct lookup.
        shots:                Shot dicts [{shot_index, start_time, end_time, ...}] used to
                              always emit an appearance at shot boundaries and to attach
                              shot_index to each record for context logging.

    Returns:
        A dict with keys:
        - ``"persons"``:         list[PersonRecord]
        - ``"appearances"``:     list[FaceAppearanceRecord]
        - ``"timeline_events"``: list[FaceTimelineEvent]
    """
    # ------------------------------------------------------------------
    # 1. Import face pipeline using safe importlib approach that avoids
    #    the ``pipeline/`` package name collision.
    # ------------------------------------------------------------------
    try:
        AppConfig, AnalysisPipeline, PipelineCallbacks, CastDatabase, *_ = _load_face_modules()
    except ImportError as exc:
        logger.warning(
            "face_analysis-model could not be imported (%s) — "
            "returning empty face analysis result.",
            exc,
        )
        return {"persons": [], "appearances": [], "timeline_events": []}
    except Exception as exc:
        logger.warning(
            "Unexpected error loading face_analysis-model (%s) — "
            "returning empty face analysis result.",
            exc,
        )
        return {"persons": [], "appearances": [], "timeline_events": []}

    # ------------------------------------------------------------------
    # 2. Build config (defaults are fine; disable GUI overlays / logging
    #    to file to avoid side effects in the pipeline process).
    # ------------------------------------------------------------------
    config = AppConfig.load()
    config.logging.to_file = False
    config.logging.profile = False
    config.gui.overlay_boxes = False

    # ------------------------------------------------------------------
    # Tuned recognition thresholds for production pipelines.
    #
    # Defaults are conservative (0.38 / 3 votes) which works well for
    # clear frontal faces but misses cast members who:
    #   - appear at partial profile or partial occlusion
    #   - have only one low-diversity reference image
    #   - appear briefly (< 3 frames in vote window)
    #
    # 0.32 similarity: catches ~15% more true matches on single-ref-image
    #   casts at acceptable false-positive rate (4-person cast is small).
    # min_votes=2: brief appearances (cut-aways) now confirmed.
    # min_det_score=0.45: matches SCRFD score_threshold — no early rejection.
    # reid_interval=30: re-identify every 30 sampled frames (~1 fps × 30s)
    #   so long shots re-confirm identity rather than relying on first lock.
    # ------------------------------------------------------------------
    # Force GPU — always runs on Modal L40S.
    try:
        config.gpu.device = "cuda"
        config.gpu.fp16   = True
    except Exception:
        pass

    # Tuned recognition thresholds — see build_cast_db_from_cast_list comments.
    # reid_interval is NOT set here — it's duration-scaled in the auto-tune
    # block below alongside frame_stride, same reasoning (real-video finding:
    # face_analysis was the dominant, still-unfixed L2 cost on a real
    # talking-head podcast; low-motion static-framing content doesn't need
    # re-identification as often as the flat default assumed).
    try:
        config.recognition.similarity_threshold = 0.32
        config.recognition.margin_threshold     = 0.04
        config.recognition.min_votes            = 2
        config.quality.min_det_score            = 0.45
        logger.info(
            "GPU forced. Thresholds: similarity=0.32 margin=0.04 "
            "min_votes=2 min_det_score=0.45"
        )
    except Exception as _cfg_exc:
        logger.warning("Could not override recognition thresholds: %s", _cfg_exc)

    # Auto-tune sampling stride and batch sizes based on video duration.
    # Default stride=4 (7.5fps) is fine for short clips; long videos can use
    # larger stride because ByteTrack + adaptive mode still maintain track identity.
    # Larger ONNX batches fill A100 significantly better than the default bs=4.
    # During L2, Qwen is idle (inference hasn't started) so ~64 GB VRAM is free
    # for ONNX — raise the arena limit from the conservative 3 GB default.
    try:
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(video_path)
        _fps_vid = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
        _total_frames_vid = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        _cap.release()
        _duration_s = _total_frames_vid / max(_fps_vid, 1.0)

        # Real-video finding: a 68.9min talking-head podcast (3 people,
        # faces on-screen almost every sampled frame) landed in the >30min
        # bucket at 17m37s for face_analysis alone — the dominant L2 cost,
        # still unfixed at the model/inference level (TensorRT EP, IOBinding
        # remain real options, not yet implemented — need live GPU iteration
        # to do safely). These two bumps are the safe, config-only levers
        # available without that: reid_interval scaled 2-3x looser (low-
        # motion/static-framing content doesn't need to re-confirm identity
        # as often as fast-cut dynamic content), and the >30min bucket's
        # frame_stride bumped to match the >90min bucket's value — real
        # accuracy/track-continuity tradeoff (ByteTrack bridges longer gaps
        # between samples), not a free win.
        if _duration_s > 5400:          # > 90 min: coarse stride, still safe
            config.sampling.frame_stride = 8
            config.sampling.max_stride   = 24
            config.recognition.reid_interval = 90
        elif _duration_s > 1800:        # > 30 min
            config.sampling.frame_stride = 8    # was 6 — bumped to match >90min bucket
            config.sampling.max_stride   = 20   # was 16 — scaled proportionally
            config.recognition.reid_interval = 60  # was 30 (flat default)
        else:                            # < 30 min: default
            config.sampling.frame_stride = 4
            config.sampling.max_stride   = 8
            config.recognition.reid_interval = 30

        config.detector.batch_size     = 8    # 4 → 8: doubles SCRFD throughput on A100
        config.recognition.batch_size  = 32   # 16 → 32
        config.emotion.batch_size      = 32   # 16 → 32
        config.gpu.gpu_mem_limit_mb    = 8192  # 3 GB → 8 GB ONNX arena (Qwen idle during L2)

        logger.info(
            "Face pipeline auto-tuned: duration=%.0fs → stride=%d max_stride=%d "
            "reid_interval=%d det_bs=%d rec_bs=%d emo_bs=%d ort_mem=%dMB",
            _duration_s,
            config.sampling.frame_stride,
            config.sampling.max_stride,
            config.recognition.reid_interval,
            config.detector.batch_size,
            config.recognition.batch_size,
            config.emotion.batch_size,
            config.gpu.gpu_mem_limit_mb,
        )
    except Exception as _tune_exc:
        logger.warning("Face pipeline auto-tune failed (non-fatal): %s", _tune_exc)

    # ------------------------------------------------------------------
    # 3. Override cast DB paths when caller supplies one.
    # ------------------------------------------------------------------
    if cast_db_path is not None:
        cast_db_abs = Path(cast_db_path).resolve()
        config.paths.cast_db = str(cast_db_abs)
        config.paths.cast_pickle = str(cast_db_abs.with_suffix(".pkl"))

    # ------------------------------------------------------------------
    # 4. Build the pipeline and run it synchronously.
    # ------------------------------------------------------------------
    pipeline = AnalysisPipeline(config, PipelineCallbacks())
    try:
        pipeline.prepare()
        logger.info(
            "Face pipeline ready — cast DB has %d actors",
            pipeline.matcher.num_actors,
        )
        document = pipeline.analyze(video_path)
    except Exception as exc:
        logger.warning(
            "face_analysis-model pipeline raised an exception (%s) — "
            "returning empty face analysis result.",
            exc,
        )
        return {"persons": [], "appearances": [], "timeline_events": []}
    finally:
        try:
            pipeline.close()
        except Exception:
            pass

    logger.info(
        "Face analysis complete: %d events, %d actors",
        len(document.events),
        len(document.actors),
    )

    # ------------------------------------------------------------------
    # 5. Map actor_name -> stable PID.
    #
    #    Known actors (matched via cast DB) get P1, P2, … ordered by
    #    first_seen timestamp.  Unknown actors get PU1, PU2, …
    # ------------------------------------------------------------------
    # document.actors is a list[ActorEntry], ordered by how they appear in
    # the timeline engine — usually by first_seen ascending.  We sort
    # explicitly to guarantee the order is deterministic.
    known_actors = sorted(
        [a for a in document.actors if a.name != "Unknown"],
        key=lambda a: a.first_seen,
    )
    unknown_actors = sorted(
        [a for a in document.actors if a.name == "Unknown"],
        key=lambda a: a.first_seen,
    )

    # actor_id (int from the pipeline) -> PID string
    actor_id_to_pid: dict[int, str] = {}
    known_counter = 0
    for actor in known_actors:
        known_counter += 1
        actor_id_to_pid[actor.id] = f"P{known_counter}"

    unknown_counter = 0
    for actor in unknown_actors:
        unknown_counter += 1
        actor_id_to_pid[actor.id] = f"PU{unknown_counter}"

    # actor_id (int) -> PersonRecord UUID
    actor_id_to_uuid: dict[int, str] = {}
    # actor_name (str) -> PersonRecord UUID  (for known actors only).
    # Unknown actors are intentionally excluded: multiple unknown actors all
    # share the name "Unknown", so a name-keyed dict would silently collapse
    # them to the last entry.  Unknown actor events are resolved exclusively
    # via actor_id_to_uuid using event.actor_id.
    actor_name_to_uuid: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 6. Retrieve centroid embeddings from the cast DB for known actors.
    # ------------------------------------------------------------------
    centroid_by_actor_id: dict[int, list[float]] = {}
    if cast_db_path is not None:
        try:
            db = CastDatabase(
                Path(cast_db_path).resolve(),
                Path(cast_db_path).resolve().with_suffix(".pkl"),
            )
            for actor_record in db.list_actors(with_samples=False):
                if actor_record.centroid.size:
                    centroid_by_actor_id[actor_record.id] = actor_record.centroid.tolist()
            db.close()
        except Exception as exc:
            logger.warning(
                "Could not read centroids from cast DB: %s", exc
            )

    # We also need to map internal pipeline actor_id to the cast DB actor_id.
    # The pipeline's IdentityRegistry stores actor_id as the gallery owner id
    # returned by CastDatabase.build_gallery.  So pipeline actor_id == DB actor_id.

    # ------------------------------------------------------------------
    # 7. Build PersonRecords.
    # ------------------------------------------------------------------
    persons: list[PersonRecord] = []
    for actor in known_actors + unknown_actors:
        person_uuid = gen_id()
        actor_id_to_uuid[actor.id] = person_uuid
        # Only map known actors by name; unknown actors are resolved via
        # actor_id_to_uuid so that multiple "Unknown" actors don't clobber
        # each other in the name-keyed dict.
        if actor.name != "Unknown":
            actor_name_to_uuid[actor.name] = person_uuid

        # Use cast DB centroid embedding when available.
        # centroid is already list[float] after .tolist() above.
        embedding: list[float] | None = centroid_by_actor_id.get(actor.id)

        pid = actor_id_to_pid[actor.id]
        display_name: str | None = actor.name if actor.name != "Unknown" else None

        persons.append(
            PersonRecord(
                id=person_uuid,
                video_id=video_id,
                pid=pid,
                display_name=display_name,
                arcface_embedding=embedding,
            )
        )

    # ------------------------------------------------------------------
    # 8. Collect face appearances across the FULL video.
    #
    #    Dense sampling is intentional: the combined face-appearance +
    #    HSEmotion records form a per-person emotion timeline across the
    #    entire video, which the L5 reasoning model uses to build identity
    #    arcs and resolve ambiguous person IDs.
    #
    #    Sampling strategy:
    #    • 1 fps baseline across the entire video (full temporal coverage).
    #    • Keyframe timestamps added on top so L3 Qwen cross-reference is
    #      exact (no interpolation needed at inference time).
    #    • Shot boundary timestamps added so first appearance in each shot
    #      is always captured even when no keyframe falls there.
    #
    #    Change-detection (5-second window):
    #    • If the same person shows the same emotion within 5 s of their
    #      last record AND the sample point is not a keyframe/shot boundary,
    #      the record is skipped.  This removes pure consecutive duplicates
    #      while preserving the full emotional arc (emotion transitions,
    #      shot entries, and all keyframe-aligned records are always kept).
    # ------------------------------------------------------------------
    _EMOTION_WINDOW_S = 5.0   # dedupe window — small enough to keep emotional arc intact
    _FORCED_POINTS: set[float] = set()  # keyframes + shot boundaries always emitted

    appearances: list[FaceAppearanceRecord] = []

    try:
        import cv2  # type: ignore[import]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture could not open {video_path!r}")

        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 1 fps baseline across full video.
        stride = max(1, int(round(fps_vid)))
        _ts_set: set[float] = {round(i / fps_vid, 3) for i in range(0, total_frames, stride)}

        # Add keyframe timestamps (forced emit points).
        if keyframe_timestamps:
            for ts in keyframe_timestamps:
                _ts_set.add(round(ts, 3))
                _FORCED_POINTS.add(round(ts, 3))
            logger.info("Face sampling: +%d keyframe timestamps", len(keyframe_timestamps))

        # Add shot boundary timestamps (forced emit points).
        _shot_starts: set[float] = set()
        if shots:
            for sh in shots:
                st = sh.get("start_time")
                if st is not None:
                    ts_r = round(float(st), 3)
                    _ts_set.add(ts_r)
                    _FORCED_POINTS.add(ts_r)
                    _shot_starts.add(ts_r)
            logger.info("Face sampling: +%d shot boundary timestamps", len(_shot_starts))

        sample_timestamps = sorted(_ts_set)
        logger.info(
            "Face sampling: %d total sample points (1fps=%d + %d forced keyframe/boundary)",
            len(sample_timestamps),
            int(total_frames / stride),
            len(_FORCED_POINTS),
        )

        all_events = list(document.events)

        # Change-detection state: person_uuid → (last_emotion, last_emit_timestamp_s)
        last_emitted: dict[str | None, tuple[str | None, float]] = {}

        for timestamp_s in sample_timestamps:
            frame_idx = min(int(round(timestamp_s * fps_vid)), total_frames - 1)
            is_forced = round(timestamp_s, 3) in _FORCED_POINTS

            # At shot boundaries, reset state for persons no longer visible
            # so their first appearance in the new shot is always recorded.
            if round(timestamp_s, 3) in _shot_starts:
                active_pids = set()
                for ev in all_events:
                    if ev.start <= timestamp_s <= ev.end:
                        p = actor_name_to_uuid.get(ev.actor)
                        if p is None and hasattr(ev, "actor_id"):
                            p = actor_id_to_uuid.get(ev.actor_id)
                        if p is not None:
                            active_pids.add(p)
                for pid in list(last_emitted):
                    if pid not in active_pids:
                        last_emitted.pop(pid, None)

            active_events = [
                ev for ev in all_events
                if ev.start <= timestamp_s <= ev.end
            ]

            for event in active_events:
                p_uuid = actor_name_to_uuid.get(event.actor)
                if p_uuid is None and hasattr(event, "actor_id"):
                    p_uuid = actor_id_to_uuid.get(event.actor_id)

                prev = last_emitted.get(p_uuid)
                if (
                    prev is not None
                    and prev[0] == event.emotion
                    and (timestamp_s - prev[1]) < _EMOTION_WINDOW_S
                    and not is_forced
                ):
                    continue  # same emotion, within 5s, not a keyframe/boundary — skip

                last_emitted[p_uuid] = (event.emotion, timestamp_s)

                track_id: int | None = (
                    event.track_ids[0] if event.track_ids else None
                )

                try:
                    appearances.append(
                        FaceAppearanceRecord(
                            id=gen_id(),
                            video_id=video_id,
                            frame_index=frame_idx,
                            timestamp_s=round(timestamp_s, 3),
                            person_id=p_uuid,
                            track_id=track_id,
                            bbox=None,
                            emotion=event.emotion,
                            emotion_conf=round(float(event.confidence), 4),
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping appearance record at ts=%.2fs: %s", timestamp_s, exc
                    )

        logger.info(
            "Face appearance sampling done: %d records from %d sample points "
            "(deduped with %.0fs emotion window — all keyframes/boundaries forced)",
            len(appearances), len(sample_timestamps), _EMOTION_WINDOW_S,
        )

    except Exception as exc:
        logger.warning(
            "Face appearance sampling failed, appearances will be empty: %s", exc
        )

    # ------------------------------------------------------------------
    # 9. Build FaceTimelineEvents from the pipeline document.
    # ------------------------------------------------------------------
    timeline_events: list[FaceTimelineEvent] = []

    for event in document.events:
        try:
            # Prefer UUID lookup via actor_name; fall back via actor_id field.
            p_uuid = actor_name_to_uuid.get(event.actor)
            if p_uuid is None and hasattr(event, "actor_id"):
                p_uuid = actor_id_to_uuid.get(event.actor_id)

            timeline_events.append(
                FaceTimelineEvent(
                    id=gen_id(),
                    video_id=video_id,
                    # person_id is the UUID string of the PersonRecord.
                    person_id=p_uuid,
                    emotion=event.emotion,
                    start_time=round(float(event.start), 3),
                    end_time=round(float(event.end), 3),
                    confidence=round(float(event.confidence), 4),
                )
            )
        except Exception as exc:
            logger.warning(
                "Skipping timeline event for actor %r: %s", event.actor, exc
            )

    logger.info(
        "Face runner complete — %d persons, %d appearances, %d timeline events",
        len(persons),
        len(appearances),
        len(timeline_events),
    )

    return {
        "persons": persons,
        "appearances": appearances,
        "timeline_events": timeline_events,
    }
