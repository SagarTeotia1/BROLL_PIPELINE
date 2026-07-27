"""Level-2 color grading runner.

Wraps the color-grading ColorGradingEngine and produces one ColorGradeRecord
per shot, sampling the midpoint frame of each shot for analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from shared.types import ColorGradeRecord, ShotRecord
from shared.utils import gen_id

logger = logging.getLogger(__name__)

# Root of the monorepo — four levels up from this file:
#   video-kb-pipeline/pipeline/level2/color_runner.py
#   -> video-kb-pipeline/pipeline/level2/
#   -> video-kb-pipeline/pipeline/
#   -> video-kb-pipeline/
#   -> <monorepo root>/
_MONOREPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# Modal containers mount color-grading at /app_modules/color_grading
_COLOR_GRADING_ROOT = (
    Path("/app_modules/color_grading")
    if Path("/app_modules/color_grading").exists()
    else _MONOREPO_ROOT / "color-grading"
)

# Parameter names from schema.PARAM_NAMES — these define the 45 parameters
# emitted by the decision engine.  We keep a local copy here so that the
# parameters dict can be constructed without importing the schema at module
# level (avoiding a top-level sys.path mutation).
_PARAM_NAMES: tuple[str, ...] | None = None


def _ensure_color_grading_on_path() -> None:
    """Idempotently add the color-grading root to sys.path."""
    color_grading_root = str(_COLOR_GRADING_ROOT)
    if color_grading_root not in sys.path:
        sys.path.insert(0, color_grading_root)


def _get_param_names() -> tuple[str, ...]:
    """Return the canonical list of 45 grade parameter names."""
    global _PARAM_NAMES
    if _PARAM_NAMES is None:
        _ensure_color_grading_on_path()
        from color_analyzer.analyzer.schema import PARAM_NAMES  # type: ignore[import]
        _PARAM_NAMES = PARAM_NAMES
    return _PARAM_NAMES


def _grade_to_parameters(grade_doc: dict[str, Any]) -> dict[str, Any]:
    """Convert a decision-engine grade document into the flat parameters dict.

    The grade document has the shape produced by
    ``color_analyzer.analyzer.schema.assemble()``::

        {
            "meta":    {...},
            "style":   {...},
            "grade": {
                "white_balance.temperature": {"current": 6200, "recommended": 6000, "delta": -200},
                ...
            },
            "notes":   [...],
            "palette": [...],
        }

    We re-emit the per-parameter dicts keyed by parameter name so that
    downstream consumers always see all 45 keys.  All values are plain Python
    scalars (int, float, bool, str, None) — no numpy types.
    """
    grade_section: dict[str, Any] = grade_doc.get("grade", {})
    param_names = _get_param_names()
    parameters: dict[str, Any] = {}

    for name in param_names:
        entry = grade_section.get(name, {})
        param_entry: dict[str, Any] = {
            "current": _to_scalar(entry.get("current")),
        }
        if "recommended" in entry:
            param_entry["recommended"] = _to_scalar(entry["recommended"])
        if "delta" in entry:
            param_entry["delta"] = _to_scalar(entry["delta"])
        parameters[name] = param_entry

    return parameters


def _to_scalar(value: Any) -> Any:
    """Convert numpy scalars to plain Python types so the dict is JSON-safe."""
    if value is None:
        return None
    # numpy scalar detection without importing numpy at module level.
    type_name = type(value).__module__
    if type_name == "numpy" or type_name.startswith("numpy."):
        # Covers np.float32, np.int64, np.bool_, etc.
        return value.item()
    return value


def _extract_style_tags(grade_doc: dict[str, Any]) -> list[str]:
    """Derive style tags from the summary section of the grade document.

    The decision engine's ``assemble()`` call places human-readable summary
    fields in the ``"style"`` and ``"meta"`` sections.  ``GradingSummary``
    populates ``top_styles`` (a list of ``(name, score)`` tuples) and ``mood``
    (the top-ranked style name); either or both may appear in those sections
    depending on the decision engine version.
    """
    style_section: dict[str, Any] = grade_doc.get("style", {})
    meta_section: dict[str, Any] = grade_doc.get("meta", {})

    tags: list[str] = []

    # top_styles may appear in either "meta" or "style" section.
    for section in (meta_section, style_section):
        for style_entry in section.get("top_styles", []):
            if isinstance(style_entry, (list, tuple)) and style_entry:
                tag = str(style_entry[0])
            elif isinstance(style_entry, str):
                tag = style_entry
            else:
                continue
            if tag and tag not in tags:
                tags.append(tag)

    # "mood" is the single top-ranked style name.
    for section in (style_section, meta_section):
        mood = section.get("mood")
        if mood and isinstance(mood, str) and mood not in tags:
            tags.append(mood)
            break

    return tags


def _read_frame_with_fallback(
    cap,  # cv2.VideoCapture
    target_frame: int,
    max_adjacent: int = 5,
):
    """Seek to *target_frame* and read it; try adjacent frames on failure.

    Returns ``(frame_index_used, frame)`` on success, or ``(None, None)`` if
    neither the target nor any adjacent frame could be read.
    """
    import cv2  # type: ignore[import]

    for offset in range(max_adjacent + 1):
        for sign in (1, -1) if offset > 0 else (0,):
            candidate = target_frame + sign * offset
            if candidate < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
            ret, frame = cap.read()
            if ret and frame is not None:
                return candidate, frame
    return None, None


def run_color_grading(
    video_path: str,
    video_id: str,
    shots: list[ShotRecord],
) -> list[ColorGradeRecord]:
    """Analyse the color grade of the midpoint frame for each shot.

    Args:
        video_path: Absolute path to the video file.
        video_id:   Stable identifier for this video in the knowledge base.
        shots:      Shot records produced by the Level-1 shot-detection step.

    Returns:
        A list of :class:`ColorGradeRecord`, one per successfully analysed
        shot.  Shots whose midpoint frame cannot be read are skipped with a
        warning rather than crashing the pipeline.
    """
    # ------------------------------------------------------------------
    # 1. Extend sys.path so color-grading imports resolve correctly.
    # ------------------------------------------------------------------
    _ensure_color_grading_on_path()

    # ------------------------------------------------------------------
    # 2. Import the engine after sys.path is set.
    # ------------------------------------------------------------------
    try:
        from color_analyzer.analyzer.engine import ColorGradingEngine  # type: ignore[import]
    except ImportError as exc:
        logger.error("Failed to import color-grading engine: %s", exc)
        raise

    # ------------------------------------------------------------------
    # 3. Instantiate the engine once and reuse it for all shots.
    #    deep=False is the fast path (core 45-parameter grade only).
    #
    #    ColorGradingEngine(prefer_gpu, deep) — matches the actual
    #    signature: __init__(self, backend=None, prefer_gpu=True, deep=False)
    # ------------------------------------------------------------------
    engine = ColorGradingEngine(prefer_gpu=True, deep=False)
    logger.info(
        "ColorGradingEngine ready — analysing %d shots from %s",
        len(shots),
        video_path,
    )

    # ------------------------------------------------------------------
    # 4. Open the video capture.
    # ------------------------------------------------------------------
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        logger.error("cv2 (OpenCV) is not installed: %s", exc)
        raise

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"cv2.VideoCapture could not open video: {video_path!r}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: list[ColorGradeRecord] = []

    # ------------------------------------------------------------------
    # 5. Iterate shots, seek to midpoint, analyse.
    # ------------------------------------------------------------------
    for shot in shots:
        start_frame = shot.start_frame
        end_frame = shot.end_frame

        # Gracefully handle missing frame boundaries.
        if start_frame is None or end_frame is None:
            logger.warning(
                "Shot %s has no frame range (start=%s, end=%s) — skipping",
                shot.id,
                start_frame,
                end_frame,
            )
            continue

        mid_frame = (start_frame + end_frame) // 2

        # Compute the midpoint timestamp.
        timestamp_s = mid_frame / fps if fps else None

        # Seek to the midpoint; try adjacent frames if the read fails.
        actual_frame, frame = _read_frame_with_fallback(cap, mid_frame)
        if actual_frame is None or frame is None:
            logger.warning(
                "Could not read frame %d (or adjacent frames) for shot %s — skipping",
                mid_frame,
                shot.id,
            )
            continue

        # If we ended up on a different frame, adjust the timestamp.
        if actual_frame != mid_frame:
            logger.debug(
                "Shot %s: sought frame %d, fell back to frame %d",
                shot.id,
                mid_frame,
                actual_frame,
            )
            timestamp_s = actual_frame / fps if fps else timestamp_s

        try:
            # engine.grade(source, **kwargs) dispatches on source type:
            #   str  -> analyze_path()
            #   else -> analyze_array(np.asarray(source), **kwargs)
            # Passing is_bgr=True tells the engine the array is BGR (OpenCV
            # default) rather than RGB.
            grade_doc: dict[str, Any] = engine.grade(frame, is_bgr=True)

            parameters = _grade_to_parameters(grade_doc)
            style_tags = _extract_style_tags(grade_doc)

            results.append(
                ColorGradeRecord(
                    id=gen_id(),
                    video_id=video_id,
                    shot_id=shot.id,
                    frame_index=actual_frame,
                    timestamp_s=round(timestamp_s, 3) if timestamp_s is not None else None,
                    # parameters is a plain dict of plain Python scalars — JSON-safe.
                    parameters=parameters,
                    style_tags=style_tags,
                )
            )
        except Exception as exc:
            logger.warning(
                "Color grading failed for shot %s (frame %d): %s — skipping",
                shot.id,
                actual_frame,
                exc,
            )
            continue

    cap.release()

    logger.info(
        "Color grading complete — %d/%d shots analysed",
        len(results),
        len(shots),
    )
    return results
