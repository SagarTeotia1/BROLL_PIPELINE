"""Level-2 background matting runner (CLAUDE.md PIPELINE ADDENDUM A1).

Per-shot alpha-matte inference, run in parallel with color grading (no
cut-order dependency — same reasoning as ``color_runner.py``, but over the
full shot span rather than a single midpoint frame; a matte needs temporal
context to stay stable, a color grade does not).

Uses Robust Video Matting (RVM) via ``torch.hub`` — this repo does not
vendor RVM or MODNet, so inference goes through a small ``MattingModel``
wrapper class rather than a bespoke per-call ``torch.hub.load``. The wrapper
keeps the model loaded once and reused across shots, matching the
``ColorGradingEngine`` pattern in ``color_runner.py`` (instantiate once,
call per shot).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from shared.config import settings
from shared.types import ShotMatteRecord, ShotRecord
from shared.utils import gen_id

logger = logging.getLogger(__name__)


def _model_version() -> str:
    """Stable identifier for the matting model+variant currently in use.

    Stored on every ``ShotMatteRecord`` so shots processed by different
    model versions are never ambiguous (rule 6 / PID-stability style
    reasoning, applied to model provenance instead of identity).
    """
    return f"rvm_{settings.MATTING_MODEL_VARIANT}"


class MattingModel:
    """Thin wrapper around a torch.hub Robust Video Matting model.

    Loaded once per process and reused across shots — mirrors
    ``ColorGradingEngine`` in ``color_runner.py`` and the pipeline object in
    ``face_runner.py`` (build once, call repeatedly, close/release at the end).
    """

    def __init__(self, device: str = "cuda", downsample_ratio: float = 0.25) -> None:
        self.device = device
        self.downsample_ratio = downsample_ratio
        self._model: Any = None

    def load(self) -> None:
        """Load the RVM model via ``torch.hub``.

        ``TORCH_HOME`` is already pointed at a persistent volume for the
        face/color Modal image (see ``modal_app.py`` ``face_color_image``),
        so the repo + checkpoint download only happens once per volume, not
        once per video.
        """
        if self._model is not None:
            return

        import torch  # type: ignore[import]

        logger.info(
            "Loading RVM model %s/%s (device=%s)",
            settings.MATTING_HUB_REPO,
            settings.MATTING_MODEL_VARIANT,
            self.device,
        )
        model = torch.hub.load(
            settings.MATTING_HUB_REPO,
            settings.MATTING_MODEL_VARIANT,
        )
        model = model.to(self.device).eval()
        self._model = model
        logger.info("RVM model loaded")

    def matte_clip(
        self,
        frames_bgr: list[Any],
        frame_stride: int = 1,
    ) -> list[Any]:
        """Run RVM over a sequence of BGR frames and return alpha mattes.

        Args:
            frames_bgr: List of HxWx3 uint8 BGR numpy arrays (as read via
                cv2), in temporal order for one shot.
            frame_stride: Only run real RVM inference every ``frame_stride``
                frames; skipped frames reuse the most recent computed alpha
                (nearest-neighbour hold, not interpolation — alpha changes
                slowly frame-to-frame for background-swap purposes, so a
                short hold is visually fine and far cheaper than inferring
                every native-fps frame). ``1`` (default) infers every frame,
                matching the original behaviour when the caller wants full
                precision. RVM's recurrent state (``rec``) still advances
                only on frames that actually run inference — the model's
                temporal context isn't stepped forward on held frames.

        Returns:
            List of HxW float32 numpy arrays (alpha in [0, 1]), same length
            and order as ``frames_bgr``.
        """
        if self._model is None:
            raise RuntimeError("MattingModel.load() must be called before matte_clip()")

        import cv2  # type: ignore[import]
        import numpy as np
        import torch  # type: ignore[import]

        alphas: list[Any] = []
        rec = [None] * 4  # RVM's recurrent state — reset per shot (independent clips)
        last_alpha: Any = None

        with torch.no_grad():
            for i, frame_bgr in enumerate(frames_bgr):
                if frame_stride > 1 and i % frame_stride != 0 and last_alpha is not None:
                    alphas.append(last_alpha)
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tensor = (
                    torch.from_numpy(frame_rgb)
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                    .to(self.device)
                )
                fgr, pha, *rec = self._model(tensor, *rec, downsample_ratio=self.downsample_ratio)
                alpha = pha[0, 0].detach().cpu().numpy().astype(np.float32)
                alphas.append(alpha)
                last_alpha = alpha

        return alphas

    def close(self) -> None:
        """Release the model reference so CUDA memory can be freed."""
        self._model = None


def _write_alpha_video(alphas: list[Any], fps: float, output_path: str) -> None:
    """Encode a sequence of float32 [0,1] alpha frames as a grayscale mp4.

    Uses ``cv2.VideoWriter`` (opencv-python-headless, already a base dep —
    see ``modal_app.py`` ``_BASE_PKGS``) with the same "no new heavy dep"
    discipline as the rest of Level-2.
    """
    import cv2  # type: ignore[import]
    import numpy as np

    if not alphas:
        raise ValueError("No alpha frames to write")

    h, w = alphas[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, max(fps, 1.0), (w, h), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open output path: {output_path!r}")

    try:
        for alpha in alphas:
            frame_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            writer.write(frame_u8)
    finally:
        writer.release()


def run_background_matting(
    video_path: str,
    video_id: str,
    shots: list[ShotRecord],
    device: str = "cuda",
    frame_stride: int | None = None,
) -> list[ShotMatteRecord]:
    """Run background matting for every shot and upload each alpha matte to R2.

    Follows the exact shape of ``run_color_grading``: same signature order
    (video_path, video_id, shots), same "best-effort per shot, log + skip on
    failure rather than aborting the whole run" error handling, same
    single-engine-instance-reused-across-shots pattern.

    Args:
        video_path: Absolute path to the video file.
        video_id:   Stable identifier for this video in the knowledge base.
        shots:      Shot records produced by the Level-1 shot-detection step.
        device:     torch device string. Always "cuda" on Modal L40S; "cpu"
                    is accepted for local smoke tests only.
        frame_stride: Only run real RVM inference every Nth frame per shot
                    (see ``MattingModel.matte_clip``). Defaults to
                    ``settings.MATTING_FRAME_STRIDE`` when None — real-video
                    measurement showed unbatched, every-native-fps-frame
                    inference over a full shot list is the dominant L2 cost
                    (near-zero GPU utilization, Python-loop-bound single-frame
                    calls), so a default > 1 is load-bearing, not cosmetic.

    Returns:
        A list of :class:`ShotMatteRecord`, one per successfully matted
        shot. Shots whose frame range cannot be read, or that fail
        inference, are skipped with a warning rather than crashing the
        pipeline (matches ``run_color_grading``'s per-shot resilience).
    """
    from knowledge_base.r2.uploader import upload_shot_matte

    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        logger.error("cv2 (OpenCV) is not installed: %s", exc)
        raise

    if frame_stride is None:
        frame_stride = settings.MATTING_FRAME_STRIDE

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open video: {video_path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model = MattingModel(device=device)
    try:
        model.load()
    except Exception as exc:
        logger.warning(
            "Could not load background matting model (%s) — "
            "returning no shot mattes for video %s.",
            exc,
            video_id,
        )
        return []

    model_version = _model_version()
    results: list[ShotMatteRecord] = []

    logger.info(
        "Background matting ready (model_version=%s) — matting %d shots from %s",
        model_version,
        len(shots),
        video_path,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open video: {video_path!r}")

    try:
        for shot in shots:
            start_frame = shot.start_frame
            end_frame = shot.end_frame

            if start_frame is None or end_frame is None:
                logger.warning(
                    "Shot %s has no frame range (start=%s, end=%s) — skipping matting",
                    shot.id,
                    start_frame,
                    end_frame,
                )
                continue

            clamped_end = min(end_frame, total_frames - 1)
            if clamped_end < start_frame:
                logger.warning(
                    "Shot %s frame range invalid after clamping (start=%d, end=%d) — skipping",
                    shot.id,
                    start_frame,
                    clamped_end,
                )
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames_bgr: list[Any] = []
            for _ in range(start_frame, clamped_end + 1):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames_bgr.append(frame)

            if not frames_bgr:
                logger.warning(
                    "Could not read any frames for shot %s (frames %d-%d) — skipping matting",
                    shot.id,
                    start_frame,
                    clamped_end,
                )
                continue

            try:
                alphas = model.matte_clip(frames_bgr, frame_stride=frame_stride)
            except Exception as exc:
                logger.warning(
                    "Matting inference failed for shot %s (%d frames): %s — skipping",
                    shot.id,
                    len(frames_bgr),
                    exc,
                )
                continue

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                _write_alpha_video(alphas, fps, tmp_path)

                r2_key = upload_shot_matte(tmp_path, video_id, shot.id)

                results.append(
                    ShotMatteRecord(
                        id=gen_id(),
                        video_id=video_id,
                        shot_id=shot.id,
                        r2_key=r2_key,
                        model_version=model_version,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to encode/upload matte for shot %s: %s — skipping",
                    shot.id,
                    exc,
                )
                continue
            finally:
                if tmp_path is not None:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except OSError:
                        pass
    finally:
        cap.release()
        model.close()

    logger.info(
        "Background matting complete — %d/%d shots matted",
        len(results),
        len(shots),
    )
    return results
