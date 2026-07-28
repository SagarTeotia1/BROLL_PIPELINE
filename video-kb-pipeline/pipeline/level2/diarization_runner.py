"""Level-2 Stage 0: speaker diarization runner.

Extracts audio from the source video and runs pyannote.audio speaker
diarization to produce raw speech turns (cluster label + time range). Output
is *not* identity-resolved yet — cluster labels ("SPEAKER_00", ...) are
arbitrary per-run IDs. Fusion with ArcFace identity happens in
:mod:`pipeline.level2.fusion`.
"""
from __future__ import annotations

import logging
import os

from shared.config import settings
from shared.utils import extract_audio

logger = logging.getLogger(__name__)


def run_diarization(
    video_path: str,
    video_id: str,
    tmp_dir: str,
    num_speakers: int | None = None,
) -> list[dict]:
    """Run speaker diarization on the audio track of *video_path*.

    Parameters
    ----------
    video_path:
        Absolute path to the source video file.
    video_id:
        Identifier of the video — used only for logging here.
    tmp_dir:
        Directory in which to write the temporary WAV file. Must exist and
        be writable; the function does *not* create it.
    num_speakers:
        Known cast size, when available (e.g. from the user-provided cast
        list, or the ArcFace person count from Level-2 Pass 1). Passing this
        explicitly is far more reliable than letting pyannote guess the
        speaker count — it removes the single biggest source of diarization
        error (over/under-clustering). Pass ``None`` only when truly unknown.

    Returns
    -------
    list[dict]
        Raw turns sorted by start time, each
        ``{"cluster_label": str, "start_time": float, "end_time": float}``.
        Empty list if diarization fails or audio has no detected speech —
        callers must treat this as non-fatal (L2 continues without
        speaker_turns; L5 falls back to face-only attribution).
    """
    audio_path = os.path.join(tmp_dir, "diarization_audio.wav")

    logger.info(
        "Diarization: extracting audio for video_id=%s from %s → %s "
        "(num_speakers=%s)",
        video_id, video_path, audio_path, num_speakers,
    )

    try:
        try:
            extract_audio(video_path, audio_path)
        except Exception as exc:
            logger.warning(
                "Diarization audio extraction failed for video_id=%s: %s — "
                "returning no speaker turns.", video_id, exc,
            )
            return []

        try:
            from pyannote.audio import Pipeline  # type: ignore[import]
        except Exception as exc:
            # Catches ImportError (not installed) as well as torch/torchvision
            # import-time failures (e.g. a concurrent first-import race with
            # another thread — see the pre-warm import in modal_app.py).
            logger.warning(
                "pyannote.audio import failed (%s) — returning no speaker "
                "turns for this run.", exc,
            )
            return []

        # pyannote.audio renamed Pipeline.from_pretrained's auth kwarg from
        # use_auth_token= to token= in newer releases (following
        # huggingface_hub's own rename). Try the current name first, fall
        # back to the old one so this works across pinned versions.
        try:
            try:
                pipeline = Pipeline.from_pretrained(
                    settings.PYANNOTE_MODEL, token=settings.HF_TOKEN
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    settings.PYANNOTE_MODEL, use_auth_token=settings.HF_TOKEN
                )
        except Exception as exc:
            logger.warning(
                "Could not load pyannote pipeline %r (%s) — returning no "
                "speaker turns. Check HF_TOKEN has accepted the model license.",
                settings.PYANNOTE_MODEL, exc,
            )
            return []

        try:
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception as exc:
            logger.warning("Could not move pyannote pipeline to GPU (%s) — running on CPU.", exc)

        try:
            # Load audio ourselves into a waveform tensor rather than passing
            # audio_path directly — pyannote's default file-path loading goes
            # through torchcodec, which needs CUDA 13's libnvrtc.so.13 (not
            # present in this CUDA 12.4 image) and has no fallback decoder in
            # this pyannote version. torchaudio.load() sidesteps torchcodec
            # entirely (uses soundfile/sox backend), which is the workaround
            # pyannote's own warning suggests.
            import torchaudio  # type: ignore[import]
            waveform, sample_rate = torchaudio.load(audio_path)

            kwargs = {}
            if num_speakers is not None and num_speakers > 0:
                kwargs["num_speakers"] = num_speakers
            diarization = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}, **kwargs
            )
        except Exception as exc:
            logger.warning(
                "pyannote diarization failed for video_id=%s (%s) — "
                "returning no speaker turns.", video_id, exc,
            )
            return []

        # pyannote.audio 4.x pipelines (e.g. the community-1 pipeline behind
        # speaker-diarization-3.1) return a DiarizeOutput wrapper whose
        # .speaker_diarization attribute is the pyannote.core.Annotation —
        # older pipelines return the Annotation directly. Unwrap either way.
        annotation = getattr(diarization, "speaker_diarization", diarization)

        turns: list[dict] = []
        for segment, _track, speaker_label in annotation.itertracks(yield_label=True):
            turns.append({
                "cluster_label": str(speaker_label),
                "start_time": round(float(segment.start), 3),
                "end_time": round(float(segment.end), 3),
            })
        turns.sort(key=lambda t: t["start_time"])

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError as rm_err:
                logger.warning(
                    "Could not remove temporary diarization audio file %s: %s",
                    audio_path, rm_err,
                )

    n_speakers_found = len({t["cluster_label"] for t in turns})
    logger.info(
        "Diarization runner complete for video_id=%s: %d turns, %d distinct clusters.",
        video_id, len(turns), n_speakers_found,
    )
    return turns
