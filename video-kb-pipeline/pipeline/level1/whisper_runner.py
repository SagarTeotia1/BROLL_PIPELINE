"""Whisper transcription runner for the Level 1 video understanding pipeline.

Extracts audio from a video file and produces a list of
:class:`~shared.types.TranscriptSegment` objects via :mod:`models.whisper_model`.
"""
from __future__ import annotations

import logging
import os

from models.whisper_model import WhisperModel
from shared.types import TranscriptSegment
from shared.utils import extract_audio

logger = logging.getLogger(__name__)


def run_whisper(
    video_path: str,
    video_id: str,
    tmp_dir: str,
) -> list[TranscriptSegment]:
    """Transcribe the audio track of *video_path* using Whisper.

    Extracts a temporary mono 16 kHz WAV file, runs Whisper transcription,
    then removes the temporary audio file before returning.

    Parameters
    ----------
    video_path:
        Absolute path to the source video file.
    video_id:
        Identifier of the video; embedded into every returned
        :class:`~shared.types.TranscriptSegment` via
        :meth:`~models.whisper_model.WhisperModel.transcribe`.
    tmp_dir:
        Directory in which to write the temporary WAV file.  Must exist
        and be writable; the function does *not* create it.

    Returns
    -------
    list[TranscriptSegment]
        One segment per phrase/sentence from Whisper.  Each segment's
        ``chunk_id`` is ``None`` — call :func:`pipeline.level1.aligner.align_transcript_to_chunks`
        afterwards to assign chunk membership.
    """
    audio_path = os.path.join(tmp_dir, "audio.wav")

    logger.info(
        "Extracting audio for video_id=%s from %s → %s",
        video_id,
        video_path,
        audio_path,
    )

    try:
        try:
            extract_audio(video_path, audio_path)
        except Exception as exc:
            raise RuntimeError(
                f"Audio extraction failed for video_id={video_id}: {exc}"
            ) from exc

        logger.info("Audio extracted. Loading Whisper model and transcribing.")

        try:
            model = WhisperModel(device="cuda", compute_type="float16")
            segments = model.transcribe(audio_path, video_id)
        except Exception as exc:
            raise RuntimeError(
                f"Whisper transcription failed for video_id={video_id}: {exc}"
            ) from exc
    finally:
        # Always clean up the temporary WAV — even on audio-extraction failure
        # (ffmpeg may have written a partial file) and on transcription failure.
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.debug("Removed temporary audio file: %s", audio_path)
            except OSError as rm_err:
                logger.warning(
                    "Could not remove temporary audio file %s: %s",
                    audio_path,
                    rm_err,
                )

    logger.info(
        "Whisper runner complete for video_id=%s: %d segments.",
        video_id,
        len(segments),
    )
    return segments
