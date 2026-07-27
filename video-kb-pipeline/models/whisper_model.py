"""faster-whisper wrapper.

Loads the Whisper model once and exposes a ``transcribe`` method that returns
a list of :class:`shared.types.TranscriptSegment` objects, one per sentence /
phrase emitted by Whisper.
"""
from __future__ import annotations

import logging
from typing import Literal

from faster_whisper import WhisperModel as FasterWhisperModel

from shared.types import TranscriptSegment
from shared.utils import gen_id

logger = logging.getLogger(__name__)


class WhisperModel:
    """Thin wrapper around :class:`faster_whisper.WhisperModel`.

    The underlying model is loaded once at construction time and reused across
    all ``transcribe`` calls, so the GPU/CPU memory cost is paid only once per
    process.

    Parameters
    ----------
    device:
        ``"cuda"`` to run on GPU (requires CUDA-enabled PyTorch) or ``"cpu"``
        for CPU-only inference.  Defaults to ``"cuda"``.
    model_size_or_path:
        Whisper model variant to load (e.g. ``"large-v3"``, ``"base"``) or an
        absolute path to a locally downloaded model directory.  ``"large-v3"``
        on GPU gives the highest accuracy; ``"base"`` is recommended for CPU to
        keep inference fast.  Defaults to ``"large-v3"``.
    compute_type:
        Quantisation precision.  ``"float16"`` is the default for GPU
        throughput; use ``"int8"`` for lower VRAM usage or ``"float32"`` for
        CPU.  Defaults to ``"float16"``.
    """

    def __init__(
        self,
        device: Literal["cuda", "cpu"] = "cuda",
        model_size_or_path: str = "large-v3",
        compute_type: str = "float16",
    ) -> None:
        logger.info(
            "Loading Whisper model '%s' on %s (compute_type=%s).",
            model_size_or_path,
            device,
            compute_type,
        )
        self._model = FasterWhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
        )
        self._device = device
        self._model_size = model_size_or_path
        logger.info("Whisper model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str, video_id: str) -> list[TranscriptSegment]:
        """Transcribe *audio_path* and return one segment per phrase.

        Language is auto-detected by Whisper on the first 30 seconds of audio.
        Word-level timestamps are requested so callers can perform fine-grained
        alignment later.

        Parameters
        ----------
        audio_path:
            Path to a WAV (or any ffmpeg-readable) audio file.
        video_id:
            The parent video's identifier; embedded into every returned
            :class:`~shared.types.TranscriptSegment`.

        Returns
        -------
        list[TranscriptSegment]
            One entry per segment emitted by Whisper.  Each segment's
            ``chunk_id`` is set to ``None`` — the downstream aligner is
            responsible for assigning chunk membership.
        """
        logger.info("Starting Whisper transcription for video_id=%s, audio=%s", video_id, audio_path)

        segments_iter, info = self._model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5,
            language=None,  # auto-detect
        )

        logger.info(
            "Detected language '%s' (probability=%.3f).",
            info.language,
            info.language_probability,
        )

        result: list[TranscriptSegment] = []

        for seg in segments_iter:
            # Build the word-level list.  faster-whisper exposes words as
            # NamedTuple objects with .word, .start, .end, .probability.
            words: list[dict] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        {
                            "word": w.word,
                            "start": float(w.start),
                            "end": float(w.end),
                            "probability": float(w.probability),
                        }
                    )

            # Confidence = average word probability for this segment.
            # Fall back to None when words are absent (should not happen with
            # word_timestamps=True, but guard defensively).
            if words:
                confidence: float | None = sum(w["probability"] for w in words) / len(words)
            else:
                confidence = None

            transcript_segment = TranscriptSegment(
                id=gen_id(),
                video_id=video_id,
                chunk_id=None,  # set later by aligner
                text=seg.text.strip(),
                start_time=float(seg.start),
                end_time=float(seg.end),
                confidence=confidence,
                words=words,
            )
            result.append(transcript_segment)

        logger.info(
            "Transcription complete: %d segments for video_id=%s.",
            len(result),
            video_id,
        )
        return result
