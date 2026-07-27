from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

import cv2
import ffmpeg
import numpy as np
from nanoid import generate
from tenacity import retry as tenacity_retry
from tenacity import stop_after_attempt, wait_fixed

from shared.types import TranscriptSegment

# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

_NANOID_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-"
_NANOID_SIZE = 21


def gen_id() -> str:
    """Return a UUID4 string — compatible with PostgreSQL UUID columns."""
    import uuid
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from *video_path* and write a mono 16 kHz WAV to *output_path*.

    Uses ffmpeg-python under the hood.  Overwrites *output_path* if it exists.

    Args:
        video_path: Absolute path to the source video file.
        output_path: Destination path for the extracted WAV file.

    Returns:
        *output_path* (convenience so callers can chain).
    """
    (
        ffmpeg.input(video_path)
        .output(
            output_path,
            ac=1,           # mono
            ar=16000,       # 16 kHz sample rate
            acodec="pcm_s16le",
            format="wav",
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    """Encode an OpenCV BGR frame as PNG and return the raw bytes.

    Args:
        frame: A numpy array in BGR format as returned by ``cv2.VideoCapture``.

    Returns:
        PNG-encoded bytes suitable for writing to disk or uploading to R2.
    """
    success, buffer = cv2.imencode(".png", frame)
    if not success:
        raise ValueError("cv2.imencode failed — frame may be malformed or empty.")
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

F = TypeVar("F")


def retry(max_attempts: int = 3, wait_seconds: float = 2.0) -> Callable[[F], F]:
    """Decorator that retries the wrapped function on any exception.

    Uses tenacity internally.  The final attempt exception is re-raised if all
    attempts are exhausted.

    Args:
        max_attempts: Total number of tries (including the first attempt).
        wait_seconds: Fixed delay in seconds between retries.

    Example::

        @retry(max_attempts=3, wait_seconds=1.5)
        def flaky_call() -> str:
            ...
    """
    def decorator(func: F) -> F:
        wrapped = tenacity_retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(wait_seconds),
            reraise=True,
        )(func)
        # Preserve the original function's metadata.
        functools.update_wrapper(wrapped, func)  # type: ignore[arg-type]
        return wrapped  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------


def get_transcript_snippet(
    segments: list[TranscriptSegment],
    timestamp: float,
    window: float = 10.0,
) -> str:
    """Return transcript text from segments within *window* seconds of *timestamp*.

    Selects all segments whose ``start_time`` falls within
    ``[timestamp - window, timestamp + window]`` and joins their ``text``
    fields with a single space.

    Args:
        segments: All transcript segments for a video or chunk.
        timestamp: The reference timestamp in seconds.
        window: Half-width of the search window in seconds (default 10 s).

    Returns:
        A single string of joined segment texts, or an empty string if no
        segments fall within the window.
    """
    if not segments:
        return ""
    lo = timestamp - window
    hi = timestamp + window
    matching = [seg.text for seg in segments if seg.start_time is not None and lo <= seg.start_time <= hi]
    if matching:
        return " ".join(matching)
    # Fallback: nearest segment when nothing falls in window
    valid = [seg for seg in segments if seg.start_time is not None]
    if not valid:
        return ""
    nearest = min(valid, key=lambda s: abs(s.start_time - timestamp))
    return nearest.text


# ---------------------------------------------------------------------------
# Progress tracking with ETA (logs to standard logger — works in Modal)
# ---------------------------------------------------------------------------


import time as _time


class ProgressTracker:
    """Thread-safe progress logger with ETA estimation.

    Usage::

        p = ProgressTracker("Qwen inference", total=213, logger=logger)
        for batch in batches:
            process(batch)
            p.update(len(batch))
        p.done()
    """

    def __init__(self, phase: str, total: int, logger, log_every: int = 1) -> None:
        self.phase = phase
        self.total = total
        self.logger = logger
        self.log_every = log_every
        self._done = 0
        self._start = _time.monotonic()
        self._last_log = -1
        logger.info("[PROGRESS] %s — starting (%d total)", phase, total)

    def update(self, n: int = 1) -> None:
        self._done += n
        pct = int(self._done * 100 / self.total) if self.total else 100
        # only log when pct crosses a log_every boundary (or 100%)
        bucket = pct // self.log_every
        if bucket != self._last_log or self._done >= self.total:
            self._last_log = bucket
            elapsed = _time.monotonic() - self._start
            if self._done > 0:
                eta = elapsed * (self.total - self._done) / self._done
                self.logger.info(
                    "[PROGRESS] %s — %d/%d (%d%%) | elapsed %.0fs | ETA %.0fs",
                    self.phase, self._done, self.total, pct, elapsed, eta,
                )
            else:
                self.logger.info(
                    "[PROGRESS] %s — %d/%d (%d%%) | elapsed %.0fs",
                    self.phase, self._done, self.total, pct, elapsed,
                )

    def done(self) -> None:
        elapsed = _time.monotonic() - self._start
        self.logger.info(
            "[PROGRESS] %s — DONE %d items in %.0fs (%.1f items/s)",
            self.phase, self.total, elapsed,
            self.total / elapsed if elapsed > 0 else 0,
        )


# ---------------------------------------------------------------------------
# Pipeline timing
# ---------------------------------------------------------------------------


class StepTimer:
    """Accumulate wall-clock timing for named pipeline steps.

    Usage::

        t = StepTimer()
        with t.step("model_load"):
            load_model()
        with t.step("inference"):
            run_inference()
        print(t.summary())          # {"model_load": 12.3, "inference": 45.6, "total_s": 57.9}
    """

    def __init__(self) -> None:
        self._steps: dict[str, float] = {}
        self._wall_start = _time.perf_counter()

    class _Step:
        def __init__(self, timer: "StepTimer", name: str) -> None:
            self._timer = timer
            self._name = name
            self._t0: float = 0.0

        def __enter__(self) -> "StepTimer._Step":
            self._t0 = _time.perf_counter()
            return self

        def __exit__(self, *_: object) -> None:
            elapsed = _time.perf_counter() - self._t0
            self._timer._steps[self._name] = round(elapsed, 2)

    def step(self, name: str) -> "_Step":
        return self._Step(self, name)

    def record(self, name: str, seconds: float) -> None:
        """Manually record a duration (e.g. when wrapping an external call)."""
        self._steps[name] = round(seconds, 2)

    def summary(self) -> dict:
        total = round(_time.perf_counter() - self._wall_start, 2)
        return {"steps": dict(self._steps), "total_s": total}

    def log(self, logger, label: str = "") -> None:
        """Print a formatted timing table to the given logger at INFO level."""
        summary = self.summary()
        steps = summary["steps"]
        total = summary["total_s"]

        header = f"⏱  TIMING — {label}" if label else "⏱  TIMING"
        sep = "─" * 46
        lines = [sep, header, sep]

        for name, secs in steps.items():
            if isinstance(secs, (int, float)):
                bar_len = min(int(secs / max(total, 1) * 20), 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                pct = secs / total * 100 if total > 0 else 0
                m, s = divmod(int(secs), 60)
                human = f"{m}m {s:02d}s" if m else f"{s}s"
                lines.append(f"  {name:<22} {bar}  {human:>6}  ({pct:4.1f}%)")

        m_tot, s_tot = divmod(int(total), 60)
        human_tot = f"{m_tot}m {s_tot:02d}s" if m_tot else f"{s_tot}s"
        lines += [sep, f"  {'TOTAL':<22} {'':20}  {human_tot:>6}", sep]
        logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------


def dataclass_to_dict(obj) -> dict:
    """Recursively convert a dataclass instance to a plain dict.

    Delegates to :func:`dataclasses.asdict` which handles nested dataclasses,
    lists, and dicts automatically.

    Args:
        obj: A dataclass instance.

    Returns:
        A plain Python dict representation of *obj*.
    """
    import dataclasses
    return dataclasses.asdict(obj)
