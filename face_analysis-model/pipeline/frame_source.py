"""Video decoding: PyAV (with NVDEC when available) and an OpenCV fallback.

Two independent consumers need frames:

* the **analysis pipeline**, which wants every frame as fast as the disk allows,
* the **GUI player**, which wants frames at wall-clock pace and must be able to seek.

Both are served by :class:`VideoSource`, which is a plain iterator, plus
:class:`DecodeThread`, which pushes into a bounded queue so the decoder can run ahead
without ever blocking playback.

Hardware decode: PyAV is opened with a CUDA hwaccel context; if the codec or driver
refuses, it silently falls back to multi-threaded software decoding, then to OpenCV.
The active path is reported in :attr:`VideoSource.decoder_name` and lands in the output
JSON so a benchmark result is never ambiguous about what decoded the file.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2

from pipeline.types import RawFrame
from utils.logging_utils import get_logger

log = get_logger(__name__)

try:  # PyAV is optional at runtime; OpenCV covers every case functionally.
    import av  # type: ignore

    _HAS_AV = True
except Exception:  # pragma: no cover - depends on install
    av = None  # type: ignore
    _HAS_AV = False


@dataclass
class VideoMetadata:
    """Container/stream properties probed before decoding starts."""

    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration: float = 0.0
    codec: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "duration": round(self.duration, 3),
            "codec": self.codec,
        }


def probe(path: str | Path) -> VideoMetadata:
    """Read metadata without decoding the whole file.

    PyAV is preferred (it knows the real duration and codec); OpenCV is the fallback.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    if _HAS_AV:
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate or stream.guessed_rate or 0.0)
                duration = 0.0
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
                elif container.duration is not None:
                    duration = container.duration / 1_000_000.0
                frames = int(stream.frames or 0)
                if frames <= 0 and fps > 0 and duration > 0:
                    frames = int(round(duration * fps))
                return VideoMetadata(
                    path=str(path),
                    width=int(stream.codec_context.width),
                    height=int(stream.codec_context.height),
                    fps=fps,
                    frame_count=frames,
                    duration=duration,
                    codec=str(stream.codec_context.name),
                )
        except Exception as exc:  # pragma: no cover - corrupt files
            log.warning("PyAV probe failed (%s); falling back to OpenCV", exc)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta = VideoMetadata(
        path=str(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps,
        frame_count=frames,
        duration=(frames / fps) if fps > 0 else 0.0,
        codec="unknown",
    )
    cap.release()
    return meta


class VideoSource:
    """Iterable frame source with an explicit decoder backend.

    Args:
        path: video file.
        backend: ``"auto" | "pyav" | "opencv"``.
        hwaccel: try NVDEC before software decoding.
        threads: PyAV software-decode thread count.
        start_time: seek here before the first frame (seconds).
        target_long_side: when set and smaller than the source, frames are downscaled
            **during** colour conversion (swscale, in C) instead of being converted at
            full resolution and resized afterwards. On a decode-bound pipeline this is
            the single biggest win available: a 1080p clip converted straight to 1280
            wide costs roughly half as much as convert-then-resize.
    """

    def __init__(
        self,
        path: str | Path,
        backend: str = "auto",
        hwaccel: bool = True,
        threads: int = 4,
        start_time: float = 0.0,
        target_long_side: Optional[int] = None,
    ) -> None:
        self.path = str(path)
        self.meta = probe(self.path)
        self.backend = backend
        self.hwaccel = hwaccel
        self.threads = max(1, threads)
        self.start_time = max(0.0, start_time)
        self.decoder_name = "unset"
        self.output_size: Optional[Tuple[int, int]] = None
        self.output_scale = 1.0
        self._configure_output(target_long_side)
        self._container = None
        self._stream = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _configure_output(self, target_long_side: Optional[int]) -> None:
        """Decide the decoder's output resolution and the coordinate scale factor."""
        source_long = max(self.meta.width, self.meta.height)
        if not target_long_side or source_long <= 0 or source_long <= target_long_side:
            self.output_size = None
            self.output_scale = 1.0
            return
        factor = target_long_side / source_long
        # Keep both dimensions even: some swscale paths dislike odd sizes for YUV.
        width = max(2, int(round(self.meta.width * factor)) // 2 * 2)
        height = max(2, int(round(self.meta.height * factor)) // 2 * 2)
        self.output_size = (width, height)
        self.output_scale = self.meta.width / width
        log.info(
            "Decoder will emit %dx%d (from %dx%d, scale %.3f)",
            width, height, self.meta.width, self.meta.height, self.output_scale,
        )

    # -- open / close -------------------------------------------------------
    def _open(self) -> None:
        want_av = _HAS_AV and self.backend in ("auto", "pyav")
        if want_av and self._open_pyav():
            return
        if self.backend == "pyav" and not _HAS_AV:
            log.warning("PyAV requested but not installed; using OpenCV")
        self._open_opencv()

    def _open_pyav(self) -> bool:
        # 1) hardware decode
        if self.hwaccel:
            try:
                hw = av.codec.hwaccel.HWAccel(device_type="cuda", allow_software_fallback=True)
                self._container = av.open(self.path, hwaccel=hw)
                self._stream = self._container.streams.video[0]
                self._stream.thread_type = "AUTO"
                self.decoder_name = f"pyav+nvdec({self._stream.codec_context.name})"
                log.info("Decoder: NVDEC via PyAV (%s)", self._stream.codec_context.name)
                self._seek_initial()
                return True
            except Exception as exc:
                log.info("NVDEC unavailable (%s); using threaded software decode", exc)
                self._safe_close_container()

        # 2) threaded software decode
        try:
            self._container = av.open(self.path)
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
            self._stream.codec_context.thread_count = self.threads
            self.decoder_name = f"pyav+sw({self._stream.codec_context.name})"
            log.info("Decoder: PyAV software, %d threads", self.threads)
            self._seek_initial()
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("PyAV open failed (%s); using OpenCV", exc)
            self._safe_close_container()
            return False

    def _open_opencv(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")
        if self.start_time > 0:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time * 1000.0)
        self.decoder_name = "opencv"
        log.info("Decoder: OpenCV VideoCapture")

    def _seek_initial(self) -> None:
        if self.start_time > 0:
            self.seek(self.start_time)

    def _safe_close_container(self) -> None:
        try:
            if self._container is not None:
                self._container.close()
        except Exception:
            pass
        self._container = None
        self._stream = None

    # -- iteration ----------------------------------------------------------
    def __iter__(self) -> Iterator[RawFrame]:
        return self.frames()

    def frames(self) -> Iterator[RawFrame]:
        """Yield every frame from the current position to the end."""
        if self._container is not None:
            yield from self._frames_pyav()
        else:
            yield from self._frames_opencv()

    def _frames_pyav(self) -> Iterator[RawFrame]:
        assert self._container is not None and self._stream is not None
        time_base = float(self._stream.time_base) if self._stream.time_base else 0.0
        fps = self.meta.fps or 30.0
        index = int(round(self.start_time * fps))
        size = self.output_size
        for packet_frame in self._container.decode(video=0):
            if packet_frame.pts is not None and time_base:
                ts = float(packet_frame.pts * time_base)
            else:
                ts = index / fps
            # Downscaling here means swscale does it while it is already touching the
            # pixels for the YUV->BGR conversion; doing it afterwards would cost a
            # full-resolution buffer plus a second pass.
            if size is not None:
                image = packet_frame.to_ndarray(format="bgr24", width=size[0], height=size[1])
            else:
                image = packet_frame.to_ndarray(format="bgr24")
            yield RawFrame(
                index=index,
                timestamp=ts,
                image=image,
                is_keyframe=bool(getattr(packet_frame, "key_frame", False)),
                scale=self.output_scale,
            )
            index += 1

    def _frames_opencv(self) -> Iterator[RawFrame]:
        assert self._cap is not None
        fps = self.meta.fps or 30.0
        size = self.output_size
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            index = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            ts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = (ts_ms / 1000.0) if ts_ms and ts_ms > 0 else index / fps
            if size is not None:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            yield RawFrame(
                index=max(index, 0), timestamp=ts, image=image, scale=self.output_scale
            )

    # -- random access ------------------------------------------------------
    def seek(self, seconds: float) -> None:
        """Seek to (approximately) the given time; the next frame follows it."""
        seconds = max(0.0, seconds)
        if self._container is not None and self._stream is not None:
            try:
                time_base = self._stream.time_base
                target = int(seconds / float(time_base)) if time_base else int(seconds)
                self._container.seek(target, stream=self._stream, backward=True, any_frame=False)
                return
            except Exception as exc:  # pragma: no cover
                log.warning("PyAV seek failed (%s)", exc)
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)

    def read_at(self, seconds: float) -> Optional[RawFrame]:
        """Decode a single frame at the given timestamp (used for thumbnails)."""
        self.seek(seconds)
        for frame in self.frames():
            return frame
        return None

    def close(self) -> None:
        self._safe_close_container()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DecodeThread(threading.Thread):
    """Producer thread pushing :class:`RawFrame` objects into a bounded queue.

    ``drop_when_full=False`` (analysis) gives back-pressure - the decoder waits, which
    keeps memory flat. ``drop_when_full=True`` (preview) discards the oldest frame so a
    slow consumer can never stall playback.
    """

    SENTINEL = None

    def __init__(
        self,
        source: VideoSource,
        out_queue: "queue.Queue[Optional[RawFrame]]",
        drop_when_full: bool = False,
        name: str = "DecodeThread",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.source = source
        self.out_queue = out_queue
        self.drop_when_full = drop_when_full
        # NB: must not be called ``_stop`` - that name shadows ``threading.Thread._stop``
        # and breaks ``join()`` with "'Event' object is not callable".
        self._stop_event = threading.Event()
        self.frames_decoded = 0
        self.frames_dropped = 0
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            for frame in self.source.frames():
                if self._stop_event.is_set():
                    break
                self.frames_decoded += 1
                if self.drop_when_full:
                    self._put_dropping(frame)
                else:
                    while not self._stop_event.is_set():
                        try:
                            self.out_queue.put(frame, timeout=0.2)
                            break
                        except queue.Full:
                            continue
        except BaseException as exc:  # noqa: BLE001 - reported to the owner
            self.error = exc
            log.exception("Decoder thread failed: %s", exc)
        finally:
            try:
                self.out_queue.put(self.SENTINEL, timeout=1.0)
            except queue.Full:  # pragma: no cover
                pass

    def _put_dropping(self, frame: RawFrame) -> None:
        try:
            self.out_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.out_queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(frame)
            except queue.Full:  # pragma: no cover
                self.frames_dropped += 1

    def stop(self) -> None:
        """Ask the decoder to stop after the current frame."""
        self._stop_event.set()


__all__ = ["VideoMetadata", "VideoSource", "DecodeThread", "probe"]
