"""Video player widget and its decoding thread.

Playback is deliberately **independent** of the analysis pipeline: a separate
:class:`VideoSource` decodes at wall-clock pace on its own thread and hands frames to
the widget through a Qt signal. If inference falls behind, the video keeps playing and
the overlay simply lags - it never stutters.

Overlay boxes come from the most recent :class:`~pipeline.types.FrameResult` whose
timestamp is at or before the playhead, kept in a short ring buffer.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from emotion.hsemotion import EMOTION_COLORS
from pipeline.frame_source import VideoSource, probe
from pipeline.types import FrameResult
from utils.logging_utils import get_logger

log = get_logger(__name__)


class PlaybackController(QObject):
    """Decodes a video on a worker thread and emits frames at wall-clock pace."""

    frameReady = Signal(object, float)   # (BGR ndarray, timestamp seconds)
    positionChanged = Signal(float)
    durationChanged = Signal(float)
    stateChanged = Signal(bool)          # True = playing
    ended = Signal()

    def __init__(self, preview_width: int = 960, fps_cap: int = 60) -> None:
        super().__init__()
        self.preview_width = preview_width
        self.fps_cap = max(1, fps_cap)
        self._path: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._playing = threading.Event()
        self._seek_to: Optional[float] = None
        self._lock = threading.Lock()
        self.duration = 0.0
        self.fps = 30.0
        self.position = 0.0

    # -- control ------------------------------------------------------------
    def load(self, path: str | Path) -> None:
        """Open a file, show its first frame and stay paused."""
        self.stop()
        meta = probe(path)
        self._path = str(path)
        self.duration = meta.duration
        self.fps = meta.fps or 30.0
        self.position = 0.0
        self.durationChanged.emit(self.duration)
        self._stop.clear()
        self._playing.clear()
        self._thread = threading.Thread(target=self._run, name="PlaybackThread", daemon=True)
        self._thread.start()

    def play(self) -> None:
        if self._path is None:
            return
        self._playing.set()
        self.stateChanged.emit(True)

    def pause(self) -> None:
        self._playing.clear()
        self.stateChanged.emit(False)

    def toggle(self) -> None:
        if self._playing.is_set():
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._seek_to = max(0.0, min(seconds, max(0.0, self.duration - 0.05)))

    def stop(self) -> None:
        self._stop.set()
        self._playing.set()          # release the pause wait so the thread can exit
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._playing.clear()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set() and not self._stop.is_set()

    # -- worker -------------------------------------------------------------
    def _run(self) -> None:
        assert self._path is not None
        source: Optional[VideoSource] = None
        try:
            source = VideoSource(
                self._path, backend="auto", hwaccel=True, threads=2,
                # Scale to preview size during colour conversion: playback then costs a
                # fraction of a full-resolution decode and never competes with analysis.
                target_long_side=self.preview_width,
            )
            iterator = source.frames()
            # Show the very first frame immediately so the panel is never blank.
            first = next(iterator, None)
            if first is not None:
                self._emit(first.image, first.timestamp)

            wall_start = time.perf_counter()
            media_start = first.timestamp if first is not None else 0.0
            min_interval = 1.0 / self.fps_cap

            while not self._stop.is_set():
                pending_seek = self._take_seek()
                if pending_seek is not None:
                    source.seek(pending_seek)
                    iterator = source.frames()
                    wall_start = time.perf_counter()
                    media_start = pending_seek
                    frame = next(iterator, None)
                    if frame is not None:
                        self._emit(frame.image, frame.timestamp)
                        media_start = frame.timestamp
                    continue

                if not self._playing.is_set():
                    time.sleep(0.02)
                    wall_start = time.perf_counter()
                    media_start = self.position
                    continue

                frame = next(iterator, None)
                if frame is None:
                    self.ended.emit()
                    self._playing.clear()
                    self.stateChanged.emit(False)
                    time.sleep(0.05)
                    continue

                target = wall_start + (frame.timestamp - media_start)
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, 0.5))
                elif delay < -0.25:
                    # Far behind (heavy GPU load): resync instead of racing ahead.
                    wall_start = time.perf_counter()
                    media_start = frame.timestamp
                self._emit(frame.image, frame.timestamp)
                time.sleep(min_interval * 0.1)
        except BaseException as exc:  # noqa: BLE001
            log.exception("Playback thread failed: %s", exc)
        finally:
            if source is not None:
                source.close()

    def _take_seek(self) -> Optional[float]:
        with self._lock:
            value, self._seek_to = self._seek_to, None
        return value

    def _emit(self, image: np.ndarray, timestamp: float) -> None:
        h, w = image.shape[:2]
        if w > self.preview_width:
            scale = self.preview_width / w
            image = cv2.resize(
                image, (self.preview_width, int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self.position = timestamp
        self.frameReady.emit(np.ascontiguousarray(image), timestamp)
        self.positionChanged.emit(timestamp)


class VideoWidget(QWidget):
    """Paints the current frame plus detection overlays."""

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(True)

        self._image: Optional[QImage] = None
        self._buffer: Optional[np.ndarray] = None      # keeps QImage memory alive
        self._source_size: Tuple[int, int] = (0, 0)    # (w, h) of the analysed frame
        self._results: Deque[FrameResult] = deque(maxlen=48)
        self._timestamp = 0.0
        self.show_overlay = True

    # -- input --------------------------------------------------------------
    def set_frame(self, image: np.ndarray, timestamp: float) -> None:
        """Display a BGR frame (called on the GUI thread via a queued signal)."""
        self._buffer = image
        h, w = image.shape[:2]
        self._image = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_BGR888)
        self._timestamp = timestamp
        self.update()

    def set_video_size(self, width: int, height: int) -> None:
        """Native resolution, used to map analysis coordinates onto the preview."""
        self._source_size = (width, height)

    def add_result(self, result: FrameResult) -> None:
        """Store an analysis result for overlay lookup."""
        self._results.append(result)

    def clear(self) -> None:
        self._image = None
        self._buffer = None
        self._results.clear()
        self.update()

    # -- painting -----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d0f13"))
        if self._image is None:
            painter.setPen(QColor("#5a626e"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Open a video to begin\n(File -> Open Video)",
            )
            painter.end()
            return

        target = self._fit_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, self._image)

        if self.show_overlay:
            self._draw_overlay(painter, target)
        painter.end()

    def _fit_rect(self) -> QRectF:
        assert self._image is not None
        iw, ih = self._image.width(), self._image.height()
        if iw <= 0 or ih <= 0:
            return QRectF(self.rect())
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def _current_result(self) -> Optional[FrameResult]:
        best: Optional[FrameResult] = None
        for res in self._results:
            if res.timestamp <= self._timestamp + 0.12:
                if best is None or res.timestamp > best.timestamp:
                    best = res
        if best is None and self._results:
            best = self._results[-1]
        if best is not None and abs(best.timestamp - self._timestamp) > 1.5:
            return None
        return best

    def _draw_overlay(self, painter: QPainter, target: QRectF) -> None:
        result = self._current_result()
        if result is None or not result.faces:
            return
        src_w, src_h = self._source_size
        if src_w <= 0 or src_h <= 0:
            return
        sx = target.width() / src_w
        sy = target.height() / src_h

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for face in result.faces:
            x1, y1, x2, y2 = (float(v) for v in face.bbox[:4])
            rect = QRectF(
                target.left() + x1 * sx,
                target.top() + y1 * sy,
                max(2.0, (x2 - x1) * sx),
                max(2.0, (y2 - y1) * sy),
            )
            color = QColor(EMOTION_COLORS.get(face.emotion, "#8a8f98"))
            if face.actor_id < 0:
                color = QColor("#6b7684")
            pen = QPen(color, 2.0)
            painter.setPen(pen)
            painter.drawRect(rect)

            label = f"{face.actor_name}"
            if face.actor_id >= 0:
                label += f" {face.similarity:.2f}"
            label += f" | {face.emotion} {face.emotion_confidence:.2f}"
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(label) + 10
            th = metrics.height() + 4
            box = QRectF(rect.left(), max(target.top(), rect.top() - th), tw, th)
            painter.fillRect(box, color)
            painter.setPen(QColor("#0d0f13"))
            painter.drawText(box.adjusted(5, 1, 0, 0), Qt.AlignmentFlag.AlignVCenter, label)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.clicked.emit()
        super().mousePressEvent(event)


__all__ = ["VideoWidget", "PlaybackController"]
