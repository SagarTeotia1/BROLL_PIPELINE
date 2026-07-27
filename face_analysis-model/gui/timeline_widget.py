"""Bottom timeline: one lane per actor, coloured by emotion, click to seek.

Custom-painted rather than built from widgets - a two-hour clip can produce thousands of
events and a widget per event would be unusable. Painting is O(visible events) and the
whole thing repaints in well under a millisecond.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from emotion.hsemotion import EMOTION_COLORS
from timeline.events import TimelineEvent

LANE_HEIGHT = 26
LANE_GAP = 4
LABEL_WIDTH = 132
RULER_HEIGHT = 20


class TimelineWidget(QWidget):
    """Interactive emotion timeline."""

    seekRequested = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(RULER_HEIGHT + LANE_HEIGHT + 16)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)

        self._events: List[TimelineEvent] = []
        self._lanes: Dict[str, int] = {}
        self.duration: float = 1.0
        self.playhead: float = 0.0

    # -- data ---------------------------------------------------------------
    def set_duration(self, seconds: float) -> None:
        self.duration = max(0.001, seconds)
        self.update()

    def add_event(self, event: TimelineEvent) -> None:
        """Append one finalised event (called as the pipeline closes them)."""
        if event.actor not in self._lanes:
            self._lanes[event.actor] = len(self._lanes)
            self._resize_for_lanes()
        self._events.append(event)
        self.update()

    def set_events(self, events: List[TimelineEvent]) -> None:
        """Replace the whole timeline (e.g. after loading a saved JSON)."""
        self._events = list(events)
        self._lanes = {}
        for ev in self._events:
            if ev.actor not in self._lanes:
                self._lanes[ev.actor] = len(self._lanes)
        self._resize_for_lanes()
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self.playhead = seconds
        self.update()

    def clear(self) -> None:
        self._events.clear()
        self._lanes.clear()
        self._resize_for_lanes()
        self.update()

    @property
    def event_count(self) -> int:
        return len(self._events)

    def _resize_for_lanes(self) -> None:
        lanes = max(1, len(self._lanes))
        self.setMinimumHeight(RULER_HEIGHT + lanes * (LANE_HEIGHT + LANE_GAP) + 12)

    # -- geometry -----------------------------------------------------------
    def _plot_rect(self) -> QRectF:
        return QRectF(
            LABEL_WIDTH, RULER_HEIGHT,
            max(1.0, self.width() - LABEL_WIDTH - 8),
            max(1.0, self.height() - RULER_HEIGHT - 6),
        )

    def _time_to_x(self, seconds: float) -> float:
        plot = self._plot_rect()
        return plot.left() + (seconds / self.duration) * plot.width()

    def _x_to_time(self, x: float) -> float:
        plot = self._plot_rect()
        if plot.width() <= 0:
            return 0.0
        return max(0.0, min(self.duration, (x - plot.left()) / plot.width() * self.duration))

    def _lane_rect(self, lane: int) -> QRectF:
        plot = self._plot_rect()
        top = plot.top() + lane * (LANE_HEIGHT + LANE_GAP)
        return QRectF(plot.left(), top, plot.width(), LANE_HEIGHT)

    # -- painting -----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#12141a"))

        self._draw_ruler(painter)
        self._draw_lanes(painter)
        self._draw_events(painter)
        self._draw_playhead(painter)
        painter.end()

    def _draw_ruler(self, painter: QPainter) -> None:
        plot = self._plot_rect()
        painter.setPen(QPen(QColor("#2a2f39"), 1))
        painter.drawLine(
            QPointF(plot.left(), RULER_HEIGHT - 1),
            QPointF(plot.right(), RULER_HEIGHT - 1),
        )
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#78808e"))

        step = self._tick_step()
        t = 0.0
        while t <= self.duration + 1e-6:
            x = self._time_to_x(t)
            painter.drawLine(QPointF(x, RULER_HEIGHT - 6), QPointF(x, RULER_HEIGHT - 1))
            painter.drawText(QPointF(x + 3, RULER_HEIGHT - 7), _format_time(t))
            t += step

    def _tick_step(self) -> float:
        plot_w = max(1.0, self._plot_rect().width())
        target_ticks = max(4.0, plot_w / 110.0)
        raw = self.duration / target_ticks
        for candidate in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
            if raw <= candidate:
                return float(candidate)
        return 3600.0

    def _draw_lanes(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        metrics = QFontMetrics(painter.font())
        for actor, lane in self._lanes.items():
            rect = self._lane_rect(lane)
            painter.fillRect(rect, QColor("#171a21"))
            label = metrics.elidedText(actor, Qt.TextElideMode.ElideRight, LABEL_WIDTH - 14)
            painter.setPen(QColor("#c7cdd8"))
            painter.drawText(
                QRectF(6, rect.top(), LABEL_WIDTH - 12, rect.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                label,
            )

    def _draw_events(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        metrics = QFontMetrics(painter.font())
        for ev in self._events:
            lane = self._lanes.get(ev.actor)
            if lane is None:
                continue
            rect = self._lane_rect(lane)
            x1 = self._time_to_x(ev.start)
            x2 = self._time_to_x(ev.end)
            bar = QRectF(x1, rect.top() + 3, max(2.0, x2 - x1), rect.height() - 6)
            color = QColor(EMOTION_COLORS.get(ev.emotion, "#8a8f98"))
            painter.fillRect(bar, color)
            if bar.width() > metrics.horizontalAdvance(ev.emotion) + 12:
                painter.setPen(QColor("#10131a"))
                painter.drawText(
                    bar, int(Qt.AlignmentFlag.AlignCenter), ev.emotion
                )

    def _draw_playhead(self, painter: QPainter) -> None:
        x = self._time_to_x(min(self.playhead, self.duration))
        painter.setPen(QPen(QColor("#f5f7fa"), 1.4))
        painter.drawLine(QPointF(x, RULER_HEIGHT - 6), QPointF(x, self.height() - 2))
        painter.setPen(QColor("#f5f7fa"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(QPointF(x + 4, RULER_HEIGHT + 9), _format_time(self.playhead))

    # -- interaction --------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if pos.x() >= self._plot_rect().left():
                self.seekRequested.emit(self._x_to_time(pos.x()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        ev = self._event_at(event.position())
        if ev is not None:
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"{ev.actor} - {ev.emotion}\n"
                f"{_format_time(ev.start)} to {_format_time(ev.end)} "
                f"({ev.duration:.2f}s)\nconfidence {ev.confidence:.2f}",
                self,
            )
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def _event_at(self, pos: QPointF) -> Optional[TimelineEvent]:
        for actor, lane in self._lanes.items():
            rect = self._lane_rect(lane)
            if not (rect.top() <= pos.y() <= rect.bottom()):
                continue
            t = self._x_to_time(pos.x())
            for ev in self._events:
                if ev.actor == actor and ev.start <= t <= ev.end:
                    return ev
        return None


def _format_time(seconds: float) -> str:
    """``mm:ss`` (or ``h:mm:ss`` past an hour)."""
    seconds = max(0.0, seconds)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


__all__ = ["TimelineWidget"]
