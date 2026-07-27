"""Live expression-change table - the primary analysis view.

Every row is one emotion span: the actor, what they showed, when it started and ended,
how long it lasted and how confident the model was. Rows appear the moment the pipeline
*closes* an event, so the table fills in while the analysis runs.

This is the on-screen twin of the exported JSON ``events`` array: same rows, same order,
same numbers.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from emotion.hsemotion import EMOTION_COLORS
from timeline.events import TimelineEvent

COLUMNS = ("Actor", "Emotion", "Start", "End", "Duration", "Conf", "Sim")


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:06.3f}"
    return f"{int(minutes):02d}:{secs:06.3f}"


class EventsTable(QWidget):
    """Sortable table of closed emotion events, with a live change counter."""

    rowActivated = Signal(float)     # emitted with the event start time (seek target)

    def __init__(self, max_rows: int = 500, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.max_rows = max_rows
        self._events: List[TimelineEvent] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        header = QHBoxLayout()
        title = QLabel("Expression changes")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        title.setStyleSheet("color:#c9cfda;")
        header.addWidget(title)
        header.addStretch(1)

        self.count_label = QLabel("0 events")
        self.count_label.setFont(QFont("Consolas", 9))
        self.count_label.setStyleSheet("color:#8b93a1;")
        header.addWidget(self.count_label)
        layout.addLayout(header)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFont(QFont("Consolas", 9))
        self.table.setStyleSheet(
            "QTableWidget{background:#171a21;border:1px solid #262b35;border-radius:6px;"
            "gridline-color:#262b35;alternate-background-color:#1b1f27;}"
            "QHeaderView::section{background:#1f242d;color:#9aa1ab;border:none;"
            "padding:5px;font-weight:600;}"
            "QTableWidget::item{padding:3px 6px;}"
            "QTableWidget::item:selected{background:#2f4a7a;}"
        )
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(COLUMNS)):
            head.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table, 1)

    # -- data ---------------------------------------------------------------
    def add_event(self, event: TimelineEvent) -> None:
        """Append one closed event."""
        self._events.append(event)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._fill_row(row, event)

        # Keep the widget bounded on very long videos; the JSON keeps everything.
        while self.table.rowCount() > self.max_rows:
            self.table.removeRow(0)
        self.table.scrollToBottom()
        self._update_count()

    def set_events(self, events: List[TimelineEvent]) -> None:
        """Replace the whole table (final document, or a loaded JSON)."""
        self._events = sorted(events, key=lambda e: (e.start, e.actor))
        self.table.setRowCount(0)
        for event in self._events[-self.max_rows :]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._fill_row(row, event)
        self._update_count()

    def clear(self) -> None:
        self._events.clear()
        self.table.setRowCount(0)
        self._update_count()

    # -- internals ----------------------------------------------------------
    def _fill_row(self, row: int, event: TimelineEvent) -> None:
        color = QColor(EMOTION_COLORS.get(event.emotion, "#8a8f98"))
        values = (
            event.actor,
            event.emotion,
            _clock(event.start),
            _clock(event.end),
            f"{event.duration:.2f}s",
            f"{event.confidence:.2f}",
            f"{event.similarity:.2f}" if event.similarity else "-",
        )
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col >= 2:
                item.setTextAlignment(
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                )
            if col == 1:
                item.setForeground(QBrush(color))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            elif col == 0:
                item.setForeground(QBrush(QColor("#e8ebef")))
            else:
                item.setForeground(QBrush(QColor("#aab1bd")))
            item.setData(Qt.ItemDataRole.UserRole, event.start)
            self.table.setItem(row, col, item)

    def _update_count(self) -> None:
        total = len(self._events)
        # Same definition as the exported JSON: a transition to a *different* emotion.
        # The first event is an appearance, and a same-emotion resume is not a change.
        previous: dict[str, str] = {}
        changes = 0
        for event in sorted(self._events, key=lambda e: e.start):
            last = previous.get(event.actor)
            if last is not None and last != event.emotion:
                changes += 1
            previous[event.actor] = event.emotion
        self.count_label.setText(
            f"{total} events  |  {changes} expression changes  |  {len(previous)} actors"
        )

    def _on_double_click(self, row: int, column: int) -> None:
        item = self.table.item(row, column)
        if item is not None:
            start = item.data(Qt.ItemDataRole.UserRole)
            if start is not None:
                self.rowActivated.emit(float(start))


__all__ = ["EventsTable"]
