"""Status bar: throughput, GPU load, queue depths and the progress bar.

Every number comes from :class:`pipeline.types.ProgressUpdate`, which is assembled from
the same :class:`utils.profiling.Profiler` the log uses - the GUI never estimates
anything on its own.
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pipeline.types import ProgressUpdate


class StatTile(QFrame):
    """A small labelled readout."""

    def __init__(self, title: str, unit: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.unit = unit
        self.setObjectName("statTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(0)

        self.title_label = QLabel(title.upper())
        self.title_label.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        self.title_label.setStyleSheet("color:#79818f; letter-spacing:0.5px;")
        layout.addWidget(self.title_label)

        self.value_label = QLabel("-")
        self.value_label.setFont(QFont("Consolas", 11, QFont.Weight.DemiBold))
        self.value_label.setStyleSheet("color:#e8ebef;")
        layout.addWidget(self.value_label)

    def set_value(self, text: str, color: str = "#e8ebef") -> None:
        self.value_label.setText(text)
        self.value_label.setStyleSheet(f"color:{color};")


class StatsBar(QWidget):
    """Bottom status strip."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 6)
        outer.setSpacing(5)

        tiles = QHBoxLayout()
        tiles.setSpacing(6)
        self.elapsed_tile = StatTile("analysis time")
        self.eta_tile = StatTile("remaining")
        self.time_tile = StatTile("timestamp")
        self.decode_tile = StatTile("decode fps")
        self.analysis_tile = StatTile("processing fps")
        self.speed_tile = StatTile("realtime")
        self.gpu_tile = StatTile("gpu")
        self.vram_tile = StatTile("vram")
        self.changes_tile = StatTile("changes")
        for tile in self._tiles():
            tile.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            tiles.addWidget(tile)
        outer.addLayout(tiles)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        row.addWidget(self.progress, 1)

        self.queue_label = QLabel("idle")
        self.queue_label.setFont(QFont("Consolas", 8))
        self.queue_label.setStyleSheet("color:#79818f;")
        row.addWidget(self.queue_label, 0)
        outer.addLayout(row)

    def _tiles(self) -> tuple:
        return (
            self.elapsed_tile, self.eta_tile, self.time_tile, self.decode_tile,
            self.analysis_tile, self.speed_tile, self.gpu_tile, self.vram_tile,
            self.changes_tile,
        )

    # -- updates ------------------------------------------------------------
    def update_progress(self, update: ProgressUpdate) -> None:
        """Refresh every tile from one telemetry sample."""
        self.elapsed_tile.set_value(
            _clock(update.elapsed), "#3ecf8e" if update.finished else "#e8ebef"
        )
        self.eta_tile.set_value(_eta(update))
        self.time_tile.set_value(
            f"{_clock(update.timestamp)} / {_clock(update.duration)}"
        )
        self.decode_tile.set_value(f"{update.decode_fps:6.1f}")
        self.analysis_tile.set_value(f"{update.analysis_fps:6.2f}")

        speed_color = "#3ecf8e" if update.realtime_factor >= 1.0 else "#f5b73d"
        self.speed_tile.set_value(f"x{update.realtime_factor:5.2f}", speed_color)

        gpu_color = "#3ecf8e" if update.gpu_utilization < 92 else "#e5484d"
        self.gpu_tile.set_value(f"{update.gpu_utilization:4.0f}%", gpu_color)

        if update.gpu_memory_total_mb > 0:
            self.vram_tile.set_value(
                f"{update.gpu_memory_mb / 1024:.1f}/{update.gpu_memory_total_mb / 1024:.1f}G"
            )
        else:
            self.vram_tile.set_value(f"{update.gpu_memory_mb:.0f}M")

        self.changes_tile.set_value(f"{update.expression_changes:5d}")
        self.progress.setValue(int(update.percent * 10))
        self.queue_label.setText(_format_queues(update.queue_depths))

    def show_completion(self, elapsed: float, realtime: float, changes: int) -> None:
        """Freeze the bar on the final numbers once the run finishes."""
        self.elapsed_tile.set_value(_clock(elapsed), "#3ecf8e")
        self.eta_tile.set_value("done", "#3ecf8e")
        self.speed_tile.set_value(f"x{realtime:5.2f}", "#3ecf8e")
        self.changes_tile.set_value(f"{changes:5d}")
        self.progress.setValue(1000)
        self.queue_label.setText(
            f"analysed in {_clock(elapsed)}  ({realtime:.2f}x realtime)"
        )

    def set_idle(self, message: str = "idle") -> None:
        self.queue_label.setText(message)
        self.progress.setValue(0)

    def reset(self) -> None:
        for tile in self._tiles():
            tile.set_value("-")
        self.progress.setValue(0)
        self.queue_label.setText("idle")


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _eta(update: ProgressUpdate) -> str:
    """Remaining wall time, extrapolated from the decode rate achieved so far."""
    if update.finished:
        return "done"
    if update.total_frames <= 0 or update.frames_decoded <= 0 or update.elapsed <= 0:
        return "--:--"
    rate = update.frames_decoded / update.elapsed
    if rate <= 0:
        return "--:--"
    remaining = max(0.0, (update.total_frames - update.frames_decoded) / rate)
    return _clock(remaining)


def _format_queues(depths: Dict[str, int]) -> str:
    if not depths:
        return "idle"
    order = ("raw", "sampled", "detected", "recog", "emotion", "release")
    parts = [f"{name}:{depths[name]}" for name in order if name in depths]
    return "queues  " + "  ".join(parts)


__all__ = ["StatsBar", "StatTile"]
