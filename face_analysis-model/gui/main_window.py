"""Main application window.

Layout::

    +--------------------------------------------------+-----------------+
    |                                                  |  Detected faces |
    |                  Video player                    |  actor          |
    |             (independent playback)               |  emotion        |
    |                                                  |  confidence     |
    |                                                  |  similarity     |
    +--------------------------------------------------+-----------------+
    |  Timeline (one lane per actor, click to seek)                      |
    +--------------------------------------------------------------------+
    |  timestamp | decode fps | processing fps | GPU | VRAM | progress    |
    +--------------------------------------------------------------------+

The player and the analysis pipeline never touch each other: playback decodes on its own
thread at wall-clock pace, and analysis results arrive as queued Qt signals which are
only *drawn on top*. A GPU stall can delay overlays; it cannot stutter the video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from configs.config import AppConfig
from gui.bridge import PipelineBridge
from gui.events_table import EventsTable
from gui.faces_panel import FacesPanel
from gui.registration_dialog import RegistrationDialog
from gui.stats_bar import StatsBar
from gui.timeline_widget import TimelineWidget
from gui.video_widget import PlaybackController, VideoWidget
from pipeline.frame_source import probe
from pipeline.types import FrameResult, ProgressUpdate
from pipeline.worker import AnalysisPipeline
from recognition.registration import CastRegistrar
from timeline.events import TimelineDocument
from timeline.exporters import export_all
from utils.logging_utils import get_logger

log = get_logger(__name__)

STYLESHEET = """
QMainWindow, QWidget { background: #12141a; color: #dfe3e9; }
QSplitter::handle { background: #1c2028; }
QPushButton {
    background: #232833; border: 1px solid #2e3542; border-radius: 5px;
    padding: 5px 12px; color: #dfe3e9;
}
QPushButton:hover { background: #2b3140; }
QPushButton:disabled { color: #5d6572; background: #1a1e26; }
QFrame#faceCard, QFrame#statTile { background: #171a21; border-radius: 7px; }
QProgressBar { background: #1c2029; border: none; border-radius: 4px; }
QProgressBar::chunk { background: #3d7ff5; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: #262b35; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #dfe3e9; width: 11px; margin: -4px 0; border-radius: 5px;
}
QListWidget, QTextEdit, QLineEdit {
    background: #171a21; border: 1px solid #262b35; border-radius: 5px;
    selection-background-color: #2f4a7a;
}
QMenuBar { background: #12141a; } QMenuBar::item:selected { background: #232833; }
QMenu { background: #171a21; border: 1px solid #262b35; }
QMenu::item:selected { background: #2b3140; }
QStatusBar { background: #12141a; color: #79818f; }
"""


class MainWindow(QMainWindow):
    """Top-level window wiring the player, the pipeline and the timeline together."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.cfg = config
        self.setWindowTitle(config.gui.window_title)
        self.resize(1500, 940)
        self.setStyleSheet(STYLESHEET)

        self.video_path: Optional[Path] = None
        self.document: Optional[TimelineDocument] = None
        self._analysing = False

        self.pipeline = AnalysisPipeline(config)
        self.bridge = PipelineBridge(self)
        self.pipeline.callbacks = self.bridge.callbacks()
        self.player = PlaybackController(
            preview_width=config.gui.preview_width, fps_cap=config.gui.playback_fps_cap
        )

        self._build_ui()
        self._connect()
        self._build_menu()
        self.statusBar().showMessage("Ready - open a video, then Analyze")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Analysis-first layout: the player is a narrow monitor column, the event
        table and the cast panel own the rest of the window."""
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        upper = QSplitter(Qt.Orientation.Horizontal)

        # --- left: compact player + transport + live cast cards ---------------
        left = QWidget()
        panel_width = self.cfg.gui.video_panel_width
        left.setMaximumWidth(panel_width + 40)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.video_widget = VideoWidget()
        # Fixed 16:9 box the width of the column: the player keeps a predictable size
        # instead of stealing space from the analysis panels as the window grows.
        self.video_widget.setMinimumSize(240, 135)
        self.video_widget.setFixedHeight(int(panel_width * 9 / 16))
        left_layout.addWidget(self.video_widget, 0)
        left_layout.addLayout(self._build_transport())

        self.faces_panel = FacesPanel(
            max_cards=self.cfg.gui.max_face_cards, thumb_size=self.cfg.gui.face_thumb_size
        )
        left_layout.addWidget(self.faces_panel, 1)
        upper.addWidget(left)

        # --- right: the analysis itself ---------------------------------------
        self.events_table = EventsTable(max_rows=self.cfg.gui.max_event_rows)
        self.events_table.setMinimumWidth(460)
        upper.addWidget(self.events_table)

        upper.setStretchFactor(0, 0)
        upper.setStretchFactor(1, 1)
        upper.setSizes([panel_width, 1000])
        root.addWidget(upper, 1)

        self.timeline_widget = TimelineWidget()
        root.addWidget(self.timeline_widget, 0)

        self.stats_bar = StatsBar()
        root.addWidget(self.stats_bar, 0)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    def _build_transport(self) -> QVBoxLayout:
        """Two compact rows under the small player: scrub, then actions."""
        column = QVBoxLayout()
        column.setSpacing(4)

        scrub = QHBoxLayout()
        scrub.setSpacing(6)
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.setFixedWidth(64)
        self.play_btn.clicked.connect(self.player.toggle)
        scrub.addWidget(self.play_btn)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderMoved.connect(self._slider_seek)
        self.position_slider.sliderReleased.connect(self._slider_released)
        scrub.addWidget(self.position_slider, 1)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFont(QFont("Consolas", 8))
        scrub.addWidget(self.time_label)
        column.addLayout(scrub)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.open_btn = QPushButton("Open video")
        self.open_btn.clicked.connect(self.open_video)
        actions.addWidget(self.open_btn)

        self.cast_btn = QPushButton("Cast")
        self.cast_btn.clicked.connect(self.open_registration)
        actions.addWidget(self.cast_btn)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.toggle_analysis)
        actions.addWidget(self.analyze_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        actions.addWidget(self.export_btn)
        column.addLayout(actions)
        return column

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open video...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)

        export_action = QAction("&Export timeline...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        cast_menu = self.menuBar().addMenu("&Cast")
        register_action = QAction("&Register actors...", self)
        register_action.setShortcut(QKeySequence("Ctrl+R"))
        register_action.triggered.connect(self.open_registration)
        cast_menu.addAction(register_action)

        reload_action = QAction("Re&load gallery", self)
        reload_action.triggered.connect(self._reload_gallery)
        cast_menu.addAction(reload_action)

        playback_menu = self.menuBar().addMenu("&Playback")
        play_action = QAction("Play / Pause", self)
        play_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        play_action.triggered.connect(self.player.toggle)
        playback_menu.addAction(play_action)

        self.overlay_action = QAction("Show detection overlay", self, checkable=True)
        self.overlay_action.setChecked(True)
        self.overlay_action.toggled.connect(self._toggle_overlay)
        playback_menu.addAction(self.overlay_action)

    def _connect(self) -> None:
        self.player.frameReady.connect(self._on_playback_frame)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.stateChanged.connect(self._on_play_state)
        self.player.ended.connect(lambda: self.play_btn.setText("Play"))

        self.bridge.frameAnalysed.connect(self._on_frame_analysed)
        self.bridge.eventClosed.connect(self._on_event_closed)
        self.bridge.progressed.connect(self._on_progress)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.failed.connect(self._on_failed)

        self.timeline_widget.seekRequested.connect(self._seek)
        self.events_table.rowActivated.connect(self._seek)
        self.video_widget.clicked.connect(self.player.toggle)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def open_video(self) -> None:
        """Pick a video, load it into the player and reset the analysis state."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.wmv);;All files (*.*)",
        )
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str | Path) -> None:
        """Load ``path`` into the player without starting analysis."""
        try:
            meta = probe(path)
        except (FileNotFoundError, RuntimeError) as exc:
            QMessageBox.critical(self, "Cannot open video", str(exc))
            return

        self.video_path = Path(path)
        self.document = None
        self.video_widget.clear()
        self.video_widget.set_video_size(meta.width, meta.height)
        self.faces_panel.clear()
        self.events_table.clear()
        self.timeline_widget.clear()
        self.timeline_widget.set_duration(meta.duration)
        self.stats_bar.reset()

        self.player.load(path)
        self.play_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.setWindowTitle(f"{self.cfg.gui.window_title} - {self.video_path.name}")
        self.statusBar().showMessage(
            f"{self.video_path.name} | {meta.width}x{meta.height} @ {meta.fps:.2f} fps "
            f"| {meta.duration:.1f}s | codec {meta.codec}"
        )

    def toggle_analysis(self) -> None:
        """Start or stop the analysis pipeline."""
        if self._analysing:
            self.pipeline.stop()
            self.statusBar().showMessage("Analysis stopped")
            self._set_analysing(False)
            return
        if self.video_path is None:
            return

        self.timeline_widget.clear()
        self.events_table.clear()
        self.timeline_widget.set_duration(self.player.duration)
        self.statusBar().showMessage("Loading models...")
        QApplication.processEvents()
        try:
            self.pipeline.prepare()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model load failed", str(exc))
            self.statusBar().showMessage("Model load failed")
            return

        if self.pipeline.matcher.is_empty:
            QMessageBox.information(
                self, "No cast registered",
                "No actors are registered yet, so every face will be reported as "
                "Unknown.\n\nUse Cast -> Register actors... to enrol the cast.",
            )
        try:
            self.pipeline.start(self.video_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot start analysis", str(exc))
            return
        self._set_analysing(True)
        self.statusBar().showMessage("Analysing...")
        self.player.play()

    def open_registration(self) -> None:
        """Show the cast registration dialog (loads models on first use)."""
        self.statusBar().showMessage("Loading models...")
        QApplication.processEvents()
        try:
            self.pipeline.prepare()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model load failed", str(exc))
            return
        assert self.pipeline.detector is not None and self.pipeline.embedder is not None

        registrar = CastRegistrar(
            self.cfg, self.pipeline.detector, self.pipeline.embedder, self.pipeline.db
        )
        dialog = RegistrationDialog(registrar, self.pipeline.db, self)
        dialog.castChanged.connect(self._reload_gallery)
        dialog.exec()
        self.statusBar().showMessage("Ready")

    def export_results(self) -> None:
        """Write JSON + CSV + PNGs for the completed analysis."""
        if self.document is None:
            QMessageBox.information(
                self, "Nothing to export", "Run an analysis first."
            )
            return
        default_dir = self.cfg.paths.resolve(self.cfg.paths.output_dir)
        directory = QFileDialog.getExistingDirectory(
            self, "Choose an export folder", str(default_dir)
        )
        if not directory:
            return
        stem = (self.video_path.stem if self.video_path else "timeline")
        try:
            paths = export_all(self.document, Path(directory), stem)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self, "Export complete",
            "Written:\n" + "\n".join(str(p) for p in paths.values()),
        )
        self.statusBar().showMessage(f"Exported to {directory}")

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _on_playback_frame(self, image, timestamp: float) -> None:
        self.video_widget.set_frame(image, timestamp)

    def _on_position(self, seconds: float) -> None:
        if not self.position_slider.isSliderDown() and self.player.duration > 0:
            self.position_slider.setValue(
                int(1000 * seconds / max(self.player.duration, 1e-6))
            )
        self.time_label.setText(
            f"{_clock(seconds)} / {_clock(self.player.duration)}"
        )
        self.timeline_widget.set_playhead(seconds)

    def _on_duration(self, seconds: float) -> None:
        self.timeline_widget.set_duration(seconds)
        self.time_label.setText(f"00:00 / {_clock(seconds)}")

    def _on_play_state(self, playing: bool) -> None:
        self.play_btn.setText("Pause" if playing else "Play")

    def _on_frame_analysed(self, result: FrameResult) -> None:
        self.video_widget.add_result(result)
        self.faces_panel.update_frame(result)

    def _on_event_closed(self, event) -> None:
        self.timeline_widget.add_event(event)
        self.events_table.add_event(event)

    def _on_progress(self, update: ProgressUpdate) -> None:
        self.stats_bar.update_progress(update)

    def _on_finished(self, document: TimelineDocument) -> None:
        self.document = document
        self._set_analysing(False)
        self.export_btn.setEnabled(True)
        self.timeline_widget.set_events(document.events)
        self.events_table.set_events(document.events)

        info = document.analysis
        self.stats_bar.show_completion(
            info.elapsed_seconds, info.realtime_factor, info.expression_changes
        )
        self.statusBar().showMessage(
            f"Analysed in {info.elapsed_seconds:.1f}s ({info.realtime_factor:.2f}x realtime) "
            f"|  {info.expression_changes} expression changes across {len(document.actors)} "
            f"cast members  |  {len(document.events)} events  "
            f"|  {info.frames_analysed} frames analysed"
        )

    def _on_failed(self, exc: BaseException) -> None:
        self._set_analysing(False)
        QMessageBox.critical(self, "Analysis failed", str(exc))
        self.statusBar().showMessage(f"Analysis failed: {exc}")

    def _reload_gallery(self) -> None:
        count = self.pipeline.reload_gallery()
        self.statusBar().showMessage(f"Cast gallery reloaded: {count} actors")

    def _toggle_overlay(self, enabled: bool) -> None:
        self.video_widget.show_overlay = enabled
        self.video_widget.update()

    def _slider_seek(self, value: int) -> None:
        if self.player.duration > 0:
            self.time_label.setText(
                f"{_clock(value / 1000 * self.player.duration)} / {_clock(self.player.duration)}"
            )

    def _slider_released(self) -> None:
        self._seek(self.position_slider.value() / 1000.0 * self.player.duration)

    def _seek(self, seconds: float) -> None:
        self.player.seek(seconds)
        self.timeline_widget.set_playhead(seconds)

    def _set_analysing(self, running: bool) -> None:
        self._analysing = running
        self.analyze_btn.setText("Stop" if running else "Analyze")
        self.open_btn.setEnabled(not running)

    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
        """Tear down the player and the pipeline before the window disappears."""
        try:
            self.player.stop()
            self.pipeline.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            log.exception("Error during shutdown")
        super().closeEvent(event)


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


__all__ = ["MainWindow"]
