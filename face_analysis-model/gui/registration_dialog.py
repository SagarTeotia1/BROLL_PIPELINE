"""Cast registration dialog (Phase 1 of the spec).

Left: the actors already in the database, with their photo count.
Right: a name field, the photos picked for the current actor, and the run button.

Enrolment runs on a worker thread (it hits the GPU) and reports per-image outcomes -
including *why* a photo was rejected - so the user can fix bad reference material
instead of wondering why recognition is poor.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from recognition.cast_database import CastDatabase
from recognition.registration import (
    CastRegistrar,
    RegistrationResult,
    collect_images,
    imread_unicode,
)
from utils.image_ops import resize_long_side
from utils.logging_utils import get_logger

log = get_logger(__name__)


class _RegistrationWorker(QObject):
    """Runs :meth:`CastRegistrar.register` off the GUI thread."""

    progressed = Signal(int, int, str)
    completed = Signal(object)          # RegistrationResult
    failed = Signal(str)

    def __init__(self, registrar: CastRegistrar) -> None:
        super().__init__()
        self.registrar = registrar
        self._thread: Optional[threading.Thread] = None

    def start(self, name: str, paths: List[str]) -> None:
        self._thread = threading.Thread(
            target=self._run, args=(name, paths), name="RegistrationWorker", daemon=True
        )
        self._thread.start()

    def _run(self, name: str, paths: List[str]) -> None:
        try:
            result = self.registrar.register(
                name, paths, progress=lambda i, n, f: self.progressed.emit(i, n, f)
            )
            self.completed.emit(result)
        except BaseException as exc:  # noqa: BLE001
            log.exception("Registration failed")
            self.failed.emit(str(exc))


class RegistrationDialog(QDialog):
    """Add, extend and remove cast members."""

    castChanged = Signal()

    def __init__(
        self,
        registrar: CastRegistrar,
        database: CastDatabase,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.registrar = registrar
        self.db = database
        self._paths: List[str] = []

        self.setWindowTitle("Cast registration")
        self.setMinimumSize(880, 560)

        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        self.worker = _RegistrationWorker(registrar)
        self.worker.progressed.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)

        self.refresh_actors()

    # -- layout -------------------------------------------------------------
    def _build_left(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Registered cast")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(title)

        self.actor_list = QListWidget()
        self.actor_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.actor_list.itemDoubleClicked.connect(self._load_actor_name)
        layout.addWidget(self.actor_list, 1)

        row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.clicked.connect(self._delete_actor)
        row.addWidget(self.delete_btn)
        layout.addLayout(row)

        self.db_label = QLabel("")
        self.db_label.setStyleSheet("color:#8b93a1;")
        layout.addWidget(self.db_label)
        return panel

    def _build_right(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Actor name"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. John")
        name_row.addWidget(self.name_edit, 1)
        layout.addLayout(name_row)

        pick_row = QHBoxLayout()
        add_files = QPushButton("Add photos...")
        add_files.clicked.connect(self._pick_files)
        pick_row.addWidget(add_files)
        add_folder = QPushButton("Add folder...")
        add_folder.clicked.connect(self._pick_folder)
        pick_row.addWidget(add_folder)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_photos)
        pick_row.addWidget(clear_btn)
        pick_row.addStretch(1)
        layout.addLayout(pick_row)

        self.photo_list = QListWidget()
        self.photo_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.photo_list.setIconSize(QPixmap(96, 96).size())
        self.photo_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.photo_list.setSpacing(6)
        layout.addWidget(self.photo_list, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.register_btn = QPushButton("Register / update actor")
        self.register_btn.clicked.connect(self._start_registration)
        layout.addWidget(self.register_btn)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setFixedHeight(140)
        self.report.setFont(QFont("Consolas", 8))
        layout.addWidget(self.report)
        return panel

    # -- actions ------------------------------------------------------------
    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select reference photos", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        self._add_paths(files)

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of photos")
        if folder:
            self._add_paths([folder])

    def _add_paths(self, paths: List[str]) -> None:
        images = collect_images(paths)
        for path in images:
            if str(path) in self._paths:
                continue
            self._paths.append(str(path))
            item = QListWidgetItem(path.name)
            thumb = self._thumbnail(path)
            if thumb is not None:
                item.setIcon(QIcon(thumb))
            item.setToolTip(str(path))
            self.photo_list.addItem(item)
        self.report.append(f"{len(images)} image(s) added ({len(self._paths)} total)")

    @staticmethod
    def _thumbnail(path: Path) -> Optional[QPixmap]:
        image = imread_unicode(path)
        if image is None:
            return None
        small, _ = resize_long_side(image, 96)
        from gui.faces_panel import numpy_to_pixmap  # local import avoids a cycle

        return numpy_to_pixmap(small)

    def _clear_photos(self) -> None:
        self._paths.clear()
        self.photo_list.clear()
        self.progress.setValue(0)

    def _start_registration(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Enter the actor's name first.")
            return
        if not self._paths:
            QMessageBox.warning(self, "No photos", "Add at least one reference photo.")
            return
        self.register_btn.setEnabled(False)
        self.report.append(f"\nRegistering '{name}' from {len(self._paths)} photo(s)...")
        self.worker.start(name, list(self._paths))

    def _on_progress(self, index: int, total: int, filename: str) -> None:
        self.progress.setValue(int(100 * index / max(1, total)))
        self.progress.setFormat(f"{filename}  ({index}/{total})")

    def _on_completed(self, result: RegistrationResult) -> None:
        self.register_btn.setEnabled(True)
        self.progress.setValue(100)
        self.report.append(result.summary())
        for outcome in result.outcomes:
            if not outcome.accepted:
                self.report.append(
                    f"  rejected {Path(outcome.path).name}: {outcome.reason or 'unknown'}"
                )
        self.refresh_actors()
        self.castChanged.emit()
        if result.ok:
            self._clear_photos()

    def _on_failed(self, message: str) -> None:
        self.register_btn.setEnabled(True)
        QMessageBox.critical(self, "Registration failed", message)

    def _delete_actor(self) -> None:
        item = self.actor_list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        confirm = QMessageBox.question(
            self, "Delete actor",
            f"Remove '{name}' and every stored embedding?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        if self.registrar.remove_actor(name):
            self.report.append(f"Deleted '{name}'")
            self.refresh_actors()
            self.castChanged.emit()

    def _load_actor_name(self, item: QListWidgetItem) -> None:
        self.name_edit.setText(item.data(Qt.ItemDataRole.UserRole))

    # -- data ---------------------------------------------------------------
    def refresh_actors(self) -> None:
        """Reload the actor list from the database."""
        self.actor_list.clear()
        for actor in self.db.list_actors():
            item = QListWidgetItem(f"{actor.name}   ({actor.num_images} photos)")
            item.setData(Qt.ItemDataRole.UserRole, actor.name)
            self.actor_list.addItem(item)
        stats = self.db.stats()
        self.db_label.setText(
            f"{stats['actors']} actors, {stats['embeddings']} embeddings"
        )


__all__ = ["RegistrationDialog"]
