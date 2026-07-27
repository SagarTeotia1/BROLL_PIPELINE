"""Right-hand panel: one card per currently detected face.

Each card shows the thumbnail, the recognised actor, the cosine similarity, the
smoothed emotion and its confidence. Cards are recycled (never rebuilt) so updating at
7.5 Hz costs nothing on the GUI thread.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from emotion.hsemotion import EMOTION_COLORS
from pipeline.types import FaceObservation, FrameResult
from recognition.matcher import UNKNOWN


def numpy_to_pixmap(image: np.ndarray) -> QPixmap:
    """Convert a contiguous BGR array to a :class:`QPixmap` (copies once)."""
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    qimage = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimage.copy())


class FaceCard(QFrame):
    """A single face entry in the panel."""

    def __init__(self, thumb_size: int = 84, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.thumb_size = thumb_size
        self.setObjectName("faceCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        self.thumb = QLabel()
        self.thumb.setFixedSize(thumb_size, thumb_size)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("background:#1b1f27; border-radius:6px;")
        layout.addWidget(self.thumb)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        self.name_label = QLabel(UNKNOWN)
        name_font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
        self.name_label.setFont(name_font)
        text_col.addWidget(self.name_label)

        self.emotion_label = QLabel("-")
        self.emotion_label.setFont(QFont("Segoe UI", 9))
        text_col.addWidget(self.emotion_label)

        self.meta_label = QLabel("")
        self.meta_label.setFont(QFont("Segoe UI", 8))
        self.meta_label.setStyleSheet("color:#8b93a1;")
        text_col.addWidget(self.meta_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(5)
        text_col.addWidget(self.confidence_bar)

        layout.addLayout(text_col, 1)

    def update_face(self, face: FaceObservation) -> None:
        """Refresh the card from one observation."""
        if face.thumbnail is not None and face.thumbnail.size:
            self.thumb.setPixmap(
                numpy_to_pixmap(face.thumbnail).scaled(
                    self.thumb_size, self.thumb_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.thumb.clear()

        known = face.actor_id >= 0
        self.name_label.setText(face.actor_name if known else UNKNOWN)
        self.name_label.setStyleSheet(
            "color:#eef1f5;" if known else "color:#98a0ad;"
        )

        color = EMOTION_COLORS.get(face.emotion, "#8a8f98")
        self.emotion_label.setText(f"{face.emotion}  {face.emotion_confidence * 100:.0f}%")
        self.emotion_label.setStyleSheet(f"color:{color}; font-weight:600;")

        sim = f"sim {face.similarity:.3f}" if known else "sim -"
        lock = "locked" if face.locked else "voting"
        self.meta_label.setText(
            f"track #{face.track_id}  |  {sim}  |  {lock}  |  q {face.quality:.2f}"
        )
        self.confidence_bar.setValue(int(round(face.emotion_confidence * 100)))
        self.confidence_bar.setStyleSheet(
            "QProgressBar{background:#20242c;border:none;border-radius:2px;}"
            f"QProgressBar::chunk{{background:{color};border-radius:2px;}}"
        )


class FacesPanel(QWidget):
    """Scroll-free stack of :class:`FaceCard` widgets."""

    def __init__(self, max_cards: int = 8, thumb_size: int = 84, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.max_cards = max_cards
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)

        header = QLabel("Cast on screen")
        header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.setStyleSheet("color:#c9cfda; padding:2px 4px;")
        self._layout.addWidget(header)

        self.empty_label = QLabel("No registered cast on screen yet")
        self.empty_label.setStyleSheet("color:#6d7685; padding:8px 6px;")
        self._layout.addWidget(self.empty_label)

        self._cards: List[FaceCard] = []
        for _ in range(max_cards):
            card = FaceCard(thumb_size)
            card.hide()
            self._cards.append(card)
            self._layout.addWidget(card)
        self._layout.addStretch(1)

    def update_frame(self, result: FrameResult) -> None:
        """Show the faces of one analysed frame, biggest/most-confident first."""
        faces = sorted(
            result.faces,
            key=lambda f: (f.actor_id >= 0, f.quality, f.det_score),
            reverse=True,
        )[: self.max_cards]

        self.empty_label.setVisible(not faces)
        for i, card in enumerate(self._cards):
            if i < len(faces):
                card.update_face(faces[i])
                card.show()
            else:
                card.hide()

    def clear(self) -> None:
        for card in self._cards:
            card.hide()
        self.empty_label.setVisible(True)


__all__ = ["FacesPanel", "FaceCard", "numpy_to_pixmap"]
