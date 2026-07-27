"""Qt bridge between the worker threads and the GUI thread.

Pipeline callbacks fire on arbitrary worker threads; touching widgets from there would
crash. :class:`PipelineBridge` is a ``QObject`` living in the GUI thread whose signals
are emitted from the workers - Qt then delivers them as *queued* connections, i.e. on
the GUI thread's event loop. That is the whole trick that keeps playback smooth while
inference runs flat out.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Signal

from pipeline.types import FrameResult, ProgressUpdate
from pipeline.worker import PipelineCallbacks
from timeline.events import TimelineDocument, TimelineEvent


class PipelineBridge(QObject):
    """Re-emits pipeline callbacks as Qt signals."""

    frameAnalysed = Signal(object)      # FrameResult
    eventClosed = Signal(object)        # TimelineEvent
    progressed = Signal(object)         # ProgressUpdate
    finished = Signal(object)           # TimelineDocument
    failed = Signal(object)             # BaseException

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

    def callbacks(self) -> PipelineCallbacks:
        """Build the callback bundle to hand to :class:`AnalysisPipeline`."""
        return PipelineCallbacks(
            on_frame=self._on_frame,
            on_event=self._on_event,
            on_progress=self._on_progress,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    # -- worker-thread entry points ----------------------------------------
    def _on_frame(self, result: FrameResult) -> None:
        self.frameAnalysed.emit(result)

    def _on_event(self, event: TimelineEvent) -> None:
        self.eventClosed.emit(event)

    def _on_progress(self, update: ProgressUpdate) -> None:
        self.progressed.emit(update)

    def _on_finished(self, document: TimelineDocument) -> None:
        self.finished.emit(document)

    def _on_error(self, exc: BaseException) -> None:
        self.failed.emit(exc)


__all__ = ["PipelineBridge"]
