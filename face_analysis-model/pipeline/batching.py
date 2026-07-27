"""Cross-frame batch collection.

A single frame usually holds 1-3 faces - far too few to saturate a GPU. The collector
drains a queue until it has ``batch_size`` items **or** ``max_latency_ms`` has elapsed
since the first item arrived, so batches stay full on busy scenes and latency stays
bounded on sparse ones.

This is what makes "never infer one face at a time" true in practice: faces from
several consecutive frames are embedded in the same ArcFace call.
"""

from __future__ import annotations

import queue
import time
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class BatchCollector(Generic[T]):
    """Latency-bounded batch accumulator over a :class:`queue.Queue`.

    Args:
        source: the queue to drain.
        batch_size: maximum items per batch.
        max_latency_ms: flush a partial batch after this long.
        sentinel: the object that signals end-of-stream (default ``None``).
    """

    def __init__(
        self,
        source: "queue.Queue[Optional[T]]",
        batch_size: int,
        max_latency_ms: float = 25.0,
        sentinel: object = None,
    ) -> None:
        self.source = source
        self.batch_size = max(1, int(batch_size))
        self.max_latency = max(0.0, max_latency_ms) / 1000.0
        self.sentinel = sentinel
        self.finished = False
        self.batches = 0
        self.items = 0

    def next_batch(self, poll_timeout: float = 0.05) -> List[T]:
        """Return the next batch (possibly empty if the queue stayed idle).

        Sets :attr:`finished` once the sentinel has been observed; the sentinel is not
        included in the returned batch, and any items collected before it are still
        returned so nothing is lost at end-of-stream.
        """
        batch: List[T] = []
        if self.finished:
            return batch

        # Block for the first item so an idle pipeline does not spin.
        try:
            first = self.source.get(timeout=poll_timeout)
        except queue.Empty:
            return batch
        if first is self.sentinel:
            self.finished = True
            return batch
        batch.append(first)

        deadline = time.perf_counter() + self.max_latency
        while len(batch) < self.batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self.source.get(timeout=remaining)
            except queue.Empty:
                break
            if item is self.sentinel:
                self.finished = True
                break
            batch.append(item)

        self.batches += 1
        self.items += len(batch)
        return batch

    @property
    def mean_batch(self) -> float:
        """Average batch size so far - the headline number for GPU efficiency."""
        return self.items / self.batches if self.batches else 0.0

    def drain(self) -> List[T]:
        """Non-blocking drain of whatever is queued right now."""
        batch: List[T] = []
        while len(batch) < self.batch_size:
            try:
                item = self.source.get_nowait()
            except queue.Empty:
                break
            if item is self.sentinel:
                self.finished = True
                break
            batch.append(item)
        if batch:
            self.batches += 1
            self.items += len(batch)
        return batch


__all__ = ["BatchCollector"]
