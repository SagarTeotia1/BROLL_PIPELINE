"""Throughput / latency instrumentation.

Every pipeline stage owns a :class:`RateMeter`. A single :class:`Profiler` collects
them so the GUI status bar and the log both read the same numbers:

    prof = Profiler()
    with prof.timer("detect"):
        boxes = detector(frames)
    prof.mark("detect", n=len(frames))
    print(prof.snapshot()["detect"].fps)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterator


@dataclass
class StageStats:
    """Immutable view of one stage's counters."""

    name: str
    count: int = 0
    fps: float = 0.0
    avg_ms: float = 0.0
    p95_ms: float = 0.0
    last_ms: float = 0.0
    queue_depth: int = 0

    def as_dict(self) -> Dict[str, float | int | str]:
        return {
            "name": self.name,
            "count": self.count,
            "fps": round(self.fps, 2),
            "avg_ms": round(self.avg_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "last_ms": round(self.last_ms, 2),
            "queue_depth": self.queue_depth,
        }


class RateMeter:
    """Sliding-window rate + latency estimator for one stage.

    ``mark(n)`` records that ``n`` items completed; ``observe(ms)`` records how long
    a call took. Both windows are bounded so memory never grows.
    """

    def __init__(self, name: str, window: int = 120) -> None:
        self.name = name
        self._window = window
        self._events: Deque[tuple[float, int]] = deque(maxlen=window)
        self._latencies: Deque[float] = deque(maxlen=window)
        self._total = 0
        self._last_ms = 0.0
        self._queue_depth = 0
        self._lock = threading.Lock()

    def mark(self, n: int = 1) -> None:
        now = time.perf_counter()
        with self._lock:
            self._events.append((now, n))
            self._total += n

    def observe(self, milliseconds: float) -> None:
        with self._lock:
            self._latencies.append(milliseconds)
            self._last_ms = milliseconds

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = depth

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._latencies.clear()
            self._total = 0
            self._last_ms = 0.0

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def stats(self) -> StageStats:
        with self._lock:
            fps = 0.0
            if len(self._events) >= 2:
                span = self._events[-1][0] - self._events[0][0]
                if span > 1e-6:
                    # Exclude the first sample's count: it delimits the window start.
                    items = sum(n for _, n in list(self._events)[1:])
                    fps = items / span
            lat = sorted(self._latencies)
            avg = sum(lat) / len(lat) if lat else 0.0
            p95 = lat[max(0, int(0.95 * len(lat)) - 1)] if lat else 0.0
            return StageStats(
                name=self.name,
                count=self._total,
                fps=fps,
                avg_ms=avg,
                p95_ms=p95,
                last_ms=self._last_ms,
                queue_depth=self._queue_depth,
            )


class Profiler:
    """Registry of :class:`RateMeter` objects, safe to share across threads."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._meters: Dict[str, RateMeter] = {}
        self._lock = threading.Lock()
        self._start = time.perf_counter()

    def meter(self, name: str) -> RateMeter:
        with self._lock:
            m = self._meters.get(name)
            if m is None:
                m = RateMeter(name)
                self._meters[name] = m
            return m

    def mark(self, name: str, n: int = 1) -> None:
        if self.enabled:
            self.meter(name).mark(n)

    def observe(self, name: str, milliseconds: float) -> None:
        if self.enabled:
            self.meter(name).observe(milliseconds)

    def set_queue_depth(self, name: str, depth: int) -> None:
        if self.enabled:
            self.meter(name).set_queue_depth(depth)

    @contextmanager
    def timer(self, name: str, mark: int = 0) -> Iterator[None]:
        """Time a block; optionally also ``mark`` that many completed items."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - t0) * 1000.0)
            if mark:
                self.mark(name, mark)

    def snapshot(self) -> Dict[str, StageStats]:
        with self._lock:
            meters = list(self._meters.values())
        return {m.name: m.stats() for m in meters}

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self) -> None:
        with self._lock:
            for m in self._meters.values():
                m.reset()
            self._start = time.perf_counter()

    def format_table(self) -> str:
        """One-line-per-stage summary for the log."""
        snap = self.snapshot()
        if not snap:
            return "(no stages recorded)"
        width = max(len(k) for k in snap)
        lines = [f"{'stage'.ljust(width)} |    count |    fps |  avg ms |  p95 ms | queue"]
        lines.append("-" * len(lines[0]))
        for name in sorted(snap):
            s = snap[name]
            lines.append(
                f"{name.ljust(width)} | {s.count:8d} | {s.fps:6.2f} | "
                f"{s.avg_ms:7.2f} | {s.p95_ms:7.2f} | {s.queue_depth:5d}"
            )
        return "\n".join(lines)


@dataclass
class Stopwatch:
    """Trivial elapsed-time helper used by the benchmark harness."""

    start: float = field(default_factory=time.perf_counter)

    def reset(self) -> None:
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000.0


__all__ = ["StageStats", "RateMeter", "Profiler", "Stopwatch"]
