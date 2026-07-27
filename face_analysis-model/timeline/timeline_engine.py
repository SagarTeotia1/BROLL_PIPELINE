"""Event assembly: turns a stream of per-frame observations into emotion spans.

The engine keeps one *open* event per identity. An observation either extends the open
event (same emotion) or closes it and opens a new one. Nothing is emitted per frame.

Identity keying matters for quality: events are keyed by **actor id** when the identity
is known, so a track that dies behind an occlusion and comes back as a new track ID
continues the same event instead of fragmenting it. Unrecognised faces fall back to a
per-track key (``track:7``) and are only exported when
``config.timeline.include_unknown`` is set.

Post-processing on close:

* an event shorter than ``min_event_duration`` is absorbed into the previous one
  (a 130 ms blip is a smoothing artefact, not an emotional beat),
* consecutive events with the same emotion separated by less than ``merge_gap``
  (the actor turned away for half a second) are merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from configs.config import TimelineConfig
from recognition.matcher import UNKNOWN
from timeline.events import (
    ActorEntry,
    AnalysisInfo,
    TimelineDocument,
    TimelineEvent,
    VideoInfo,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)

EventKey = Tuple[str, int]  # ("actor", actor_id) or ("track", track_id)


@dataclass
class _OpenEvent:
    """Mutable accumulator for the event currently being built for one identity."""

    actor: str
    actor_id: int
    emotion: str
    start: float
    end: float
    start_frame: int
    end_frame: int
    conf_sum: float = 0.0
    sim_sum: float = 0.0
    samples: int = 0
    track_ids: List[int] = field(default_factory=list)
    last_update: float = 0.0

    def add(self, confidence: float, similarity: float, timestamp: float, frame: int, track_id: int) -> None:
        self.end = max(self.end, timestamp)
        self.end_frame = max(self.end_frame, frame)
        self.conf_sum += confidence
        self.sim_sum += similarity
        self.samples += 1
        self.last_update = timestamp
        if track_id not in self.track_ids:
            self.track_ids.append(track_id)

    def finish(self, end_time: Optional[float] = None) -> TimelineEvent:
        end = self.end if end_time is None else max(self.end, end_time)
        n = max(1, self.samples)
        return TimelineEvent(
            actor=self.actor,
            emotion=self.emotion,
            start=self.start,
            end=end,
            confidence=self.conf_sum / n,
            actor_id=self.actor_id,
            track_ids=list(self.track_ids),
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            samples=self.samples,
            similarity=self.sim_sum / n,
        )


class TimelineEngine:
    """Builds the event timeline incrementally as observations arrive.

    Thread affinity: all mutating methods are expected to be called from the single
    timeline thread. :meth:`snapshot` is safe to call from the GUI thread because it
    only reads finished events (a shallow copy of the list).
    """

    def __init__(self, config: TimelineConfig) -> None:
        self.cfg = config
        self._open: Dict[EventKey, _OpenEvent] = {}
        self._closed: Dict[EventKey, List[TimelineEvent]] = {}
        self._actor_names: Dict[int, str] = {}
        self._similarity: Dict[int, List[float]] = {}
        self.last_timestamp: float = 0.0

    # -- ingestion ----------------------------------------------------------
    def observe(
        self,
        track_id: int,
        actor_id: int,
        actor_name: str,
        emotion: str,
        confidence: float,
        timestamp: float,
        frame_index: int,
        similarity: float = 0.0,
    ) -> Optional[TimelineEvent]:
        """Feed one smoothed observation.

        Returns the event that was *closed* by this observation, if any - the GUI uses
        that to append rows to the timeline widget as soon as they are final.
        """
        self.last_timestamp = max(self.last_timestamp, timestamp)
        if actor_id < 0 and not self.cfg.include_unknown:
            return None

        key: EventKey = ("actor", actor_id) if actor_id >= 0 else ("track", track_id)
        name = actor_name or (UNKNOWN if actor_id < 0 else f"Actor {actor_id}")
        if actor_id >= 0:
            self._actor_names[actor_id] = name
            self._similarity.setdefault(actor_id, []).append(similarity)

        current = self._open.get(key)
        if current is None:
            self._open[key] = _OpenEvent(
                actor=name, actor_id=actor_id, emotion=emotion,
                start=timestamp, end=timestamp,
                start_frame=frame_index, end_frame=frame_index,
                last_update=timestamp,
            )
            self._open[key].add(confidence, similarity, timestamp, frame_index, track_id)
            return None

        if current.emotion == emotion:
            current.actor = name  # a late identity lock renames the open event
            current.add(confidence, similarity, timestamp, frame_index, track_id)
            return None

        # Emotion changed -> close the previous span at this instant and open a new one.
        closed = self._close(key, end_time=timestamp)
        self._open[key] = _OpenEvent(
            actor=name, actor_id=actor_id, emotion=emotion,
            start=timestamp, end=timestamp,
            start_frame=frame_index, end_frame=frame_index,
            last_update=timestamp,
        )
        self._open[key].add(confidence, similarity, timestamp, frame_index, track_id)
        return closed

    def close_stale(self, now: float) -> List[TimelineEvent]:
        """Close events whose identity has not been observed recently."""
        closed: List[TimelineEvent] = []
        for key in list(self._open):
            open_ev = self._open[key]
            if now - open_ev.last_update > self.cfg.close_track_after:
                ev = self._close(key)
                if ev is not None:
                    closed.append(ev)
        return closed

    def close_all(self, end_time: Optional[float] = None) -> List[TimelineEvent]:
        """Close every open event (end of video)."""
        closed: List[TimelineEvent] = []
        for key in list(self._open):
            ev = self._close(key, end_time=end_time)
            if ev is not None:
                closed.append(ev)
        return closed

    # -- internals ----------------------------------------------------------
    def _close(self, key: EventKey, end_time: Optional[float] = None) -> Optional[TimelineEvent]:
        open_ev = self._open.pop(key, None)
        if open_ev is None:
            return None
        event = open_ev.finish(end_time)
        bucket = self._closed.setdefault(key, [])

        if bucket:
            prev = bucket[-1]
            # Same emotion resuming after a short gap -> one continuous event.
            if prev.emotion == event.emotion and (event.start - prev.end) <= self.cfg.merge_gap:
                total = prev.samples + event.samples
                prev.confidence = (
                    prev.confidence * prev.samples + event.confidence * event.samples
                ) / max(1, total)
                prev.similarity = (
                    prev.similarity * prev.samples + event.similarity * event.samples
                ) / max(1, total)
                prev.samples = total
                prev.end = max(prev.end, event.end)
                prev.end_frame = max(prev.end_frame, event.end_frame)
                prev.track_ids = sorted(set(prev.track_ids) | set(event.track_ids))
                return None
            # Too short to be a real beat -> absorb into the previous event.
            if event.duration < self.cfg.min_event_duration:
                prev.end = max(prev.end, event.end)
                prev.end_frame = max(prev.end_frame, event.end_frame)
                return None
        elif event.duration < self.cfg.min_event_duration:
            # No predecessor to absorb it: keep it only if it carries real confidence.
            if event.samples < 2:
                return None

        bucket.append(event)
        return event

    # -- output -------------------------------------------------------------
    def events(self, include_open: bool = False) -> List[TimelineEvent]:
        """All finished events (optionally including the currently open ones)."""
        out: List[TimelineEvent] = []
        for bucket in self._closed.values():
            out.extend(bucket)
        if include_open:
            out.extend(ev.finish() for ev in self._open.values())
        out.sort(key=lambda e: (e.start, e.actor, e.emotion))
        return out

    def snapshot(self) -> List[TimelineEvent]:
        """Cheap read-only copy for the GUI."""
        return self.events(include_open=True)

    def build_document(
        self,
        video: VideoInfo,
        stats: Optional[Dict[str, object]] = None,
        analysis: Optional[AnalysisInfo] = None,
    ) -> TimelineDocument:
        """Close everything and assemble the final document."""
        self.close_all(end_time=video.duration if video.duration > 0 else None)
        events = self.events()

        actors: Dict[int, ActorEntry] = {}
        for ev in events:
            if ev.actor_id < 0 and not self.cfg.include_unknown:
                continue
            entry = actors.get(ev.actor_id)
            if entry is None:
                entry = ActorEntry(id=ev.actor_id, name=ev.actor, first_seen=ev.start)
                actors[ev.actor_id] = entry
            entry.first_seen = min(entry.first_seen, ev.start)
            entry.last_seen = max(entry.last_seen, ev.end)
            entry.screen_time += ev.duration
            entry.events += 1
            entry.emotions[ev.emotion] = entry.emotions.get(ev.emotion, 0.0) + ev.duration
        for actor_id, entry in actors.items():
            sims = self._similarity.get(actor_id, [])
            entry.mean_similarity = float(sum(sims) / len(sims)) if sims else 0.0

        # Count real transitions per actor: walk their events in time order and count
        # only the moves to a *different* emotion.
        for actor_id, entry in actors.items():
            previous: Optional[str] = None
            changes = 0
            for ev in events:
                if ev.actor_id != actor_id:
                    continue
                if previous is not None and ev.emotion != previous:
                    changes += 1
                previous = ev.emotion
            entry.expression_changes = changes

        ordered = sorted(actors.values(), key=lambda a: (a.id < 0, a.first_seen))
        info = analysis or AnalysisInfo()
        # The document is the single source of truth for this number, so the top-level
        # total and the per-actor totals can never disagree.
        info.expression_changes = sum(a.expression_changes for a in ordered)
        doc = TimelineDocument(
            video=video, actors=ordered, events=events,
            stats=dict(stats or {}), analysis=info,
        )
        log.info(
            "Timeline built: %d events / %d expression changes across %d actors (%.1fs of video)",
            len(events), info.expression_changes, len(ordered), video.duration,
        )
        return doc

    def reset(self) -> None:
        self._open.clear()
        self._closed.clear()
        self._actor_names.clear()
        self._similarity.clear()
        self.last_timestamp = 0.0

    @property
    def open_count(self) -> int:
        return len(self._open)

    @property
    def event_count(self) -> int:
        return sum(len(b) for b in self._closed.values())


__all__ = ["TimelineEngine", "EventKey"]
