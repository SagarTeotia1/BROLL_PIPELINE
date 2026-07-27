"""Timeline data model: events, video metadata and the serialisable document.

The output contract is *event based*: one record per emotion **span**, never per frame.

    {"actor": "John", "emotion": "Happy", "start": 1.52, "end": 8.32, "confidence": 0.96}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class VideoInfo:
    """Static description of the analysed clip."""

    path: str = ""
    duration: float = 0.0
    fps: float = 0.0
    processed_fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    frames_analysed: int = 0
    decoder: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "video_duration": round(self.duration, 3),
            "fps": round(self.fps, 3),
            "processed_fps": round(self.processed_fps, 3),
            "frame_count": self.frame_count,
            "frames_analysed": self.frames_analysed,
            "width": self.width,
            "height": self.height,
            "decoder": self.decoder,
        }


@dataclass
class AnalysisInfo:
    """How long the analysis took and how much work it did.

    Surfaced at the top level of the JSON because "how long did this take" is a
    first-class question for an editor deciding whether to run a batch overnight.
    """

    elapsed_seconds: float = 0.0        # wall-clock time of the analysis run
    realtime_factor: float = 0.0        # video seconds processed per wall second
    frames_decoded: int = 0
    frames_analysed: int = 0
    faces_processed: int = 0
    expression_changes: int = 0         # total emotion switches across all actors
    recognition_calls: int = 0
    device: str = ""
    decoder: str = ""
    started_at: str = ""
    finished_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "elapsed_readable": _readable_duration(self.elapsed_seconds),
            "realtime_factor": round(self.realtime_factor, 3),
            "frames_decoded": self.frames_decoded,
            "frames_analysed": self.frames_analysed,
            "faces_processed": self.faces_processed,
            "expression_changes": self.expression_changes,
            "recognition_calls": self.recognition_calls,
            "device": self.device,
            "decoder": self.decoder,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _readable_duration(seconds: float) -> str:
    """``72.4 s`` / ``1 m 12 s`` / ``1 h 03 m`` for humans reading the JSON."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} m"
    return f"{minutes} m {secs:02d} s"


@dataclass
class ActorEntry:
    """An actor that actually appears in the video."""

    id: int
    name: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    screen_time: float = 0.0
    events: int = 0
    mean_similarity: float = 0.0
    emotions: Dict[str, float] = field(default_factory=dict)
    # Transitions to a *different* emotion. Deliberately not ``events - 1``: an actor
    # who leaves frame and comes back still Neutral produces two Neutral events but has
    # not changed expression, and the first event is an appearance, not a change.
    expression_changes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "screen_time": round(self.screen_time, 3),
            "events": self.events,
            "expression_changes": self.expression_changes,
            "mean_similarity": round(self.mean_similarity, 4),
            "emotion_totals": {k: round(v, 3) for k, v in self.emotions.items()},
        }


@dataclass
class TimelineEvent:
    """A contiguous span during which one actor held one emotion."""

    actor: str
    emotion: str
    start: float
    end: float
    confidence: float = 0.0
    actor_id: int = -1
    track_ids: List[int] = field(default_factory=list)
    start_frame: int = -1
    end_frame: int = -1
    samples: int = 0
    similarity: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlaps(self, other: "TimelineEvent") -> bool:
        return self.start < other.end and other.start < self.end

    def as_dict(self, verbose: bool = False) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "actor": self.actor,
            "emotion": self.emotion,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "confidence": round(self.confidence, 4),
        }
        if verbose:
            base.update(
                {
                    "actor_id": self.actor_id,
                    "track_ids": sorted(set(self.track_ids)),
                    "start_frame": self.start_frame,
                    "end_frame": self.end_frame,
                    "samples": self.samples,
                    "similarity": round(self.similarity, 4),
                }
            )
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineEvent":
        return cls(
            actor=data["actor"],
            emotion=data["emotion"],
            start=float(data["start"]),
            end=float(data["end"]),
            confidence=float(data.get("confidence", 0.0)),
            actor_id=int(data.get("actor_id", -1)),
            track_ids=list(data.get("track_ids", [])),
            start_frame=int(data.get("start_frame", -1)),
            end_frame=int(data.get("end_frame", -1)),
            samples=int(data.get("samples", 0)),
            similarity=float(data.get("similarity", 0.0)),
        )


@dataclass
class TimelineDocument:
    """The complete analysis result, ready for JSON/CSV export."""

    video: VideoInfo = field(default_factory=VideoInfo)
    actors: List[ActorEntry] = field(default_factory=list)
    events: List[TimelineEvent] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    analysis: AnalysisInfo = field(default_factory=AnalysisInfo)

    def as_dict(self, verbose: bool = False) -> Dict[str, Any]:
        """Serialise the document.

        The default (``verbose=False``) is the answer to "who reacted how, and when":
        video header, analysis timing, the cast, and the event list. The per-stage
        profiler dump and the per-event debug fields (track ids, frame numbers, sample
        counts) only appear with ``verbose=True`` - they are diagnostics, not results,
        and they drown the part anyone actually reads.
        """
        info = self.video.as_dict()
        payload: Dict[str, Any] = {
            "video": Path(self.video.path).name,
            "video_duration": info["video_duration"],
            "fps": info["fps"],
            "processed_fps": info["processed_fps"],
            "analysis_time_seconds": round(self.analysis.elapsed_seconds, 3),
            "analysis_time": _readable_duration(self.analysis.elapsed_seconds),
            "frames_total": info["frame_count"],
            "frames_analysed": info["frames_analysed"],
            "frames_analysed_percent": (
                round(100.0 * info["frames_analysed"] / info["frame_count"], 1)
                if info["frame_count"] else 0.0
            ),
            "expression_changes": sum(a.expression_changes for a in self.actors),
            "actors": [a.as_dict() for a in self.actors],
            "events": [e.as_dict(verbose) for e in self.events],
        }
        if verbose:
            payload["analysis"] = self.analysis.as_dict()
            payload["video_info"] = info
            payload["stats"] = self.stats
        return payload

    def to_json(self, indent: int = 2, verbose: bool = False) -> str:
        return json.dumps(self.as_dict(verbose), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "TimelineDocument":
        """Load either the lean or the verbose form."""
        data = json.loads(text)
        info_raw = data.get("video_info", {})
        video = VideoInfo(
            path=info_raw.get("path", data.get("video", "")),
            duration=float(info_raw.get("video_duration", data.get("video_duration", 0.0))),
            fps=float(info_raw.get("fps", data.get("fps", 0.0))),
            processed_fps=float(info_raw.get("processed_fps", data.get("processed_fps", 0.0))),
            frame_count=int(info_raw.get("frame_count", 0)),
            width=int(info_raw.get("width", 0)),
            height=int(info_raw.get("height", 0)),
            frames_analysed=int(info_raw.get("frames_analysed", 0)),
            decoder=info_raw.get("decoder", ""),
        )
        actors = [
            ActorEntry(
                id=int(a["id"]),
                name=a["name"],
                first_seen=float(a.get("first_seen", 0.0)),
                last_seen=float(a.get("last_seen", 0.0)),
                screen_time=float(a.get("screen_time", 0.0)),
                events=int(a.get("events", 0)),
                mean_similarity=float(a.get("mean_similarity", 0.0)),
                emotions=dict(a.get("emotion_totals", {})),
                expression_changes=int(a.get("expression_changes", 0)),
            )
            for a in data.get("actors", [])
        ]
        events = [TimelineEvent.from_dict(e) for e in data.get("events", [])]
        raw_analysis = data.get("analysis", {})
        analysis = AnalysisInfo(
            elapsed_seconds=float(
                raw_analysis.get(
                    "elapsed_seconds", data.get("analysis_time_seconds", 0.0)
                )
            ),
            realtime_factor=float(raw_analysis.get("realtime_factor", 0.0)),
            frames_decoded=int(raw_analysis.get("frames_decoded", 0)),
            frames_analysed=int(raw_analysis.get("frames_analysed", 0)),
            faces_processed=int(raw_analysis.get("faces_processed", 0)),
            expression_changes=int(
                raw_analysis.get("expression_changes", data.get("expression_changes", 0))
            ),
            recognition_calls=int(raw_analysis.get("recognition_calls", 0)),
            device=raw_analysis.get("device", ""),
            decoder=raw_analysis.get("decoder", ""),
            started_at=raw_analysis.get("started_at", ""),
            finished_at=raw_analysis.get("finished_at", ""),
        )
        return cls(
            video=video, actors=actors, events=events,
            stats=data.get("stats", {}), analysis=analysis,
        )

    # -- queries ------------------------------------------------------------
    def events_for(self, actor: str) -> List[TimelineEvent]:
        return [e for e in self.events if e.actor == actor]

    def at(self, timestamp: float) -> List[TimelineEvent]:
        """Every event active at a given time (used by the GUI playhead)."""
        return [e for e in self.events if e.start <= timestamp <= e.end]

    def emotion_totals(self) -> Dict[str, float]:
        """Total seconds spent in each emotion across all actors."""
        totals: Dict[str, float] = {}
        for e in self.events:
            totals[e.emotion] = totals.get(e.emotion, 0.0) + e.duration
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    def __len__(self) -> int:
        return len(self.events)


__all__ = [
    "VideoInfo",
    "AnalysisInfo",
    "ActorEntry",
    "TimelineEvent",
    "TimelineDocument",
]
