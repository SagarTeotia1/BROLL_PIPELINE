"""Per-track identity state and the ArcFace scheduling policy.

The tracker gives persistent IDs; this module decides **when a track needs ArcFace**
and turns a stream of noisy matches into a locked identity through majority voting.

Scheduling rules (a track is queued for recognition when any holds):

* it has never been embedded (new track),
* its identity is not locked yet and it has fewer than ``min_votes`` agreeing votes,
* the last match was ambiguous (small margin to the runner-up),
* ``reid_interval`` sampled frames have passed since the last embedding - guards against
  identity swaps after an occlusion,
* the track was just re-activated after being lost.

In steady state this costs roughly one ArcFace call per actor per ``reid_interval``
sampled frames instead of one per frame, which is where most of the speed-up lives.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Sequence

import numpy as np

from configs.config import RecognitionConfig
from recognition.matcher import UNKNOWN, MatchResult
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class TrackIdentity:
    """Identity state accumulated for one track ID."""

    track_id: int
    actor_id: int = -1
    name: str = UNKNOWN
    similarity: float = 0.0
    margin: float = 0.0
    locked: bool = False
    votes: Deque[int] = field(default_factory=lambda: deque(maxlen=8))
    vote_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    embeddings_run: int = 0
    frames_since_reid: int = 10_000
    first_frame: int = -1
    last_frame: int = -1
    first_time: float = 0.0
    last_time: float = 0.0
    pending: bool = False          # an ArcFace call is already queued for this track

    @property
    def is_known(self) -> bool:
        return self.actor_id >= 0

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "actor_id": self.actor_id,
            "name": self.name,
            "similarity": round(self.similarity, 4),
            "locked": self.locked,
            "embeddings_run": self.embeddings_run,
        }


class IdentityRegistry:
    """Owns :class:`TrackIdentity` records and the recognition schedule."""

    def __init__(self, config: RecognitionConfig) -> None:
        self.cfg = config
        self._tracks: Dict[int, TrackIdentity] = {}

    # -- accessors ----------------------------------------------------------
    def get(self, track_id: int) -> TrackIdentity:
        rec = self._tracks.get(track_id)
        if rec is None:
            rec = TrackIdentity(track_id=track_id)
            rec.votes = deque(maxlen=max(2, self.cfg.vote_window))
            rec.vote_scores = deque(maxlen=max(2, self.cfg.vote_window))
            self._tracks[track_id] = rec
        return rec

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._tracks

    def __len__(self) -> int:
        return len(self._tracks)

    def all(self) -> List[TrackIdentity]:
        return list(self._tracks.values())

    # -- scheduling ---------------------------------------------------------
    def tick(self, active_ids: Sequence[int]) -> None:
        """Advance the per-track re-identification counters by one sampled frame."""
        for tid in active_ids:
            self.get(tid).frames_since_reid += 1

    def needs_recognition(self, track_id: int) -> bool:
        """Whether ArcFace should run for this track on the current frame."""
        rec = self.get(track_id)
        if rec.pending:
            return False
        if rec.embeddings_run == 0:
            return True
        if not rec.locked:
            return True
        if rec.frames_since_reid >= self.cfg.reid_interval:
            return True
        # A *named* track whose similarity has decayed is worth re-checking - it may be
        # drifting onto a different person. A track locked as Unknown is always below
        # the threshold by definition, so the same rule there would re-embed every
        # background face on every frame; it waits for ``reid_interval`` instead.
        if rec.is_known and rec.similarity < self.cfg.similarity_threshold:
            return True
        return False

    def mark_pending(self, track_id: int) -> None:
        """Flag that a recognition request is in flight (prevents duplicate work)."""
        self.get(track_id).pending = True

    # -- updates ------------------------------------------------------------
    def update(
        self,
        track_id: int,
        match: MatchResult,
        frame_index: int,
        timestamp: float,
    ) -> TrackIdentity:
        """Fold a new match into the track's identity vote.

        The locked name is the majority of the last ``vote_window`` matches, and only
        becomes locked once ``min_votes`` of them agree - so one bad frame cannot
        rename an actor mid-shot.
        """
        rec = self.get(track_id)
        rec.pending = False
        rec.embeddings_run += 1
        rec.frames_since_reid = 0
        rec.last_frame = frame_index
        rec.last_time = timestamp
        if rec.first_frame < 0:
            rec.first_frame = frame_index
            rec.first_time = timestamp

        rec.votes.append(match.actor_id)
        rec.vote_scores.append(match.similarity)

        counts = Counter(rec.votes)
        winner_id, winner_votes = counts.most_common(1)[0]
        needed = min(self.cfg.min_votes, max(1, len(rec.votes)))

        if winner_id >= 0 and winner_votes >= needed:
            if not rec.locked or winner_id != rec.actor_id:
                if rec.locked and winner_id != rec.actor_id:
                    log.debug(
                        "Track %d identity changed %s -> id %d", track_id, rec.name, winner_id
                    )
                rec.actor_id = winner_id
                rec.name = match.name if match.actor_id == winner_id else rec.name
            rec.locked = True
        elif winner_id < 0 and winner_votes >= needed:
            rec.actor_id = -1
            rec.name = UNKNOWN
            rec.locked = True

        # Name may still be empty if the winning vote came from an earlier match.
        if match.actor_id == rec.actor_id and match.name:
            rec.name = match.name
        if rec.actor_id < 0:
            rec.name = UNKNOWN

        # Report the mean similarity of the winning votes: less jumpy than the last one.
        winning = [
            s for a, s in zip(rec.votes, rec.vote_scores) if a == rec.actor_id
        ]
        rec.similarity = float(np.mean(winning)) if winning else match.similarity
        rec.margin = match.margin
        return rec

    def touch(self, track_id: int, frame_index: int, timestamp: float) -> TrackIdentity:
        """Record that a track was seen this frame (no recognition performed)."""
        rec = self.get(track_id)
        rec.last_frame = frame_index
        rec.last_time = timestamp
        if rec.first_frame < 0:
            rec.first_frame = frame_index
            rec.first_time = timestamp
        return rec

    def unlock(self, track_id: int) -> None:
        """Force a re-identification (used after scene cuts)."""
        rec = self.get(track_id)
        rec.locked = False
        rec.frames_since_reid = 10_000

    # -- lifecycle ----------------------------------------------------------
    def prune(self, alive: Sequence[int]) -> List[int]:
        """Drop records for tracks that no longer exist."""
        alive_set = set(alive)
        dead = [tid for tid in self._tracks if tid not in alive_set]
        for tid in dead:
            del self._tracks[tid]
        return dead

    def prune_stale(self, current_frame: int, max_age: int = 600) -> List[int]:
        """Drop records not observed for ``max_age`` source frames.

        Preferred over :meth:`prune` inside the pipeline: it needs no access to the
        tracker's live state, so it is safe to call from the timeline thread.
        """
        dead = [
            tid
            for tid, rec in self._tracks.items()
            if rec.last_frame >= 0 and (current_frame - rec.last_frame) > max_age
        ]
        for tid in dead:
            del self._tracks[tid]
        return dead

    def reset(self) -> None:
        self._tracks.clear()

    def known_actors(self) -> Dict[int, str]:
        """Actor id -> name for every identity seen so far."""
        return {
            rec.actor_id: rec.name
            for rec in self._tracks.values()
            if rec.actor_id >= 0
        }


__all__ = ["TrackIdentity", "IdentityRegistry"]
