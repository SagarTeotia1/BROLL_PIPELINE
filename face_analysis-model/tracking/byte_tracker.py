"""ByteTrack: two-stage IoU association with a Kalman motion model.

Why ByteTrack for this pipeline: it recovers a face through short occlusions and motion
blur by giving *low-confidence* detections a second association pass. That directly cuts
ArcFace work - a track that survives an occlusion does not need to be re-identified.

Association order per sampled frame:

1. predict every track,
2. match **high-score** detections to tracked+lost tracks (IoU cost, Hungarian),
3. match the remaining tracked ones to **low-score** detections (recovery pass),
4. unmatched tracks go to ``Lost``; unmatched high-score detections start new tracks,
5. tracks lost for more than ``max_time_lost`` sampled frames are removed.

Distances use a fused IoU/score cost so a confident detection is preferred at equal
overlap.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from configs.config import TrackerConfig
from detector.base import Detection
from tracking.kalman_filter import KalmanFilterXYAH
from utils.logging_utils import get_logger

log = get_logger(__name__)


class TrackState(IntEnum):
    """Lifecycle of a track."""

    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """A single tracked face with Kalman state and detection payload."""

    _count = 0

    def __init__(self, detection: Detection, frame_id: int) -> None:
        self.det = detection
        self.tlbr = np.asarray(detection.bbox, dtype=np.float32).copy()
        self.score = float(detection.score)
        self.landmarks: Optional[np.ndarray] = (
            None if detection.landmarks is None else detection.landmarks.copy()
        )
        self.track_id: int = -1
        self.state: TrackState = TrackState.NEW
        self.is_activated = False

        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        self.time_since_update = 0

        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

    # -- id ------------------------------------------------------------------
    @staticmethod
    def next_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id_counter() -> None:
        """Restart IDs at 1 (called when a new video is loaded)."""
        STrack._count = 0

    # -- conversions ---------------------------------------------------------
    @staticmethod
    def tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        return np.array(
            [tlbr[0] + w * 0.5, tlbr[1] + h * 0.5, w / max(h, 1e-6), h], dtype=np.float32
        )

    @staticmethod
    def xyah_to_tlbr(xyah: np.ndarray) -> np.ndarray:
        h = xyah[3]
        w = xyah[2] * h
        return np.array(
            [xyah[0] - w * 0.5, xyah[1] - h * 0.5, xyah[0] + w * 0.5, xyah[1] + h * 0.5],
            dtype=np.float32,
        )

    @property
    def bbox(self) -> np.ndarray:
        """Current box as ``[x1, y1, x2, y2]`` (Kalman estimate once activated)."""
        if self.mean is None:
            return self.tlbr
        return self.xyah_to_tlbr(self.mean[:4])

    @property
    def area(self) -> float:
        b = self.bbox
        return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))

    # -- lifecycle -----------------------------------------------------------
    def activate(self, kalman: KalmanFilterXYAH, frame_id: int) -> None:
        self.track_id = STrack.next_id()
        self.mean, self.covariance = kalman.initiate(self.tlbr_to_xyah(self.tlbr))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = frame_id <= 1
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

    def re_activate(
        self, kalman: KalmanFilterXYAH, new_track: "STrack", frame_id: int, new_id: bool = False
    ) -> None:
        assert self.mean is not None and self.covariance is not None
        self.mean, self.covariance = kalman.update(
            self.mean, self.covariance, self.tlbr_to_xyah(new_track.tlbr)
        )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.det = new_track.det
        self.landmarks = new_track.landmarks
        self.time_since_update = 0
        if new_id:
            self.track_id = STrack.next_id()

    def update(self, kalman: KalmanFilterXYAH, new_track: "STrack", frame_id: int) -> None:
        assert self.mean is not None and self.covariance is not None
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = kalman.update(
            self.mean, self.covariance, self.tlbr_to_xyah(new_track.tlbr)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self.det = new_track.det
        self.landmarks = new_track.landmarks
        self.time_since_update = 0

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    @property
    def end_frame(self) -> int:
        return self.frame_id

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"STrack(id={self.track_id}, frames={self.start_frame}-{self.end_frame})"


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------
def iou_matrix(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``[x1, y1, x2, y2]`` boxes."""
    if atlbrs.size == 0 or btlbrs.size == 0:
        return np.zeros((atlbrs.shape[0], btlbrs.shape[0]), dtype=np.float32)
    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])

    lt = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    rb = np.minimum(atlbrs[:, None, 2:4], btlbrs[None, :, 2:4])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def iou_distance(tracks: Sequence[STrack], detections: Sequence[STrack]) -> np.ndarray:
    """``1 - IoU`` cost matrix."""
    a = np.array([t.bbox for t in tracks], dtype=np.float32).reshape(-1, 4)
    b = np.array([d.tlbr for d in detections], dtype=np.float32).reshape(-1, 4)
    return 1.0 - iou_matrix(a, b)


def fuse_score(cost: np.ndarray, detections: Sequence[STrack]) -> np.ndarray:
    """Bias the cost towards confident detections (ByteTrack's ``fuse_score``)."""
    if cost.size == 0:
        return cost
    iou_sim = 1.0 - cost
    det_scores = np.array([d.score for d in detections], dtype=np.float32)
    fused = iou_sim * np.tile(det_scores, (cost.shape[0], 1))
    return 1.0 - fused


def linear_assignment(
    cost: np.ndarray, threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian matching with a cost cut-off.

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)``.
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches: List[Tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= threshold:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_rows = [i for i in range(cost.shape[0]) if i not in matched_rows]
    unmatched_cols = [i for i in range(cost.shape[1]) if i not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class ByteTracker:
    """Multi-face tracker producing stable track IDs across sampled frames."""

    def __init__(self, config: TrackerConfig) -> None:
        self.cfg = config
        self.kalman = KalmanFilterXYAH()
        self.tracked: List[STrack] = []
        self.lost: List[STrack] = []
        self.removed: List[STrack] = []
        self.frame_id = 0

    def reset(self) -> None:
        """Forget everything (new video)."""
        self.tracked.clear()
        self.lost.clear()
        self.removed.clear()
        self.frame_id = 0
        STrack.reset_id_counter()

    # -- main loop ----------------------------------------------------------
    def update(self, detections: Sequence[Detection]) -> List[STrack]:
        """Advance the tracker one sampled frame.

        Args:
            detections: detections for this frame, in frame coordinates.

        Returns:
            The tracks that are currently active and confirmed.
        """
        self.frame_id += 1
        activated: List[STrack] = []
        refound: List[STrack] = []
        lost_now: List[STrack] = []
        removed_now: List[STrack] = []

        scores = np.array([d.score for d in detections], dtype=np.float32)
        high_mask = scores >= self.cfg.high_threshold
        low_mask = (scores >= self.cfg.low_threshold) & ~high_mask

        dets_high = [STrack(detections[i], self.frame_id) for i in np.nonzero(high_mask)[0]]
        dets_low = [STrack(detections[i], self.frame_id) for i in np.nonzero(low_mask)[0]]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked_pool = [t for t in self.tracked if t.is_activated]

        # --- predict --------------------------------------------------------
        pool: List[STrack] = tracked_pool + self.lost
        self._multi_predict(pool)

        # --- pass 1: high-score detections ----------------------------------
        # ``match_threshold`` is the maximum accepted *cost* (1 - fused IoU), matching
        # the original ByteTrack convention: 0.8 admits overlaps down to ~0.2 IoU.
        cost = fuse_score(iou_distance(pool, dets_high), dets_high)
        matches, u_track, u_det = linear_assignment(cost, self.cfg.match_threshold)
        for it, idet in matches:
            track, det = pool[it], dets_high[idet]
            if track.state == TrackState.TRACKED:
                track.update(self.kalman, det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(self.kalman, det, self.frame_id)
                refound.append(track)

        # --- pass 2: low-score recovery -------------------------------------
        remaining = [pool[i] for i in u_track if pool[i].state == TrackState.TRACKED]
        cost_low = iou_distance(remaining, dets_low)
        matches_low, u_track_low, _ = linear_assignment(
            cost_low, self.cfg.match_threshold_low
        )
        for it, idet in matches_low:
            track, det = remaining[it], dets_low[idet]
            if track.state == TrackState.TRACKED:
                track.update(self.kalman, det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(self.kalman, det, self.frame_id)
                refound.append(track)
        for i in u_track_low:
            track = remaining[i]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost_now.append(track)

        # Tracks from pass 1 that were never in ``remaining`` (already lost) stay lost.
        for i in u_track:
            track = pool[i]
            if track.state == TrackState.TRACKED and track not in remaining:
                track.mark_lost()
                lost_now.append(track)

        # --- unconfirmed tracks (born last frame) ---------------------------
        leftover_high = [dets_high[i] for i in u_det]
        cost_unconf = fuse_score(iou_distance(unconfirmed, leftover_high), leftover_high)
        matches_u, u_unconf, u_det2 = linear_assignment(
            cost_unconf, self.cfg.match_threshold_new
        )
        for it, idet in matches_u:
            unconfirmed[it].update(self.kalman, leftover_high[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for i in u_unconf:
            unconfirmed[i].mark_removed()
            removed_now.append(unconfirmed[i])

        # --- births ---------------------------------------------------------
        for i in u_det2:
            det = leftover_high[i]
            if det.score < self.cfg.new_track_threshold:
                continue
            det.activate(self.kalman, self.frame_id)
            activated.append(det)

        # --- retire stale tracks --------------------------------------------
        for track in self.lost:
            if self.frame_id - track.end_frame > self.cfg.max_time_lost:
                track.mark_removed()
                removed_now.append(track)

        # --- bookkeeping -----------------------------------------------------
        self.tracked = [t for t in self.tracked if t.state == TrackState.TRACKED]
        self.tracked = _merge(self.tracked, activated)
        self.tracked = _merge(self.tracked, refound)
        self.lost = _subtract(self.lost, self.tracked)
        self.lost.extend(lost_now)
        self.lost = _subtract(self.lost, removed_now)
        self.removed.extend(removed_now)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)
        # Keep the removed list bounded on long videos.
        if len(self.removed) > 4096:
            self.removed = self.removed[-1024:]

        return [
            t for t in self.tracked
            if t.is_activated or (self.frame_id - t.start_frame + 1) >= self.cfg.min_hits
        ]

    # -- helpers ------------------------------------------------------------
    def _multi_predict(self, tracks: Sequence[STrack]) -> None:
        if not tracks:
            return
        means = np.stack([t.mean for t in tracks], axis=0)
        covs = np.stack([t.covariance for t in tracks], axis=0)
        # A lost track has no velocity observation; freeze aspect/height velocity.
        for i, t in enumerate(tracks):
            if t.state != TrackState.TRACKED:
                means[i][7] = 0.0
        means, covs = self.kalman.multi_predict(means, covs)
        for i, t in enumerate(tracks):
            t.mean, t.covariance = means[i], covs[i]
            t.time_since_update += 1

    @property
    def active_ids(self) -> List[int]:
        return [t.track_id for t in self.tracked]


def _merge(base: List[STrack], extra: Sequence[STrack]) -> List[STrack]:
    seen = {t.track_id for t in base}
    out = list(base)
    for t in extra:
        if t.track_id not in seen:
            seen.add(t.track_id)
            out.append(t)
    return out


def _subtract(base: Sequence[STrack], remove: Sequence[STrack]) -> List[STrack]:
    ids = {t.track_id for t in remove}
    return [t for t in base if t.track_id not in ids]


def _remove_duplicates(
    a: List[STrack], b: List[STrack]
) -> Tuple[List[STrack], List[STrack]]:
    """Drop near-identical boxes that exist in both pools, keeping the older track."""
    if not a or not b:
        return a, b
    dist = iou_distance(a, b)
    pairs = np.nonzero(dist < 0.15)
    dup_a: set[int] = set()
    dup_b: set[int] = set()
    for i, j in zip(*pairs):
        len_a = a[i].frame_id - a[i].start_frame
        len_b = b[j].frame_id - b[j].start_frame
        if len_a > len_b:
            dup_b.add(int(j))
        else:
            dup_a.add(int(i))
    return (
        [t for i, t in enumerate(a) if i not in dup_a],
        [t for j, t in enumerate(b) if j not in dup_b],
    )


__all__ = [
    "ByteTracker",
    "STrack",
    "TrackState",
    "iou_matrix",
    "iou_distance",
    "linear_assignment",
]
