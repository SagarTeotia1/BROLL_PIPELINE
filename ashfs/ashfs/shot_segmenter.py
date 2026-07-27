"""Shot segmentation module for ASH-FS pipeline.

Pipeline:
  1. PySceneDetect (ContentDetector, threshold=27.0) — hard cuts.
  2. DINOv2-embedding GLRT refinement — soft boundaries via windowed cosine distance.
  3. Optical-flow secondary boundary detection (opt-in, improvement #5).
  4. Min/max shot duration enforcement — merge short shots, force-split long shots.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import cv2
import numpy as np

from ashfs.config import Config
from ashfs.text_change_detector import TextChangeDetector
from ashfs.types import Shot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCENEDETECT_THRESHOLD = 27.0


# ---------------------------------------------------------------------------
# Helper: cosine distance
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 1] between two L2-normalised vectors."""
    dot = float(np.dot(a, b))
    # Clamp to [-1, 1] to guard against fp rounding past unit norm
    return 1.0 - max(-1.0, min(1.0, dot))


def _mean_cosine_dist(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Mean pairwise cosine distance between two groups of embeddings.

    Both arrays must have shape (N, D) and (M, D) respectively.
    Returns 0.0 when either group is empty.
    """
    if len(group_a) == 0 or len(group_b) == 0:
        return 0.0
    # group_a: (N, D), group_b: (M, D) → dot products: (N, M)
    dots = np.clip(group_a @ group_b.T, -1.0, 1.0)
    return float(np.mean(1.0 - dots))


# ---------------------------------------------------------------------------
# ShotSegmenter
# ---------------------------------------------------------------------------

class ShotSegmenter:
    """Detects shots in a video using hard-cut detection + DINOv2 soft boundaries.

    Parameters
    ----------
    config:
        Parsed pipeline configuration.  The relevant sub-config is
        ``config.shot_segmentation``.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config.shot_segmentation
        self._cfg_root = config  # retained for cross-config access in text-change method

    # ------------------------------------------------------------------
    # Public: frame sampling
    # ------------------------------------------------------------------

    def sample_candidate_frames(
        self, video_path: str
    ) -> tuple[list[int], Callable[[int], np.ndarray]]:
        """Sample frame indices at ``candidate_fps`` from the video.

        Returns
        -------
        frame_indices:
            Sorted list of absolute frame indices selected for embedding.
        frame_loader_fn:
            Callable ``(idx: int) -> np.ndarray`` — loads frame *idx* on
            demand via cv2.  The returned image is RGB uint8 (H, W, 3).
            Raises ``ValueError`` on seek/decode failure.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video for frame sampling: {video_path!r}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if total_frames <= 0 or video_fps <= 0:
            raise ValueError(
                f"Video has invalid properties (frames={total_frames}, fps={video_fps}): "
                f"{video_path!r}"
            )

        # Adaptive candidate_fps: scale with video duration so candidate count
        # stays proportional.  Short clips (<5 min) get full 4fps resolution;
        # long clips (>60 min) drop to 2fps to keep memory/time manageable.
        # Breakpoints: <5min→4fps, 5-20min→3fps, 20-60min→2.5fps, >60min→2fps
        duration_s = total_frames / video_fps
        if duration_s < 300:          # < 5 min
            eff_fps = min(self._cfg.candidate_fps, 4.0)
        elif duration_s < 1200:       # 5–20 min
            eff_fps = min(self._cfg.candidate_fps, 3.0)
        elif duration_s < 3600:       # 20–60 min
            eff_fps = min(self._cfg.candidate_fps, 2.5)
        else:                         # > 60 min
            eff_fps = min(self._cfg.candidate_fps, 2.0)

        logger.info(
            "Video duration %.1fs — adaptive candidate_fps: %.1f (config: %.1f)",
            duration_s, eff_fps, self._cfg.candidate_fps,
        )

        step = max(1, round(video_fps / eff_fps))
        frame_indices = list(range(0, total_frames, step))

        # Sequential reader: one persistent VideoCapture, uses grab() to skip
        # frames without decoding — 100x faster than re-opening per frame.
        class _SequentialReader:
            def __init__(self, path: str) -> None:
                self._path = path
                self._cap = cv2.VideoCapture(path)
                self._pos = 0

            def __call__(self, idx: int) -> np.ndarray:
                if idx < self._pos:
                    # Backward seek — rare, but handle by reopening
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self._path)
                    self._pos = 0
                # Skip forward without decoding
                while self._pos < idx:
                    self._cap.grab()
                    self._pos += 1
                ok, frame = self._cap.read()
                self._pos += 1
                if not ok or frame is None:
                    raise ValueError(
                        f"Failed to decode frame {idx} from {self._path!r}"
                    )
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            def __del__(self) -> None:
                try:
                    self._cap.release()
                except Exception:
                    pass

        return frame_indices, _SequentialReader(video_path)

    # ------------------------------------------------------------------
    # Public: main entry point
    # ------------------------------------------------------------------

    def detect_shots(
        self,
        video_path: str,
        embedding_manager,
        frame_indices: list[int] | None = None,
        embeddings_array: np.ndarray | None = None,
    ) -> list[Shot]:
        """Detect shots in *video_path* and return a list of :class:`Shot` objects.

        Parameters
        ----------
        video_path:
            Absolute path to the video file.
        embedding_manager:
            Pre-initialised embedding manager whose ``.get_embeddings(frame_indices,
            frame_loader_fn)`` returns an ``np.ndarray`` of shape ``(N, D)``.
            This method does **not** create a new EmbeddingManager — it receives
            the shared instance from the caller so embeddings are extracted once
            and reused across pipeline stages.
        frame_indices:
            Optional pre-sampled candidate frame indices.  When provided together
            with *embeddings_array* the internal re-extraction is skipped (avoids
            double extraction when called from the pipeline).
        embeddings_array:
            Optional pre-extracted embeddings of shape ``(N, D)`` in the same
            order as *frame_indices*.  Both must be provided together or both
            must be ``None``.
        """
        # --- validate video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path!r}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if total_frames <= 0 or video_fps <= 0:
            raise ValueError(
                f"Video has invalid properties (frames={total_frames}, fps={video_fps}): "
                f"{video_path!r}"
            )

        # ----------------------------------------------------------------
        # Step 1 — PySceneDetect hard cuts
        # ----------------------------------------------------------------
        hard_boundaries = self._detect_hard_cuts(video_path)

        # ----------------------------------------------------------------
        # Step 2 — DINOv2 soft boundaries (GLRT-style)
        # ----------------------------------------------------------------
        # Use pre-extracted data if provided to avoid double extraction.
        if frame_indices is not None and embeddings_array is not None:
            candidate_frame_indices = frame_indices
            # embeddings_array already provided — no re-extraction needed
        else:
            candidate_frame_indices, frame_loader = self.sample_candidate_frames(video_path)
            embeddings_array = embedding_manager.get_embeddings(
                candidate_frame_indices, frame_loader
            )
        # embeddings_array: shape (len(candidate_frame_indices), D)

        soft_boundaries = self._detect_soft_boundaries(
            candidate_frame_indices=candidate_frame_indices,
            embeddings=embeddings_array,
            hard_boundaries=hard_boundaries,
        )

        # ----------------------------------------------------------------
        # Step 3 (opt-in) — optical-flow secondary boundary detection
        # ----------------------------------------------------------------
        flow_boundaries: list[int] = []
        if self._cfg.flow_boundary_enabled:
            # We need a frame_loader — create one if not already available
            if frame_indices is not None and embeddings_array is not None:
                # frame_loader was not returned from the pre-extracted path;
                # open a new one just for flow (cheap: only reads resized grays)
                _fi_inner, _fl_inner = self.sample_candidate_frames(video_path)
                _flow_loader = _fl_inner
            else:
                # frame_loader was assigned above in the else branch but is
                # not in scope here — re-open.  sample_candidate_frames is
                # idempotent and uses grab()-based skipping, so this is safe.
                _fi_inner, _flow_loader = self.sample_candidate_frames(video_path)
            flow_boundaries = self._detect_flow_boundaries(
                candidate_frame_indices=candidate_frame_indices,
                frame_loader_fn=_flow_loader,
                hard_boundaries=hard_boundaries,
            )
            if flow_boundaries:
                logger.debug(
                    "Optical-flow detected %d additional boundaries", len(flow_boundaries)
                )

        # ----------------------------------------------------------------
        # Merge hard + soft + flow boundaries → raw shots
        # ----------------------------------------------------------------
        all_boundaries: set[int] = set(hard_boundaries) | set(soft_boundaries) | set(flow_boundaries)
        all_boundaries.add(0)
        all_boundaries.add(total_frames - 1)
        sorted_boundaries = sorted(all_boundaries)

        raw_shots = self._boundaries_to_shots(
            sorted_boundaries, total_frames=total_frames, fps=video_fps
        )

        # ----------------------------------------------------------------
        # Step 3 — enforce min / max shot duration
        # ----------------------------------------------------------------
        shots = self._enforce_duration_constraints(raw_shots, fps=video_fps)

        return shots

    # ------------------------------------------------------------------
    # Private: Step 1 — hard cut detection
    # ------------------------------------------------------------------

    def _detect_hard_cuts(self, video_path: str) -> list[int]:
        """Detect shot boundaries via TransNetV2 (GPU) or PySceneDetect (CPU fallback).

        TransNetV2 is preferred — it runs on GPU (~10-20x faster) and detects
        both hard cuts and gradual transitions.  PySceneDetect is used when
        TransNetV2 is not installed.

        Returns a list of *start* frame indices for each detected scene
        (excluding frame 0).
        """
        from .gpu_shot_detector import GPUShotDetector
        detector = GPUShotDetector(
            threshold=self._cfg.gpu_shot_threshold,
            pyscenedetect_threshold=_SCENEDETECT_THRESHOLD,
            min_shot_duration_s=self._cfg.min_shot_duration_s,
        )
        boundaries = detector.detect_boundaries(video_path)

        return boundaries

    # ------------------------------------------------------------------
    # Private: Step 2 — soft boundary detection
    # ------------------------------------------------------------------

    def _detect_soft_boundaries(
        self,
        candidate_frame_indices: list[int],
        embeddings: np.ndarray,
        hard_boundaries: list[int],
    ) -> list[int]:
        """Identify soft boundary frames via windowed cosine distance.

        For each candidate frame NOT on a hard boundary:
          - Determine the approximate segment length (distance to nearest
            hard boundary on each side).
          - window_size = clamp(segment_length // 4, min=3, max=15)
          - half = window_size // 2
          - boundary_score(i) = mean_cosine_dist(emb[i-half:i], emb[i:i+half])
          - Mark as soft boundary if score > threshold **and** local max.

        Threshold is adaptive: derived from this video's own consecutive-frame
        cosine distance distribution (median + 1.5 * IQR).  The config value
        ``glrt_threshold`` acts as a floor so low-variance videos don't
        over-segment on noise.
        """
        n = len(candidate_frame_indices)
        if n < 3:
            return []

        hard_set = set(hard_boundaries)
        min_w, max_w = self._cfg.window_size_range

        # --- adaptive threshold from this video's embedding distribution ---
        if n >= 4:
            consec_dists = np.array([
                _cosine_distance(embeddings[i], embeddings[i + 1])
                for i in range(n - 1)
            ], dtype=np.float32)
            q75, q25 = np.percentile(consec_dists, [75, 25])
            iqr = float(q75 - q25)
            adaptive = float(np.median(consec_dists)) + 1.5 * iqr
            # floor = config value so we never fire on pure noise in static video
            threshold = float(np.clip(adaptive, self._cfg.glrt_threshold, 0.50))
            logger.info(
                "GLRT adaptive threshold: %.4f  (median=%.4f iqr=%.4f config_floor=%.4f)",
                threshold, float(np.median(consec_dists)), iqr, self._cfg.glrt_threshold,
            )
        else:
            threshold = self._cfg.glrt_threshold

        # Build a sorted list of hard boundary frame positions projected onto
        # candidate_frame_indices so we can compute segment lengths.
        hard_emb_positions: list[int] = []
        for hb in sorted(hard_set):
            # Find the nearest candidate frame index for this hard boundary
            nearest = min(range(n), key=lambda i: abs(candidate_frame_indices[i] - hb))
            hard_emb_positions.append(nearest)
        hard_emb_positions = sorted(set(hard_emb_positions))

        def _segment_length_at(emb_idx: int) -> int:
            """Approximate segment length (in candidate frames) around emb_idx."""
            prev_hard = 0
            next_hard = n - 1
            for hp in hard_emb_positions:
                if hp < emb_idx:
                    prev_hard = hp
                elif hp > emb_idx:
                    next_hard = hp
                    break
            return next_hard - prev_hard

        # Compute scores for all candidate positions
        scores = np.zeros(n, dtype=np.float32)
        for i in range(1, n - 1):
            abs_frame = candidate_frame_indices[i]
            if abs_frame in hard_set:
                continue  # skip hard boundaries

            seg_len = _segment_length_at(i)
            raw_w = max(min_w, min(max_w, seg_len // 4))
            half = raw_w // 2
            if half < 1:
                half = 1

            left_start = max(0, i - half)
            right_end = min(n, i + half)

            group_a = embeddings[left_start:i]
            group_b = embeddings[i:right_end]
            scores[i] = _mean_cosine_dist(group_a, group_b)

        # Find positions where score > threshold AND local maximum
        soft_boundaries: list[int] = []
        for i in range(1, n - 1):
            if scores[i] < threshold:
                continue
            # Local maximum check (immediate neighbours) — strict inequality so
            # that plateau regions (equal scores) do not all fire as boundaries.
            # BUG-SS-3: was >=, which marks every frame in a flat plateau as a
            # soft boundary; must be strictly greater than both neighbours.
            if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
                soft_boundaries.append(candidate_frame_indices[i])

        return soft_boundaries

    # ------------------------------------------------------------------
    # Private: Step 3 (opt-in) — optical-flow secondary boundaries
    # ------------------------------------------------------------------

    def _detect_flow_boundaries(
        self,
        candidate_frame_indices: list[int],
        frame_loader_fn: Callable[[int], np.ndarray],
        hard_boundaries: list[int],
    ) -> list[int]:
        """Detect shot boundaries via dense optical flow magnitude spikes.

        Computes mean optical flow magnitude between consecutive candidate
        frames.  A sustained spike (> ``flow_threshold`` for >
        ``flow_min_spike_frames`` consecutive pairs) that is NOT already near
        a hard boundary is reported as a soft boundary.

        Uses ``cv2.calcOpticalFlowFarneback`` — no extra dependencies.

        Parameters
        ----------
        candidate_frame_indices:
            Sorted list of absolute frame indices to compare.
        frame_loader_fn:
            Callable ``(abs_frame_idx) -> np.ndarray`` returning RGB uint8.
        hard_boundaries:
            Already-detected hard boundary frame indices.  Flow boundaries
            within ``flow_min_gap_frames`` of any hard boundary are skipped.

        Returns
        -------
        List of absolute frame indices where flow-based boundaries detected.
        Empty on any error or if feature is unavailable.
        """
        n = len(candidate_frame_indices)
        if n < 3:
            return []

        threshold = self._cfg.flow_threshold
        min_gap = self._cfg.flow_min_gap_frames
        hard_set = set(hard_boundaries)

        _FLOW_W, _FLOW_H = 256, 144  # resize target for speed

        magnitudes: list[float] = []  # one entry per consecutive pair
        pair_frame_indices: list[int] = []  # frame index for pair (i, i+1) → index i+1

        try:
            prev_idx = candidate_frame_indices[0]
            prev_rgb = frame_loader_fn(prev_idx)
            prev_small = cv2.resize(prev_rgb, (_FLOW_W, _FLOW_H))
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)

            for curr_idx in candidate_frame_indices[1:]:
                try:
                    curr_rgb = frame_loader_fn(curr_idx)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Flow: frame load failed at %d: %s", curr_idx, exc)
                    # Reset chain — skip this pair
                    prev_gray = None  # type: ignore[assignment]
                    pair_frame_indices.append(curr_idx)
                    magnitudes.append(0.0)
                    prev_idx = curr_idx
                    continue

                curr_small = cv2.resize(curr_rgb, (_FLOW_W, _FLOW_H))
                curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, curr_gray,
                        None,
                        0.5,   # pyr_scale
                        3,     # levels
                        15,    # winsize
                        3,     # iterations
                        5,     # poly_n
                        1.2,   # poly_sigma
                        0,     # flags
                    )
                    mag = float(np.mean(
                        cv2.magnitude(flow[..., 0], flow[..., 1])
                    ))
                else:
                    mag = 0.0

                magnitudes.append(mag)
                pair_frame_indices.append(curr_idx)
                prev_gray = curr_gray
                prev_idx = curr_idx

        except Exception as exc:  # noqa: BLE001
            logger.warning("_detect_flow_boundaries error: %s", exc)
            return []

        if not magnitudes:
            return []

        # 3-frame rolling mean smoothing
        mags = np.array(magnitudes, dtype=np.float32)
        kernel = np.ones(3, dtype=np.float32) / 3.0
        if len(mags) >= 3:
            smoothed = np.convolve(mags, kernel, mode="same")
        else:
            smoothed = mags

        # Adaptive flow threshold from this video's own flow distribution.
        # Use 85th percentile — fires on the top 15% of motion events.
        # Config floor prevents noise spikes from over-firing on static footage.
        if len(smoothed) >= 10:
            adaptive_flow = float(np.percentile(smoothed, 85))
            flow_threshold = max(adaptive_flow, threshold)
            logger.info(
                "Flow adaptive threshold: %.4f  (p85=%.4f config_floor=%.4f)",
                flow_threshold, adaptive_flow, threshold,
            )
        else:
            flow_threshold = threshold

        # Find frames where smoothed magnitude exceeds threshold, excluding
        # positions already near a hard boundary
        flow_boundaries: list[int] = []
        for k, abs_idx in enumerate(pair_frame_indices):
            if smoothed[k] <= flow_threshold:
                continue
            # Skip if within min_gap of any hard boundary
            too_close = any(abs(abs_idx - hb) < min_gap for hb in hard_set)
            if too_close:
                continue
            flow_boundaries.append(abs_idx)

        return flow_boundaries

    # ------------------------------------------------------------------
    # Public: text-change forced keyframes (improvement #2)
    # ------------------------------------------------------------------

    def find_text_change_keyframes(
        self,
        shots: list[Shot],
        frame_indices: list[int],
        frame_loader_fn: Callable[[int], np.ndarray],
        existing_keyframes_per_shot: dict[int, list[int]],
    ) -> dict[int, list[int]]:
        """Return additional forced keyframes per shot where text changed.

        Intended to be called AFTER shot detection but BEFORE keyframe
        selection.  The caller (pipeline orchestrator) merges the returned
        frame indices into its "must-include" set for each shot.

        Parameters
        ----------
        shots:
            Shot list as returned by ``detect_shots()``.
        frame_indices:
            All candidate frame indices (full video).
        frame_loader_fn:
            Callable ``(abs_frame_idx) -> np.ndarray`` returning RGB uint8.
        existing_keyframes_per_shot:
            Mapping from shot index → list of already-selected keyframe
            indices.  Used to enforce minimum spacing.

        Returns
        -------
        Mapping from shot index → list of additional keyframe indices where
        text changes were detected.  Missing shot indices have no additions.
        Empty dict if ``text_change.enabled`` is False.
        """
        if not self._cfg_root.text_change.enabled:
            return {}

        tc_cfg = self._cfg_root.text_change
        detector = TextChangeDetector(
            min_iou_change=tc_cfg.min_iou_change,
            min_region_count_change=tc_cfg.min_region_count_change,
            min_spacing_frames=self._cfg_root.dual_keyframe.min_spacing_frames,
        )

        frame_set = set(frame_indices)
        result: dict[int, list[int]] = {}

        for shot_idx, shot in enumerate(shots):
            # Gather candidate frames belonging to this shot
            shot_frames = sorted(
                f for f in frame_set
                if shot.start_frame <= f <= shot.end_frame
            )
            if len(shot_frames) < 2:
                continue

            existing_kf = existing_keyframes_per_shot.get(shot_idx, [])
            extras = detector.find_text_change_frames(
                shot_frame_indices=shot_frames,
                frame_loader_fn=frame_loader_fn,
                existing_keyframes=existing_kf,
            )
            if extras:
                result[shot_idx] = extras

        return result

    # ------------------------------------------------------------------
    # Private: boundary list → Shot list
    # ------------------------------------------------------------------

    @staticmethod
    def _boundaries_to_shots(
        sorted_boundaries: list[int],
        total_frames: int,
        fps: float,
    ) -> list[Shot]:
        """Convert a sorted list of boundary frame indices into :class:`Shot` objects.

        ``sorted_boundaries`` must include frame 0 (first element) and
        ``total_frames - 1`` (last element), as guaranteed by the caller.

        Convention used here:
          - boundaries[0..N-2] are each the *start* of a shot.
          - The end of shot i  = boundaries[i+1] - 1   (i < N-2)
          - The end of the final shot = total_frames - 1  (= boundaries[-1])

        So the last element of sorted_boundaries is used only as the inclusive
        end of the final shot; it is NOT treated as the start of a new shot.
        """
        shots: list[Shot] = []
        n_boundaries = len(sorted_boundaries)
        if n_boundaries < 2:
            # Degenerate: single-frame or empty video
            if n_boundaries == 1:
                start = sorted_boundaries[0]
                end = total_frames - 1
                shots.append(Shot(
                    start_frame=start,
                    end_frame=end,
                    duration_s=(end - start + 1) / fps,
                    fps=fps,
                ))
            return shots

        for i in range(n_boundaries - 1):
            start = sorted_boundaries[i]
            if i < n_boundaries - 2:
                # Normal interior shot: ends one frame before the next boundary starts
                end = sorted_boundaries[i + 1] - 1
            else:
                # Final shot: inclusive end is the last boundary (= total_frames - 1)
                end = sorted_boundaries[i + 1]

            if end < start:
                end = start
            duration_s = (end - start + 1) / fps
            shots.append(
                Shot(
                    start_frame=start,
                    end_frame=end,
                    duration_s=duration_s,
                    fps=fps,
                )
            )
        return shots

    # ------------------------------------------------------------------
    # Private: Step 3 — duration constraint enforcement
    # ------------------------------------------------------------------

    def _enforce_duration_constraints(
        self, shots: list[Shot], fps: float
    ) -> list[Shot]:
        """Merge shots that are too short; force-split shots that are too long.

        Merging strategy: merge with the temporally adjacent neighbor that has
        the *shorter* duration (greedy left-to-right pass; iterate until stable).

        Force-split strategy: split into equal sub-segments; each sub-segment
        carries ``is_forced_split=True`` in its metadata (stored on the Shot
        dataclass via a monkey-patched attribute since Shot has no metadata
        field — see note below).

        Note on ``is_forced_split``:
            :class:`Shot` has no ``metadata`` field.  We attach the boolean as
            a plain attribute (``shot.is_forced_split``) so downstream modules
            can detect budget sub-segments without schema changes.

        ``max_shot_duration_s`` is adapted per video: for raw continuous footage
        the detected shots may be very long (no hard cuts), so we scale the
        ceiling proportionally to total video duration to keep segment density
        sensible.
        """
        min_frames = math.ceil(self._cfg.min_shot_duration_s * fps)

        # Adaptive max shot duration: scale with total video duration so we
        # produce consistent segment density regardless of clip length.
        # Infer total duration from the last shot's end frame.
        if shots:
            total_frames_est = shots[-1].end_frame + 1
            total_duration_s = total_frames_est / fps
        else:
            total_duration_s = 0.0

        # Target ~1 forced-split segment per 8s for short clips,
        # relaxing to 12s for very long clips (>60 min) to avoid too many tiny segments.
        if total_duration_s < 600:        # < 10 min
            adaptive_max_s = min(self._cfg.max_shot_duration_s, 6.0)
        elif total_duration_s < 3600:     # 10 min – 1 hr
            adaptive_max_s = min(self._cfg.max_shot_duration_s, 8.0)
        else:                             # > 1 hr
            adaptive_max_s = min(self._cfg.max_shot_duration_s, 12.0)

        logger.info(
            "Adaptive max_shot_duration: %.1fs  (config: %.1fs, video_duration: %.1fs)",
            adaptive_max_s, self._cfg.max_shot_duration_s, total_duration_s,
        )
        max_frames = math.floor(adaptive_max_s * fps)

        if not shots:
            return shots

        # --- merge short shots (iterate until no more merges needed) ---
        shots = self._merge_short_shots(shots, min_frames=min_frames, fps=fps)

        # --- force-split long shots ---
        shots = self._split_long_shots(shots, max_frames=max_frames, fps=fps)

        return shots

    def _merge_short_shots(
        self, shots: list[Shot], min_frames: int, fps: float
    ) -> list[Shot]:
        """Greedy merge pass: remove shots shorter than ``min_frames``."""
        # Iterate until stable (a merge may create another short shot)
        changed = True
        while changed:
            changed = False
            merged: list[Shot] = []
            i = 0
            while i < len(shots):
                shot = shots[i]
                fc = shot.frame_count
                if fc < min_frames and len(shots) > 1:
                    # Decide which neighbour to merge into.
                    # ``has_prev`` reflects whether we have an actual predecessor
                    # already placed in ``merged`` — not just whether i > 0.  On
                    # a merge-with-next pass we advance i without appending to
                    # merged, so a later short shot could see ``i > 0`` while
                    # ``merged`` is still empty.  Use len(merged) for the check.
                    has_prev = len(merged) > 0
                    has_next = i < len(shots) - 1

                    if has_prev and has_next:
                        # BUG-SS-2: shots[i-1] may already have been merged into
                        # merged[-1], making its frame_count stale.  Use merged[-1]
                        # (the actual current predecessor) when has_prev is True.
                        prev_fc = merged[-1].frame_count
                        next_fc = shots[i + 1].frame_count
                        merge_with_prev = prev_fc <= next_fc
                    elif has_prev:
                        merge_with_prev = True
                    else:
                        merge_with_prev = False

                    if merge_with_prev:
                        # Absorb into the last shot already placed in `merged`
                        prev = merged[-1]
                        new_end = shot.end_frame
                        new_duration = (new_end - prev.start_frame + 1) / fps
                        merged[-1] = Shot(
                            start_frame=prev.start_frame,
                            end_frame=new_end,
                            duration_s=new_duration,
                            fps=fps,
                        )
                    else:
                        # Absorb into next shot — modify shots list in place
                        # and re-process next position
                        next_shot = shots[i + 1]
                        new_start = shot.start_frame
                        new_duration = (next_shot.end_frame - new_start + 1) / fps
                        shots[i + 1] = Shot(
                            start_frame=new_start,
                            end_frame=next_shot.end_frame,
                            duration_s=new_duration,
                            fps=fps,
                        )
                        changed = True  # BUG-SS-1: was missing; `continue` below skipped the flag
                        i += 1
                        continue

                    changed = True
                    i += 1
                else:
                    merged.append(shot)
                    i += 1
            shots = merged

        return shots

    def _split_long_shots(
        self, shots: list[Shot], max_frames: int, fps: float
    ) -> list[Shot]:
        """Force-split any shot longer than ``max_frames`` into equal sub-segments."""
        result: list[Shot] = []
        for shot in shots:
            fc = shot.frame_count
            if fc <= max_frames:
                result.append(shot)
                continue

            # Number of equal sub-segments required
            n_splits = math.ceil(fc / max_frames)
            sub_size = fc // n_splits
            remainder = fc % n_splits

            current_start = shot.start_frame
            for k in range(n_splits):
                # Distribute remainder frames among the first `remainder` segments
                extra = 1 if k < remainder else 0
                seg_frames = sub_size + extra
                seg_end = current_start + seg_frames - 1

                sub = Shot(
                    start_frame=current_start,
                    end_frame=seg_end,
                    duration_s=seg_frames / fps,
                    fps=fps,
                )
                sub.is_forced_split = True  # type: ignore[attr-defined]
                result.append(sub)
                current_start = seg_end + 1

        return result
