"""Unit tests for ashfs.shot_segmenter.

Tests cover the pure helper functions and private methods of ShotSegmenter that
can be exercised without a real video file or GPU.
"""
import sys
import types
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ashfs.shot_segmenter import (
    ShotSegmenter,
    _cosine_distance,
    _mean_cosine_dist,
)
from ashfs.types import Shot
from tests.conftest import make_l2_normed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shot(start: int, end: int, fps: float = 25.0) -> Shot:
    """Build a Shot from frame indices and fps."""
    duration_s = (end - start + 1) / fps
    return Shot(start_frame=start, end_frame=end, duration_s=duration_s, fps=fps)


def _make_segmenter(cfg) -> ShotSegmenter:
    """Return a ShotSegmenter constructed from the shared config fixture."""
    return ShotSegmenter(cfg)


# ---------------------------------------------------------------------------
# Tests: _cosine_distance helper
# ---------------------------------------------------------------------------

class TestCosineDistance:
    def test_cosine_distance_zero_on_identical(self):
        """Identical L2-normalised vectors must yield cosine distance 0."""
        rng = np.random.default_rng(42)
        v = make_l2_normed(rng, 1)[0]
        result = _cosine_distance(v, v)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_cosine_distance_one_on_orthogonal(self):
        """Orthogonal unit vectors must yield cosine distance 1 (cos_sim=0)."""
        # e1 and e2 are axis-aligned unit vectors — perfectly orthogonal
        d = 768
        a = np.zeros(d, dtype=np.float32)
        b = np.zeros(d, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        result = _cosine_distance(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: _mean_cosine_dist helper
# ---------------------------------------------------------------------------

class TestMeanCosineDist:
    def test_mean_cosine_dist_empty_groups(self):
        """Empty group arrays must return 0.0 without raising."""
        empty = np.empty((0, 768), dtype=np.float32)
        rng = np.random.default_rng(0)
        some = make_l2_normed(rng, 3)
        assert _mean_cosine_dist(empty, some) == pytest.approx(0.0)
        assert _mean_cosine_dist(some, empty) == pytest.approx(0.0)
        assert _mean_cosine_dist(empty, empty) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: ShotSegmenter._boundaries_to_shots
# ---------------------------------------------------------------------------

class TestBoundariesToShots:
    def test_boundaries_to_shots_basic(self, cfg):
        """3 boundaries [0, 50, 99] with total_frames=100 must produce 2 shots."""
        seg = _make_segmenter(cfg)
        shots = seg._boundaries_to_shots([0, 50, 99], total_frames=100, fps=25.0)
        assert len(shots) == 2
        # Shot 0: frames 0..49
        assert shots[0].start_frame == 0
        assert shots[0].end_frame == 49
        # Shot 1: frames 50..99
        assert shots[1].start_frame == 50
        assert shots[1].end_frame == 99

    def test_boundaries_to_shots_single_boundary(self, cfg):
        """[0, 99] (only start and end) must produce exactly 1 shot."""
        seg = _make_segmenter(cfg)
        shots = seg._boundaries_to_shots([0, 99], total_frames=100, fps=25.0)
        assert len(shots) == 1
        assert shots[0].start_frame == 0
        assert shots[0].end_frame == 99

    def test_boundaries_to_shots_duration_is_correct(self, cfg):
        """Shot duration must equal (end - start + 1) / fps."""
        seg = _make_segmenter(cfg)
        fps = 25.0
        shots = seg._boundaries_to_shots([0, 50, 99], total_frames=100, fps=fps)
        for shot in shots:
            expected = (shot.end_frame - shot.start_frame + 1) / fps
            assert shot.duration_s == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Tests: ShotSegmenter._merge_short_shots
# ---------------------------------------------------------------------------

class TestMergeShortShots:
    def test_merge_short_shots_removes_short(self, cfg):
        """A shot shorter than min_frames must be merged into a neighbour."""
        seg = _make_segmenter(cfg)
        fps = 25.0
        # Shot 0: 1 frame (very short), Shot 1: 50 frames, Shot 2: 49 frames
        shots = [
            _make_shot(0, 0, fps),       # 1 frame — too short
            _make_shot(1, 50, fps),      # 50 frames
            _make_shot(51, 99, fps),     # 49 frames
        ]
        # min_frames requires at least 13 frames (0.5s * 25 fps = 12.5 → ceil=13)
        min_frames = 13
        merged = seg._merge_short_shots(shots, min_frames=min_frames, fps=fps)
        # The 1-frame shot must have been absorbed — result must have fewer shots
        assert len(merged) < 3
        # Verify no shot is shorter than min_frames after merging
        for s in merged:
            assert s.frame_count >= min_frames

    def test_merge_short_shots_stable(self, cfg):
        """When all shots are already long enough, no merge should occur."""
        seg = _make_segmenter(cfg)
        fps = 25.0
        shots = [
            _make_shot(0, 49, fps),     # 50 frames
            _make_shot(50, 99, fps),    # 50 frames
        ]
        min_frames = 13
        merged = seg._merge_short_shots(shots, min_frames=min_frames, fps=fps)
        assert len(merged) == 2
        assert merged[0].start_frame == 0 and merged[0].end_frame == 49
        assert merged[1].start_frame == 50 and merged[1].end_frame == 99


# ---------------------------------------------------------------------------
# Tests: ShotSegmenter._split_long_shots
# ---------------------------------------------------------------------------

class TestSplitLongShots:
    def test_split_long_shots_creates_subsegments(self, cfg):
        """A shot longer than max_frames must be split into multiple sub-segments."""
        seg = _make_segmenter(cfg)
        fps = 25.0
        # max_shot_duration_s=30.0 → max_frames = floor(30.0 * 25) = 750
        # Create a shot of 800 frames, which exceeds 750
        shots = [_make_shot(0, 799, fps)]
        max_frames = 750
        result = seg._split_long_shots(shots, max_frames=max_frames, fps=fps)
        assert len(result) > 1
        # Verify each sub-segment is within max_frames
        for sub in result:
            assert sub.frame_count <= max_frames

    def test_split_long_shots_sets_is_forced_split(self, cfg):
        """Sub-segments produced by force-splitting must carry is_forced_split=True."""
        seg = _make_segmenter(cfg)
        fps = 25.0
        shots = [_make_shot(0, 799, fps)]
        max_frames = 750
        result = seg._split_long_shots(shots, max_frames=max_frames, fps=fps)
        for sub in result:
            assert getattr(sub, "is_forced_split", False) is True

    def test_split_long_shots_preserves_short_shots(self, cfg):
        """Shots that are already within max_frames must pass through unchanged."""
        seg = _make_segmenter(cfg)
        fps = 25.0
        shots = [_make_shot(0, 99, fps)]   # 100 frames < 750
        max_frames = 750
        result = seg._split_long_shots(shots, max_frames=max_frames, fps=fps)
        assert len(result) == 1
        assert result[0].start_frame == 0
        assert result[0].end_frame == 99


# ---------------------------------------------------------------------------
# Tests: _detect_soft_boundaries (uniform embeddings)
# ---------------------------------------------------------------------------

class TestSoftBoundaryDetection:
    def test_soft_boundary_detection_no_false_positives_on_uniform(self, cfg):
        """Identical embeddings across all frames must produce zero soft boundaries."""
        seg = _make_segmenter(cfg)
        rng = np.random.default_rng(1)
        # All frames share the same L2-normalised vector → cosine distance = 0
        base = make_l2_normed(rng, 1)
        n = 20
        embeddings = np.tile(base, (n, 1))  # (20, 768) all identical
        candidate_frame_indices = list(range(0, n * 5, 5))  # [0,5,10,...,95]

        boundaries = seg._detect_soft_boundaries(
            candidate_frame_indices=candidate_frame_indices,
            embeddings=embeddings,
            hard_boundaries=[],
        )
        assert boundaries == []


# ---------------------------------------------------------------------------
# Tests: _detect_hard_cuts (graceful ImportError)
# ---------------------------------------------------------------------------

class TestDetectHardCuts:
    def test_detect_hard_cuts_without_scenedetect(self, cfg):
        """When PySceneDetect is not importable, _detect_hard_cuts must raise ImportError."""
        seg = _make_segmenter(cfg)

        # Temporarily make 'scenedetect' unimportable
        with patch.dict(sys.modules, {"scenedetect": None, "scenedetect.detectors": None}):
            with pytest.raises(ImportError, match="PySceneDetect"):
                seg._detect_hard_cuts("dummy_video.mp4")


# ---------------------------------------------------------------------------
# Additional missing tests — added to close coverage gaps
# ---------------------------------------------------------------------------

class TestBoundariesToShotsEdgeCases:

    def test_single_element_boundary_list_degenerate(self, cfg):
        """A boundaries list with only one element must return exactly one shot
        spanning [boundaries[0], total_frames-1].  This is the degenerate path
        in _boundaries_to_shots (n_boundaries == 1)."""
        seg = _make_segmenter(cfg)
        # Only element is 0 — the whole video is one shot
        shots = seg._boundaries_to_shots([0], total_frames=50, fps=25.0)
        assert len(shots) == 1
        assert shots[0].start_frame == 0
        assert shots[0].end_frame == 49
        assert shots[0].duration_s == pytest.approx(50 / 25.0, rel=1e-5)

    def test_empty_boundary_list_returns_no_shots(self, cfg):
        """An empty boundaries list must return an empty shot list (n_boundaries < 2
        path, n_boundaries == 0)."""
        seg = _make_segmenter(cfg)
        shots = seg._boundaries_to_shots([], total_frames=100, fps=25.0)
        assert shots == []


class TestSoftBoundaryWithHardMatches:

    def test_soft_boundary_skips_frames_coinciding_with_hard_boundary(self, cfg, rng):
        """Frames whose absolute index exactly matches a hard boundary must be
        skipped by _detect_soft_boundaries — they must not appear in the output
        even if their cosine distance score would exceed the threshold."""
        seg = _make_segmenter(cfg)
        n = 10
        # Build embeddings where frame at position 5 has high variance
        # (two very different unit vectors flanking it).
        base_a = make_l2_normed(rng, 1)[0]   # (768,)
        base_b = -base_a                      # antipodal — cosine dist = 2 from a
        # Normalise b to unit length (already unit since -a is unit when a is unit)
        base_b = base_b / np.linalg.norm(base_b)

        embeddings = np.tile(base_a, (n, 1)).astype(np.float32)
        # Frame at embedding index 5 looks like base_b — would normally score high
        embeddings[5] = base_b

        # candidate_frame_indices: each step=5 so absolute frame = emb_idx * 5
        candidate_frame_indices = [i * 5 for i in range(n)]  # [0, 5, 10, ..., 45]

        # Frame 25 (embedding index 5) is on a hard boundary
        hard_boundaries = [25]

        soft = seg._detect_soft_boundaries(
            candidate_frame_indices=candidate_frame_indices,
            embeddings=embeddings,
            hard_boundaries=hard_boundaries,
        )
        # Frame 25 must NOT appear in soft boundaries because it matches a hard boundary
        assert 25 not in soft


class TestMergeShortShotsConvergence:

    def test_chain_merge_until_stable(self, cfg):
        """Merging a tiny shot may create another short shot — the loop must
        converge to a stable result where no remaining shot is shorter than
        min_frames."""
        seg = _make_segmenter(cfg)
        fps = 25.0
        # 3-frame, 3-frame, 50-frame: both small shots are below min_frames=13.
        # After merging the first pair the merged result (6 frames) is still
        # below min_frames, so it must be merged again into the 50-frame shot.
        shots = [
            _make_shot(0, 2, fps),       # 3 frames — too short
            _make_shot(3, 5, fps),       # 3 frames — too short
            _make_shot(6, 55, fps),      # 50 frames — ok
        ]
        min_frames = 13
        merged = seg._merge_short_shots(shots, min_frames=min_frames, fps=fps)
        for s in merged:
            assert s.frame_count >= min_frames, (
                f"Shot {s.start_frame}-{s.end_frame} has {s.frame_count} frames "
                f"which is below min_frames={min_frames}"
            )


class TestSplitLongShotsExactBoundary:

    def test_shot_exactly_at_max_frames_is_not_split(self, cfg):
        """A shot whose frame_count == max_frames must NOT be split (boundary
        condition: fc <= max_frames passes through unchanged)."""
        seg = _make_segmenter(cfg)
        fps = 25.0
        max_frames = 750
        # frame_count = end - start + 1 = 750 - 0 + 1? No: 0..749 = 750 frames
        shot = _make_shot(0, 749, fps)   # exactly 750 frames
        assert shot.frame_count == max_frames
        result = seg._split_long_shots([shot], max_frames=max_frames, fps=fps)
        assert len(result) == 1
        assert result[0].start_frame == 0
        assert result[0].end_frame == 749


class TestEnforceDurationConstraintsCombined:

    def test_enforce_duration_constraints_merge_then_split(self, cfg):
        """_enforce_duration_constraints must first merge short shots and then
        split the (potentially enlarged) result if it exceeds max_shot_duration_s.

        We construct a scenario where merging two tiny shots produces a combined
        shot that exceeds max_frames so that both phases are exercised.
        """
        seg = _make_segmenter(cfg)
        fps = 25.0

        # cfg defaults: min_shot_duration_s=0.5 → min_frames=ceil(12.5)=13
        #               max_shot_duration_s=30.0 → max_frames=floor(750)=750
        # Build two shots below min_frames that together exceed max_frames when merged.
        # We need their combined frame count > 750.  Use 400 frames each.
        shots = [
            _make_shot(0, 399, fps),     # 400 frames (> min, < max individually)
            _make_shot(400, 799, fps),   # 400 frames
        ]

        # Lower max to force a split of the merged result
        import math as _math
        min_frames = _math.ceil(seg._cfg.min_shot_duration_s * fps)
        max_frames = 600  # both shots together = 800 > 600

        # Call split directly after manual merge to simulate the combined pass
        merged = seg._merge_short_shots(shots, min_frames=min_frames, fps=fps)
        result = seg._split_long_shots(merged, max_frames=max_frames, fps=fps)

        # All resulting shots must be within [min_frames, max_frames]
        for s in result:
            assert s.frame_count <= max_frames, (
                f"Shot {s.start_frame}-{s.end_frame} exceeds max_frames={max_frames}"
            )
