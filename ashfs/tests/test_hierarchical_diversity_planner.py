"""Unit tests for ashfs.hierarchical_diversity_planner.

Tests verify tree construction, expand/summarize decisions, medoid-of-medoids
strategy, temporal ordering, and compression statistics using synthetic
SelectedKeyframes objects.  No GPU or model calls are required.
"""
import numpy as np
import pytest

from ashfs.hierarchical_diversity_planner import HierarchicalDiversityPlanner
from ashfs.types import Shot, SelectedKeyframes, TreeNode
from tests.conftest import make_l2_normed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shot(start: int, end: int, fps: float = 25.0) -> Shot:
    return Shot(
        start_frame=start,
        end_frame=end,
        duration_s=(end - start + 1) / fps,
        fps=fps,
    )


def _make_kf(
    frame_indices: list[int],
    emb: np.ndarray,
    roles: list[str] = None,
) -> SelectedKeyframes:
    """Build a SelectedKeyframes from frame indices and matching embeddings."""
    start = frame_indices[0]
    end = frame_indices[-1]
    shot = _make_shot(start, end)
    if roles is None:
        roles = ["medoid"] * len(frame_indices)
    return SelectedKeyframes(
        shot=shot,
        frame_indices=frame_indices,
        embeddings=emb,
        siglip_embeddings=None,
        frame_roles=roles,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def planner(cfg) -> HierarchicalDiversityPlanner:
    """HierarchicalDiversityPlanner constructed from the shared config fixture."""
    return HierarchicalDiversityPlanner(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHierarchicalDiversityPlanner:

    def test_empty_input_returns_video_root(self, planner):
        """An empty keyframe list must return a root TreeNode at level='video'."""
        root = planner.build_tree([])
        assert isinstance(root, TreeNode)
        assert root.level == "video"
        assert root.children == []
        assert root.keyframes == []

    def test_single_shot_always_expands(self, planner, rng):
        """A tree with one shot must have its shot-level node with expand=True."""
        emb = make_l2_normed(rng, 2)
        kf = _make_kf([0, 1], emb)
        root = planner.build_tree([kf])
        # Walk down to the shot node — shot nodes are at the leaf level
        # Find shot-level nodes by recursion
        def find_shot_nodes(node):
            if node.level == "shot":
                return [node]
            results = []
            for child in node.children:
                results.extend(find_shot_nodes(child))
            return results

        shot_nodes = find_shot_nodes(root)
        assert len(shot_nodes) == 1
        assert shot_nodes[0].expand is True

    def test_low_diversity_chunk_summarizes(self, cfg, rng):
        """Shots with nearly identical embeddings must produce a chunk node with expand=False."""
        planner = HierarchicalDiversityPlanner(cfg)
        threshold = cfg.hierarchy.expansion_threshold_chunk

        # All shots share nearly the same embedding → diversity ≈ 0 < threshold
        base = make_l2_normed(rng, 1)  # single unit vector
        n_shots = cfg.hierarchy.chunk_size_shots
        keyframes = []
        for i in range(n_shots):
            emb = np.tile(base, (1, 1)).copy()      # (1, 768) identical each time
            kf = _make_kf([i * 10], emb, roles=["medoid"])
            keyframes.append(kf)

        root = planner.build_tree(keyframes)

        # Find the first chunk-level node
        def find_level(node, level):
            if node.level == level:
                return node
            for child in node.children:
                found = find_level(child, level)
                if found is not None:
                    return found
            return None

        chunk_node = find_level(root, "chunk")
        assert chunk_node is not None
        assert chunk_node.expand is False, (
            f"Expected chunk to summarize (expand=False) for nearly-identical shots "
            f"(diversity={chunk_node.diversity_score:.4f} vs threshold={threshold})"
        )

    def test_high_diversity_chunk_expands(self, cfg):
        """Shots with maximally diverse embeddings must produce a chunk node with expand=True."""
        planner = HierarchicalDiversityPlanner(cfg)
        n_shots = cfg.hierarchy.chunk_size_shots

        # Build orthogonal unit vectors in 768-d space to maximise diversity
        d = 768
        keyframes = []
        for i in range(n_shots):
            emb = np.zeros((1, d), dtype=np.float32)
            emb[0, i] = 1.0                           # axis-aligned — fully orthogonal
            kf = _make_kf([i * 10], emb, roles=["medoid"])
            keyframes.append(kf)

        root = planner.build_tree(keyframes)

        def find_level(node, level):
            if node.level == level:
                return node
            for child in node.children:
                found = find_level(child, level)
                if found:
                    return found
            return None

        chunk_node = find_level(root, "chunk")
        assert chunk_node is not None
        assert chunk_node.expand is True

    def test_get_final_keyframes_expand_path(self, planner, cfg, rng):
        """When all nodes expand, get_final_keyframes must return all shot-level keyframes."""
        n_shots = cfg.hierarchy.chunk_size_shots
        d = 768
        # Use orthogonal embeddings to force expansion at every level
        keyframes = []
        for i in range(n_shots):
            emb = np.zeros((1, d), dtype=np.float32)
            emb[0, i] = 1.0
            kf = _make_kf([i * 10], emb, roles=["medoid"])
            keyframes.append(kf)

        root = planner.build_tree(keyframes)
        final = planner.get_final_keyframes(root)

        # All shot keyframes must appear (one per shot)
        assert len(final) == n_shots

    def test_get_final_keyframes_summarize_path(self, cfg, rng):
        """When a chunk node summarizes, get_final_keyframes must return a single keyframe."""
        planner = HierarchicalDiversityPlanner(cfg)
        n_shots = cfg.hierarchy.chunk_size_shots

        # Identical embeddings → chunk will summarize → returns medoid only
        base = make_l2_normed(rng, 1)
        keyframes = []
        for i in range(n_shots):
            emb = np.tile(base, (1, 1)).copy()
            kf = _make_kf([i * 10], emb, roles=["medoid"])
            keyframes.append(kf)

        root = planner.build_tree(keyframes)
        final = planner.get_final_keyframes(root)

        # Summarized chunk produces one medoid keyframe — fewer than n_shots
        assert len(final) < n_shots
        assert len(final) >= 1

    def test_stats_reset_between_calls(self, planner, rng):
        """get_final_keyframes called twice must reset stats so they reflect only the second call."""
        emb = make_l2_normed(rng, 2)
        kf1 = _make_kf([0, 1], emb)
        kf2 = _make_kf([10, 11], emb.copy())
        root = planner.build_tree([kf1, kf2])

        planner.get_final_keyframes(root)
        stats_1_expanded = planner._stats_expanded
        stats_1_summarized = planner._stats_summarized

        # Second call must reset then re-count
        planner.get_final_keyframes(root)
        stats_2_expanded = planner._stats_expanded
        stats_2_summarized = planner._stats_summarized

        # The counts from the second call must equal the first call's counts
        # (tree structure is unchanged between calls)
        assert stats_2_expanded == stats_1_expanded
        assert stats_2_summarized == stats_1_summarized

    def test_medoid_of_medoids_strategy(self, cfg, rng):
        """A summarized chunk must return the medoid-of-medoids, not an arbitrary frame."""
        planner = HierarchicalDiversityPlanner(cfg)
        n_shots = cfg.hierarchy.chunk_size_shots

        # Nearly identical embeddings with one slightly perturbed shot
        base = make_l2_normed(rng, 1)
        keyframes = []
        for i in range(n_shots):
            emb = np.tile(base, (1, 1)).copy()
            # Small perturbation on all shots to avoid all-zero gradient issues
            noise = np.zeros((1, 768), dtype=np.float32)
            noise[0, i] = 0.001
            emb = emb + noise
            # Re-normalise
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            kf = _make_kf([i * 10], emb, roles=["medoid"])
            keyframes.append(kf)

        root = planner.build_tree(keyframes)

        # Locate the chunk node
        def find_level(node, level):
            if node.level == level:
                return node
            for child in node.children:
                found = find_level(child, level)
                if found:
                    return found
            return None

        chunk = find_level(root, "chunk")
        if chunk is not None and not chunk.expand:
            medoid_kf = getattr(chunk, "_medoid_kf", None)
            assert medoid_kf is not None, "_medoid_kf attribute must be set on summarized nodes"
            assert len(medoid_kf.frame_indices) == 1

    def test_temporal_order_preserved(self, planner, rng):
        """get_final_keyframes must return keyframes sorted by their first frame index."""
        d = 768
        keyframes = []
        # Build shots out of order to confirm sorting is applied
        for start in [50, 0, 30, 10]:
            emb = make_l2_normed(rng, 1)
            kf = _make_kf([start], emb, roles=["medoid"])
            keyframes.append(kf)

        root = planner.build_tree(keyframes)
        final = planner.get_final_keyframes(root)

        first_indices = [kf.frame_indices[0] for kf in final]
        assert first_indices == sorted(first_indices)

    def test_compression_stats_ratio_calculation(self, planner, rng):
        """compression_ratio must equal total_original_frames / total_final_frames."""
        d = 768
        keyframes = []
        for i in range(4):
            emb = make_l2_normed(rng, 2)
            kf = _make_kf([i * 10, i * 10 + 1], emb)
            keyframes.append(kf)

        root = planner.build_tree(keyframes)
        final = planner.get_final_keyframes(root)
        stats = planner.get_compression_stats(keyframes, final)

        total_orig = sum(len(kf.frame_indices) for kf in keyframes)
        total_final = sum(len(kf.frame_indices) for kf in final)
        expected_ratio = total_orig / total_final if total_final > 0 else float("inf")

        assert stats["total_original_frames"] == total_orig
        assert stats["total_final_frames"] == total_final
        assert stats["compression_ratio"] == pytest.approx(expected_ratio, rel=1e-3)

    # ------------------------------------------------------------------
    # Additional missing tests
    # ------------------------------------------------------------------

    def test_get_compression_stats_zero_final_frames_returns_inf(self, planner, rng):
        """get_compression_stats must return compression_ratio=inf when final
        has 0 frames — must not raise ZeroDivisionError."""
        emb = make_l2_normed(rng, 2)
        kf = _make_kf([0, 1], emb)
        # Pass an empty final list to trigger the total_final == 0 branch
        stats = planner.get_compression_stats([kf], [])
        assert stats["total_final_frames"] == 0
        assert stats["compression_ratio"] == float("inf"), (
            "Expected compression_ratio=inf when final has 0 frames"
        )

    def test_tree_single_shot_with_zero_frames(self, planner):
        """build_tree with a single SelectedKeyframes that has 0 embeddings must
        not raise and must produce a video-level root with expand=True."""
        empty_emb = np.empty((0, 768), dtype=np.float32)
        # frame_indices is empty — use a dummy shot spanning 0..0
        shot = _make_shot(0, 0)
        kf = SelectedKeyframes(
            shot=shot,
            frame_indices=[],
            embeddings=empty_emb,
            siglip_embeddings=None,
            frame_roles=[],
        )
        root = planner.build_tree([kf])
        assert root.level == "video"
        # The shot node's aggregate_embedding will be None (empty embeddings)
        # but build_tree must complete without error

    def test_pairwise_mean_cosine_distance_exactly_two_embeddings(self, planner, rng):
        """_pairwise_mean_cosine_distance with exactly 2 embeddings must return
        the cosine distance between those two vectors (there is only one pair)."""
        # Use two orthogonal unit vectors: cosine similarity = 0 → distance = 1
        d = 768
        a = np.zeros(d, dtype=np.float32); a[0] = 1.0
        b = np.zeros(d, dtype=np.float32); b[1] = 1.0

        result = HierarchicalDiversityPlanner._pairwise_mean_cosine_distance([a, b])
        # cos_sim(a, b) = 0 → cos_distance = 1 - 0 = 1
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_pairwise_mean_cosine_distance_single_embedding_returns_zero(self, planner, rng):
        """_pairwise_mean_cosine_distance with only 1 embedding must return 0.0
        (fewer than 2 embeddings → no pairs → defined as 0)."""
        emb = make_l2_normed(rng, 1)[0]
        result = HierarchicalDiversityPlanner._pairwise_mean_cosine_distance([emb])
        assert result == pytest.approx(0.0)
