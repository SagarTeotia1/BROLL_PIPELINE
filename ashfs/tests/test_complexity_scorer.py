"""Unit tests for ashfs.complexity_scorer.

Exercises ComplexityScorer with synthetic L2-normalised embeddings to verify
variance_score, motion_score, weighted sum, batch scoring, and reproducibility
of random pair sampling.
"""
import numpy as np
import pytest

from ashfs.complexity_scorer import ComplexityScorer, _PAIRWISE_THRESHOLD
from ashfs.types import Shot, ShotEmbeddings
from tests.conftest import make_l2_normed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shot(start: int = 0, end: int = 99, fps: float = 25.0) -> Shot:
    return Shot(
        start_frame=start,
        end_frame=end,
        duration_s=(end - start + 1) / fps,
        fps=fps,
    )


def _make_shot_embeddings(emb: np.ndarray, shot: Shot = None) -> ShotEmbeddings:
    if shot is None:
        shot = _make_shot(0, max(0, len(emb) - 1))
    frame_indices = list(range(shot.start_frame, shot.start_frame + len(emb)))
    return ShotEmbeddings(shot=shot, frame_indices=frame_indices, embeddings=emb)


# ---------------------------------------------------------------------------
# Fixture: scorer constructed from the session-scoped cfg
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer(cfg) -> ComplexityScorer:
    """ComplexityScorer built from the shared config fixture."""
    return ComplexityScorer(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComplexityScorer:

    def test_score_single_frame_returns_zero(self, scorer, rng):
        """A one-frame shot must produce variance_score=0, motion_score=0, complexity=0."""
        emb = make_l2_normed(rng, 1)
        se = _make_shot_embeddings(emb, _make_shot(0, 0))
        result = scorer.score(se)
        assert result.variance_score == pytest.approx(0.0)
        assert result.motion_score == pytest.approx(0.0)
        assert result.complexity == pytest.approx(0.0)

    def test_score_empty_embeddings_returns_zero(self, scorer):
        """A shot with zero frames must return all scores as 0.0."""
        empty_emb = np.empty((0, 768), dtype=np.float32)
        se = _make_shot_embeddings(empty_emb, _make_shot(0, 0))
        result = scorer.score(se)
        assert result.variance_score == pytest.approx(0.0)
        assert result.motion_score == pytest.approx(0.0)
        assert result.complexity == pytest.approx(0.0)

    def test_variance_score_identical_frames(self, scorer, rng):
        """All identical embeddings in a shot must yield variance_score=0."""
        base = make_l2_normed(rng, 1)
        emb = np.tile(base, (5, 1))   # 5 copies of the same vector
        se = _make_shot_embeddings(emb)
        result = scorer.score(se)
        # Cosine distance between identical unit vectors is 0
        assert result.variance_score == pytest.approx(0.0, abs=1e-5)

    def test_variance_score_orthogonal_frames(self, scorer):
        """Pairs of perfectly orthogonal unit vectors must yield variance_score=1."""
        # Build 2 orthogonal unit vectors in 768-d space
        d = 768
        a = np.zeros((1, d), dtype=np.float32)
        b = np.zeros((1, d), dtype=np.float32)
        a[0, 0] = 1.0
        b[0, 1] = 1.0
        emb = np.vstack([a, b])      # (2, 768) — orthogonal pair
        se = _make_shot_embeddings(emb)
        result = scorer.score(se)
        # cos_sim = 0 → cos_dist = 1 → variance_score = 1
        assert result.variance_score == pytest.approx(1.0, abs=1e-5)

    def test_motion_score_static_shot(self, scorer, rng):
        """Consecutive identical frames must produce motion_score=0."""
        base = make_l2_normed(rng, 1)
        emb = np.tile(base, (6, 1))
        se = _make_shot_embeddings(emb)
        result = scorer.score(se)
        assert result.motion_score == pytest.approx(0.0, abs=1e-5)

    def test_motion_score_large_jumps(self, scorer):
        """Frames that alternate between two antipodal unit vectors must give high motion_score."""
        d = 768
        pos = np.zeros((1, d), dtype=np.float32)
        neg = np.zeros((1, d), dtype=np.float32)
        pos[0, 0] = 1.0
        neg[0, 0] = -1.0   # antipodal — L2 dist = 2, cos_dist = 2
        emb = np.vstack([pos, neg, pos, neg])
        se = _make_shot_embeddings(emb)
        result = scorer.score(se)
        # L2(pos, neg) = sqrt(2 - 2*(-1)) = sqrt(4) = 2; summed 3 times / 4 = 1.5
        assert result.motion_score > 0.5

    def test_complexity_weighted_sum(self, cfg, rng):
        """complexity must exactly equal w1*variance_score + w2*motion_score."""
        scorer = ComplexityScorer(cfg)
        emb = make_l2_normed(rng, 5)
        se = _make_shot_embeddings(emb)
        result = scorer.score(se)
        w1 = cfg.complexity.variance_weight
        w2 = cfg.complexity.motion_weight
        expected = w1 * result.variance_score + w2 * result.motion_score
        assert result.complexity == pytest.approx(expected, rel=1e-5)

    def test_score_all_batch(self, scorer, rng):
        """score_all must return the same number of ScoredShots as the input list."""
        n_shots = 5
        shot_emb_list = []
        for i in range(n_shots):
            emb = make_l2_normed(rng, 4)
            shot = _make_shot(i * 10, i * 10 + 3)
            shot_emb_list.append(_make_shot_embeddings(emb, shot))
        results = scorer.score_all(shot_emb_list)
        assert len(results) == n_shots

    def test_pairwise_sampling_reproducible(self, scorer):
        """For shots with >15 frames, calling score twice must return the same variance_score.

        This verifies that the internal seed=42 random pair sampler is deterministic.
        """
        rng1 = np.random.default_rng(99)
        n = _PAIRWISE_THRESHOLD + 5   # e.g. 20 frames → triggers random sampling
        emb = make_l2_normed(rng1, n)
        se = _make_shot_embeddings(emb)

        result_a = scorer.score(se)
        result_b = scorer.score(se)
        assert result_a.variance_score == pytest.approx(result_b.variance_score)

    def test_score_preserves_frame_indices(self, scorer, rng):
        """ScoredShot must carry the same frame_indices as the input ShotEmbeddings."""
        emb = make_l2_normed(rng, 3)
        shot = _make_shot(10, 12)
        se = ShotEmbeddings(shot=shot, frame_indices=[10, 11, 12], embeddings=emb)
        result = scorer.score(se)
        assert result.frame_indices == [10, 11, 12]

    # ------------------------------------------------------------------
    # Additional missing tests
    # ------------------------------------------------------------------

    def test_score_exactly_two_frames(self, scorer):
        """A shot with exactly 2 frames must produce non-zero variance and motion
        scores (both scores require >= 2 frames; this exercises the n==2 path
        in both _compute_variance_score and _compute_motion_score)."""
        d = 768
        a = np.zeros((1, d), dtype=np.float32); a[0, 0] = 1.0
        b = np.zeros((1, d), dtype=np.float32); b[0, 1] = 1.0
        emb = np.vstack([a, b])   # 2 orthogonal unit vectors
        se = _make_shot_embeddings(emb, _make_shot(0, 1))
        result = scorer.score(se)
        # Two orthogonal vectors: cosine distance = 1 → variance_score = 1
        assert result.variance_score == pytest.approx(1.0, abs=1e-5)
        # motion_score > 0 (sequential L2 distance between orthogonal unit vecs)
        assert result.motion_score > 0.0
        # complexity = w1 * 1 + w2 * motion_score > 0
        assert result.complexity > 0.0

    def test_non_default_weights_w1_plus_w2_not_one(self, cfg, rng):
        """ComplexityScorer must correctly handle weights that do not sum to 1.

        We create a custom Config copy with w1=0.3, w2=0.9 (sum=1.2) and verify
        that complexity == 0.3*variance + 0.9*motion exactly."""
        import copy
        custom_cfg = copy.deepcopy(cfg)
        custom_cfg.complexity.variance_weight = 0.3
        custom_cfg.complexity.motion_weight = 0.9

        custom_scorer = ComplexityScorer(custom_cfg)
        emb = make_l2_normed(rng, 5)
        se = _make_shot_embeddings(emb)
        result = custom_scorer.score(se)

        expected = 0.3 * result.variance_score + 0.9 * result.motion_score
        assert result.complexity == pytest.approx(expected, rel=1e-5)

    def test_score_all_empty_list(self, scorer):
        """score_all with an empty input list must return an empty list without raising."""
        results = scorer.score_all([])
        assert results == []
