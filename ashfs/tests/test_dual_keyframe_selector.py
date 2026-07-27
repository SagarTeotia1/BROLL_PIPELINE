"""Unit tests for ashfs.dual_keyframe_selector.

Verifies the InfoShot λ/α math, medoid selection, spacing enforcement,
temporal ordering, and edge-case handling using synthetic L2-normalised
embeddings.  All tests are CPU-only and require no model loading.
"""
import numpy as np
import pytest

from ashfs.dual_keyframe_selector import DualKeyframeSelector
from ashfs.types import Shot, ScoredShot, BudgetedShot, SelectedKeyframes
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


def _make_budgeted(
    emb: np.ndarray,
    budget: int,
    start: int = 0,
    fps: float = 25.0,
) -> BudgetedShot:
    """Wrap embeddings in the full BudgetedShot dataclass chain."""
    n = emb.shape[0]
    end = start + n - 1
    shot = _make_shot(start, end, fps)
    frame_indices = list(range(start, start + n))
    scored = ScoredShot(
        shot=shot,
        frame_indices=frame_indices,
        embeddings=emb,
        variance_score=0.0,
        motion_score=0.0,
        complexity=0.0,
    )
    return BudgetedShot(scored_shot=scored, budget=budget)


# ---------------------------------------------------------------------------
# Fixture: selector built from the session config
# ---------------------------------------------------------------------------

@pytest.fixture
def selector(cfg) -> DualKeyframeSelector:
    """DualKeyframeSelector built from the shared config fixture."""
    return DualKeyframeSelector(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDualKeyframeSelector:

    def test_budget_1_selects_medoid(self, selector, rng):
        """With budget=1, the selected frame must be the most typical (argmax typicality)."""
        emb = make_l2_normed(rng, 5)
        bs = _make_budgeted(emb, budget=1)
        result = selector.select(bs)
        # Compute typicality manually (mean cosine sim to all others)
        normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sim = normed @ normed.T
        np.fill_diagonal(sim, 0.0)
        g = sim.sum(axis=1) / (5 - 1)
        expected_idx = int(np.argmax(g))
        assert result.frame_indices == [expected_idx]
        assert result.frame_roles == ["medoid"]

    def test_budget_2_returns_common_and_unique(self, selector, rng):
        """With budget=2, the result must contain exactly one 'common' and one 'unique' frame."""
        emb = make_l2_normed(rng, 6)
        bs = _make_budgeted(emb, budget=2)
        result = selector.select(bs)
        assert len(result.frame_indices) == 2
        assert "common" in result.frame_roles
        assert "unique" in result.frame_roles

    def test_lambda_math_budget2(self, cfg):
        """The common frame must be argmax(λ*ĝ - (1-λ)*v̂) where λ=lambda_common from config."""
        # Build 4 frames with known, hand-controlled embeddings
        d = 768
        rng = np.random.default_rng(7)
        emb = make_l2_normed(rng, 4, d=d)

        selector = DualKeyframeSelector(cfg)
        lam = cfg.dual_keyframe.lambda_common

        # Replicate the selector's internal math
        normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sim = normed @ normed.T
        g = (sim.sum(axis=1) - 1.0) / (4 - 1)

        k = cfg.dual_keyframe.neighborhood_k
        v = np.zeros(4)
        for i in range(4):
            lo, hi = max(0, i - k), min(3, i + k)
            neighbours = [j for j in range(lo, hi + 1) if j != i]
            sims = (normed[neighbours] @ normed[i])
            v[i] = 1.0 - sims.mean()

        lo_g, hi_g = g.min(), g.max()
        g_hat = np.zeros_like(g) if np.isclose(lo_g, hi_g) else (g - lo_g) / (hi_g - lo_g)
        lo_v, hi_v = v.min(), v.max()
        v_hat = np.zeros_like(v) if np.isclose(lo_v, hi_v) else (v - lo_v) / (hi_v - lo_v)

        common_score = lam * g_hat - (1.0 - lam) * v_hat
        expected_common = int(np.argmax(common_score))

        bs = _make_budgeted(emb, budget=2)
        result = selector.select(bs)

        # Find which result frame has the 'common' role
        common_role_pos = result.frame_roles.index("common")
        selected_common_abs = result.frame_indices[common_role_pos]
        assert selected_common_abs == expected_common

    def test_alpha_math_budget2(self, cfg):
        """The unique frame must be argmax(α*(1-ĝ) + (1-α)*v̂) where α=alpha_unique from config."""
        d = 768
        rng = np.random.default_rng(13)
        emb = make_l2_normed(rng, 5, d=d)

        selector = DualKeyframeSelector(cfg)
        alpha = cfg.dual_keyframe.alpha_unique

        normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        n = 5
        sim = normed @ normed.T
        g = (sim.sum(axis=1) - 1.0) / (n - 1)

        k = cfg.dual_keyframe.neighborhood_k
        v = np.zeros(n)
        for i in range(n):
            lo, hi = max(0, i - k), min(n - 1, i + k)
            neighbours = [j for j in range(lo, hi + 1) if j != i]
            sims = normed[neighbours] @ normed[i]
            v[i] = 1.0 - sims.mean()

        def minmax(arr):
            lo, hi = arr.min(), arr.max()
            return np.zeros_like(arr) if np.isclose(lo, hi) else (arr - lo) / (hi - lo)

        g_hat = minmax(g)
        v_hat = minmax(v)

        lam = cfg.dual_keyframe.lambda_common
        common_score = lam * g_hat - (1.0 - lam) * v_hat
        common_idx = int(np.argmax(common_score))

        unique_score = alpha * (1.0 - g_hat) + (1.0 - alpha) * v_hat
        unique_score_masked = unique_score.copy()
        unique_score_masked[common_idx] = -np.inf
        expected_unique = int(np.argmax(unique_score_masked))

        bs = _make_budgeted(emb, budget=2)
        result = selector.select(bs)

        unique_role_pos = result.frame_roles.index("unique")
        selected_unique_abs = result.frame_indices[unique_role_pos]
        assert selected_unique_abs == expected_unique

    def test_common_and_unique_never_same_frame(self, selector, rng):
        """The common and unique frames must always be different frame indices."""
        emb = make_l2_normed(rng, 8)
        bs = _make_budgeted(emb, budget=2)
        result = selector.select(bs)
        common_pos = result.frame_roles.index("common")
        unique_pos = result.frame_roles.index("unique")
        assert result.frame_indices[common_pos] != result.frame_indices[unique_pos]

    def test_budget_gt2_spacing_enforced(self, cfg):
        """With budget=5 and min_spacing=3, no two selected frames must be within 3 of each other."""
        rng = np.random.default_rng(77)
        emb = make_l2_normed(rng, 20)
        bs = _make_budgeted(emb, budget=5)

        # Ensure min_spacing from config is what we expect
        min_spacing = cfg.dual_keyframe.min_spacing_frames  # 3 per config.yaml
        selector = DualKeyframeSelector(cfg)
        result = selector.select(bs)

        indices = sorted(result.frame_indices)
        # "min_spacing" is the *minimum allowed* gap: |a - b| >= min_spacing.
        # So a gap of exactly min_spacing is allowed; excluded are gaps < min_spacing.
        for i in range(1, len(indices)):
            gap = indices[i] - indices[i - 1]
            assert gap >= min_spacing, (
                f"Frames {indices[i-1]} and {indices[i]} are only {gap} apart "
                f"(min_spacing={min_spacing})"
            )

    def test_all_identical_frames_returns_medoid(self, selector, rng):
        """When all embeddings are identical, the selector must return a single medoid frame."""
        base = make_l2_normed(rng, 1)
        emb = np.tile(base, (6, 1))
        bs = _make_budgeted(emb, budget=3)
        result = selector.select(bs)
        assert len(result.frame_indices) == 1
        assert result.frame_roles == ["medoid"]

    def test_frame_roles_temporal_order(self, selector, rng):
        """Selected frame_indices must always be in ascending (temporal) order."""
        emb = make_l2_normed(rng, 15)
        for budget in [1, 2, 4]:
            bs = _make_budgeted(emb, budget=budget)
            result = selector.select(bs)
            assert result.frame_indices == sorted(result.frame_indices), (
                f"frame_indices not sorted for budget={budget}: {result.frame_indices}"
            )

    def test_single_frame_shot(self, selector, rng):
        """A one-frame shot must return that frame as the medoid."""
        emb = make_l2_normed(rng, 1)
        bs = _make_budgeted(emb, budget=1, start=42)
        result = selector.select(bs)
        assert result.frame_indices == [42]
        assert result.frame_roles == ["medoid"]

    def test_minmax_norm_constant_array(self, selector):
        """_minmax_norm on a constant array must return all zeros without dividing by zero."""
        arr = np.full(5, 3.14)
        result = DualKeyframeSelector._minmax_norm(arr)
        assert np.all(result == 0.0)
        assert result.shape == arr.shape

    def test_volatility_edge_frames(self, selector, rng):
        """_volatility must not raise IndexError for first and last frames with k=1."""
        emb = make_l2_normed(rng, 4)
        normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        # Should not raise; k=1 from config.yaml
        v = selector._volatility(normed)
        assert v.shape == (4,)
        # Edge frames (i=0, i=3) have fewer neighbours — just verify finite values
        assert np.all(np.isfinite(v))

    def test_select_all_length_matches_input(self, selector, rng):
        """select_all must return the same number of SelectedKeyframes as input shots."""
        budgeted_shots = [
            _make_budgeted(make_l2_normed(rng, 5), budget=2, start=i * 10)
            for i in range(4)
        ]
        results = selector.select_all(budgeted_shots)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, SelectedKeyframes)

    # ------------------------------------------------------------------
    # Additional missing tests
    # ------------------------------------------------------------------

    def test_budget_exceeds_available_frames_is_capped(self, selector, rng):
        """When budget > number of available frames, the selector must cap it to
        the number of available frames and not raise an IndexError."""
        n_available = 3
        emb = make_l2_normed(rng, n_available)
        # Request budget=10 — far more than n_available
        bs = _make_budgeted(emb, budget=10)
        result = selector.select(bs)
        # Must select at most n_available frames
        assert len(result.frame_indices) <= n_available
        # Must not duplicate any frame index
        assert len(result.frame_indices) == len(set(result.frame_indices))

    def test_pick_extra_returns_none_when_all_within_spacing(self, selector, rng):
        """_pick_extra must return None when every frame index is within
        min_spacing of an already-selected frame."""
        n = 5
        emb = make_l2_normed(rng, n)
        normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        # Build v_hat as a uniform constant array so no index is preferred
        v_hat = np.full(n, 0.5, dtype=np.float64)

        # Select frames 0 and n-1 with min_spacing large enough to cover all
        min_spacing = n  # spacing covers the entire range
        already_selected = [0, n - 1]

        result = selector._pick_extra(v_hat, already_selected, min_spacing, n)
        assert result is None, (
            f"Expected None when all frames are within spacing={min_spacing}, "
            f"got {result}"
        )

    def test_two_frame_shot_with_budget_2(self, selector, rng):
        """A 2-frame shot with budget=2 must select both available frames,
        one as 'common' and one as 'unique' — there are no other options."""
        d = 768
        # Use two orthogonal unit vectors so they are clearly distinct
        a = np.zeros((1, d), dtype=np.float32); a[0, 0] = 1.0
        b = np.zeros((1, d), dtype=np.float32); b[0, 1] = 1.0
        emb = np.vstack([a, b])   # (2, 768)

        bs = _make_budgeted(emb, budget=2, start=0)
        result = selector.select(bs)

        # Both frames must be selected
        assert len(result.frame_indices) == 2
        assert sorted(result.frame_indices) == [0, 1]
        # Roles must be 'common' and 'unique' (in temporal order, so order may vary)
        assert set(result.frame_roles) == {"common", "unique"}

    def test_large_shot_budget_3_exact_infoshot_math(self, cfg, rng):
        """Replicate the full InfoShot λ/α computation in Python for a 10-frame
        shot with budget=3 and assert that the selector chooses the exact same
        common and unique indices."""
        n = 10
        d = 768
        emb = make_l2_normed(rng, n, d=d)

        selector = DualKeyframeSelector(cfg)
        lam = cfg.dual_keyframe.lambda_common
        alpha = cfg.dual_keyframe.alpha_unique
        k = cfg.dual_keyframe.neighborhood_k

        # Replicate internal math exactly ----------------------------------
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = emb / norms

        # Typicality
        sim = normed @ normed.T      # (n, n)
        row_sums = sim.sum(axis=1) - 1.0
        g = row_sums / (n - 1)

        # Volatility
        v = np.zeros(n, dtype=np.float64)
        for i in range(n):
            lo, hi = max(0, i - k), min(n - 1, i + k)
            neighbours = [j for j in range(lo, hi + 1) if j != i]
            if neighbours:
                sims = normed[neighbours] @ normed[i]
                v[i] = 1.0 - float(sims.mean())

        # Min-max normalise
        def _minmax(arr):
            lo_, hi_ = arr.min(), arr.max()
            if abs(hi_ - lo_) < 1e-9:
                return np.zeros_like(arr, dtype=np.float64)
            return (arr - lo_) / (hi_ - lo_)

        g_hat = _minmax(g)
        v_hat = _minmax(v)

        common_score = lam * g_hat - (1.0 - lam) * v_hat
        expected_common = int(np.argmax(common_score))

        unique_score = alpha * (1.0 - g_hat) + (1.0 - alpha) * v_hat
        unique_masked = unique_score.copy()
        unique_masked[expected_common] = -np.inf
        expected_unique = int(np.argmax(unique_masked))
        # ------------------------------------------------------------------

        bs = _make_budgeted(emb, budget=3, start=0)
        result = selector.select(bs)

        common_pos = result.frame_roles.index("common")
        unique_pos = result.frame_roles.index("unique")

        assert result.frame_indices[common_pos] == expected_common, (
            f"common frame mismatch: got {result.frame_indices[common_pos]}, "
            f"expected {expected_common}"
        )
        assert result.frame_indices[unique_pos] == expected_unique, (
            f"unique frame mismatch: got {result.frame_indices[unique_pos]}, "
            f"expected {expected_unique}"
        )
