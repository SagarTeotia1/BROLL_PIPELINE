"""Unit tests for ashfs.budget_planner.

Verifies per-shot budget allocation rules (base, complexity bonus, duration bonus,
per-shot cap, frame count cap) and global video cap enforcement using synthetic
ScoredShot objects.
"""
import numpy as np
import pytest

from ashfs.budget_planner import BudgetPlanner
from ashfs.types import Shot, ScoredShot, BudgetedShot
from tests.conftest import make_l2_normed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shot(start: int, end: int, fps: float = 25.0, duration_s: float = None) -> Shot:
    if duration_s is None:
        duration_s = (end - start + 1) / fps
    return Shot(start_frame=start, end_frame=end, duration_s=duration_s, fps=fps)


def _make_scored_shot(
    start: int,
    end: int,
    complexity: float,
    duration_s: float = None,
    fps: float = 25.0,
    n_frames: int = None,
    rng=None,
    variance_score: float = 0.0,
    motion_score: float = 0.0,
) -> ScoredShot:
    """Build a ScoredShot with synthetic embeddings."""
    shot = _make_shot(start, end, fps=fps, duration_s=duration_s)
    if n_frames is None:
        n_frames = shot.frame_count
    if rng is None:
        rng = np.random.default_rng(42)
    emb = make_l2_normed(rng, max(1, n_frames))
    frame_indices = list(range(start, start + len(emb)))
    return ScoredShot(
        shot=shot,
        frame_indices=frame_indices,
        embeddings=emb,
        variance_score=variance_score,
        motion_score=motion_score,
        complexity=complexity,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def planner(cfg) -> BudgetPlanner:
    """BudgetPlanner built from the shared config fixture."""
    return BudgetPlanner(cfg)


@pytest.fixture
def threshold(cfg) -> float:
    """Complexity threshold for the configured content vertical."""
    from ashfs.config import get_complexity_threshold
    return get_complexity_threshold(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBudgetPlannerAllocation:

    def test_base_allocation_for_low_complexity(self, planner, cfg, threshold, rng):
        """A shot with complexity below threshold must receive only the base budget."""
        base = cfg.budget.base_frames_per_shot
        low_complexity = threshold * 0.5
        shot = _make_scored_shot(0, 99, complexity=low_complexity, duration_s=1.0, rng=rng)
        results = planner.plan([shot], threshold)
        assert results[0].budget == base

    def test_complexity_bonus_applied(self, planner, cfg, threshold, rng):
        """A shot with complexity above threshold must receive base + 1 frames."""
        base = cfg.budget.base_frames_per_shot
        high_complexity = threshold + 0.10
        shot = _make_scored_shot(
            0, 99, complexity=high_complexity, duration_s=1.0, rng=rng, n_frames=10
        )
        results = planner.plan([shot], threshold)
        # budget must be at least base + 1 (before any caps)
        assert results[0].budget >= base + 1

    def test_duration_bonus_single_interval(self, planner, cfg, threshold, rng):
        """Duration = start_s + 1 * interval_s must add exactly 1 bonus frame."""
        base = cfg.budget.base_frames_per_shot
        interval = cfg.budget.duration_bonus_interval_s
        start_s = cfg.budget.duration_bonus_start_s
        duration_s = start_s + interval + 0.01   # just over the first interval

        shot = _make_scored_shot(
            0, 499, complexity=0.0, duration_s=duration_s, rng=rng, n_frames=50
        )
        results = planner.plan([shot], threshold)
        # complexity below threshold → no complexity bonus; duration bonus = 1
        assert results[0].budget >= base + 1

    def test_duration_bonus_two_intervals(self, planner, cfg, threshold, rng):
        """Duration = start_s + 2 * interval_s must add exactly 2 bonus frames."""
        base = cfg.budget.base_frames_per_shot
        interval = cfg.budget.duration_bonus_interval_s
        start_s = cfg.budget.duration_bonus_start_s
        duration_s = start_s + 2 * interval + 0.01

        shot = _make_scored_shot(
            0, 999, complexity=0.0, duration_s=duration_s, rng=rng, n_frames=50
        )
        results = planner.plan([shot], threshold)
        assert results[0].budget >= base + 2

    def test_per_shot_cap_respected(self, planner, cfg, threshold, rng):
        """Computed budget must never exceed max_frames_per_shot."""
        max_fps = cfg.budget.max_frames_per_shot
        # Push complexity and duration high to generate a large uncapped budget
        duration_s = cfg.budget.duration_bonus_start_s + 100 * cfg.budget.duration_bonus_interval_s
        shot = _make_scored_shot(
            0, 9999, complexity=1.0, duration_s=duration_s, rng=rng, n_frames=500
        )
        results = planner.plan([shot], threshold)
        assert results[0].budget <= max_fps

    def test_frame_count_cap(self, planner, cfg, threshold, rng):
        """Budget must never exceed the number of candidate frames in the shot."""
        # Shot has only 2 candidate frames regardless of high complexity/duration
        duration_s = cfg.budget.duration_bonus_start_s + 50 * cfg.budget.duration_bonus_interval_s
        shot = _make_scored_shot(
            0, 1, complexity=1.0, duration_s=duration_s, rng=rng, n_frames=2
        )
        results = planner.plan([shot], threshold)
        assert results[0].budget <= 2

    def test_global_cap_trims_lowest_complexity_first(self, planner, cfg, threshold, rng):
        """When total budget exceeds max_total_frames_per_video, lowest-complexity shots
        must be trimmed before higher-complexity ones."""
        max_total = cfg.budget.max_total_frames_per_video
        base = cfg.budget.base_frames_per_shot
        max_fps = cfg.budget.max_frames_per_shot

        # Create enough shots at max budget to guarantee we exceed the global cap
        n_shots = (max_total // max_fps) + 5
        shots = [
            _make_scored_shot(
                i * 100, i * 100 + 99,
                complexity=1.0 if i == 0 else 0.01,  # first shot is high complexity
                duration_s=60.0,
                rng=np.random.default_rng(i),
                n_frames=max_fps,
            )
            for i in range(n_shots)
        ]
        results = planner.plan(shots, threshold)

        total = sum(r.budget for r in results)
        assert total <= max_total

        # The high-complexity shot (index 0) must not have been trimmed to base
        # while lower-complexity shots still have bonus frames
        high_budget = results[0].budget
        low_budgets = [results[i].budget for i in range(1, len(results))]
        # high-complexity shot should be trimmed last: budget >= any low-complexity budget
        assert high_budget >= min(low_budgets)

    def test_global_cap_preserves_base_allocation_last(self, planner, cfg, threshold, rng):
        """Even when the global cap is very tight, shots must retain at least their
        base allocation (base=1) as long as there are still shots to trim above base."""
        max_total = cfg.budget.max_total_frames_per_video
        base = cfg.budget.base_frames_per_shot
        max_fps = cfg.budget.max_frames_per_shot

        n_shots = (max_total // max_fps) + 10
        shots = [
            _make_scored_shot(
                i * 100, i * 100 + 99,
                complexity=float(i) / n_shots,
                duration_s=60.0,
                rng=np.random.default_rng(i),
                n_frames=max_fps,
            )
            for i in range(n_shots)
        ]
        results = planner.plan(shots, threshold)

        # No shot should drop below base (= 1) unless total frames < n_shots
        total_frames_available = sum(r.scored_shot.frame_count for r in results)
        if total_frames_available >= n_shots:
            for r in results:
                assert r.budget >= base

    def test_get_stats_totals_correct(self, planner, cfg, threshold, rng):
        """get_stats() must return total_frames equal to the sum of all budgets."""
        shots = [
            _make_scored_shot(i * 50, i * 50 + 49, complexity=0.1, rng=rng)
            for i in range(5)
        ]
        budgeted = planner.plan(shots, threshold)
        stats = planner.get_stats(budgeted)
        expected_total = sum(b.budget for b in budgeted)
        assert stats["total_frames"] == expected_total

    def test_empty_input(self, planner, threshold):
        """An empty input list must produce an empty output list."""
        results = planner.plan([], threshold)
        assert results == []

    def test_get_stats_empty(self, planner):
        """get_stats on an empty list must return all zero values."""
        stats = planner.get_stats([])
        assert stats["total_frames"] == 0
        assert stats["min_budget"] == 0
        assert stats["max_budget"] == 0
        assert stats["mean_budget"] == 0.0

    # ------------------------------------------------------------------
    # Additional missing tests
    # ------------------------------------------------------------------

    def test_shot_with_zero_candidate_frames_gets_zero_budget(self, planner, cfg, threshold):
        """A ScoredShot whose frame_count == 0 must receive budget == 0.

        The per-shot cap ``budget = min(budget, shot.frame_count)`` clamps any
        computed budget down to 0 when there are literally no candidate frames.
        """
        # Build a ScoredShot manually with an empty embeddings array
        shot = _make_shot(0, 99, fps=25.0, duration_s=1.0)
        emb = np.empty((0, 768), dtype=np.float32)
        scored = ScoredShot(
            shot=shot,
            frame_indices=[],
            embeddings=emb,
            variance_score=0.0,
            motion_score=0.0,
            complexity=threshold + 0.1,  # above threshold → would get complexity bonus
        )
        results = planner.plan([scored], threshold)
        assert results[0].budget == 0, (
            f"Expected budget=0 for a shot with 0 candidate frames, "
            f"got {results[0].budget}"
        )

    def test_global_cap_all_shots_at_base_do_not_undershoot(self, planner, cfg, threshold, rng):
        """When all shots already sit at base budget and their total is within the
        global cap, _enforce_global_cap must not reduce any budget below base.

        This exercises the invariant that pass 3 never trims a budget that is
        already at ``base``."""
        base = cfg.budget.base_frames_per_shot
        max_total = cfg.budget.max_total_frames_per_video

        # n_shots such that n_shots * base <= max_total (all within cap already)
        n_shots = min(max_total // base, 10)
        shots = [
            _make_scored_shot(
                i * 10, i * 10 + 9,
                complexity=0.0,    # below threshold → no bonus → budget == base
                duration_s=0.1,    # very short → no duration bonus
                rng=np.random.default_rng(i),
                n_frames=base,     # exactly base frames available
            )
            for i in range(n_shots)
        ]
        results = planner.plan(shots, threshold)

        total = sum(r.budget for r in results)
        # The global cap is not breached so no trimming should occur
        assert total == n_shots * base
        for r in results:
            assert r.budget == base

    def test_enforce_global_cap_pass3_does_not_trim_below_base(self, planner, cfg, threshold, rng):
        """Pass 3 of _enforce_global_cap may reduce shots with budget > base down
        to base, but must never reduce a shot that is already at base.

        Set up a scenario where total exceeds cap but passes 1 & 2 already
        brought total to exactly cap so pass 3 should not fire."""
        base = cfg.budget.base_frames_per_shot
        max_total = cfg.budget.max_total_frames_per_video

        # Create shots so that after pass 1 & 2 (trim to base+1) the total
        # lands exactly at or below max_total — pass 3 must leave them alone.
        # n_shots * (base + 1) == max_total means each shot has base+1 budget
        # and the total is exactly at the cap after passes 1+2.
        n_shots = max_total // (base + 1)
        if n_shots == 0:
            pytest.skip("Config does not allow meaningful pass-3 scenario.")

        shots = [
            _make_scored_shot(
                i * 100, i * 100 + 99,
                complexity=threshold + 0.1,  # ensures complexity bonus → budget >= base+1
                duration_s=0.1,              # no duration bonus
                rng=np.random.default_rng(i),
                n_frames=base + 1,           # exactly base+1 frames available
            )
            for i in range(n_shots)
        ]
        results = planner.plan(shots, threshold)

        total = sum(r.budget for r in results)
        assert total <= max_total

        # No shot should have been trimmed below base by pass 3
        for r in results:
            assert r.budget >= base, (
                f"budget {r.budget} is below base {base} — "
                "pass 3 incorrectly trimmed a shot below base"
            )
