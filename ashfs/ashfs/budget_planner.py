"""Budget planner for ASH-FS pipeline.

Assigns per-shot keyframe budgets based on complexity, duration, and global
video-level frame cap.  Pure CPU — no model calls, no GPU.
"""
from __future__ import annotations

import math
from typing import List

from .config import Config
from .types import BudgetedShot, ScoredShot


class BudgetPlanner:
    """Assign keyframe budgets to scored shots.

    Parameters
    ----------
    cfg:
        Pipeline configuration produced by :func:`~ashfs.config.load_config`.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        scored_shots: List[ScoredShot],
        complexity_threshold: float,
    ) -> List[BudgetedShot]:
        """Compute per-shot frame budgets and enforce the global video cap.

        Parameters
        ----------
        scored_shots:
            Ordered list of :class:`~ashfs.types.ScoredShot` objects for the
            entire video.
        complexity_threshold:
            Shots whose ``complexity`` exceeds this value receive an extra
            complexity-bonus frame.

        Returns
        -------
        list[BudgetedShot]
            One :class:`~ashfs.types.BudgetedShot` per input shot, in the
            same order, with budgets already capped by both the per-shot and
            global video limits.
        """
        cfg = self._cfg.budget
        base: int = cfg.base_frames_per_shot

        budgets: List[int] = []
        for shot in scored_shots:
            budget = base

            # 1. Complexity bonus — one extra unique slot
            if shot.complexity > complexity_threshold:
                budget += 1

            # 2. Duration bonus
            if shot.shot.duration_s > cfg.duration_bonus_start_s:
                extra = int(
                    (shot.shot.duration_s - cfg.duration_bonus_start_s)
                    / cfg.duration_bonus_interval_s
                )
                extra = max(0, extra)
                budget += extra

            # 3. Duration floor — guarantees minimum keyframes for long shots
            if cfg.duration_floor_enabled:
                floor = math.ceil(shot.shot.duration_s / cfg.duration_floor_interval_s)
                floor = max(1, floor)
                budget = max(budget, floor)

            # 4. Per-shot cap — config max
            budget = min(budget, cfg.max_frames_per_shot)

            # 5. Per-shot cap — actual available frames
            budget = min(budget, shot.frame_count)

            budgets.append(budget)

        # ------------------------------------------------------------------
        # Global cap enforcement
        # ------------------------------------------------------------------
        budgets = self._enforce_global_cap(scored_shots, budgets, base)

        return [
            BudgetedShot(scored_shot=shot, budget=b)
            for shot, b in zip(scored_shots, budgets)
        ]

    def get_stats(self, budgeted: List[BudgetedShot]) -> dict:
        """Return summary statistics over a list of budgeted shots.

        Parameters
        ----------
        budgeted:
            Output of :meth:`plan`.

        Returns
        -------
        dict with keys:
            ``total_frames``, ``min_budget``, ``max_budget``,
            ``mean_budget``, ``shots_at_base``, ``shots_with_bonus``.
        """
        if not budgeted:
            return {
                "total_frames": 0,
                "min_budget": 0,
                "max_budget": 0,
                "mean_budget": 0.0,
                "shots_at_base": 0,
                "shots_with_bonus": 0,
            }

        base = self._cfg.budget.base_frames_per_shot
        bs = [b.budget for b in budgeted]
        total = sum(bs)
        shots_at_base = sum(1 for b in bs if b == base)
        shots_with_bonus = sum(1 for b in bs if b > base)

        return {
            "total_frames": total,
            "min_budget": min(bs),
            "max_budget": max(bs),
            "mean_budget": total / len(bs),
            "shots_at_base": shots_at_base,
            "shots_with_bonus": shots_with_bonus,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_global_cap(
        self,
        scored_shots: List[ScoredShot],
        budgets: List[int],
        base: int,
    ) -> List[int]:
        """Trim budgets so their sum does not exceed the global video cap.

        Trimming is performed in three passes, each iterating from
        lowest-complexity shot to highest (least important first):

        * Pass 1 — reduce any shot whose budget > base + 1 down to base + 1.
        * Pass 2 — same condition; catches any remainder left after pass 1
          that still keeps total over the cap.
        * Pass 3 — reduce any shot whose budget > base down to base
          (medoid-only shots are last to lose their frame).

        Within each pass shots are sorted by complexity ascending so the
        lowest-complexity shots are trimmed first.
        """
        max_total = self._cfg.budget.max_total_frames_per_video
        if max_total is None:
            return budgets  # no cap — fully adaptive

        if sum(budgets) <= max_total:
            return budgets

        # Work on a mutable copy
        b = list(budgets)

        # Indices sorted by complexity ASC (lowest trimmed first within pass)
        order = sorted(
            range(len(scored_shots)),
            key=lambda i: scored_shots[i].complexity,
        )

        # Pass 1 & 2 — target: budget > base + 1  →  base + 1
        for _pass in range(2):
            if sum(b) <= max_total:
                break
            for i in order:
                if sum(b) <= max_total:
                    break
                if b[i] > base + 1:
                    b[i] = base + 1

        # Pass 3 — target: budget > base  →  base
        # Intentionally stops here: every shot retains its base allocation (1
        # frame minimum).  The hierarchical diversity planner is responsible for
        # reducing below this floor via chunk-level summarisation.
        if sum(b) > max_total:
            for i in order:
                if sum(b) <= max_total:
                    break
                if b[i] > base:
                    b[i] = base

        return b
