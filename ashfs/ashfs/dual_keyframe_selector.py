"""Dual keyframe selector for ASH-FS pipeline.

Implements the InfoShot formulation:
  - Typicality  (g_i)  — how representative a frame is of the whole shot.
  - Volatility  (v_i)  — how locally distinctive a frame is from its
                          temporal neighbours.

Selection strategy per budget:
  budget == 1  →  medoid (most typical frame).
  budget == 2  →  common (typicality-dominant) + unique (atypicality-dominant).
  budget  > 2  →  common + unique + greedy high-volatility frames with
                   minimum temporal spacing.

Pure CPU — numpy only, no GPU, no model calls.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from .config import Config
from .types import BudgetedShot, SelectedKeyframes


class DualKeyframeSelector:
    """Select keyframes from each budgeted shot using the InfoShot formulation.

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

    def select(self, budgeted_shot: BudgetedShot) -> SelectedKeyframes:
        """Select keyframes for a single budgeted shot.

        Parameters
        ----------
        budgeted_shot:
            A :class:`~ashfs.types.BudgetedShot` carrying embeddings and the
            assigned frame budget.

        Returns
        -------
        SelectedKeyframes
            Contains the subset of frame indices, their embeddings, and a
            ``frame_roles`` label for each (``'medoid'``, ``'common'``,
            ``'unique'``, or ``'extra'``).
        """
        ss = budgeted_shot.scored_shot
        emb: np.ndarray = ss.embeddings         # (N, D)
        frame_indices: list = ss.frame_indices
        budget: int = budgeted_shot.budget

        n_frames = emb.shape[0]

        # Guard: zero budget means this shot was trimmed by the global cap.
        if budget == 0:
            return SelectedKeyframes(
                shot=ss.shot,
                frame_indices=[],
                embeddings=np.empty((0, emb.shape[1] if emb.ndim > 1 else 768), dtype=emb.dtype),
                siglip_embeddings=None,
                frame_roles=[],
            )

        # BUG-DKS-2: n_frames == 0 was not guarded.  _all_identical() returns
        # True for empty arrays (shape[0] <= 1), so it falls through to
        # _make_result with selected=[0], causing IndexError on frame_indices[0].
        if n_frames == 0:
            return SelectedKeyframes(
                shot=ss.shot,
                frame_indices=[],
                embeddings=emb,
                siglip_embeddings=None,
                frame_roles=[],
            )

        # Edge case — single frame or budget exhausted to 1
        if n_frames == 1:
            return self._make_result(ss.shot, frame_indices, emb, [0], ["medoid"])

        # Edge case — cap budget to available frames
        budget = min(budget, n_frames)

        # Edge case — all embeddings identical (static shot)
        if self._all_identical(emb):
            return self._make_result(ss.shot, frame_indices, emb, [0], ["medoid"])

        # Normalize rows to unit length for cosine similarity via dot product
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed: np.ndarray = emb / norms         # (N, D)

        g = self._typicality(normed)             # (N,)
        v = self._volatility(normed)             # (N,)

        g_hat = self._minmax_norm(g)             # ĝ ∈ [0, 1]
        v_hat = self._minmax_norm(v)             # v̂ ∈ [0, 1]

        if budget == 1:
            idx = int(np.argmax(g_hat))
            return self._make_result(ss.shot, frame_indices, emb, [idx], ["medoid"])

        # budget >= 2
        lam: float = self._cfg.dual_keyframe.lambda_common
        alpha: float = self._cfg.dual_keyframe.alpha_unique

        # Common frame — typicality-dominant
        common_score = lam * g_hat - (1.0 - lam) * v_hat
        common_idx = int(np.argmax(common_score))

        # Unique frame — atypicality-dominant (exclude common)
        unique_score = alpha * (1.0 - g_hat) + (1.0 - alpha) * v_hat
        unique_score_masked = unique_score.copy()
        unique_score_masked[common_idx] = -np.inf
        unique_idx = int(np.argmax(unique_score_masked))

        if budget == 2:
            selected = [common_idx, unique_idx]
            roles = ["common", "unique"]
            selected, roles = self._sort_by_index(selected, roles)
            return self._make_result(ss.shot, frame_indices, emb, selected, roles)

        # budget > 2 — greedy extra frames by volatility with spacing
        selected = [common_idx, unique_idx]
        roles = ["common", "unique"]
        min_spacing: int = self._cfg.dual_keyframe.min_spacing_frames

        remaining_budget = budget - 2
        for _ in range(remaining_budget):
            best_idx = self._pick_extra(v_hat, selected, min_spacing, n_frames)
            if best_idx is None:
                break
            selected.append(best_idx)
            roles.append("extra")

        selected, roles = self._sort_by_index(selected, roles)
        return self._make_result(ss.shot, frame_indices, emb, selected, roles)

    def select_all(
        self, budgeted_shots: List[BudgetedShot]
    ) -> List[SelectedKeyframes]:
        """Select keyframes for every shot in a video.

        Parameters
        ----------
        budgeted_shots:
            Ordered output of :meth:`~ashfs.budget_planner.BudgetPlanner.plan`.

        Returns
        -------
        list[SelectedKeyframes]
            One entry per input shot, in the same order.
        """
        return [self.select(bs) for bs in budgeted_shots]

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    def _typicality(self, normed: np.ndarray) -> np.ndarray:
        """Compute mean cosine similarity of each frame to all other frames.

        g_i = (1 / (N-1)) * sum_{j != i} cos(e_i, e_j)

        For N == 1 returns an array of zeros (handled upstream).
        """
        n = normed.shape[0]
        if n == 1:
            return np.zeros(1, dtype=np.float64)

        # Full similarity matrix via dot product of unit vectors
        sim_matrix = normed @ normed.T                     # (N, N)
        row_sums = sim_matrix.sum(axis=1) - 1.0            # subtract self-sim
        return row_sums / (n - 1)

    def _volatility(self, normed: np.ndarray) -> np.ndarray:
        """Compute local volatility for each frame.

        v_i = 1 - mean cosine similarity of frame i to its k nearest
        temporal neighbours (indices i-k .. i+k, excluding self).
        """
        n = normed.shape[0]
        k: int = self._cfg.dual_keyframe.neighborhood_k
        v = np.zeros(n, dtype=np.float64)

        for i in range(n):
            lo = max(0, i - k)
            hi = min(n - 1, i + k)
            # Collect neighbour indices in temporal window, excluding self
            neighbours = [j for j in range(lo, hi + 1) if j != i]
            if not neighbours:
                v[i] = 0.0
                continue
            neighbor_emb = normed[neighbours]              # (M, D)
            sims = neighbor_emb @ normed[i]               # (M,)
            v[i] = 1.0 - float(sims.mean())

        return v

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise array to [0, 1]. Handles constant arrays."""
        lo, hi = float(arr.min()), float(arr.max())
        if math.isclose(lo, hi):
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _all_identical(emb: np.ndarray) -> bool:
        """Return True if every row in emb is identical to the first row."""
        if emb.shape[0] <= 1:
            return True
        diffs = emb[1:] - emb[0]
        return bool(np.allclose(diffs, 0.0))

    @staticmethod
    def _pick_extra(
        v_hat: np.ndarray,
        already_selected: list,
        min_spacing: int,
        n_frames: int,
    ) -> int | None:
        """Pick the highest-volatility frame not within min_spacing of any
        already-selected frame index.

        Returns the chosen index or ``None`` if no valid candidate remains.
        """
        scores = v_hat.copy()
        for sel in already_selected:
            # BUG-DKS-1: the exclusion zone was [sel-min_spacing, sel+min_spacing],
            # masking frames at exactly min_spacing distance from sel.  The spec
            # defines min_spacing_frames as the *minimum* gap, meaning a candidate
            # at distance == min_spacing is the closest frame that IS allowed.
            # The zone to mask is therefore (sel-min_spacing, sel+min_spacing)
            # exclusive — i.e., only indices where |idx - sel| < min_spacing.
            lo = max(0, sel - min_spacing + 1)
            hi = min(n_frames - 1, sel + min_spacing - 1)
            scores[lo : hi + 1] = -np.inf

        # Also exclude already selected frames explicitly (handles sel itself,
        # which the zone above still covers, but be explicit for clarity).
        for sel in already_selected:
            scores[sel] = -np.inf

        best = int(np.argmax(scores))
        if scores[best] == -np.inf:
            return None
        return best

    @staticmethod
    def _sort_by_index(
        indices: list, roles: list
    ) -> tuple[list, list]:
        """Sort selected frame indices ascending (temporal order) and
        reorder roles to match."""
        paired = sorted(zip(indices, roles), key=lambda x: x[0])
        if not paired:
            return [], []
        sorted_indices, sorted_roles = zip(*paired)
        return list(sorted_indices), list(sorted_roles)

    @staticmethod
    def _make_result(
        shot,
        frame_indices: list,
        emb: np.ndarray,
        selected_local: list,
        roles: list,
    ) -> SelectedKeyframes:
        """Build a :class:`~ashfs.types.SelectedKeyframes` from local indices."""
        abs_indices = [frame_indices[i] for i in selected_local]
        sel_emb = emb[selected_local]
        return SelectedKeyframes(
            shot=shot,
            frame_indices=abs_indices,
            embeddings=sel_emb,
            siglip_embeddings=None,
            frame_roles=roles,
        )
