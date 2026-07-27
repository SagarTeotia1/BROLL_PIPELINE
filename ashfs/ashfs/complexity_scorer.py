"""Complexity scoring module for ASH-FS pipeline.

For each :class:`ShotEmbeddings`, computes:
  - ``variance_score``  — mean pairwise cosine distance among all embeddings
                          in the shot (capped at 100 random pairs when >15 frames).
  - ``motion_score``    — mean sequential L2 distance between consecutive
                          embeddings, normalised by the number of consecutive
                          pairs (n - 1) so the result is independent of shot
                          length.
  - ``complexity``      — weighted sum: w1 * variance_score + w2 * motion_score.

Single-frame shots and shots with no embeddings return complexity=0.0.
"""
from __future__ import annotations

import numpy as np

from ashfs.config import Config
from ashfs.types import ScoredShot, ShotEmbeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAIRWISE_CAP = 100          # max random pairs sampled for variance score
_PAIRWISE_THRESHOLD = 15     # frame count above which we sample pairs
_PAIRWISE_SEED = 42          # reproducibility seed for pair sampling


# ---------------------------------------------------------------------------
# ComplexityScorer
# ---------------------------------------------------------------------------

class ComplexityScorer:
    """Compute complexity scores for shots based on their DINOv2 embeddings.

    Parameters
    ----------
    config:
        Parsed pipeline configuration.  Only ``config.complexity`` is used.
    """

    def __init__(self, config: Config) -> None:
        self._w1: float = config.complexity.variance_weight
        self._w2: float = config.complexity.motion_weight

    # ------------------------------------------------------------------
    # Public: score a single shot
    # ------------------------------------------------------------------

    def score(self, shot_embeddings: ShotEmbeddings) -> ScoredShot:
        """Compute complexity for one shot.

        Parameters
        ----------
        shot_embeddings:
            A :class:`ShotEmbeddings` instance with ``shot``, ``frame_indices``,
            and ``embeddings`` (shape ``(N, D)`` or empty).

        Returns
        -------
        ScoredShot
            Populated with ``variance_score``, ``motion_score``, and
            ``complexity``.  For zero/one-frame shots, all scores are 0.0.
        """
        emb = shot_embeddings.embeddings
        n_frames = 0 if emb is None else len(emb)

        if n_frames <= 1 or emb is None:
            return ScoredShot(
                shot=shot_embeddings.shot,
                frame_indices=shot_embeddings.frame_indices,
                embeddings=emb if emb is not None else np.empty((0,), dtype=np.float32),
                variance_score=0.0,
                motion_score=0.0,
                complexity=0.0,
            )

        variance_score = self._compute_variance_score(emb)
        motion_score = self._compute_motion_score(emb)
        complexity = self._w1 * variance_score + self._w2 * motion_score

        return ScoredShot(
            shot=shot_embeddings.shot,
            frame_indices=shot_embeddings.frame_indices,
            embeddings=emb,
            variance_score=variance_score,
            motion_score=motion_score,
            complexity=complexity,
        )

    # ------------------------------------------------------------------
    # Public: batch scoring
    # ------------------------------------------------------------------

    def score_all(self, shots_with_embeddings: list[ShotEmbeddings]) -> list[ScoredShot]:
        """Score all shots in the list.

        Parameters
        ----------
        shots_with_embeddings:
            List of :class:`ShotEmbeddings` instances (order preserved).

        Returns
        -------
        list[ScoredShot]
            One :class:`ScoredShot` per input, in the same order.
        """
        return [self.score(se) for se in shots_with_embeddings]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_variance_score(emb: np.ndarray) -> float:
        """Mean pairwise cosine distance among all embeddings in a shot.

        When the shot has more than ``_PAIRWISE_THRESHOLD`` frames, a random
        sample of up to ``_PAIRWISE_CAP`` pairs is used (seed=42) to keep
        CPU cost bounded.

        Assumes embeddings are L2-normalised so cosine distance =
        1 - dot(a, b).
        """
        n = len(emb)
        if n < 2:
            return 0.0

        if n <= _PAIRWISE_THRESHOLD:
            # All n*(n-1)/2 pairs
            # Gram matrix — shape (n, n)
            gram = np.clip(emb @ emb.T, -1.0, 1.0)
            # Extract upper triangle (excluding diagonal)
            i_upper, j_upper = np.triu_indices(n, k=1)
            pairwise_cos_sim = gram[i_upper, j_upper]
        else:
            # Random pair sampling
            rng = np.random.default_rng(_PAIRWISE_SEED)
            indices_a = rng.integers(0, n, size=_PAIRWISE_CAP)
            indices_b = rng.integers(0, n, size=_PAIRWISE_CAP)
            # Avoid self-pairs (rare but possible)
            same = indices_a == indices_b
            if same.any():
                indices_b[same] = (indices_b[same] + 1) % n
            dots = np.einsum("nd,nd->n", emb[indices_a], emb[indices_b])
            pairwise_cos_sim = np.clip(dots, -1.0, 1.0)

        return float(np.mean(1.0 - pairwise_cos_sim))

    @staticmethod
    def _compute_motion_score(emb: np.ndarray) -> float:
        """Mean sequential L2 distance between consecutive embeddings.

        For L2-normalised vectors:
            ||a - b||_2 = sqrt(2 - 2 * dot(a, b))

        The result is normalised by the number of consecutive pairs
        (i.e. frame_count - 1) so it is independent of shot length.
        """
        n = len(emb)
        if n < 2:
            return 0.0

        # Sequential dot products: shape (n-1,)
        dots = np.clip(np.einsum("nd,nd->n", emb[:-1], emb[1:]), -1.0, 1.0)
        l2_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))

        # BUG-CS-1: the comment claimed to normalise by frame count (n) but
        # the spec calls for the *mean* frame-to-frame delta, which is the sum
        # of (n-1) consecutive distances divided by (n-1) — not by n.  Dividing
        # by n under-counts the score by (n-1)/n and makes short shots appear
        # less complex than long shots with identical per-frame motion, breaking
        # cross-shot comparability.  Fixed: divide by number of pairs (n-1).
        return float(np.sum(l2_dists) / (n - 1))
