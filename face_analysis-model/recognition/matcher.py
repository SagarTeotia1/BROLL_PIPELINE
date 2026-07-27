"""Gallery matching: cosine similarity of ArcFace embeddings against the cast.

The gallery is tiny (one 512-d centroid per actor, plus optionally every enrolment
vector), so the match is a single ``(N, dim) x (dim, M)`` matmul. It is kept on the GPU
because the query embeddings are already there in spirit and a matmul avoids a Python
loop; for CPU-only runs the same code path works with a torch CPU tensor.

Two thresholds decide the outcome:

* ``similarity_threshold`` - the best score must clear it, otherwise "Unknown".
* ``margin_threshold`` - the gap to the runner-up must clear it, otherwise the match is
  reported but flagged ``ambiguous`` so the caller can keep voting instead of locking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from utils.logging_utils import get_logger

log = get_logger(__name__)

UNKNOWN = "Unknown"


@dataclass
class MatchResult:
    """Outcome of matching one query embedding against the gallery."""

    actor_id: int
    name: str
    similarity: float
    margin: float
    runner_up: str = ""
    ambiguous: bool = False

    @property
    def is_known(self) -> bool:
        return self.actor_id >= 0

    def as_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "name": self.name,
            "similarity": round(self.similarity, 4),
            "margin": round(self.margin, 4),
            "ambiguous": self.ambiguous,
        }


class GalleryMatcher:
    """Holds the cast gallery on-device and answers nearest-identity queries.

    Args:
        device: where the gallery matrix lives.
        similarity_threshold: minimum cosine similarity for a named match.
        margin_threshold: minimum best-vs-second gap for an unambiguous match.
        use_max_over_samples: when the gallery stores every enrolment vector (not just
            centroids), score an actor by their best-matching sample instead of the
            centroid. More robust to pose variety in the reference photos.
    """

    def __init__(
        self,
        device: torch.device,
        similarity_threshold: float = 0.38,
        margin_threshold: float = 0.06,
        use_max_over_samples: bool = True,
    ) -> None:
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.use_max_over_samples = use_max_over_samples
        self._matrix: Optional[torch.Tensor] = None      # (M, dim) on device
        self._owner_ids: np.ndarray = np.zeros((0,), dtype=np.int64)  # (M,) actor id per row
        self._id_to_name: Dict[int, str] = {}
        self._actor_ids: np.ndarray = np.zeros((0,), dtype=np.int64)  # unique, sorted

    # -- gallery management -------------------------------------------------
    def set_gallery(
        self,
        vectors: np.ndarray,
        owner_ids: Sequence[int],
        id_to_name: Dict[int, str],
    ) -> None:
        """Replace the gallery.

        Args:
            vectors: ``(M, dim)`` L2-normalised embeddings (centroids and/or samples).
            owner_ids: length-``M`` actor id for each row.
            id_to_name: actor id -> display name.
        """
        if vectors.size == 0:
            self.clear()
            return
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-9)
        self._matrix = torch.from_numpy(arr).to(self.device)
        self._owner_ids = np.asarray(owner_ids, dtype=np.int64)
        self._id_to_name = dict(id_to_name)
        self._actor_ids = np.unique(self._owner_ids)
        log.info(
            "Gallery loaded: %d actors / %d vectors (dim=%d)",
            len(self._actor_ids), arr.shape[0], arr.shape[1],
        )

    def clear(self) -> None:
        """Drop the gallery (every query then returns Unknown)."""
        self._matrix = None
        self._owner_ids = np.zeros((0,), dtype=np.int64)
        self._id_to_name = {}
        self._actor_ids = np.zeros((0,), dtype=np.int64)

    @property
    def is_empty(self) -> bool:
        return self._matrix is None or self._matrix.numel() == 0

    @property
    def actor_names(self) -> List[str]:
        return [self._id_to_name.get(int(i), UNKNOWN) for i in self._actor_ids]

    @property
    def num_actors(self) -> int:
        return int(self._actor_ids.size)

    # -- querying -----------------------------------------------------------
    def match(self, embeddings: np.ndarray) -> List[MatchResult]:
        """Match ``(N, dim)`` query embeddings against the gallery."""
        n = 0 if embeddings.size == 0 else embeddings.shape[0]
        if n == 0:
            return []
        if self.is_empty:
            return [MatchResult(-1, UNKNOWN, 0.0, 0.0) for _ in range(n)]

        sims = self.similarity_matrix(embeddings)  # (N, num_actors)
        results: List[MatchResult] = []
        order = np.argsort(-sims, axis=1)
        for i in range(n):
            best_col = int(order[i, 0])
            best = float(sims[i, best_col])
            if sims.shape[1] > 1:
                second_col = int(order[i, 1])
                second = float(sims[i, second_col])
                runner_up = self._id_to_name.get(int(self._actor_ids[second_col]), UNKNOWN)
            else:
                second, runner_up = -1.0, ""
            margin = best - second if second > -1.0 else best
            actor_id = int(self._actor_ids[best_col])
            if best < self.similarity_threshold:
                results.append(MatchResult(-1, UNKNOWN, best, margin, runner_up, False))
            else:
                results.append(
                    MatchResult(
                        actor_id=actor_id,
                        name=self._id_to_name.get(actor_id, UNKNOWN),
                        similarity=best,
                        margin=margin,
                        runner_up=runner_up,
                        ambiguous=margin < self.margin_threshold,
                    )
                )
        return results

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Per-actor cosine similarity, ``(N, num_actors)``.

        Rows belonging to the same actor are reduced with max (or mean) depending on
        ``use_max_over_samples``.
        """
        assert self._matrix is not None
        q = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
        q = q.to(self.device, non_blocking=True)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)
        raw = q @ self._matrix.T                       # (N, M)
        raw_np = raw.detach().float().cpu().numpy()

        num_actors = int(self._actor_ids.size)
        out = np.full((raw_np.shape[0], num_actors), -1.0, dtype=np.float32)
        for col, actor_id in enumerate(self._actor_ids):
            mask = self._owner_ids == actor_id
            block = raw_np[:, mask]
            out[:, col] = block.max(axis=1) if self.use_max_over_samples else block.mean(axis=1)
        return out

    def match_one(self, embedding: np.ndarray) -> MatchResult:
        """Convenience wrapper for a single query vector."""
        return self.match(embedding.reshape(1, -1))[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (safe for zero vectors)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


__all__ = ["MatchResult", "GalleryMatcher", "cosine_similarity", "UNKNOWN"]
