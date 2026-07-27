"""Dense signal extractor — emotion and identity at candidate-frame density.

Runs lightweight face analysis (emotion + identity) across all candidate frames,
producing a dense time-series independent of keyframe selection.  This decouples
fast-changing signals (emotion flickers, speaker changes) from the sparse VLM
keyframe grid.

Optional dependency: deepface.  If not installed, returns empty DenseSignals
with a logged warning.  The pipeline continues normally — dense signals are
additive metadata only.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from .types import DenseSignals

logger = logging.getLogger(__name__)

_DEEPFACE_AVAILABLE: bool | None = None  # None = not yet checked


def _check_deepface() -> bool:
    global _DEEPFACE_AVAILABLE
    if _DEEPFACE_AVAILABLE is None:
        try:
            import deepface  # noqa: F401
            _DEEPFACE_AVAILABLE = True
        except ImportError:
            _DEEPFACE_AVAILABLE = False
            logger.warning(
                "deepface not installed — dense signal extraction disabled. "
                "Install with: pip install deepface"
            )
    return _DEEPFACE_AVAILABLE


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


class DenseSignalExtractor:
    """Extract emotion and identity signals at candidate-frame density.

    Parameters
    ----------
    batch_size:
        Number of frames to analyze per DeepFace batch call.
    face_detector:
        OpenCV detector backend name — 'opencv' is fastest for dense sampling.
    identity_cluster_threshold:
        Cosine distance above which two face embeddings are considered
        different identities (new cluster created).
    """

    def __init__(
        self,
        batch_size: int = 32,
        face_detector: str = "opencv",
        identity_cluster_threshold: float = 0.4,
    ) -> None:
        self._batch_size = batch_size
        self._face_detector = face_detector
        self._cluster_threshold = identity_cluster_threshold

    def extract(
        self,
        frame_indices: list[int],
        frame_loader_fn: Callable[[int], np.ndarray],
    ) -> DenseSignals:
        """Extract dense emotion + identity signals for all candidate frames.

        Parameters
        ----------
        frame_indices:
            Sorted list of absolute frame indices (the same candidate frames
            used for DINOv2 embedding — typically ~2fps density).
        frame_loader_fn:
            Callable ``(idx: int) -> np.ndarray`` returning RGB uint8 (H, W, 3).

        Returns
        -------
        DenseSignals
            Per-frame emotion dicts and identity cluster labels.
            If deepface unavailable, returns empty DenseSignals with
            ``extractor_available=False``.
        """
        if not _check_deepface():
            return DenseSignals(
                frame_indices=frame_indices,
                emotions=[None] * len(frame_indices),
                identities=[None] * len(frame_indices),
                extraction_fps=0.0,
                extractor_available=False,
            )

        from deepface import DeepFace

        emotions: list[dict | None] = []
        face_embeddings: list[np.ndarray | None] = []

        for i in range(0, len(frame_indices), self._batch_size):
            batch_indices = frame_indices[i : i + self._batch_size]
            for idx in batch_indices:
                try:
                    frame = frame_loader_fn(idx)
                    result = DeepFace.analyze(
                        img_path=frame,
                        actions=["emotion"],
                        detector_backend=self._face_detector,
                        enforce_detection=False,
                        silent=True,
                    )
                    face_data = result[0] if isinstance(result, list) else result
                    emotions.append({
                        "dominant_emotion": face_data.get("dominant_emotion"),
                        "scores": face_data.get("emotion", {}),
                    })
                    # Extract face embedding for identity clustering
                    try:
                        emb_result = DeepFace.represent(
                            img_path=frame,
                            model_name="Facenet",
                            detector_backend=self._face_detector,
                            enforce_detection=False,
                        )
                        emb = np.array(emb_result[0]["embedding"], dtype=np.float32)
                        face_embeddings.append(emb)
                    except Exception:
                        face_embeddings.append(None)
                except Exception as exc:
                    logger.debug("Dense signal extraction failed for frame %d: %s", idx, exc)
                    emotions.append(None)
                    face_embeddings.append(None)

        identities = self._cluster_identities(face_embeddings)

        return DenseSignals(
            frame_indices=frame_indices,
            emotions=emotions,
            identities=identities,
            extraction_fps=0.0,  # same fps as candidate frames
            extractor_available=True,
        )

    def _cluster_identities(
        self, embeddings: list[np.ndarray | None]
    ) -> list[str | None]:
        """Assign cluster labels to face embeddings via greedy cosine distance.

        Each new face embedding is compared to all existing cluster centroids.
        If the closest centroid is within ``_cluster_threshold``, the frame is
        assigned to that cluster; otherwise a new cluster is created.

        No named recognition — output is 'speaker_0', 'speaker_1', etc.
        """
        cluster_centroids: list[np.ndarray] = []
        identities: list[str | None] = []

        for emb in embeddings:
            if emb is None:
                identities.append(None)
                continue

            if not cluster_centroids:
                cluster_centroids.append(emb.copy())
                identities.append("speaker_0")
                continue

            distances = [_cosine_distance(emb, c) for c in cluster_centroids]
            best_idx = int(np.argmin(distances))
            if distances[best_idx] < self._cluster_threshold:
                # Update centroid as running mean
                n = sum(1 for label in identities if label == f"speaker_{best_idx}")
                cluster_centroids[best_idx] = (
                    cluster_centroids[best_idx] * n + emb
                ) / (n + 1)
                identities.append(f"speaker_{best_idx}")
            else:
                new_id = len(cluster_centroids)
                cluster_centroids.append(emb.copy())
                identities.append(f"speaker_{new_id}")

        return identities
