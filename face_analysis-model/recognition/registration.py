"""Phase 1 - cast registration.

Turns a folder of reference photos into one averaged ArcFace centroid per actor:

1. read the image (unicode-safe on Windows),
2. detect faces, keep the dominant one (largest x confidence),
3. align to the 112x112 canonical crop,
4. run the quality gate - a blurry or extreme-profile reference photo poisons the
   centroid far more than it helps, so it is rejected with an explicit reason,
5. embed everything in **one batch**,
6. store the vectors and recompute the quality-weighted centroid.

The result object reports exactly which images were used and which were skipped, so
the GUI can show the user why a photo did not count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from configs.config import AppConfig
from detector.base import Detection
from detector.face_quality import FaceQualityEstimator
from detector.scrfd import SCRFDDetector
from recognition.arcface import ArcFaceEmbedder
from recognition.cast_database import CastDatabase
from utils.image_ops import align_face, resize_long_side
from utils.logging_utils import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ProgressFn = Callable[[int, int, str], None]


@dataclass
class ImageOutcome:
    """What happened to one reference photo."""

    path: str
    accepted: bool
    reason: str = ""
    quality: float = 0.0
    det_score: float = 0.0

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "accepted": self.accepted,
            "reason": self.reason,
            "quality": round(self.quality, 3),
            "det_score": round(self.det_score, 3),
        }


@dataclass
class RegistrationResult:
    """Summary of a registration run for one actor."""

    actor_name: str
    actor_id: int
    accepted: int
    rejected: int
    outcomes: List[ImageOutcome] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    thumbnails: List[np.ndarray] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.accepted > 0

    def summary(self) -> str:
        return (
            f"{self.actor_name}: {self.accepted} accepted, {self.rejected} rejected"
            + (
                ""
                if self.ok
                else " - no usable reference photo, actor NOT enrolled"
            )
        )


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """``cv2.imread`` replacement that survives non-ASCII paths on Windows."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image
    except Exception as exc:  # pragma: no cover - filesystem dependent
        log.warning("Could not read %s: %s", path, exc)
        return None


def collect_images(paths: Sequence[str | Path]) -> List[Path]:
    """Expand a mixed list of files and folders into a sorted list of image files."""
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(
                sorted(
                    q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
        elif p.suffix.lower() in IMAGE_EXTENSIONS and p.exists():
            found.append(p)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: List[Path] = []
    for p in found:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


class CastRegistrar:
    """Builds and updates the cast database.

    Args:
        config: application config.
        detector: shared SCRFD instance.
        embedder: shared ArcFace instance.
        database: destination store.
    """

    def __init__(
        self,
        config: AppConfig,
        detector: SCRFDDetector,
        embedder: ArcFaceEmbedder,
        database: CastDatabase,
    ) -> None:
        self.cfg = config
        self.detector = detector
        self.embedder = embedder
        self.db = database
        self.quality = FaceQualityEstimator(config.quality)

    # -- public API ---------------------------------------------------------
    def register(
        self,
        actor_name: str,
        image_paths: Sequence[str | Path],
        progress: Optional[ProgressFn] = None,
        keep_thumbnails: bool = True,
    ) -> RegistrationResult:
        """Enrol (or extend) one actor from reference photos."""
        images = collect_images(image_paths)
        if not images:
            return RegistrationResult(actor_name, -1, 0, 0, [])

        outcomes: List[ImageOutcome] = []
        crops: List[np.ndarray] = []
        weights: List[float] = []
        sources: List[str] = []
        thumbnails: List[np.ndarray] = []
        # Rejected images that still yielded a face, kept for the fallback below.
        salvageable: List[Tuple[str, np.ndarray, float]] = []

        total = len(images)
        for idx, path in enumerate(images):
            if progress is not None:
                progress(idx, total, path.name)
            outcome, crop, weight = self._process_image(path)
            outcomes.append(outcome)
            if outcome.accepted and crop is not None:
                crops.append(crop)
                weights.append(weight)
                sources.append(path.name)
                if keep_thumbnails:
                    thumbnails.append(crop.copy())
            elif crop is not None:
                salvageable.append((str(path), crop, weight))

        # Fallback: reference photos are chosen deliberately by the user, so refusing
        # every one of them would leave the actor un-enrollable. If the gate rejected
        # everything, keep the best-scoring detections instead of failing - with an
        # explicit warning, and only for images where a face was actually found.
        if not crops and salvageable:
            salvageable.sort(key=lambda item: item[2], reverse=True)
            keep = salvageable[: max(1, min(3, len(salvageable)))]
            log.warning(
                "Every reference photo for '%s' failed the quality gate; keeping the "
                "%d best (%s). Recognition accuracy will suffer - prefer sharp, "
                "near-frontal photos.",
                actor_name, len(keep), ", ".join(Path(p).name for p, _, _ in keep),
            )
            for path_str, crop, score in keep:
                crops.append(crop)
                weights.append(max(score, 0.05))
                sources.append(Path(path_str).name)
                if keep_thumbnails:
                    thumbnails.append(crop.copy())
                for outcome in outcomes:
                    if outcome.path == path_str:
                        outcome.accepted = True
                        outcome.reason = f"salvaged ({outcome.reason})"

        accepted = len(crops)
        rejected = total - accepted
        if accepted == 0:
            log.warning("No usable reference photo for '%s'", actor_name)
            return RegistrationResult(actor_name, -1, 0, rejected, outcomes)

        # One batched ArcFace pass for every accepted crop.
        embeddings = self.embedder.embed_aligned(crops)
        actor_id = self.db.add_actor(actor_name)
        self.db.add_embeddings(actor_id, embeddings, sources, weights)
        centroid = self.db.recompute_centroid(actor_id)
        self.db.export_pickle()

        if progress is not None:
            progress(total, total, "done")
        result = RegistrationResult(
            actor_name=actor_name,
            actor_id=actor_id,
            accepted=accepted,
            rejected=rejected,
            outcomes=outcomes,
            centroid=centroid,
            thumbnails=thumbnails,
        )
        log.info(result.summary())
        return result

    def update_actor(
        self,
        actor_name: str,
        image_paths: Sequence[str | Path],
        progress: Optional[ProgressFn] = None,
    ) -> RegistrationResult:
        """Add more photos to an already registered actor (alias of :meth:`register`)."""
        return self.register(actor_name, image_paths, progress)

    def remove_actor(self, actor_name: str) -> bool:
        """Delete an actor by name; returns ``True`` when something was removed."""
        actor_id = self.db.get_actor_id(actor_name)
        if actor_id is None:
            return False
        self.db.delete_actor(actor_id)
        self.db.export_pickle()
        return True

    # -- internals ----------------------------------------------------------
    def _process_image(
        self, path: Path
    ) -> Tuple[ImageOutcome, Optional[np.ndarray], float]:
        image = imread_unicode(path)
        if image is None:
            return ImageOutcome(str(path), False, "unreadable"), None, 0.0

        # Reference photos are often huge; detection does not need 4K.
        image, _ = resize_long_side(image, max(self.cfg.video.analysis_long_side, 1280))
        detections = self.detector.detect(image)
        if not detections:
            return ImageOutcome(str(path), False, "no face detected"), None, 0.0

        best = self._dominant_face(detections)
        if best.landmarks is None:
            return ImageOutcome(str(path), False, "no landmarks"), None, 0.0

        crop = align_face(image, best.landmarks, self.cfg.recognition.input_size)
        report = self.quality.evaluate(best, crop)
        if not report.passed:
            # The crop is still returned so ``register`` can salvage the best rejects
            # when nothing at all passes the gate.
            return (
                ImageOutcome(str(path), False, report.reason, report.score, best.score),
                crop,
                report.score * 0.7 + best.score * 0.3,
            )
        return (
            ImageOutcome(str(path), True, "", report.score, best.score),
            crop,
            report.score,
        )

    @staticmethod
    def _dominant_face(detections: Sequence[Detection]) -> Detection:
        """Pick the subject of a portrait: biggest face weighted by confidence."""
        return max(detections, key=lambda d: d.area * (0.5 + 0.5 * d.score))


__all__ = [
    "CastRegistrar",
    "RegistrationResult",
    "ImageOutcome",
    "collect_images",
    "imread_unicode",
]
