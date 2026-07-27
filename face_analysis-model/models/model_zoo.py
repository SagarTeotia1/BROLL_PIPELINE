"""Model registry: declares where every ONNX graph comes from and fetches it once.

Weights live in ``models/weights`` and are downloaded on first use:

* **SCRFD-10G-BNKPS** (``det_10g.onnx``) - face detector with 5 landmarks.
* **ArcFace R50 / w600k** (``w600k_r50.onnx``) - 512-d recognition embedding.
* **HSEmotion EfficientNet-B0 / B2** - 8-class AffectNet emotion classifier.

Each spec lists several mirrors; the first that responds wins. A model can be
provided either as a direct ``.onnx`` URL or as a member of a zip archive
(InsightFace ships ``buffalo_l.zip``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from utils.downloader import download_first_available, extract_member
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to obtain and run one ONNX graph."""

    key: str
    filename: str
    task: str                              # "detection" | "recognition" | "emotion"
    input_size: int
    urls: Sequence[str] = field(default_factory=tuple)
    archive_urls: Sequence[str] = field(default_factory=tuple)
    archive_member: Optional[str] = None
    sha256: Optional[str] = None
    description: str = ""
    # Preprocessing contract, consumed by the wrappers.
    mean: Sequence[float] = (127.5, 127.5, 127.5)
    std: Sequence[float] = (128.0, 128.0, 128.0)
    bgr_to_rgb: bool = True


_IMAGENET_MEAN_255 = (0.485 * 255, 0.456 * 255, 0.406 * 255)
_IMAGENET_STD_255 = (0.229 * 255, 0.224 * 255, 0.225 * 255)

_BUFFALO_ZIP = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "https://huggingface.co/spaces/lianzhou/insightface/resolve/main/buffalo_l.zip",
)

REGISTRY: Dict[str, ModelSpec] = {
    "scrfd_10g_bnkps": ModelSpec(
        key="scrfd_10g_bnkps",
        filename="det_10g.onnx",
        task="detection",
        input_size=640,
        urls=(
            "https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx",
            "https://huggingface.co/nickgardner/arcface/resolve/main/det_10g.onnx",
        ),
        archive_urls=_BUFFALO_ZIP,
        archive_member="det_10g.onnx",
        description="SCRFD 10GF with 5-point landmarks (InsightFace buffalo_l)",
        mean=(127.5, 127.5, 127.5),
        std=(128.0, 128.0, 128.0),
    ),
    "scrfd_2.5g_bnkps": ModelSpec(
        key="scrfd_2.5g_bnkps",
        filename="det_2.5g.onnx",
        task="detection",
        input_size=640,
        urls=(
            "https://huggingface.co/immich-app/buffalo_s/resolve/main/detection/model.onnx",
        ),
        archive_urls=(
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        ),
        archive_member="det_500m.onnx",
        description="Lighter SCRFD for low-power fallback",
    ),
    "arcface_r50": ModelSpec(
        key="arcface_r50",
        filename="w600k_r50.onnx",
        task="recognition",
        input_size=112,
        urls=(
            "https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx",
            "https://huggingface.co/nickgardner/arcface/resolve/main/w600k_r50.onnx",
        ),
        archive_urls=_BUFFALO_ZIP,
        archive_member="w600k_r50.onnx",
        description="ArcFace ResNet50 trained on WebFace600K, 512-d embedding",
        mean=(127.5, 127.5, 127.5),
        std=(127.5, 127.5, 127.5),
    ),
    "hsemotion_enet_b0_8": ModelSpec(
        key="hsemotion_enet_b0_8",
        filename="enet_b0_8_best_vgaf.onnx",
        task="emotion",
        input_size=224,
        urls=(
            "https://github.com/av-savchenko/face-emotion-recognition/raw/main/"
            "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx",
            "https://raw.githubusercontent.com/av-savchenko/face-emotion-recognition/"
            "main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx",
        ),
        description="HSEmotion EfficientNet-B0, 8 AffectNet classes",
        mean=_IMAGENET_MEAN_255,
        std=_IMAGENET_STD_255,
    ),
    "hsemotion_enet_b2_8": ModelSpec(
        key="hsemotion_enet_b2_8",
        filename="enet_b2_8.onnx",
        task="emotion",
        input_size=260,
        urls=(
            "https://github.com/av-savchenko/face-emotion-recognition/raw/main/"
            "models/affectnet_emotions/onnx/enet_b2_8.onnx",
            "https://raw.githubusercontent.com/av-savchenko/face-emotion-recognition/"
            "main/models/affectnet_emotions/onnx/enet_b2_8.onnx",
        ),
        description="HSEmotion EfficientNet-B2, higher accuracy, ~2x cost",
        mean=_IMAGENET_MEAN_255,
        std=_IMAGENET_STD_255,
    ),
}


def get_spec(key: str) -> ModelSpec:
    """Look up a model spec, with a helpful error listing valid keys."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown model '{key}'. Available: {sorted(REGISTRY)}"
        ) from None


def ensure_model(key: str, weights_dir: Path, progress=None) -> Path:
    """Return the local path to a model, downloading it if necessary.

    Direct URLs are tried first; if all of them fail and the spec declares an
    archive, the archive is downloaded to ``weights_dir/_archives`` and the member
    is extracted. The archive is kept so that the *other* model from the same pack
    (detector + recogniser share ``buffalo_l.zip``) is a local extraction away.
    """
    spec = get_spec(key)
    weights_dir = Path(weights_dir)
    target = weights_dir / spec.filename
    if target.exists() and target.stat().st_size > 0:
        return target

    weights_dir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    if spec.urls:
        try:
            return download_first_available(list(spec.urls), target, spec.sha256, progress)
        except RuntimeError as exc:
            errors.append(str(exc))
            log.warning("Direct download failed for %s, trying archive mirror", key)

    if spec.archive_urls and spec.archive_member:
        archive_dir = weights_dir / "_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_name = Path(spec.archive_urls[0]).name
        archive_path = archive_dir / archive_name
        if not archive_path.exists():
            try:
                download_first_available(list(spec.archive_urls), archive_path, None, progress)
            except RuntimeError as exc:
                errors.append(str(exc))
                raise RuntimeError(
                    f"could not obtain model '{key}'.\n" + "\n".join(errors)
                    + f"\nManual fix: place '{spec.filename}' in {weights_dir}"
                ) from exc
        return extract_member(archive_path, spec.archive_member, target)

    raise RuntimeError(
        f"could not obtain model '{key}'.\n" + "\n".join(errors)
        + f"\nManual fix: place '{spec.filename}' in {weights_dir}"
    )


def list_models(task: Optional[str] = None) -> List[ModelSpec]:
    """All registered specs, optionally filtered by task."""
    specs = list(REGISTRY.values())
    if task:
        specs = [s for s in specs if s.task == task]
    return sorted(specs, key=lambda s: s.key)


__all__ = ["ModelSpec", "REGISTRY", "get_spec", "ensure_model", "list_models"]
