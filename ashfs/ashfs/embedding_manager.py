"""Embedding extraction for ASH-FS pipeline.

DINOv2 (via torch.hub) is the primary embedding model.
SigLIP2 (via timm) is a stub stored for future retrieval use — it is NOT
consumed by the ASH-FS sampling pipeline itself.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ashfs.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_INPUT_SIZE = 224  # DINOv2 ViT-B/14 expects 224×224


class EmbeddingManager:
    """Manages DINOv2 (and optionally SigLIP2) embedding extraction.

    Parameters
    ----------
    cfg:
        Fully-loaded :class:`~ashfs.config.Config` instance.  Device,
        model name, batch size, and siglip2 flags are all read from it.
    cache_maxsize:
        Maximum number of per-frame embeddings to keep in the LRU cache.
        Defaults to 2048 which comfortably covers a typical 90-minute film
        at 2 fps candidate sampling.
    """

    def __init__(self, cfg: Config, cache_maxsize: int = 2048) -> None:
        self._cfg = cfg
        self._batch_size: int = cfg.embeddings.dinov2_batch_size
        self._siglip2_enabled: bool = cfg.embeddings.siglip2_enabled
        self._siglip2_batch_size: int = cfg.embeddings.siglip2_batch_size

        # Determine primary device; fall back to CPU if CUDA is unavailable.
        requested_device = cfg.embeddings.dinov2_device
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available — falling back to CPU."
            )
            requested_device = "cpu"
        self._device = torch.device(requested_device)

        # LRU cache: OrderedDict keyed by frame_index (int) → np.ndarray (768,)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_maxsize = cache_maxsize

        # Lazy-loaded models — not initialised until first use.
        self._dinov2: Optional[torch.nn.Module] = None
        self._siglip2: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dinov2(self) -> torch.nn.Module:
        """Load DINOv2 from torch.hub (once) and move to target device."""
        if self._dinov2 is not None:
            return self._dinov2

        logger.info(
            "Loading DINOv2 model %r from torch.hub on device %s …",
            self._cfg.embeddings.dinov2_model,
            self._device,
        )
        model = torch.hub.load(
            "facebookresearch/dinov2",
            self._cfg.embeddings.dinov2_model,
            pretrained=True,
        )
        model.eval()
        model = model.to(self._device)
        self._dinov2 = model
        return model

    @staticmethod
    def _preprocess_frames(frames: list[np.ndarray]) -> torch.Tensor:
        """Resize + normalise a list of HWC uint8 frames to a (N,3,224,224) tensor."""
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)

        # Image.Resampling.BILINEAR is the preferred form in Pillow ≥ 9.1;
        # fall back to the legacy attribute for older installs.
        _resample = getattr(Image, "Resampling", Image).BILINEAR

        tensors: list[torch.Tensor] = []
        for frame in frames:
            img = Image.fromarray(frame).resize(
                (_INPUT_SIZE, _INPUT_SIZE), _resample
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0      # (H, W, 3) in [0, 1]
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            tensors.append(t)

        batch = torch.cat(tensors, dim=0)   # (N, 3, H, W)
        batch = (batch - mean) / std
        return batch

    def _cache_put(self, frame_index: int, embedding: np.ndarray) -> None:
        """Insert into LRU cache, evicting the oldest entry when full."""
        if frame_index in self._cache:
            self._cache.move_to_end(frame_index)
        else:
            if len(self._cache) >= self._cache_maxsize:
                self._cache.popitem(last=False)   # evict LRU (front)
            self._cache[frame_index] = embedding

    def _run_dinov2_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        """Run DINOv2 inference on a list of frames, returning L2-normalised
        embeddings of shape (N, 768).  Falls back to CPU on CUDA OOM."""
        model = self._load_dinov2()
        batch = self._preprocess_frames(frames)

        try:
            batch = batch.to(self._device)
            with torch.no_grad():
                # FP16 autocast: 2x faster on RTX/A-series GPUs, identical results
                with torch.amp.autocast("cuda", enabled=(self._device.type == "cuda")):
                    feats: torch.Tensor = model(batch)      # (N, 768)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self._device.type != "cpu":
                logger.warning(
                    "CUDA OOM during DINOv2 inference — retrying on CPU."
                )
                torch.cuda.empty_cache()
                model = model.to(torch.device("cpu"))
                self._device = torch.device("cpu")
                self._dinov2 = model
                batch = batch.to(self._device)
                with torch.no_grad():
                    feats = model(batch)
            else:
                raise

        # L2 normalise
        feats = F.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_embeddings(self, frames: list[np.ndarray]) -> np.ndarray:
        """Batch-extract DINOv2 embeddings for *frames*.

        Parameters
        ----------
        frames:
            List of HWC uint8 numpy arrays (any spatial resolution).

        Returns
        -------
        np.ndarray
            Shape ``(N, 768)``, float32, L2-normalised.
        """
        if not frames:
            return np.empty((0, 768), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(frames), self._batch_size):
            chunk = frames[start : start + self._batch_size]
            embs = self._run_dinov2_batch(chunk)
            all_embeddings.append(embs)

        return np.concatenate(all_embeddings, axis=0)

    def extract_for_frames(
        self,
        frame_indices: list[int],
        frame_loader: Callable[[int], np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Return embeddings for *frame_indices*, using cache where possible.

        Parameters
        ----------
        frame_indices:
            Absolute frame indices within the video.
        frame_loader:
            Callable that accepts a single frame index (int) and returns a
            HWC uint8 numpy array.  Only called for cache-missing frames.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping of frame_index → L2-normalised embedding of shape (768,).
        """
        result: dict[int, np.ndarray] = {}

        # Split into cached and missing
        cached_indices: list[int] = []
        missing_indices: list[int] = []
        for idx in frame_indices:
            if idx in self._cache:
                cached_indices.append(idx)
            else:
                missing_indices.append(idx)

        # Satisfy cache hits — mark as recently used
        for idx in cached_indices:
            self._cache.move_to_end(idx)
            result[idx] = self._cache[idx]

        if not missing_indices:
            return result

        try:
            from tqdm import tqdm as _tqdm
            _bar: Any = _tqdm(
                total=len(missing_indices),
                desc="  DINOv2",
                unit="frame",
                leave=False,
                dynamic_ncols=True,
            )
        except ImportError:
            _bar = None

        # Split indices into batches
        batches = [
            missing_indices[s : s + self._batch_size]
            for s in range(0, len(missing_indices), self._batch_size)
        ]

        all_emb_parts: list[np.ndarray] = []

        # Prefetch: load next batch on CPU thread while GPU runs current batch.
        # Uses a single-worker ThreadPoolExecutor so frame loading overlaps GPU.
        def _load_batch(idxs: list[int]) -> list[np.ndarray]:
            return [frame_loader(i) for i in idxs]

        with ThreadPoolExecutor(max_workers=1) as pool:
            # Kick off load of first batch immediately
            future = pool.submit(_load_batch, batches[0]) if batches else None

            for i, batch_indices in enumerate(batches):
                frames = future.result()  # wait for current batch frames

                # Prefetch next batch in background while GPU runs
                if i + 1 < len(batches):
                    future = pool.submit(_load_batch, batches[i + 1])

                embs = self._run_dinov2_batch(frames)
                all_emb_parts.append(embs)

                if _bar is not None:
                    _bar.update(len(batch_indices))

        if _bar is not None:
            _bar.close()

        all_embs = np.concatenate(all_emb_parts, axis=0) if all_emb_parts else np.empty((0, 768), dtype=np.float32)

        for i, idx in enumerate(missing_indices):
            emb = all_embs[i]
            self._cache_put(idx, emb)
            result[idx] = emb

        return result

    # ------------------------------------------------------------------
    # SigLIP2 stub
    # stored for future retrieval use, not consumed by ASH-FS itself
    # ------------------------------------------------------------------

    def extract_siglip2(self, frames: list[np.ndarray]) -> Optional[np.ndarray]:
        """Extract SigLIP2 embeddings for *frames*.

        .. note::
            stored for future retrieval use, not consumed by ASH-FS itself

        If ``siglip2_enabled`` is ``False`` in config this method returns
        ``None`` immediately without loading any model.

        Parameters
        ----------
        frames:
            List of HWC uint8 numpy arrays.

        Returns
        -------
        np.ndarray or None
            Shape ``(N, D)`` float32 L2-normalised, or ``None`` when disabled.
        """
        if not self._siglip2_enabled:
            return None

        try:
            import timm  # type: ignore[import]
            from timm.data import resolve_data_config, create_transform  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "timm is required for SigLIP2 extraction. "
                "Install it with: pip install timm"
            ) from exc

        if self._siglip2 is None:
            logger.info("Loading SigLIP2 model via timm …")
            model = timm.create_model(
                "hf-hub:timm/ViT-SO400M-14-SigLIP2-512",
                pretrained=True,
                num_classes=0,    # remove classifier head; return pooled features
            )
            model.eval()
            model = model.to(self._device)
            self._siglip2 = model

        # Build timm-specific transform for SigLIP2
        data_cfg = resolve_data_config(model=self._siglip2)
        transform = create_transform(**data_cfg)

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(frames), self._siglip2_batch_size):
            chunk = frames[start : start + self._siglip2_batch_size]
            pil_imgs = [Image.fromarray(f) for f in chunk]
            tensors = torch.stack([transform(img) for img in pil_imgs])  # type: ignore[arg-type]

            try:
                tensors = tensors.to(self._device)
                with torch.no_grad():
                    feats: torch.Tensor = self._siglip2(tensors)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self._device.type != "cpu":
                    logger.warning(
                        "CUDA OOM during SigLIP2 inference — retrying on CPU."
                    )
                    torch.cuda.empty_cache()
                    self._siglip2 = self._siglip2.to(torch.device("cpu"))
                    self._device = torch.device("cpu")
                    tensors = tensors.to(self._device)
                    with torch.no_grad():
                        feats = self._siglip2(tensors)
                else:
                    raise

            feats = F.normalize(feats, p=2, dim=1)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 0), dtype=np.float32)

    def get_embeddings(
        self,
        frame_indices: list[int],
        frame_loader: Callable[[int], np.ndarray],
    ) -> np.ndarray:
        """Extract embeddings for *frame_indices* and return as ordered (N, D) array.

        Convenience wrapper around :meth:`extract_for_frames` that returns a
        numpy array in the same order as *frame_indices* rather than a dict.
        Used by shot_segmenter for soft-boundary detection.
        """
        if not frame_indices:
            return np.empty((0, 768), dtype=np.float32)
        cache = self.extract_for_frames(frame_indices, frame_loader)
        # Stack in exactly the same order as frame_indices so callers that
        # pair this array with frame_indices by position get correct alignment.
        # extract_for_frames guarantees every requested index is in the cache,
        # but guard with a fallback zero-vector to avoid a hard crash.
        dim = 768
        rows: list[np.ndarray] = []
        for idx in frame_indices:
            if idx in cache:
                rows.append(cache[idx])
            else:
                logger.warning("Embedding missing for frame %d — substituting zeros.", idx)
                rows.append(np.zeros(dim, dtype=np.float32))
        return np.stack(rows, axis=0)
