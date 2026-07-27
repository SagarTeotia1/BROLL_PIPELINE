"""Unit tests for ashfs.embedding_manager.

All DINOv2 model loading (torch.hub.load) is mocked — no GPU, no network,
no model weights required.  Tests verify LRU cache behaviour, extract_for_frames
correctness, siglip2 disabled path, device fallback, and preprocessing shape.
"""
from __future__ import annotations

import sys
import types
from collections import OrderedDict
from unittest.mock import patch, MagicMock, call

import numpy as np
import pytest
import torch

from ashfs.config import Config
from ashfs.embedding_manager import EmbeddingManager
from tests.conftest import make_l2_normed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_dinov2_model(output_dim: int = 768):
    """Return a callable mock that simulates DINOv2 forward pass.

    When called with a tensor of shape (N, 3, H, W) it returns (N, output_dim).
    """
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model

    def _forward(batch: torch.Tensor) -> torch.Tensor:
        n = batch.shape[0]
        # Return random unit-normalised vectors so L2 norm ≈ 1 after F.normalize
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((n, output_dim)).astype(np.float32)
        return torch.from_numpy(arr)

    model.side_effect = _forward
    model.__call__ = _forward
    return model


def _make_rgb_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random uint8 HWC RGB array."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _build_manager(cfg: Config, cache_maxsize: int = 2048) -> EmbeddingManager:
    """Build an EmbeddingManager with torch.hub.load patched to avoid model download."""
    fake_model = _fake_dinov2_model()
    with patch("torch.hub.load", return_value=fake_model):
        manager = EmbeddingManager(cfg, cache_maxsize=cache_maxsize)
    # Pre-inject the fake model so _load_dinov2 returns it immediately
    manager._dinov2 = fake_model
    return manager


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def manager(cfg) -> EmbeddingManager:
    """EmbeddingManager with a mocked DINOv2 model."""
    return _build_manager(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLRUCache:

    def test_lru_cache_evicts_oldest(self, cfg):
        """Filling the cache beyond maxsize must evict the oldest (LRU) entry."""
        manager = _build_manager(cfg, cache_maxsize=3)

        # Manually populate the cache with 3 entries
        dummy_emb = np.ones(768, dtype=np.float32)
        manager._cache_put(0, dummy_emb.copy())
        manager._cache_put(1, dummy_emb.copy())
        manager._cache_put(2, dummy_emb.copy())
        assert len(manager._cache) == 3

        # Adding a 4th entry must evict key 0 (oldest)
        manager._cache_put(3, dummy_emb.copy())
        assert len(manager._cache) == 3
        assert 0 not in manager._cache, "LRU entry (key=0) should have been evicted"
        assert 3 in manager._cache

    def test_cache_hit_avoids_reextraction(self, cfg):
        """A second call for the same frame_index must not invoke frame_loader again."""
        manager = _build_manager(cfg)

        loader_calls = []

        def fake_loader(idx: int) -> np.ndarray:
            loader_calls.append(idx)
            return _make_rgb_frame()

        frame_indices = [5]
        # First call — cache miss, loader called
        manager.extract_for_frames(frame_indices, fake_loader)
        assert loader_calls == [5]

        # Second call — cache hit, loader NOT called again
        manager.extract_for_frames(frame_indices, fake_loader)
        assert loader_calls == [5], "frame_loader must not be called on cache hit"


class TestExtractForFrames:

    def test_extract_for_frames_returns_correct_shape(self, cfg):
        """extract_for_frames must return a dict with the correct frame keys and (768,) values."""
        manager = _build_manager(cfg)
        frame_indices = [0, 1, 2]
        loader = lambda idx: _make_rgb_frame()

        result = manager.extract_for_frames(frame_indices, loader)

        assert set(result.keys()) == {0, 1, 2}
        for idx, emb in result.items():
            assert emb.shape == (768,), f"Embedding for frame {idx} has wrong shape: {emb.shape}"
            assert emb.dtype == np.float32

    def test_get_embeddings_returns_ordered_array(self, cfg):
        """get_embeddings must return rows in the same order as frame_indices."""
        manager = _build_manager(cfg)
        frame_indices = [10, 5, 20]
        loader = lambda idx: _make_rgb_frame()

        # Pre-populate cache with deterministic embeddings keyed by index
        for idx in frame_indices:
            emb = np.full(768, fill_value=float(idx), dtype=np.float32)
            emb /= np.linalg.norm(emb)
            manager._cache[idx] = emb

        result_array = manager.get_embeddings(frame_indices, loader)

        assert result_array.shape == (3, 768)
        # Verify each row corresponds to the right frame
        for row, idx in zip(result_array, frame_indices):
            np.testing.assert_allclose(row, manager._cache[idx], rtol=1e-5)


class TestSigLIP2:

    def test_siglip2_returns_none_when_disabled(self, cfg):
        """extract_siglip2 must return None immediately when siglip2_enabled=False in config."""
        # config.yaml sets siglip2_enabled: false — assert this is the case
        assert cfg.embeddings.siglip2_enabled is False, (
            "This test requires siglip2_enabled=false in config.yaml"
        )
        manager = _build_manager(cfg)
        frames = [_make_rgb_frame()]
        result = manager.extract_siglip2(frames)
        assert result is None


class TestDeviceFallback:

    def test_cpu_fallback_on_cuda_unavailable(self, cfg):
        """When CUDA is not available but requested, EmbeddingManager must silently fall back to CPU."""
        import ashfs.embedding_manager as em_module

        # Patch torch.cuda.is_available to return False
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.hub.load", return_value=_fake_dinov2_model()):
                # Config requests 'cuda' but CUDA is unavailable
                manager = EmbeddingManager(cfg, cache_maxsize=16)

        assert manager._device == torch.device("cpu"), (
            f"Expected CPU fallback, got {manager._device}"
        )


class TestPreprocessFrames:

    def test_preprocess_frames_output_shape(self):
        """_preprocess_frames must convert 2 HWC frames into a (2, 3, 224, 224) tensor."""
        frames = [_make_rgb_frame(128, 128), _make_rgb_frame(64, 64)]
        result = EmbeddingManager._preprocess_frames(frames)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3, 224, 224), (
            f"Expected shape (2, 3, 224, 224), got {tuple(result.shape)}"
        )
        assert result.dtype == torch.float32

    def test_preprocess_frames_single_frame(self):
        """_preprocess_frames must handle a list with exactly one frame."""
        frames = [_make_rgb_frame()]
        result = EmbeddingManager._preprocess_frames(frames)
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_frames_values_normalised(self):
        """Output pixel values must be normalised (not raw [0, 1] floats — ImageNet mean/std applied)."""
        # A pure white frame (all 255) should have normalised values not equal to 1.0
        frame = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = EmbeddingManager._preprocess_frames([frame])
        # After (1.0 - mean) / std, values won't all be 1.0
        assert not torch.allclose(result, torch.ones_like(result))


# ---------------------------------------------------------------------------
# Additional missing tests
# ---------------------------------------------------------------------------

class TestExtractEmbeddingsEdgeCases:

    def test_extract_embeddings_empty_list_returns_empty_array(self, cfg):
        """extract_embeddings([]) must return an (0, 768) float32 array without
        calling the DINOv2 model at all."""
        manager = _build_manager(cfg)
        result = manager.extract_embeddings([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 768)
        assert result.dtype == np.float32


class TestGetEmbeddingsEdgeCases:

    def test_get_embeddings_empty_frame_indices_returns_empty_array(self, cfg):
        """get_embeddings with an empty frame_indices list must return an (0, 768)
        array without invoking frame_loader."""
        manager = _build_manager(cfg)

        loader_calls = []

        def fake_loader(idx):
            loader_calls.append(idx)
            return _make_rgb_frame()

        result = manager.get_embeddings([], fake_loader)
        assert result.shape == (0, 768)
        assert result.dtype == np.float32
        assert loader_calls == [], "frame_loader must not be called for empty indices"


class TestCacheEvictionMaxsizeOne:

    def test_cache_eviction_maxsize_1(self, cfg):
        """With cache_maxsize=1, inserting a second entry must evict the first."""
        manager = _build_manager(cfg, cache_maxsize=1)

        dummy = np.ones(768, dtype=np.float32)
        manager._cache_put(0, dummy.copy())
        assert 0 in manager._cache

        # Insert frame 1 — must evict frame 0
        manager._cache_put(1, dummy.copy())
        assert len(manager._cache) == 1
        assert 0 not in manager._cache, "Frame 0 should have been evicted (LRU)"
        assert 1 in manager._cache
