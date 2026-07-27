"""Shared pytest fixtures: synthetic images with known grading properties."""

from __future__ import annotations

import numpy as np
import pytest

from color_analyzer.analyzer.utils import Backend, ImageContext


@pytest.fixture(scope="session")
def backend() -> Backend:
    # Tests run on whatever backend is available; force CPU for determinism if
    # no GPU is present (the default), but honour a GPU if the host has one.
    return Backend()


def _ctx(img: np.ndarray) -> ImageContext:
    return ImageContext(img.astype(np.float32))


@pytest.fixture
def gray_image() -> np.ndarray:
    """A flat neutral 50% grey image (no cast, no colour)."""
    return np.full((64, 64, 3), 0.5, dtype=np.float32)


@pytest.fixture
def gray_ctx(gray_image) -> ImageContext:
    return _ctx(gray_image)


@pytest.fixture
def teal_orange_image() -> np.ndarray:
    """Left half warm-orange, right half cool-teal (complementary)."""
    img = np.zeros((128, 128, 3), np.float32)
    img[:, :64] = (0.9, 0.5, 0.2)   # orange
    img[:, 64:] = (0.1, 0.45, 0.5)  # teal
    return img


@pytest.fixture
def teal_orange_ctx(teal_orange_image) -> ImageContext:
    return _ctx(teal_orange_image)


@pytest.fixture
def dark_image() -> np.ndarray:
    return np.full((64, 64, 3), 0.08, dtype=np.float32)


@pytest.fixture
def bright_image() -> np.ndarray:
    return np.full((64, 64, 3), 0.92, dtype=np.float32)


@pytest.fixture
def split_tone_image() -> np.ndarray:
    """Luminance ramp with teal shadows and orange highlights."""
    w = 256
    lum = np.linspace(0, 1, w)[None, :].repeat(128, 0)
    # R/G rise strongly with luminance (warm, bright highlights >0.66),
    # B falls (teal shadows <0.33) so both tonal zones are populated.
    img = np.stack([lum, 0.3 + lum * 0.5, 0.5 - lum * 0.45], -1).astype(np.float32)
    return np.clip(img, 0, 1)


@pytest.fixture
def split_tone_ctx(split_tone_image) -> ImageContext:
    return _ctx(split_tone_image)
