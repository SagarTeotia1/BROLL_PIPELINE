"""Correctness tests for the colour-space conversions."""

from __future__ import annotations

import numpy as np

from color_analyzer.analyzer.utils import (
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_luminance,
    rgb_to_ycrcb,
)


def _one(rgb):
    return np.asarray(rgb, np.float32).reshape(1, 1, 3)


def test_white_pixel_hsv_and_lab():
    white = _one((1.0, 1.0, 1.0))
    hsv = rgb_to_hsv(np, white).reshape(3)
    assert hsv[1] < 1e-4          # saturation ~ 0
    assert abs(hsv[2] - 1.0) < 1e-4  # value == 1
    lab = rgb_to_lab(np, white).reshape(3)
    assert abs(lab[0] - 100.0) < 0.5  # L* ~ 100 for white


def test_black_pixel_lab():
    lab = rgb_to_lab(np, _one((0.0, 0.0, 0.0))).reshape(3)
    assert abs(lab[0]) < 0.5  # L* ~ 0


def test_pure_red_hue():
    hsv = rgb_to_hsv(np, _one((1.0, 0.0, 0.0))).reshape(3)
    assert hsv[0] < 1.0 or hsv[0] > 359.0  # hue ~ 0 deg
    assert abs(hsv[1] - 1.0) < 1e-4        # fully saturated


def test_pure_green_and_blue_hue():
    g = rgb_to_hsv(np, _one((0.0, 1.0, 0.0))).reshape(3)
    b = rgb_to_hsv(np, _one((0.0, 0.0, 1.0))).reshape(3)
    assert abs(g[0] - 120.0) < 1.0
    assert abs(b[0] - 240.0) < 1.0


def test_gray_has_neutral_lab_and_chroma():
    lab = rgb_to_lab(np, _one((0.5, 0.5, 0.5))).reshape(3)
    assert abs(lab[1]) < 1.0  # a* ~ 0
    assert abs(lab[2]) < 1.0  # b* ~ 0


def test_gray_ycrcb_neutral_chroma():
    ycc = rgb_to_ycrcb(np, _one((0.5, 0.5, 0.5))).reshape(3)
    assert abs(ycc[1] - 0.5) < 1e-4  # Cr neutral
    assert abs(ycc[2] - 0.5) < 1e-4  # Cb neutral


def _lum(rgb) -> float:
    return float(rgb_to_luminance(np, _one(rgb)).reshape(-1)[0])


def test_luminance_monotonic():
    assert _lum((0.9, 0.9, 0.9)) > _lum((0.1, 0.1, 0.1))
    # green contributes most to luminance under Rec.709 weights.
    assert _lum((0.0, 1.0, 0.0)) > _lum((1.0, 0.0, 0.0)) > _lum((0.0, 0.0, 1.0))
