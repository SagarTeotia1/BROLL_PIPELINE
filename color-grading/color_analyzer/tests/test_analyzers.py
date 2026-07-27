"""Behavioural tests for individual analyzers."""

from __future__ import annotations

import numpy as np

from color_analyzer.analyzer.cinematic import CinematicAnalyzer
from color_analyzer.analyzer.contrast import ContrastAnalyzer
from color_analyzer.analyzer.dominant_colors import DominantColorAnalyzer
from color_analyzer.analyzer.exposure import ExposureAnalyzer
from color_analyzer.analyzer.harmony import HarmonyAnalyzer
from color_analyzer.analyzer.lab import LabAnalyzer
from color_analyzer.analyzer.rgb import RGBAnalyzer
from color_analyzer.analyzer.split_toning import SplitToningAnalyzer
from color_analyzer.analyzer.utils import ImageContext
from color_analyzer.analyzer.white_balance import WhiteBalanceAnalyzer


def _ctx(img):
    return ImageContext(np.asarray(img, np.float32))


def test_rgb_ratios_on_warm_image():
    img = np.zeros((32, 32, 3), np.float32)
    img[:] = (0.8, 0.5, 0.2)
    f = RGBAnalyzer().analyze(_ctx(img))
    assert f.r_over_b > 1.0            # red dominates blue -> warm
    assert f.red.mean > f.blue.mean


def test_exposure_dark_vs_bright(dark_image, bright_image):
    dark = ExposureAnalyzer().analyze(_ctx(dark_image))
    bright = ExposureAnalyzer().analyze(_ctx(bright_image))
    assert dark.shadow_percentage > 0.9
    assert bright.highlight_percentage > 0.9
    assert bright.mean_brightness > dark.mean_brightness


def test_lab_warmth_sign():
    warm = LabAnalyzer().analyze(_ctx(np.full((16, 16, 3), 0.0, np.float32) + (0.8, 0.5, 0.2)))
    cool = LabAnalyzer().analyze(_ctx(np.full((16, 16, 3), 0.0, np.float32) + (0.2, 0.4, 0.8)))
    assert warm.warmth_score > cool.warmth_score


def test_white_balance_neutral_on_gray(gray_ctx):
    f = WhiteBalanceAnalyzer().analyze(gray_ctx)
    assert f.neutrality_score > 0.95
    assert abs(f.gain_r - 1.0) < 0.05
    assert abs(f.gain_b - 1.0) < 0.05


def test_white_balance_detects_red_cast():
    img = np.zeros((16, 16, 3), np.float32)
    img[:] = (0.8, 0.4, 0.4)
    f = WhiteBalanceAnalyzer().analyze(_ctx(img))
    assert f.red_cast > 0.1
    assert f.neutrality_score < 0.9


def test_harmony_complementary(teal_orange_ctx):
    f = HarmonyAnalyzer().analyze(teal_orange_ctx)
    assert f.confidences["complementary"] >= f.confidences["triadic"]
    assert f.best_confidence > 0.2


def test_harmony_monochromatic_on_gray(gray_ctx):
    f = HarmonyAnalyzer().analyze(gray_ctx)
    assert f.best_match == "monochromatic"


def test_split_toning_separation(split_tone_ctx):
    f = SplitToningAnalyzer().analyze(split_tone_ctx)
    # teal shadows vs orange highlights -> large hue separation.
    assert f.shadow_highlight_hue_separation > 90.0
    assert f.split_tone_confidence > 0.3


def test_contrast_orders_flat_vs_ramp(gray_ctx, split_tone_ctx):
    flat = ContrastAnalyzer().analyze(gray_ctx)
    ramp = ContrastAnalyzer().analyze(split_tone_ctx)
    assert ramp.rms_contrast > flat.rms_contrast
    assert ramp.global_contrast > flat.global_contrast


def test_dominant_colors_counts_and_coverage(teal_orange_ctx):
    f = DominantColorAnalyzer(k=4).analyze(teal_orange_ctx)
    assert 1 <= len(f.colors) <= 4
    total = sum(c.percentage for c in f.colors)
    assert abs(total - 1.0) < 0.02
    assert all(c.hex.startswith("#") for c in f.colors)


def test_cinematic_teal_orange_high(split_tone_ctx):
    f = CinematicAnalyzer().analyze(split_tone_ctx)
    assert f.teal_orange_score > 0.5
    assert 0.0 <= f.moody_score <= 1.0
