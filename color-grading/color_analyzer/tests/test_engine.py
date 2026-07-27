"""End-to-end tests for the engine, feature vector and reporting."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from color_analyzer import ColorGradingEngine
from color_analyzer.analyzer.engine import CORE_SECTIONS, DEEP_SECTIONS
from color_analyzer.analyzer.report import ReportGenerator
from color_analyzer.analyzer.visualization import Visualizer


def _make_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = 320
    lum = np.linspace(0, 1, w)[None, :].repeat(240, 0)
    img = np.stack([lum, 0.3 + lum * 0.3, 0.5 - lum * 0.4], -1).astype(np.float32)
    img = np.clip(img + rng.random((240, w, 3)).astype(np.float32) * 0.04, 0, 1)
    return img


def test_engine_runs_and_serialises():
    res = ColorGradingEngine().analyze_array(_make_image())
    d = res.to_dict()
    # Must be fully JSON serialisable with no NaN/Inf.
    text = json.dumps(d)
    assert "NaN" not in text and "Infinity" not in text
    assert set(["summary", "feature_vector", "cinematic"]).issubset(d.keys())


def test_fast_path_omits_deep_sections():
    res = ColorGradingEngine().analyze_array(_make_image())
    assert res.deep is False
    for name in DEEP_SECTIONS:
        assert getattr(res, name) is None, name
    # Omitted rather than serialised as null, so the key set stays honest.
    assert not set(DEEP_SECTIONS) & set(res.to_dict())


def test_deep_mode_populates_every_section():
    res = ColorGradingEngine(deep=True).analyze_array(_make_image())
    assert res.deep is True
    for name in CORE_SECTIONS + DEEP_SECTIONS:
        assert getattr(res, name) is not None, name


def test_deep_mode_does_not_change_the_core_sections():
    """The fast path must be the same maths, not an approximation of it.

    Deep mode adds analyzers; it must not perturb the ones the grade reads. This
    is what guards the shared ImageContext caches — if the luminance histogram
    or the local-std map ever disagreed with a per-analyzer computation, the
    core sections would drift between modes.
    """
    img = _make_image(3)
    fast = ColorGradingEngine(deep=False).analyze_array(img)
    deep = ColorGradingEngine(deep=True).analyze_array(img)
    for name in CORE_SECTIONS:
        fast_section = getattr(fast, name).to_dict()
        deep_section = getattr(deep, name).to_dict()
        if name in ("hsv", "lab"):
            # These two intentionally skip deep-only fields on the fast path;
            # compare only what the fast path claims to produce.
            deep_section = {k: deep_section[k] for k in fast_section
                            if _is_populated(fast_section[k])}
            fast_section = {k: v for k, v in fast_section.items() if _is_populated(v)}
        assert fast_section == deep_section, name


def _is_populated(value) -> bool:
    """False for the empty defaults a skipped deep-only field leaves behind.

    Recurses, because a skipped ``ChannelStats`` is a dict of zeros rather than
    a falsy value.
    """
    if isinstance(value, dict):
        return any(_is_populated(v) for v in value.values())
    if isinstance(value, list):
        return any(_is_populated(v) for v in value)
    return value not in (None, 0.0, 0)


def test_grades_match_between_fast_and_deep():
    img = _make_image(4)
    fast = ColorGradingEngine(deep=False).grade(img)
    deep = ColorGradingEngine(deep=True).grade(img)
    assert fast["grade"] == deep["grade"]


def test_feature_vector_consistency_and_finiteness():
    res = ColorGradingEngine(deep=True).analyze_array(_make_image())
    fv = res.feature_vector
    assert len(fv) == len(fv.names) == fv.values.shape[0]
    assert len(fv) > 100
    assert all(math.isfinite(v) for v in fv.values)
    # Names are unique and deterministic (sorted within each section).
    assert len(set(fv.names)) == len(fv.names)


def test_feature_vector_stable_across_runs():
    a = ColorGradingEngine(deep=True).analyze_array(_make_image(1)).feature_vector
    b = ColorGradingEngine(deep=True).analyze_array(_make_image(1)).feature_vector
    assert a.names == b.names
    # KMeans is seeded, so the vector is reproducible run-to-run.
    assert np.allclose(a.values, b.values, atol=1e-6)


def test_reports_and_visuals_written(tmp_path):
    res = ColorGradingEngine(deep=True).analyze_array(_make_image())
    visuals = Visualizer(scatter_samples=1000).generate_all(res, str(tmp_path))
    paths = ReportGenerator().generate(res, str(tmp_path), visuals)
    assert os.path.exists(paths.summary_path)
    assert os.path.exists(paths.html_path)
    # A representative visualisation exists.
    assert os.path.exists(visuals.files["rgb_histogram"])
    assert os.path.exists(visuals.files["dominant_palette"])
    # summary.txt is non-trivial.
    with open(paths.summary_path, encoding="utf-8") as fh:
        assert "COLOUR GRADING ANALYSIS SUMMARY" in fh.read()


def test_visuals_refuse_a_fast_analysis(tmp_path):
    res = ColorGradingEngine(deep=False).analyze_array(_make_image())
    with pytest.raises(ValueError, match="deep"):
        Visualizer(scatter_samples=1000).generate_all(res, str(tmp_path))


def test_summary_fields_present():
    res = ColorGradingEngine().analyze_array(_make_image())
    s = res.summary
    assert s.overall_grading and s.mood and s.temperature
    assert 0.0 <= s.confidence <= 1.0
    assert len(s.dominant_colors) >= 1


def test_max_side_caps_the_analysis_resolution():
    res = ColorGradingEngine().analyze_array(_make_image(), max_side=64)
    assert max(res.width, res.height) == 64


def test_analyze_frames_reuses_the_engine():
    engine = ColorGradingEngine()
    frames = [_make_image(i) for i in range(3)]
    results = list(engine.analyze_frames(frames, is_bgr=False, max_side=128))
    assert len(results) == 3
    assert all(max(r.width, r.height) == 128 for r in results)
