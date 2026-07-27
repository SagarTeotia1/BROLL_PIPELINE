"""Integration tests for the ASH-FS pipeline.

Contains:
  - A slow end-to-end test using a real synthetic MP4 (requires --run-slow).
  - A fast (<5 s) integration test that wires the full pipeline with a mocked
    EmbeddingManager so no model loading or GPU is required.
"""
from __future__ import annotations

import math
import os
import statistics
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# NOTE: pytest_addoption / pytest_configure / pytest_collection_modifyitems
# are intentionally NOT defined here.  They live in conftest.py so they are
# registered once at session start, before any test module is imported.


# ---------------------------------------------------------------------------
# Synthetic video generation
# ---------------------------------------------------------------------------

_FPS = 25
_TOTAL_FRAMES = 15_000  # exactly 600 s at 25 fps
_HEIGHT = 128
_WIDTH = 128


def _solid_frame(r: int, g: int, b: int, noise_rng: np.random.Generator) -> np.ndarray:
    """Return a single BGR frame (for cv2) with base color + slight pixel noise."""
    noise = noise_rng.integers(-8, 9, size=(_HEIGHT, _WIDTH, 3), dtype=np.int16)
    frame = np.array([b, g, r], dtype=np.int16)  # BGR for cv2
    img = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return img


def _gradient_frame(
    t: float,
    r1: int, g1: int, b1: int,
    r2: int, g2: int, b2: int,
) -> np.ndarray:
    """Return a BGR frame linearly interpolated between two colors at ratio t∈[0,1]."""
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    img = np.full((_HEIGHT, _WIDTH, 3), [b, g, r], dtype=np.uint8)
    return img


def generate_synthetic_video(output_path: str) -> None:
    """Write a synthetic 15 000-frame MP4 to *output_path*.

    Segment layout
    --------------
    Frames 0    – 2 999 : solid red   (200, 50, 50)  + noise
    Frames 3 000 – 3 000: HARD CUT
    Frames 3 001 – 5 999: solid blue  (50, 50, 200)  + noise
    Frames 6 000 – 8 999: gradient red → blue (gradual fade, soft boundary around 7 500)
    Frames 9 000 – 9 000: HARD CUT
    Frames 9 001 –11 999: solid green  (50, 200, 50) + noise
    Frames 12 000–12 000: HARD CUT
    Frames 12 001–14 999: solid yellow (200, 200, 50) + noise
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(_FPS), (_WIDTH, _HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {output_path!r}")

    rng = np.random.default_rng(0)

    for i in range(_TOTAL_FRAMES):
        if i <= 2_999:
            # Segment 1: solid red
            frame = _solid_frame(200, 50, 50, rng)
        elif i <= 5_999:
            # Segment 2: solid blue (hard cut at frame 3 001)
            frame = _solid_frame(50, 50, 200, rng)
        elif i <= 8_999:
            # Segment 3: gradual fade red → blue
            t = (i - 6_000) / 2_999
            frame = _gradient_frame(t, 200, 50, 50, 50, 50, 200)
        elif i <= 11_999:
            # Segment 4: solid green (hard cut at frame 9 001)
            frame = _solid_frame(50, 200, 50, rng)
        else:
            # Segment 5: solid yellow (hard cut at frame 12 001)
            frame = _solid_frame(200, 200, 50, rng)

        writer.write(frame)

    writer.release()


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

GPU_HR_PER_FOOTAGE_HR = 0.37   # Qwen3-VL cost anchor from spec


@pytest.mark.slow
def test_full_pipeline_synthetic_video(tmp_path: Path) -> None:
    """End-to-end integration test on a synthetic 10-minute video.

    Covers: video generation → SamplerPipeline.run() → manifest validation
    → metrics report.
    """
    # ------------------------------------------------------------------
    # 1. Generate synthetic video
    # ------------------------------------------------------------------
    video_path = str(tmp_path / "synthetic_10min.mp4")
    t_gen_start = time.perf_counter()
    generate_synthetic_video(video_path)
    t_gen = time.perf_counter() - t_gen_start
    assert os.path.exists(video_path), "Video file was not created."
    assert os.path.getsize(video_path) > 0, "Video file is empty."

    # ------------------------------------------------------------------
    # 2. Load config and force CPU (no GPU required for CI)
    # ------------------------------------------------------------------
    _HERE = os.path.dirname(__file__)
    config_path = os.path.join(_HERE, "..", "config.yaml")

    # We must patch BEFORE constructing the pipeline so EmbeddingManager
    # reads the patched value at __init__ time.
    from ashfs.config import load_config

    cfg = load_config(config_path)
    cfg.embeddings.dinov2_device = "cpu"

    # Monkey-patch load_config so SamplerPipeline.__init__ picks up our
    # CPU-forced config without touching the YAML file on disk.
    import ashfs.config as _cfg_module
    import ashfs.sampler_pipeline as _pipeline_module

    _orig_load_config = _cfg_module.load_config

    def _patched_load_config(path: str):  # noqa: ANN202
        c = _orig_load_config(path)
        c.embeddings.dinov2_device = "cpu"
        return c

    _cfg_module.load_config = _patched_load_config
    _pipeline_module.load_config = _patched_load_config

    try:
        # ------------------------------------------------------------------
        # 3. Run the full pipeline
        # ------------------------------------------------------------------
        from ashfs.sampler_pipeline import SamplerPipeline

        t_pipeline_start = time.perf_counter()
        pipeline = SamplerPipeline(config_path)
        manifest = pipeline.run(video_path)
        t_pipeline = time.perf_counter() - t_pipeline_start

    finally:
        # Restore originals regardless of test outcome
        _cfg_module.load_config = _orig_load_config
        _pipeline_module.load_config = _orig_load_config

    # ------------------------------------------------------------------
    # 4. Assertions
    # ------------------------------------------------------------------

    # 4a. Pipeline must NOT have fallen back to uniform sampling.
    assert manifest.get("fallback") is not True, (
        f"Pipeline fell back to uniform sampling. "
        f"Error: {manifest.get('fallback_error', 'unknown')}"
    )

    # 4b. At least the 3 hard cuts must be detected as shot boundaries.
    total_shots = manifest["total_shots"]
    assert total_shots >= 3, (
        f"Expected at least 3 shots (3 hard cuts), got {total_shots}."
    )

    # 4c. At least one frame selected per shot.
    total_selected = manifest["total_selected_frames"]
    assert total_selected >= total_shots, (
        f"Selected frames ({total_selected}) < total shots ({total_shots}). "
        "Expected at least 1 frame per shot."
    )

    # 4d. Global frame budget respected.
    from ashfs.config import load_config as _real_load_config

    _real_cfg = _real_load_config(config_path)
    max_budget = _real_cfg.budget.max_total_frames_per_video
    assert total_selected <= max_budget, (
        f"Selected {total_selected} frames but budget cap is {max_budget}."
    )

    # 4e. Selected frame indices are sorted in temporal order.
    selected_indices = manifest["selected_frame_indices"]
    assert selected_indices == sorted(selected_indices), (
        "selected_frame_indices are not in sorted (temporal) order."
    )

    # 4f. frame_roles has the same length as selected_frame_indices.
    frame_roles = manifest["frame_roles"]
    assert len(frame_roles) == len(selected_indices), (
        f"frame_roles length ({len(frame_roles)}) != "
        f"selected_frame_indices length ({len(selected_indices)})."
    )

    # 4g. All roles must be from the valid set.
    valid_roles = {"medoid", "common", "unique", "extra", "fallback"}
    bad_roles = [r for r in frame_roles if r not in valid_roles]
    assert not bad_roles, (
        f"Unexpected frame role(s) found: {set(bad_roles)}. "
        f"Valid roles: {valid_roles}."
    )

    # 4h. Compression ratio must be >= 1.0 (we never select *more* than input).
    compression_ratio = manifest["compression_stats"]["compression_ratio"]
    assert compression_ratio >= 1.0, (
        f"compression_ratio = {compression_ratio} < 1.0 — "
        "more frames selected than DualKeyframeSelector received."
    )

    # ------------------------------------------------------------------
    # 5. Print report table
    # ------------------------------------------------------------------
    fps_dist: list[int] = manifest["frames_per_shot_distribution"]
    candidate_frames: int = manifest["total_candidate_frames"]

    fps_min = min(fps_dist) if fps_dist else 0
    fps_max = max(fps_dist) if fps_dist else 0
    fps_mean = statistics.mean(fps_dist) if fps_dist else 0.0
    fps_median = statistics.median(fps_dist) if fps_dist else 0.0

    # GPU-hours proxy: pipeline processes (total_frames / video_fps) hours of footage
    cap = cv2.VideoCapture(video_path)
    video_fps_actual = cap.get(cv2.CAP_PROP_FPS) or _FPS
    total_frames_actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    footage_hours = (total_frames_actual / video_fps_actual) / 3600.0
    gpu_hr_estimate = footage_hours * GPU_HR_PER_FOOTAGE_HR

    timing = manifest["timing"]
    timing_str = ", ".join(f"{k}={v:.3f}s" for k, v in timing.items())

    print()
    print("=" * 60)
    print("ASH-FS Integration Test — Synthetic 10-minute Video")
    print("=" * 60)
    print(f"Video generation time          : {t_gen:.2f}s")
    print(f"Pipeline wall-clock time       : {t_pipeline:.2f}s")
    print(f"Total shots detected           : {total_shots}")
    print(f"Candidate frames sampled       : {candidate_frames}")
    print(f"Final keyframes selected       : {total_selected}")
    print(
        f"Frames-per-shot                : "
        f"min={fps_min}  max={fps_max}  "
        f"mean={fps_mean:.2f}  median={fps_median:.1f}"
    )
    print(
        f"Compression ratio              : "
        f"{compression_ratio:.2f}x "
        f"({manifest['compression_stats']['total_original_frames']} → "
        f"{manifest['compression_stats']['total_final_frames']} frames)"
    )
    print(
        f"GPU-hours (Qwen3-VL proxy,     : {gpu_hr_estimate:.4f} "
        f"({footage_hours * 60:.1f} footage-min × {GPU_HR_PER_FOOTAGE_HR} GPU-hr/footage-hr)"
    )
    print(f"Timing breakdown               : {timing_str}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Fast integration test — mocked EmbeddingManager, no model loading, < 5 s
# ---------------------------------------------------------------------------

def _make_fake_embedding_manager(n_frames: int, dim: int = 768):
    """Return a mock EmbeddingManager that returns deterministic unit vectors.

    ``extract_for_frames`` returns a dict of {frame_idx: (dim,) array}.
    ``get_embeddings`` returns an (N, dim) ordered array.
    Neither calls torch.hub or any GPU code.
    """
    rng = np.random.default_rng(0)

    def _unit_vec(idx: int) -> np.ndarray:
        """Deterministic per-frame unit vector seeded from frame index."""
        r = np.random.default_rng(idx)
        v = r.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    mock_mgr = MagicMock()

    def extract_for_frames(frame_indices, frame_loader):
        return {idx: _unit_vec(idx) for idx in frame_indices}

    def get_embeddings(frame_indices, frame_loader):
        if not frame_indices:
            return np.empty((0, dim), dtype=np.float32)
        rows = [_unit_vec(idx) for idx in frame_indices]
        return np.stack(rows, axis=0)

    mock_mgr.extract_for_frames.side_effect = extract_for_frames
    mock_mgr.get_embeddings.side_effect = get_embeddings
    mock_mgr.extract_siglip2.return_value = None
    return mock_mgr


def _generate_tiny_video(output_path: str, n_frames: int = 100, fps: int = 25) -> None:
    """Write a tiny solid-color MP4 with one hard cut in the middle."""
    h, w = 32, 32
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed: {output_path!r}")
    mid = n_frames // 2
    for i in range(n_frames):
        # First half: red; second half: blue — guarantees at least one hard cut.
        color = (0, 0, 200) if i < mid else (200, 0, 0)  # BGR
        frame = np.full((h, w, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_fast_pipeline_with_mocked_embeddings(tmp_path, cfg):
    """Fast (<5 s) end-to-end pipeline test using mocked EmbeddingManager.

    Validates the full Stage 1→9 orchestration (sampling, shot segmentation,
    complexity scoring, budget planning, dual-keyframe selection, hierarchical
    diversity planning) with synthetic embeddings — no GPU, no model weights.

    Shot segmentation (PySceneDetect) is also mocked so the test has zero
    external dependencies beyond OpenCV for video I/O.
    """
    video_path = str(tmp_path / "tiny.mp4")
    fps = 25
    n_frames = 200   # 8 s at 25 fps
    _generate_tiny_video(video_path, n_frames=n_frames, fps=fps)

    # --- Build a minimal SamplerPipeline with injected mock components ---
    from ashfs.sampler_pipeline import SamplerPipeline
    from ashfs.shot_segmenter import ShotSegmenter
    from ashfs.complexity_scorer import ComplexityScorer
    from ashfs.dual_keyframe_selector import DualKeyframeSelector
    from ashfs.types import Shot

    pipeline = SamplerPipeline.__new__(SamplerPipeline)
    pipeline._cfg = cfg
    pipeline._config_path = ""

    # Inject mock EmbeddingManager
    mock_emb_mgr = _make_fake_embedding_manager(n_frames)
    pipeline._embedding_manager = mock_emb_mgr

    from ashfs.budget_planner import BudgetPlanner
    from ashfs.hierarchical_diversity_planner import HierarchicalDiversityPlanner
    pipeline._budget_planner = BudgetPlanner(cfg)
    pipeline._diversity_planner = HierarchicalDiversityPlanner(cfg)

    # Mock _load_shot_segmenter to return a real ShotSegmenter but with
    # _detect_hard_cuts patched to return a known single hard cut at frame 100.
    real_segmenter = ShotSegmenter(cfg)

    def _fake_detect_hard_cuts(video_path_):
        # Simulate one hard cut at the midpoint
        return [100]

    real_segmenter._detect_hard_cuts = _fake_detect_hard_cuts

    pipeline._load_shot_segmenter = lambda: real_segmenter
    pipeline._load_complexity_scorer = lambda: ComplexityScorer(cfg)
    pipeline._load_dual_keyframe_selector = lambda: DualKeyframeSelector(cfg)

    # --- Run the pipeline ---
    t0 = time.perf_counter()
    manifest = pipeline.run(video_path)
    elapsed = time.perf_counter() - t0

    # Must finish in under 5 seconds (no model loading)
    assert elapsed < 5.0, f"Fast integration test took {elapsed:.2f}s — too slow."

    # Must not have triggered fallback
    assert manifest.get("fallback") is not True, (
        f"Pipeline unexpectedly fell back: {manifest.get('fallback_error')}"
    )

    # Must have detected shots (at least 1 from our fake hard cut)
    assert manifest["total_shots"] >= 1

    # At least 1 frame selected
    assert manifest["total_selected_frames"] >= 1

    # Global budget respected
    assert manifest["total_selected_frames"] <= cfg.budget.max_total_frames_per_video

    # Frame indices must be sorted
    sel = manifest["selected_frame_indices"]
    assert sel == sorted(sel), "selected_frame_indices not in temporal order"

    # frame_roles length matches selected indices
    assert len(manifest["frame_roles"]) == len(sel)

    # All roles must be valid
    valid_roles = {"medoid", "common", "unique", "extra", "fallback"}
    bad = [r for r in manifest["frame_roles"] if r not in valid_roles]
    assert not bad, f"Invalid frame roles: {set(bad)}"

    # Compression ratio must be >= 1.0
    assert manifest["compression_stats"]["compression_ratio"] >= 1.0
