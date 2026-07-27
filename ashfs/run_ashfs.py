"""ASH-FS command-line entry point.

Usage:
    python run_ashfs.py --video path/to/video.mp4 --config config.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ashfs.sampler_pipeline import SamplerPipeline

try:
    from tqdm import tqdm as _tqdm_cls
    _TQDM = True
except ImportError:
    _TQDM = False


# ---------------------------------------------------------------------------
# Stage progress display
# ---------------------------------------------------------------------------

_STAGES = [
    ("candidate_sampling",        "Sampling candidate frames       "),
    ("dinov2_extraction",         "Extracting DINOv2 embeddings    "),
    ("shot_segmentation",         "Detecting shots                 "),
    ("shot_embedding_assembly",   "Assembling shot embeddings      "),
    ("complexity_scoring",        "Scoring complexity              "),
    ("budget_planning",           "Planning keyframe budgets       "),
    ("dual_keyframe_selection",   "Selecting keyframes             "),
    ("hierarchical_diversity",    "Hierarchical diversity planning "),
    ("novelty_estimation",        "Estimating novelty missed       "),
]
_N_STAGES = len(_STAGES)


class _StageBar:
    """Simple stage-level progress tracker printed to stdout."""

    def __init__(self) -> None:
        self._stage_idx = 0
        self._t_stage = time.time()
        self._t_total = time.time()
        if _TQDM:
            self._bar = _tqdm_cls(
                total=_N_STAGES,
                desc="ASH-FS",
                unit="stage",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} stages [{elapsed}<{remaining}]",
            )
        else:
            self._bar = None
        self._print(f"Starting pipeline — {_N_STAGES} stages")

    def advance(self, name: str) -> None:
        elapsed = time.time() - self._t_stage
        label = self._stage_label(name)
        self._print(f"  ✓ {label}  ({elapsed:.1f}s)")
        if self._bar is not None:
            self._bar.update(1)
        self._stage_idx += 1
        self._t_stage = time.time()

    def next_stage(self, name: str) -> None:
        label = self._stage_label(name)
        if self._bar is not None:
            self._bar.set_postfix_str(label.strip())
        else:
            print(f"  → {label.strip()} ...", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    @staticmethod
    def _stage_label(name: str) -> str:
        for key, label in _STAGES:
            if key == name:
                return label
        return name

    @staticmethod
    def _print(msg: str) -> None:
        if _TQDM:
            _tqdm_cls.write(msg)
        else:
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# Patched pipeline that emits stage events
# ---------------------------------------------------------------------------

class _InstrumentedPipeline(SamplerPipeline):
    """SamplerPipeline subclass that calls a stage callback after each stage."""

    def __init__(self, config_path: str, bar: _StageBar) -> None:
        super().__init__(config_path)
        self._bar = bar

    def _run_pipeline(self, video_path: str, timing: dict) -> dict:
        import numpy as np
        from ashfs.config import get_complexity_threshold

        cfg = self._cfg
        bar = self._bar

        shot_segmenter = self._load_shot_segmenter()
        complexity_scorer = self._load_complexity_scorer()
        dual_keyframe_selector = self._load_dual_keyframe_selector()

        # Stage 1
        bar.next_stage("candidate_sampling")
        t0 = time.perf_counter()
        frame_indices, frame_loader = shot_segmenter.sample_candidate_frames(video_path)
        timing["candidate_sampling"] = time.perf_counter() - t0
        bar.advance("candidate_sampling")

        # Stage 2
        bar.next_stage("dinov2_extraction")
        t0 = time.perf_counter()
        embeddings_cache = self._embedding_manager.extract_for_frames(frame_indices, frame_loader)
        timing["dinov2_extraction"] = time.perf_counter() - t0
        bar.advance("dinov2_extraction")

        # Stage 3
        bar.next_stage("shot_segmentation")
        t0 = time.perf_counter()
        embeddings_array = np.stack([embeddings_cache[idx] for idx in frame_indices])
        shots = shot_segmenter.detect_shots(
            video_path, self._embedding_manager,
            frame_indices=frame_indices, embeddings_array=embeddings_array,
        )
        timing["shot_segmentation"] = time.perf_counter() - t0
        bar.advance("shot_segmentation")

        # Stage 4
        bar.next_stage("shot_embedding_assembly")
        t0 = time.perf_counter()
        shot_embeddings = self._build_shot_embeddings(shots, frame_indices, embeddings_cache)
        timing["shot_embedding_assembly"] = time.perf_counter() - t0
        bar.advance("shot_embedding_assembly")

        # Stage 5
        bar.next_stage("complexity_scoring")
        t0 = time.perf_counter()
        scored_shots = complexity_scorer.score_all(shot_embeddings)
        timing["complexity_scoring"] = time.perf_counter() - t0
        bar.advance("complexity_scoring")

        # Stage 6 (fast — no separate bar entry)
        complexity_threshold = get_complexity_threshold(cfg)
        timing["complexity_threshold_lookup"] = 0.0

        # Stage 7
        bar.next_stage("budget_planning")
        t0 = time.perf_counter()
        budgeted_shots = self._budget_planner.plan(scored_shots, complexity_threshold)
        timing["budget_planning"] = time.perf_counter() - t0
        budget_stats = self._budget_planner.get_stats(budgeted_shots)
        bar.advance("budget_planning")

        # Stage 8
        bar.next_stage("dual_keyframe_selection")
        t0 = time.perf_counter()
        selected_keyframes = dual_keyframe_selector.select_all(budgeted_shots)
        timing["dual_keyframe_selection"] = time.perf_counter() - t0
        bar.advance("dual_keyframe_selection")

        # Stage 9
        bar.next_stage("hierarchical_diversity")
        t0 = time.perf_counter()
        tree = self._diversity_planner.build_tree(selected_keyframes)
        final_keyframes = self._diversity_planner.get_final_keyframes(tree)
        timing["hierarchical_diversity"] = time.perf_counter() - t0
        compression_stats = self._diversity_planner.get_compression_stats(selected_keyframes, final_keyframes)
        bar.advance("hierarchical_diversity")

        # Stage 10 optional: SigLIP2
        if cfg.embeddings.siglip2_enabled:
            t0 = time.perf_counter()
            self._extract_siglip2_for_final(final_keyframes, frame_loader)
            timing["siglip2_extraction"] = time.perf_counter() - t0

        # Stage 10b optional: dense signals
        if cfg.dense_signals.enabled:
            t0 = time.perf_counter()
            self._extract_dense_signals_stage(frame_indices, frame_loader)
            timing["dense_signal_extraction"] = time.perf_counter() - t0

        # Stage 10c: novelty
        bar.next_stage("novelty_estimation")
        t0 = time.perf_counter()
        novelty_stats = self._estimate_novelty_missed(
            frame_indices=frame_indices,
            embeddings_cache=embeddings_cache,
            final_keyframes=final_keyframes,
            sample_shots=20,
        )
        timing["novelty_estimation"] = time.perf_counter() - t0
        bar.advance("novelty_estimation")

        return self._build_manifest(
            video_path=video_path,
            shots=shots,
            frame_indices=frame_indices,
            final_keyframes=final_keyframes,
            budget_stats=budget_stats,
            compression_stats=compression_stats,
            novelty_stats=novelty_stats,
            timing=timing,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ashfs.py",
        description="ASH-FS — Adaptive Shot-Aware Hierarchical Frame Sampler",
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--output", default=None, help="Output manifest JSON path.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)
    output_path = (
        Path(args.output) if args.output
        else Path(__file__).parent / "output" / "manifest.json"
    )

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nVideo  : {video_path}")
    print(f"Config : {config_path}")
    print(f"Output : {output_path}\n")

    bar = _StageBar()
    t_total = time.time()

    try:
        pipeline = _InstrumentedPipeline(str(config_path), bar)
        manifest = pipeline.run(str(video_path))
    except Exception as exc:
        bar.close()
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    bar.close()
    elapsed = time.time() - t_total

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n{'=' * 52}")
    print("  ASH-FS Results")
    print(f"{'=' * 52}")
    print(f"  Shots detected       : {manifest['total_shots']}")
    print(f"  Candidate frames     : {manifest['total_candidate_frames']}")
    print(f"  Keyframes selected   : {manifest['total_selected_frames']}")
    total_cand = manifest.get("total_candidate_frames", 0)
    total_final = manifest.get("total_selected_frames", 0)
    true_cr = total_cand / total_final if total_final > 0 else float("inf")
    print(f"  Compression ratio    : {true_cr:.2f}x  (candidate→final)")
    ns = manifest.get("novelty_stats", {})
    if ns:
        print(f"  Max novelty missed   : {ns.get('max_novelty_missed', 'N/A')}")
        print(f"  Shots above 0.15     : {ns.get('shots_above_threshold', 'N/A')}/{ns.get('shots_sampled', 'N/A')}")
    print(f"  Wall-clock time      : {elapsed:.1f}s")
    print(f"{'=' * 52}")

    timing = manifest.get("timing", {})
    if timing:
        print("\n  Timing breakdown:")
        for stage, t in timing.items():
            print(f"    {stage:<35}: {t:.2f}s")

    if manifest.get("fallback"):
        print(f"\nWARNING: fallback to uniform sampling. Error: {manifest.get('fallback_error')}", file=sys.stderr)

    print(f"\n  Manifest saved → {output_path}\n")


if __name__ == "__main__":
    main()
