"""
ASH-FS Evaluation Harness

Measures recall, precision, and F1 of ASH-FS keyframe selection against
hand-labelled "must-keep" ground-truth frames.  Three methods are compared:

  - ASH-FS          — full adaptive pipeline
  - Flat-FPS        — uniform 1 frame/second baseline
  - InfoShot-2/shot — fixed budget=2 per shot, no adaptive budget

Usage:
  # Label clips (run once per clip):
  python eval_harness.py label --clip path/to/clip.mp4 --output eval/labels/clip_001.json

  # Run evaluation:
  python eval_harness.py eval --clips eval/labels/ --config config.yaml

  # Report:
  python eval_harness.py report --results eval/results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure the project root (parent of eval/) is on sys.path so `import ashfs` works
# when running this script directly (e.g. python eval/eval_harness.py ...).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ClipLabel dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClipLabel:
    """Ground-truth annotation for a single video clip."""

    clip_path: str
    content_vertical: str  # 'talking_head' | 'fast_cut' | 'edtech_slides' | 'general'
    must_keep_frames: list[int]  # absolute frame indices ground-truth "must-keep"
    label_notes: str            # free text explaining what each frame represents
    labeled_by: str
    labeled_at: str             # ISO date string


# ---------------------------------------------------------------------------
# Per-clip evaluation result
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    """Evaluation result for one method on one clip."""

    clip_path: str
    content_vertical: str
    method: str
    recall_at_k: float
    precision: float
    f1: float
    frames_selected: int
    total_video_frames: int
    compression_ratio: float
    max_novelty_missed: float = 0.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _recall_at_k(
    must_keep: list[int],
    selected: list[int],
    k: int = 15,
) -> float:
    """Fraction of must-keep frames that have ≥1 selected frame within ±k frames.

    Parameters
    ----------
    must_keep:
        Ground-truth "must-keep" absolute frame indices.
    selected:
        Frame indices chosen by a method.
    k:
        Matching tolerance in frames (default 15 = 0.5 s at 30 fps).
    """
    if not must_keep:
        return 1.0  # vacuously true
    if not selected:
        return 0.0

    selected_set = set(selected)
    hits = 0
    for gt in must_keep:
        lo, hi = gt - k, gt + k
        if any(lo <= s <= hi for s in selected_set):
            hits += 1
    return hits / len(must_keep)


def _precision_at_k(
    must_keep: list[int],
    selected: list[int],
    k: int = 15,
) -> float:
    """Fraction of selected frames that are within ±k of any must-keep frame.

    Parameters
    ----------
    must_keep:
        Ground-truth "must-keep" absolute frame indices.
    selected:
        Frame indices chosen by a method.
    k:
        Matching tolerance in frames (default 15 = 0.5 s at 30 fps).
    """
    if not selected:
        return 0.0
    if not must_keep:
        return 0.0

    hits = 0
    for s in selected:
        lo, hi = s - k, s + k
        if any(lo <= gt <= hi for gt in must_keep):
            hits += 1
    return hits / len(selected)


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------


def _flat_fps_baseline(video_path: str, fps: float = 1.0) -> list[int]:
    """Select 1 frame per second uniformly (flat-FPS baseline).

    Parameters
    ----------
    video_path:
        Path to the video file.
    fps:
        Sampling rate in frames-per-second (default 1 fps).

    Returns
    -------
    list[int]
        Sorted absolute frame indices.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required: pip install opencv-python") from exc

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path!r}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    step = max(1, int(round(video_fps / fps)))
    return list(range(0, total_frames, step))


def _infoshot_2per_shot_baseline(
    video_path: str,
    config_path: str,
) -> list[int]:
    """InfoShot-default baseline: fixed budget=2 per shot, no adaptive budget.

    Uses the same ShotSegmenter and DualKeyframeSelector as ASH-FS but
    forces every shot's budget to exactly 2 (the InfoShot default), bypassing
    the BudgetPlanner entirely.

    Parameters
    ----------
    video_path:
        Path to the video file.
    config_path:
        Path to config.yaml.

    Returns
    -------
    list[int]
        Sorted absolute frame indices.
    """
    try:
        from ashfs.config import load_config
        from ashfs.embedding_manager import EmbeddingManager
        from ashfs.shot_segmenter import ShotSegmenter
        from ashfs.complexity_scorer import ComplexityScorer
        from ashfs.dual_keyframe_selector import DualKeyframeSelector
        from ashfs.types import BudgetedShot, ShotEmbeddings
        import numpy as np
    except ImportError as exc:
        raise ImportError("ashfs package not found on sys.path.") from exc

    cfg = load_config(config_path)

    emb_mgr = EmbeddingManager(cfg)
    segmenter = ShotSegmenter(cfg)
    scorer = ComplexityScorer(cfg)
    selector = DualKeyframeSelector(cfg)

    # Stage 1+2: sample + extract
    frame_indices, frame_loader = segmenter.sample_candidate_frames(video_path)
    emb_cache = emb_mgr.extract_for_frames(frame_indices, frame_loader)

    import numpy as np

    embeddings_array = np.stack([emb_cache[idx] for idx in frame_indices])

    # Stage 3: detect shots
    shots = segmenter.detect_shots(
        video_path,
        emb_mgr,
        frame_indices=frame_indices,
        embeddings_array=embeddings_array,
    )

    # Stage 4: build ShotEmbeddings
    shot_embeddings_list: list[ShotEmbeddings] = []
    for shot in shots:
        shot_indices = [
            idx for idx in frame_indices
            if shot.start_frame <= idx <= shot.end_frame
        ]
        if not shot_indices:
            nearest = min(
                frame_indices,
                key=lambda idx: abs(idx - (shot.start_frame + shot.end_frame) // 2),
                default=None,
            )
            shot_indices = [nearest] if nearest is not None else []

        if shot_indices:
            embs = np.stack(
                [emb_cache[idx] for idx in shot_indices if idx in emb_cache],
                axis=0,
            )
        else:
            embs = np.empty((0, 768), dtype=np.float32)

        shot_embeddings_list.append(
            ShotEmbeddings(shot=shot, frame_indices=shot_indices, embeddings=embs)
        )

    # Stage 5: score (needed to build ScoredShot for BudgetedShot)
    scored_shots = scorer.score_all(shot_embeddings_list)

    # InfoShot default: FIXED budget=2 per shot (ignore BudgetPlanner)
    budgeted = [
        BudgetedShot(scored_shot=ss, budget=min(2, ss.frame_count))
        for ss in scored_shots
    ]

    selected_kf = selector.select_all(budgeted)

    selected: list[int] = sorted(
        idx for kf in selected_kf for idx in kf.frame_indices
    )
    return selected


def _get_total_frames(video_path: str) -> int:
    """Return total frame count of a video via cv2."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required.") from exc

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


# ---------------------------------------------------------------------------
# label subcommand
# ---------------------------------------------------------------------------


def cmd_label(args: argparse.Namespace) -> None:
    """Interactive CLI to label must-keep frames in a clip."""
    try:
        import cv2
    except ImportError as exc:
        print("ERROR: opencv-python is required for labeling: pip install opencv-python")
        sys.exit(1)

    clip_path = str(args.clip)
    output_path = str(args.output)

    if not os.path.exists(clip_path):
        print(f"ERROR: Clip not found: {clip_path!r}")
        sys.exit(1)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {clip_path!r}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps
    cap.release()

    print()
    print(f"Clip          : {clip_path}")
    print(f"Duration      : {duration_s:.1f}s  ({total_frames} frames @ {video_fps:.1f} fps)")
    print()

    # Show frames at 1 fps so the user can orient themselves.
    print("Frames at 1 fps:")
    step = max(1, int(round(video_fps)))
    preview_indices = list(range(0, total_frames, step))
    print(f"  Frame indices shown (1fps, {len(preview_indices)} frames): "
          f"{preview_indices[:10]}{'...' if len(preview_indices) > 10 else ''}")
    print()

    # Collect must-keep frames
    print("Enter the frame indices you want to mark as must-keep.")
    print("Separate multiple indices with spaces or commas (e.g. 150 300 750).")
    print("Press Enter with no input to finish.")
    print()

    must_keep_raw = input("Must-keep frame indices: ").strip()
    if not must_keep_raw:
        must_keep_frames: list[int] = []
    else:
        tokens = must_keep_raw.replace(",", " ").split()
        try:
            must_keep_frames = sorted(int(t) for t in tokens)
        except ValueError:
            print("ERROR: Could not parse frame indices. Please enter integers only.")
            sys.exit(1)

    # Validate indices are within range
    out_of_range = [f for f in must_keep_frames if f < 0 or f >= total_frames]
    if out_of_range:
        print(f"WARNING: The following indices are out of range [0, {total_frames - 1}] "
              f"and will be clamped: {out_of_range}")
        must_keep_frames = [
            max(0, min(total_frames - 1, f)) for f in must_keep_frames
        ]

    print()
    content_vertical = input(
        "Content vertical [talking_head / fast_cut / edtech_slides / general]: "
    ).strip()
    if content_vertical not in ("talking_head", "fast_cut", "edtech_slides", "general"):
        print(f"WARNING: Unrecognised vertical {content_vertical!r} — defaulting to 'general'.")
        content_vertical = "general"

    label_notes = input("Label notes (free text, press Enter to skip): ").strip()
    labeled_by = input("Your name / identifier: ").strip() or "unknown"

    from datetime import datetime, timezone

    labeled_at = datetime.now(timezone.utc).date().isoformat()

    label = ClipLabel(
        clip_path=clip_path,
        content_vertical=content_vertical,
        must_keep_frames=must_keep_frames,
        label_notes=label_notes,
        labeled_by=labeled_by,
        labeled_at=labeled_at,
    )

    # Write JSON
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(label), fh, indent=2)

    print()
    print(f"Label saved to: {output_path}")
    print(f"Must-keep frames recorded: {len(must_keep_frames)}")


# ---------------------------------------------------------------------------
# eval subcommand
# ---------------------------------------------------------------------------

_K_TOLERANCE = 15  # frames — half-second at 30fps


def _load_labels(labels_dir: str) -> list[ClipLabel]:
    """Load all .json label files from *labels_dir*."""
    label_files = sorted(Path(labels_dir).glob("*.json"))
    if not label_files:
        print(f"WARNING: No .json label files found in {labels_dir!r}")
        return []

    labels: list[ClipLabel] = []
    for path in label_files:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        try:
            labels.append(ClipLabel(**raw))
        except TypeError as exc:
            print(f"WARNING: Could not parse {path} as ClipLabel: {exc}")

    return labels


def _eval_clip(
    label: ClipLabel,
    config_path: str,
    results_dir: str,
) -> list[ClipResult]:
    """Evaluate all three methods on a single labelled clip.

    Parameters
    ----------
    label:
        Ground-truth annotation for the clip.
    config_path:
        Path to config.yaml for ASH-FS and InfoShot baselines.
    results_dir:
        Directory to write per-clip JSON result files.

    Returns
    -------
    list[ClipResult]
        One result per method (3 total).
    """
    clip_path = label.clip_path
    must_keep = label.must_keep_frames

    if not os.path.exists(clip_path):
        print(f"  SKIP: clip not found: {clip_path!r}")
        return []

    total_frames = _get_total_frames(clip_path)
    if total_frames <= 0:
        print(f"  SKIP: cannot read frame count from {clip_path!r}")
        return []

    print(f"  Evaluating: {os.path.basename(clip_path)}"
          f"  ({total_frames} frames, {len(must_keep)} must-keep frames)")

    results: list[ClipResult] = []

    # ------------------------------------------------------------------
    # Method 1: ASH-FS
    # ------------------------------------------------------------------
    print("    [1/3] ASH-FS ...", end=" ", flush=True)
    ashfs_novelty_missed: float = 0.0
    try:
        from ashfs.sampler_pipeline import SamplerPipeline

        pipeline = SamplerPipeline(config_path)
        manifest = pipeline.run(clip_path)
        ashfs_selected = manifest["selected_frame_indices"]
        novelty_stats = manifest.get("novelty_stats", {})
        ashfs_novelty_missed = float(novelty_stats.get("max_novelty_missed", 0.0))
    except Exception as exc:
        print(f"FAILED ({exc})")
        ashfs_selected = []

    r = _recall_at_k(must_keep, ashfs_selected, k=_K_TOLERANCE)
    p = _precision_at_k(must_keep, ashfs_selected, k=_K_TOLERANCE)
    cr = total_frames / len(ashfs_selected) if ashfs_selected else float("inf")
    results.append(
        ClipResult(
            clip_path=clip_path,
            content_vertical=label.content_vertical,
            method="ASH-FS",
            recall_at_k=round(r, 4),
            precision=round(p, 4),
            f1=round(_f1(p, r), 4),
            frames_selected=len(ashfs_selected),
            total_video_frames=total_frames,
            compression_ratio=round(cr, 2),
            max_novelty_missed=round(ashfs_novelty_missed, 6),
        )
    )
    print(f"recall={r:.3f}  precision={p:.3f}  f1={_f1(p, r):.3f}  "
          f"frames={len(ashfs_selected)}  max_novelty={ashfs_novelty_missed:.4f}")

    # ------------------------------------------------------------------
    # Method 2: Flat-FPS baseline
    # ------------------------------------------------------------------
    print("    [2/3] Flat-FPS (1 fps) ...", end=" ", flush=True)
    try:
        flat_selected = _flat_fps_baseline(clip_path, fps=1.0)
    except Exception as exc:
        print(f"FAILED ({exc})")
        flat_selected = []

    r = _recall_at_k(must_keep, flat_selected, k=_K_TOLERANCE)
    p = _precision_at_k(must_keep, flat_selected, k=_K_TOLERANCE)
    cr = total_frames / len(flat_selected) if flat_selected else float("inf")
    results.append(
        ClipResult(
            clip_path=clip_path,
            content_vertical=label.content_vertical,
            method="Flat-FPS",
            recall_at_k=round(r, 4),
            precision=round(p, 4),
            f1=round(_f1(p, r), 4),
            frames_selected=len(flat_selected),
            total_video_frames=total_frames,
            compression_ratio=round(cr, 2),
        )
    )
    print(f"recall={r:.3f}  precision={p:.3f}  f1={_f1(p, r):.3f}  "
          f"frames={len(flat_selected)}")

    # ------------------------------------------------------------------
    # Method 3: InfoShot-2/shot baseline
    # ------------------------------------------------------------------
    print("    [3/3] InfoShot-2/shot ...", end=" ", flush=True)
    try:
        info_selected = _infoshot_2per_shot_baseline(clip_path, config_path)
    except Exception as exc:
        print(f"FAILED ({exc})")
        info_selected = []

    r = _recall_at_k(must_keep, info_selected, k=_K_TOLERANCE)
    p = _precision_at_k(must_keep, info_selected, k=_K_TOLERANCE)
    cr = total_frames / len(info_selected) if info_selected else float("inf")
    results.append(
        ClipResult(
            clip_path=clip_path,
            content_vertical=label.content_vertical,
            method="InfoShot-2/shot",
            recall_at_k=round(r, 4),
            precision=round(p, 4),
            f1=round(_f1(p, r), 4),
            frames_selected=len(info_selected),
            total_video_frames=total_frames,
            compression_ratio=round(cr, 2),
        )
    )
    print(f"recall={r:.3f}  precision={p:.3f}  f1={_f1(p, r):.3f}  "
          f"frames={len(info_selected)}")

    # ------------------------------------------------------------------
    # Save per-clip result JSON
    # ------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)
    clip_stem = Path(clip_path).stem
    out_path = os.path.join(results_dir, f"{clip_stem}_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)

    return results


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation over all labelled clips."""
    labels = _load_labels(str(args.clips))
    if not labels:
        print("No clips to evaluate.")
        sys.exit(0)

    config_path = str(args.config)
    if not os.path.exists(config_path):
        print(f"ERROR: config not found: {config_path!r}")
        sys.exit(1)

    results_dir = str(getattr(args, "output", "eval/results"))

    all_results: list[ClipResult] = []
    for i, label in enumerate(labels, start=1):
        print(f"\n[{i}/{len(labels)}] {os.path.basename(label.clip_path)}")
        clip_results = _eval_clip(label, config_path, results_dir)
        all_results.extend(clip_results)

    print(f"\nEvaluation complete. Results saved to: {results_dir}/")
    _print_aggregate_table(all_results)


# ---------------------------------------------------------------------------
# report subcommand
# ---------------------------------------------------------------------------


def _load_all_results(results_dir: str) -> list[ClipResult]:
    """Load all per-clip result JSON files from *results_dir*."""
    result_files = sorted(Path(results_dir).glob("*_results.json"))
    if not result_files:
        print(f"WARNING: No *_results.json files found in {results_dir!r}")
        return []

    all_results: list[ClipResult] = []
    for path in result_files:
        with open(path, "r", encoding="utf-8") as fh:
            raw_list = json.load(fh)
        for raw in raw_list:
            try:
                all_results.append(ClipResult(**raw))
            except TypeError as exc:
                print(f"WARNING: Cannot parse result in {path}: {exc}")
    return all_results


def _avg(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _print_aggregate_table(all_results: list[ClipResult]) -> None:
    """Print the summary table — overall + breakdown by content vertical."""
    if not all_results:
        print("No results to report.")
        return

    methods = ["ASH-FS", "Flat-FPS", "InfoShot-2/shot"]
    verticals = sorted({r.content_vertical for r in all_results})

    def _row(label: str, rows: list[ClipResult], method: str) -> str:
        subset = [r for r in rows if r.method == method]
        if not subset:
            return (
                f"  {method:<20} | {'N/A':>9} | {'N/A':>9} | {'N/A':>6} "
                f"| {'N/A':>10} | {'N/A':>11} | {'N/A':>10}"
            )
        recall = _avg([r.recall_at_k for r in subset])
        prec   = _avg([r.precision for r in subset])
        f1     = _avg([r.f1 for r in subset])
        frames = _avg([r.frames_selected for r in subset])
        comp   = _avg([r.compression_ratio for r in subset])
        # MaxNovelty is only meaningful for ASH-FS; show dash for baselines.
        novelty_vals = [r.max_novelty_missed for r in subset if r.max_novelty_missed > 0.0]
        if novelty_vals:
            novelty_str = f"{_avg(novelty_vals):>10.4f}"
        else:
            novelty_str = f"{'--':>10}"
        return (
            f"  {method:<20} | {recall:>9.3f} | {prec:>9.3f} | {f1:>6.3f} "
            f"| {frames:>10.1f} | {comp:>9.1f}x | {novelty_str}"
        )

    header = (
        f"  {'Method':<20} | {'Recall@15':>9} | {'Precision':>9} "
        f"| {'F1':>6} | {'Avg Frames':>10} | {'Compression':>11} | {'MaxNovelty':>10}"
    )
    sep = (
        "  " + "-" * 20 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-"
        + "-" * 6 + "-+-" + "-" * 10 + "-+-" + "-" * 11 + "-+-" + "-" * 10
    )

    print()
    print("=" * 90)
    print("  ASH-FS Evaluation Report")
    print("=" * 90)

    # Overall table
    print(f"\n  OVERALL ({len(set(r.clip_path for r in all_results))} clips)\n")
    print(header)
    print(sep)
    for method in methods:
        print(_row("overall", all_results, method))

    # Per-vertical breakdown
    for vertical in verticals:
        v_rows = [r for r in all_results if r.content_vertical == vertical]
        clips_in_v = len(set(r.clip_path for r in v_rows))
        print(f"\n  {vertical.upper()} ({clips_in_v} clips)\n")
        print(header)
        print(sep)
        for method in methods:
            print(_row(vertical, v_rows, method))

    print()
    print("=" * 90)
    print("  Recall@15   = fraction of must-keep frames matched within ±15 frames")
    print("  Precision   = fraction of selected frames within ±15 of a must-keep frame")
    print("  F1          = harmonic mean of Recall@15 and Precision")
    print("  MaxNovelty  = avg max cosine distance from any candidate to nearest keyframe (ASH-FS only)")
    print("=" * 90)
    print()


def cmd_report(args: argparse.Namespace) -> None:
    """Print aggregate report from saved result files."""
    results_dir = str(args.results)
    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir!r}")
        sys.exit(1)

    all_results = _load_all_results(results_dir)
    _print_aggregate_table(all_results)


# ---------------------------------------------------------------------------
# novelty subcommand
# ---------------------------------------------------------------------------


def cmd_novelty(args: argparse.Namespace) -> None:
    """Run embedding extraction + novelty estimation for a single video.

    This does not run the full pipeline evaluation — it just runs the
    ASH-FS pipeline (which includes Stage 10b novelty estimation) and
    prints the novelty_stats section of the manifest.

    Usage
    -----
    python eval_harness.py novelty --video path/to/video.mp4 --config config.yaml
    """
    video_path = str(args.video)
    config_path = str(args.config)

    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path!r}")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path!r}")
        sys.exit(1)

    try:
        from ashfs.sampler_pipeline import SamplerPipeline
    except ImportError as exc:
        print(f"ERROR: ashfs package not found on sys.path: {exc}")
        sys.exit(1)

    print(f"Video  : {video_path}")
    print(f"Config : {config_path}")
    print("Running ASH-FS pipeline (includes novelty estimation) ...")

    pipeline = SamplerPipeline(config_path)
    manifest = pipeline.run(video_path)

    novelty_stats: dict = manifest.get("novelty_stats", {})

    print()
    print("=" * 50)
    print("  Novelty Missed Stats")
    print("=" * 50)
    print(f"  max_novelty_missed    : {novelty_stats.get('max_novelty_missed', 'N/A')}")
    print(f"  mean_novelty_missed   : {novelty_stats.get('mean_novelty_missed', 'N/A')}")
    print(f"  shots_sampled         : {novelty_stats.get('shots_sampled', 'N/A')}")
    print(f"  shots_above_threshold : {novelty_stats.get('shots_above_threshold', 'N/A')}")
    print("=" * 50)
    print()
    print("Full novelty_stats JSON:")
    print(json.dumps(novelty_stats, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_harness.py",
        description="ASH-FS Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # label
    p_label = sub.add_parser(
        "label",
        help="Interactively label must-keep frames in a clip.",
    )
    p_label.add_argument("--clip", required=True, help="Path to the video clip.")
    p_label.add_argument(
        "--output",
        required=True,
        help="Output JSON path (e.g. eval/labels/clip_001.json).",
    )

    # eval
    p_eval = sub.add_parser(
        "eval",
        help="Run evaluation over all labelled clips.",
    )
    p_eval.add_argument(
        "--clips",
        required=True,
        help="Directory containing .json label files.",
    )
    p_eval.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml.",
    )
    p_eval.add_argument(
        "--output",
        default="eval/results",
        help="Directory to write per-clip result files (default: eval/results).",
    )

    # report
    p_report = sub.add_parser(
        "report",
        help="Print aggregate report from saved result JSON files.",
    )
    p_report.add_argument(
        "--results",
        required=True,
        help="Directory containing *_results.json files.",
    )

    # novelty
    p_novelty = sub.add_parser(
        "novelty",
        help=(
            "Run embedding extraction + novelty estimation for a single video "
            "and print the novelty_stats from the ASH-FS manifest."
        ),
    )
    p_novelty.add_argument(
        "--video",
        required=True,
        help="Path to the video file.",
    )
    p_novelty.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "label":
        cmd_label(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "novelty":
        cmd_novelty(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
