"""Command-line entry point for the colour-grading analysis engine.

Usage examples
--------------
Analyse a single image and write all reports/visualisations::

    python -m color_analyzer.main path/to/frame.jpg -o outputs

Analyse a whole folder, only emit JSON + summary (no plots)::

    python -m color_analyzer.main frames/ -o outputs --no-visuals

Force CPU even if a GPU is available::

    python -m color_analyzer.main frame.png --cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .analyzer.engine import ColorGradingEngine
from .analyzer.report import ReportGenerator
from .analyzer.utils import Backend
from .analyzer.visualization import Visualizer

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

#: Default analysis resolution cap. Colour statistics converge well below the
#: delivery resolution, so analysing a 4K frame at full size costs ~8x the work
#: for no measurable change in the grade. Pass ``--max-side 0`` to disable.
DEFAULT_MAX_SIDE = 1024


def _collect_images(path: str) -> List[str]:
    """Return the list of image files implied by a file or directory path."""
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        ]
        return files
    return [path]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="color_analyzer",
        description="Production-grade colour-grading analysis engine.",
    )
    p.add_argument("input", help="Image file or directory of images to analyse.")
    p.add_argument("-o", "--output", default="outputs", help="Output directory.")
    p.add_argument(
        "--deep", action="store_true",
        help="Run the deep analyzers too (histograms, spatial maps, perceptual, "
             "harmony). Required for --visuals; several times slower.",
    )
    p.add_argument(
        "--visuals", action="store_true",
        help="Also generate the visualisation PNGs, report.html and feature vector. "
             "Implies --deep.",
    )
    p.add_argument("--no-report", action="store_true", help="Skip the text/HTML report.")
    p.add_argument("--no-grade", action="store_true", help="Skip the grade.json recommendation.")
    p.add_argument(
        "--grade-only", action="store_true",
        help="Emit only the grading recommendation (grade.json).",
    )
    p.add_argument(
        "--flat", action="store_true",
        help="Also write grade.flat.json: a flat {parameter: recommended} mapping.",
    )
    p.add_argument(
        "--render", action="store_true",
        help="Also render the recommended grade to graded.png.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU backend.")
    p.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help=f"Downscale the longest image side to this many pixels before analysis "
             f"(default {DEFAULT_MAX_SIDE}). Pass 0 to analyse at native resolution.",
    )
    return p


def run(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Visualisations read the histogram/spatial sections, so they need deep mode.
    deep = args.deep or args.visuals
    max_side = args.max_side if args.max_side else None

    backend = Backend(prefer_gpu=not args.cpu)
    engine = ColorGradingEngine(backend=backend, deep=deep)
    reporter = ReportGenerator()
    visualizer = Visualizer()

    images = _collect_images(args.input)
    if not images:
        print(f"No images found at: {args.input}", file=sys.stderr)
        return 1

    try:
        from rich.console import Console
        from rich.progress import track

        console: object = Console()
        iterator = track(images, description="Analysing") if len(images) > 1 else images
    except Exception:  # pragma: no cover
        console = None
        iterator = images

    for image_path in iterator:
        name = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(args.output, name) if len(images) > 1 else args.output
        os.makedirs(out_dir, exist_ok=True)

        result = engine.analyze_path(image_path, max_side=max_side)

        # The 45-parameter grade document (+ optional flat view and render).
        grade = None
        if not args.no_grade:
            grade = engine.decide(result)
            _write_grade(grade, out_dir, flat=args.flat)
            if args.render:
                _render_grade(image_path, grade, out_dir, max_side)

        # Current-grade report (descriptive): compact report.json. Visuals opt-in.
        if not args.grade_only:
            visuals = None
            if args.visuals:
                visuals = visualizer.generate_all(result, out_dir)
                _save_feature_vector(result, out_dir)
            if not args.no_report:
                reporter.generate(result, out_dir, visuals)

        # Echo the current-grade + recommendation for the operator.
        reporter.print_console(result)
        if grade is not None:
            _print_grade(grade)

    return 0


def _write_grade(grade: dict, out_dir: str, flat: bool = False) -> None:
    """Persist the 45-parameter grade document (and optionally its flat view)."""
    import json

    from .analyzer.schema import flatten

    with open(os.path.join(out_dir, "grade.json"), "w", encoding="utf-8") as fh:
        json.dump(grade, fh, indent=2)
    if flat:
        with open(os.path.join(out_dir, "grade.flat.json"), "w", encoding="utf-8") as fh:
            json.dump(flatten(grade, "recommended"), fh, indent=2)


def _render_grade(image_path: str, grade: dict, out_dir: str, max_side) -> None:
    """Apply the recommended grade to the image and save graded.png."""
    import cv2
    import numpy as np

    from .analyzer.decision_engine import to_executor_decision
    from .analyzer.grading_plan import GradingPlanExecutor
    from .analyzer.utils import downscale

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return
    bgr = downscale(bgr, max_side)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # The executor speaks the older section-shaped decision; translate rather
    # than teach it a second input format.
    plan = to_executor_decision(grade)
    graded = GradingPlanExecutor().apply(rgb, plan).image
    out = cv2.cvtColor((np.clip(graded, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, "graded.png"), out)


def _print_grade(grade: dict) -> None:
    """Print a compact summary of the grade document to the terminal."""
    g = grade.get("grade", {})
    style = grade.get("style", {})
    wb = g.get("white_balance.temperature", {})
    notes = grade.get("notes", [])

    line = (
        f"Grade -> {style.get('target', '?')} | WB {wb.get('current', '?')}K "
        f"-> {wb.get('recommended', '?')}K | " + "; ".join(notes[:3])
    )
    try:
        from rich.console import Console

        # markup=False: notes and style names are data, and rich would eat any
        # bracketed text in them as a markup tag.
        Console(safe_box=True).print(line, style="bold", markup=False)
    except Exception:  # pragma: no cover
        print(line)


def _save_feature_vector(result, out_dir: str) -> None:
    """Write the numeric feature vector + names next to the reports."""
    import numpy as np

    np.save(os.path.join(out_dir, "feature_vector.npy"), result.feature_vector.to_array())
    with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(result.feature_vector.names))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
