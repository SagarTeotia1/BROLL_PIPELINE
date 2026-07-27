"""Command-line entry point for the editorial analyzer and renderer.

Analyse::

    python -m color_analyzer.editorial.cli frame.jpg
    python -m color_analyzer.editorial.cli frames/ -o out/ --max-side 720
    python -m color_analyzer.editorial.cli clip.mp4 --every 24 -o out/

The grading loop — build a payload for a model, then render what it returns::

    python -m color_analyzer.editorial.cli frame.jpg \\
        --prompt "dark cinematic grading" -o payload.json
    # ... send payload.json to the model, save its reply as graded.json ...
    python -m color_analyzer.editorial.cli frame.jpg --apply graded.json -o graded.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import cv2

from .analyzer import DEFAULT_MAX_SIDE, DEFAULT_PALETTE_SIZE, EditorialAnalyzer
from .gpu import Backend

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="color_analyzer.editorial",
        description="Describe the editable colour state of an image, folder or video.",
    )
    parser.add_argument("input", help="Image file, directory of images, or video file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Write JSON here (a directory for multiple inputs). "
                             "Defaults to stdout.")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                        help=f"Analysis resolution cap (default {DEFAULT_MAX_SIDE}; "
                             f"0 for native).")
    parser.add_argument("--colors", type=int, default=DEFAULT_PALETTE_SIZE,
                        help=f"Palette swatches (default {DEFAULT_PALETTE_SIZE}).")
    parser.add_argument("--every", type=int, default=24,
                        help="Video only: analyse every Nth frame (default 24).")
    parser.add_argument("--cpu", action="store_true", help="Force the CPU backend.")
    parser.add_argument("--compact", action="store_true",
                        help="Emit single-line JSON instead of indented.")

    grading = parser.add_argument_group("grading loop")
    grading.add_argument("--prompt", metavar="INSTRUCTION",
                         help="Emit a payload for a grading model instead of the raw "
                              "analysis: the editable controls, read-only context, and "
                              "instructions. INSTRUCTION describes the look wanted.")
    grading.add_argument("--apply", metavar="CONTROLS.JSON",
                         help="Render a model's returned controls onto the image and "
                              "write the result (-o must name an image file).")
    grading.add_argument("--source", metavar="STATE.JSON",
                         help="Measured state to grade from. Re-measured from the image "
                              "when omitted.")
    grading.add_argument("--no-protect-skin", dest="protect_skin", action="store_false",
                         help="Apply the grade at full strength on skin as well. By "
                              "default skin is held back so faces survive a heavy grade.")
    parser.set_defaults(protect_skin=True)
    return parser


def run(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    analyzer = EditorialAnalyzer(
        backend=Backend(prefer_gpu=not args.cpu),
        max_side=args.max_side if args.max_side else None,
        palette_size=args.colors,
    )
    indent = None if args.compact else 2

    extension = os.path.splitext(args.input)[1].lower()

    if args.apply:
        if extension not in _IMAGE_EXTS:
            print("--apply takes a single image", file=sys.stderr)
            return 2
        return _apply_mode(analyzer, args)

    if os.path.isdir(args.input):
        results = _analyze_directory(analyzer, args.input)
    elif extension in _VIDEO_EXTS:
        results = _analyze_video(analyzer, args.input, args.every)
    elif extension in _IMAGE_EXTS:
        results = {os.path.basename(args.input): analyzer.analyze_path(args.input)}
    else:
        print(f"unsupported input: {args.input}", file=sys.stderr)
        return 2

    if not results:
        print(f"nothing to analyse in: {args.input}", file=sys.stderr)
        return 1

    if args.prompt:
        from .prompt import build_payload

        results = {name: build_payload(state, args.prompt)
                   for name, state in results.items()}

    _emit(results, args.output, indent)
    return 0


def _apply_mode(analyzer: EditorialAnalyzer, args: argparse.Namespace) -> int:
    """Render a model's controls onto the image and write the graded result."""
    import numpy as np

    from .apply import apply_controls

    controls = _read_json(args.apply)
    if controls is None:
        return 2
    # Accept either a bare controls object or the whole payload back.
    if isinstance(controls, dict) and "controls" in controls:
        controls = controls["controls"]

    source = None
    if args.source:
        source = _read_json(args.source)
        if source is None:
            return 2

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"could not read image: {args.input}", file=sys.stderr)
        return 2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    result = apply_controls(rgb, controls, source=source,
                            protect_skin=args.protect_skin,
                            max_side=analyzer.max_side)

    for note in result.clamped:
        print(f"clamped: {note}", file=sys.stderr)
    for note in result.ignored:
        print(f"ignored: {note}", file=sys.stderr)
    print(result.summary(), file=sys.stderr)

    destination = args.output or _default_output(args.input)
    graded = cv2.cvtColor((np.clip(result.image, 0, 1) * 255 + 0.5).astype(np.uint8),
                          cv2.COLOR_RGB2BGR)
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    if not cv2.imwrite(destination, graded):
        print(f"could not write: {destination}", file=sys.stderr)
        return 1
    print(destination)
    return 0


def _default_output(source: str) -> str:
    stem, extension = os.path.splitext(source)
    return f"{stem}.graded{extension or '.png'}"


def _read_json(path: str) -> Optional[Any]:
    """Load JSON, tolerating a model's markdown fence around it."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        print(f"could not read {path}: {error}", file=sys.stderr)
        return None

    stripped = text.strip()
    if stripped.startswith("```"):
        # ```json ... ``` — common enough in model replies to be worth handling.
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as error:
        print(f"{path} is not valid JSON: {error}", file=sys.stderr)
        return None


def _analyze_directory(analyzer: EditorialAnalyzer, path: str) -> Dict[str, Any]:
    """Analyse every image in a directory, keyed by filename."""
    names = sorted(
        name for name in os.listdir(path)
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS
    )
    return {name: analyzer.analyze_path(os.path.join(path, name)) for name in names}


def _analyze_video(analyzer: EditorialAnalyzer, path: str, every: int) -> Dict[str, Any]:
    """Analyse every ``every``-th frame of a video, keyed by frame index.

    Frames are decoded sequentially and skipped in Python rather than sought
    with ``CAP_PROP_POS_FRAMES``: seeking a long-GOP codec lands on the nearest
    keyframe, so requested and delivered frame numbers drift apart.
    """
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")

    results: Dict[str, Any] = {}
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % max(1, every) == 0:
                results[f"frame_{index:06d}"] = analyzer.analyze_bgr(frame)
            index += 1
    finally:
        capture.release()
    return results


def _emit(results: Dict[str, Any], output: Optional[str], indent: Optional[int]) -> None:
    """Write to stdout, one file, or one file per input."""
    single = len(results) == 1
    payload = next(iter(results.values())) if single else results

    if output is None:
        print(json.dumps(payload, indent=indent))
        return

    if single and not os.path.isdir(output) and os.path.splitext(output)[1]:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        _write(output, payload, indent)
        return

    os.makedirs(output, exist_ok=True)
    for name, document in results.items():
        stem = os.path.splitext(name)[0]
        _write(os.path.join(output, f"{stem}.json"), document, indent)


def _write(path: str, payload: Any, indent: Optional[int]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)


if __name__ == "__main__":
    raise SystemExit(run())
