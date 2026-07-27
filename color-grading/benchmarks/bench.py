"""Timing harness for the colour-grading engine.

Measures three things separately, because they optimise differently:

1. **Context build** — decoding + the colour-space conversions cached on
   :class:`~color_analyzer.analyzer.utils.ImageContext`.
2. **Per-analyzer** — each analyzer run against an already-built context, so the
   shared conversions are not charged to whichever analyzer happens to be first.
3. **End-to-end** — ``analyze() + decide()``, i.e. what a caller actually pays
   per frame.

Usage::

    python benchmarks/bench.py --image frame.png --runs 20
    python benchmarks/bench.py --image frame.png --runs 20 --deep
    python benchmarks/bench.py --image frame.png --max-side 1024 --json out.json

Report the median, not the mean: the first run pays for lazy imports (scipy,
numba) and OpenCV's thread pool spin-up, and one cold outlier drags a mean of 20
runs by several milliseconds.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_analyzer.analyzer.engine import ColorGradingEngine  # noqa: E402
from color_analyzer.analyzer.utils import Backend, ImageContext  # noqa: E402


def _time(fn: Callable[[], Any], runs: int, warmup: int = 2) -> Dict[str, float]:
    """Run ``fn`` and return millisecond timing statistics."""
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "mean_ms": statistics.fmean(samples),
    }


def _engine(deep: bool, cpu: bool) -> ColorGradingEngine:
    """Build an engine, tolerating a build that predates the ``deep`` flag."""
    backend = Backend(prefer_gpu=not cpu)
    try:
        return ColorGradingEngine(backend=backend, deep=deep)  # type: ignore[call-arg]
    except TypeError:
        return ColorGradingEngine(backend=backend)


def run_bench(image: str, runs: int, max_side: int | None, deep: bool, cpu: bool) -> Dict[str, Any]:
    engine = _engine(deep, cpu)
    backend = engine.backend

    ctx = ImageContext.from_path(image, backend=backend, max_side=max_side)
    out: Dict[str, Any] = {
        "image": os.path.basename(image),
        "resolution": f"{ctx.width}x{ctx.height}",
        "pixels": ctx.width * ctx.height,
        "backend": backend.describe()["backend"],
        "deep": deep,
        "runs": runs,
    }

    out["context_build"] = _time(
        lambda: ImageContext.from_path(image, backend=backend, max_side=max_side), runs
    )

    # Per-analyzer: one fresh context per repetition, then every analyzer timed
    # in registry order against it. That is exactly how production runs, so a
    # derived array shared between analyzers (luminance histogram, local std) is
    # charged to whoever asks for it first rather than being amortised away.
    # Timing each analyzer against its own fresh context instead would cost
    # len(analyzers) x runs context builds — minutes on a 4K frame.
    samples: Dict[str, List[float]] = {name: [] for name in engine._analyzers}
    for rep in range(runs + 1):
        fresh = ImageContext.from_path(image, backend=backend, max_side=max_side)
        for name, analyzer in engine._analyzers.items():
            t0 = time.perf_counter()
            analyzer.analyze(fresh)
            dt = (time.perf_counter() - t0) * 1000.0
            if rep > 0:  # first repetition is warmup (lazy imports, thread pools)
                samples[name].append(dt)
    per = {
        name: {
            "median_ms": statistics.median(vals),
            "max_ms": max(vals),
        }
        for name, vals in samples.items()
    }
    out["per_analyzer"] = dict(sorted(per.items(), key=lambda kv: -kv[1]["median_ms"]))
    out["analyzer_sum_ms"] = sum(v["median_ms"] for v in per.values())

    out["analyze"] = _time(lambda: engine.analyze_path(image, max_side=max_side), runs)

    result = engine.analyze_path(image, max_side=max_side)
    out["decide"] = _time(lambda: engine.decide(result), runs)
    out["end_to_end"] = {
        "median_ms": out["analyze"]["median_ms"] + out["decide"]["median_ms"],
    }
    return out


def _print(report: Dict[str, Any]) -> None:
    print()
    print(f"  {report['image']}  {report['resolution']}  "
          f"({report['pixels'] / 1e6:.1f} MP)  backend={report['backend']}  "
          f"deep={report['deep']}  runs={report['runs']}")
    print("  " + "-" * 62)
    print(f"  {'context build':<28} {report['context_build']['median_ms']:>8.2f} ms")
    print(f"  {'analyze (all analyzers)':<28} {report['analyze']['median_ms']:>8.2f} ms")
    print(f"  {'decide':<28} {report['decide']['median_ms']:>8.2f} ms")
    print(f"  {'END-TO-END':<28} {report['end_to_end']['median_ms']:>8.2f} ms")
    print("  " + "-" * 62)
    print(f"  per-analyzer, registry order, shared context "
          f"(sum {report['analyzer_sum_ms']:.2f} ms):")
    for name, stats in report["per_analyzer"].items():
        print(f"    {name:<26} {stats['median_ms']:>8.2f} ms")
    print()


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Colour-grading engine timing harness.")
    p.add_argument("--image", required=True, action="append",
                   help="Image to benchmark. Repeatable.")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--max-side", type=int, default=None,
                   help="Downscale long side before analysis (0 or omitted = native).")
    p.add_argument("--deep", action="store_true", help="Include the deep analyzers.")
    p.add_argument("--cpu", action="store_true", help="Force the CPU backend.")
    p.add_argument("--json", default=None, help="Also write the raw report here.")
    args = p.parse_args(argv)

    max_side = args.max_side if args.max_side else None
    reports = [run_bench(img, args.runs, max_side, args.deep, args.cpu) for img in args.image]
    for r in reports:
        _print(r)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, indent=2)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
