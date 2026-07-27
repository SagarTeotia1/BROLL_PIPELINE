"""Throughput benchmark harness.

Three modes:

``micro``
    Per-model latency/throughput sweep over batch sizes. Answers "how many faces per
    second can ArcFace do at batch 16 on this card".

``pipeline``
    Full end-to-end run on a real video, reporting realtime factor, per-stage FPS,
    sampling statistics and peak VRAM.

``sweep``
    Runs the pipeline several times with different ``frame_stride`` / batch settings and
    prints a comparison table - the fastest way to tune for a new machine.

    python benchmarks/benchmark.py micro
    python benchmarks/benchmark.py pipeline clip.mp4
    python benchmarks/benchmark.py sweep clip.mp4 --strides 2 4 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configs.config import AppConfig  # noqa: E402
from detector.scrfd import SCRFDDetector  # noqa: E402
from emotion.hsemotion import HSEmotionClassifier  # noqa: E402
from models.onnx_engine import available_providers  # noqa: E402
from pipeline.worker import AnalysisPipeline  # noqa: E402
from recognition.arcface import ArcFaceEmbedder  # noqa: E402
from utils.gpu import (  # noqa: E402
    GpuMonitor,
    apply_torch_performance_flags,
    empty_cache,
    get_device_info,
    memory_summary,
    resolve_device,
)
from utils.logging_utils import get_logger, setup_logging  # noqa: E402

log = get_logger("benchmark")


@dataclass
class MicroResult:
    """One (model, batch) measurement."""

    model: str
    batch: int
    iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    items_per_second: float

    def row(self) -> str:
        return (
            f"  {self.model:<12} bs={self.batch:<3} "
            f"mean {self.mean_ms:7.2f} ms  p50 {self.p50_ms:7.2f}  p95 {self.p95_ms:7.2f}  "
            f"-> {self.items_per_second:8.1f} items/s"
        )


def _time_calls(fn, warmup: int, iterations: int) -> List[float]:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def run_micro(
    config: AppConfig, batches: Sequence[int] = (1, 2, 4, 8, 16, 32), iterations: int = 20
) -> List[MicroResult]:
    """Benchmark each model in isolation across batch sizes."""
    device = resolve_device(config.gpu.device, config.gpu.device_id)
    apply_torch_performance_flags()
    results: List[MicroResult] = []

    print(f"\nDevice : {get_device_info(config.gpu.device_id)}")
    print(f"ORT EPs: {available_providers()}\n")

    detector = SCRFDDetector(config, device)
    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    det_batches = [b for b in batches if b <= max(1, detector.batch_size)] or [1]
    for batch in det_batches:
        images = [frame] * batch
        timings = _time_calls(lambda: detector.detect_batch(images), 3, iterations)
        results.append(_summarise("scrfd", batch, timings))
        print(results[-1].row())
    detector.close()
    empty_cache()

    embedder = ArcFaceEmbedder(config, device)
    crop112 = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    for batch in batches:
        crops = [crop112] * batch
        timings = _time_calls(lambda: embedder.embed_aligned(crops), 3, iterations)
        results.append(_summarise("arcface", batch, timings))
        print(results[-1].row())
    embedder.close()
    empty_cache()

    classifier = HSEmotionClassifier(config, device)
    size = classifier.input_size
    crop = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    for batch in batches:
        crops = [crop] * batch
        timings = _time_calls(lambda: classifier.predict_crops(crops), 3, iterations)
        results.append(_summarise("hsemotion", batch, timings))
        print(results[-1].row())
    classifier.close()
    empty_cache()

    print(f"\nPeak VRAM: {memory_summary(config.gpu.device_id)}")
    return results


def _summarise(model: str, batch: int, timings: List[float]) -> MicroResult:
    ordered = sorted(timings)
    mean = statistics.fmean(timings)
    return MicroResult(
        model=model,
        batch=batch,
        iterations=len(timings),
        mean_ms=mean,
        p50_ms=ordered[len(ordered) // 2],
        p95_ms=ordered[max(0, int(0.95 * len(ordered)) - 1)],
        items_per_second=(batch / mean) * 1000.0 if mean > 0 else 0.0,
    )


def run_pipeline(config: AppConfig, video: Path, label: str = "") -> Dict[str, object]:
    """Full end-to-end run; returns the stats block of the produced document."""
    monitor = GpuMonitor(config.gpu.device_id)
    peak_util = 0.0
    peak_mem = 0.0

    pipeline = AnalysisPipeline(config)
    pipeline.prepare()

    t0 = time.perf_counter()
    pipeline.start(video)
    while pipeline.is_running:
        sample = monitor.sample()
        peak_util = max(peak_util, sample.utilization)
        peak_mem = max(peak_mem, sample.memory_used_mb)
        time.sleep(0.25)
    document = pipeline.wait()
    elapsed = time.perf_counter() - t0
    pipeline.close()
    monitor.close()

    assert document is not None
    stats = dict(document.stats)
    stats.update(
        {
            "label": label or video.name,
            "wall_seconds": round(elapsed, 3),
            "peak_gpu_utilization": peak_util,
            "peak_gpu_memory_mb": round(peak_mem, 1),
            "events": len(document.events),
            "actors": len(document.actors),
            "processed_fps": document.video.processed_fps,
            "video_duration": document.video.duration,
        }
    )
    return stats


def print_pipeline_stats(stats: Dict[str, object]) -> None:
    print(f"\n  {'video duration':<24}{stats.get('video_duration', 0):.2f} s")
    print(f"  {'wall time':<24}{stats.get('wall_seconds', 0):.2f} s")
    print(f"  {'realtime factor':<24}x{stats.get('realtime_factor', 0)}")
    print(f"  {'frames decoded':<24}{stats.get('frames_decoded', 0)}")
    print(f"  {'frames analysed':<24}{stats.get('frames_analysed', 0)}")
    print(f"  {'faces processed':<24}{stats.get('faces_processed', 0)}")
    print(f"  {'processed fps':<24}{stats.get('processed_fps', 0):.2f}")
    print(f"  {'decoder':<24}{stats.get('decoder', '')}")
    print(f"  {'peak GPU util':<24}{stats.get('peak_gpu_utilization', 0):.0f} %")
    print(f"  {'peak GPU memory':<24}{stats.get('peak_gpu_memory_mb', 0):.0f} MB")
    print(f"  {'events':<24}{stats.get('events', 0)}")
    stages = stats.get("stages", {})
    if isinstance(stages, dict) and stages:
        print("\n  stage           fps      avg ms    p95 ms   count")
        for name in sorted(stages):
            s = stages[name]
            print(
                f"  {name:<14}{s.get('fps', 0):7.2f}  {s.get('avg_ms', 0):8.2f}  "
                f"{s.get('p95_ms', 0):8.2f}  {s.get('count', 0):6d}"
            )


def run_sweep(
    config: AppConfig, video: Path, strides: Sequence[int]
) -> List[Dict[str, object]]:
    """Compare several sampling strides on the same clip."""
    rows: List[Dict[str, object]] = []
    for stride in strides:
        cfg = AppConfig.load()
        cfg.sampling.frame_stride = stride
        cfg.sampling.adaptive = config.sampling.adaptive
        cfg.gpu = config.gpu
        print(f"\n=== stride {stride} ===")
        stats = run_pipeline(cfg, video, label=f"stride={stride}")
        print_pipeline_stats(stats)
        rows.append(stats)
        empty_cache()

    print("\n  stride   realtime   processed fps   events   wall s")
    for stride, row in zip(strides, rows):
        print(
            f"  {stride:<8} x{row.get('realtime_factor', 0):<9} "
            f"{row.get('processed_fps', 0):<15.2f} {row.get('events', 0):<8} "
            f"{row.get('wall_seconds', 0):.1f}"
        )
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark harness")
    parser.add_argument("mode", choices=["micro", "pipeline", "sweep"])
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", default=None, help="write results to this JSON file")
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)

    setup_logging("INFO", None, False)
    config = AppConfig.load(args.config)

    payload: object
    if args.mode == "micro":
        results = run_micro(config, args.batches, args.iterations)
        payload = [asdict(r) for r in results]
    else:
        if not args.video:
            raise SystemExit(f"mode '{args.mode}' needs a video path")
        video = Path(args.video)
        if not video.exists():
            raise SystemExit(f"video not found: {video}")
        if args.mode == "pipeline":
            stats = run_pipeline(config, video)
            print_pipeline_stats(stats)
            payload = stats
        else:
            payload = run_sweep(config, video, args.strides)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\n  results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
