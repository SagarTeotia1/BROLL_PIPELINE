"""Headless command line interface.

    # enrol an actor from a folder of photos
    python cli.py register --name "John" --images ./cast/john

    # analyse a video and write JSON + CSV + PNG into outputs/
    python cli.py analyze clip.mp4
    python cli.py analyze clip.mp4 --full          # add the profiler dump to the JSON

    # empty timeline? check whether the cast is actually in this clip
    python cli.py probe clip.mp4

    # list / remove cast members, inspect the model zoo
    python cli.py cast --list
    python cli.py cast --remove "John"
    python cli.py models

Use this for batch jobs and CI; the GUI shares the exact same pipeline code.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import AppConfig  # noqa: E402
from models.model_zoo import list_models  # noqa: E402
from pipeline.types import ProgressUpdate  # noqa: E402
from pipeline.worker import AnalysisPipeline, PipelineCallbacks  # noqa: E402
from recognition.cast_database import CastDatabase  # noqa: E402
from recognition.registration import CastRegistrar  # noqa: E402
from timeline.exporters import export_all  # noqa: E402
from utils.gpu import get_device_info  # noqa: E402
from utils.logging_utils import get_logger, setup_logging  # noqa: E402


def _global_options(suppress: bool) -> argparse.ArgumentParser:
    """Options accepted both before and after the sub-command.

    The sub-parser copy uses ``SUPPRESS`` defaults so that omitting an option there
    does not overwrite the value given before the sub-command - the standard argparse
    pitfall with shared parents.
    """
    default = argparse.SUPPRESS if suppress else None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=default, help="path to a YAML config")
    parser.add_argument(
        "--log-level", default=default, choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--set", action="append",
        default=(argparse.SUPPRESS if suppress else []), metavar="KEY=VALUE",
        help="override a config value, e.g. --set sampling.frame_stride=2",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = _global_options(suppress=True)
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Face recognition + emotion timeline (headless)",
        parents=[_global_options(suppress=False)],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="enrol or update an actor", parents=[common])
    p_reg.add_argument("--name", required=True, help="actor name")
    p_reg.add_argument(
        "--images", required=True, nargs="+",
        help="image files and/or folders of reference photos",
    )

    p_an = sub.add_parser("analyze", help="analyse a video", parents=[common])
    p_an.add_argument("video", help="path to the video file")
    p_an.add_argument("--output", default=None, help="output folder (default: outputs/)")
    p_an.add_argument("--no-export", action="store_true", help="print JSON to stdout only")
    p_an.add_argument("--quiet", action="store_true", help="suppress the progress line")
    p_an.add_argument(
        "--full", action="store_true",
        help="include the profiler dump and per-event debug fields in the JSON",
    )

    p_probe = sub.add_parser(
        "probe",
        help="check whether the registered cast actually appears in a video",
        parents=[common],
    )
    p_probe.add_argument("video", help="path to the video file")
    p_probe.add_argument(
        "--every", type=int, default=15, help="test 1 frame in N (default 15)"
    )
    p_probe.add_argument(
        "--max-frames", type=int, default=1200, help="stop after this many frames"
    )

    p_cast = sub.add_parser(
        "cast", help="inspect or edit the cast database", parents=[common]
    )
    p_cast.add_argument("--list", action="store_true", help="list registered actors")
    p_cast.add_argument("--remove", default=None, metavar="NAME", help="delete an actor")
    p_cast.add_argument("--export-pickle", default=None, metavar="PATH")
    p_cast.add_argument("--import-pickle", default=None, metavar="PATH")

    sub.add_parser(
        "models", help="show the model zoo and where weights are cached", parents=[common]
    )
    return parser


def apply_overrides(config: AppConfig, overrides: List[str]) -> AppConfig:
    """Apply ``--set key=value`` pairs onto the config tree."""
    if not overrides:
        return config
    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        parsed[key.strip()] = _coerce(value.strip())
    return config.override(parsed)


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_register(config: AppConfig, args: argparse.Namespace) -> int:
    log = get_logger("cli.register")
    pipeline = AnalysisPipeline(config)
    pipeline.prepare()
    assert pipeline.detector is not None and pipeline.embedder is not None

    registrar = CastRegistrar(config, pipeline.detector, pipeline.embedder, pipeline.db)

    def progress(index: int, total: int, name: str) -> None:
        print(f"\r  [{index:3d}/{total}] {name[:48]:<48}", end="", flush=True)

    result = registrar.register(args.name, args.images, progress=progress)
    print()
    log.info(result.summary())
    for outcome in result.outcomes:
        status = "OK      " if outcome.accepted else "REJECTED"
        detail = f" ({outcome.reason})" if outcome.reason else ""
        print(f"  {status} {Path(outcome.path).name}{detail} q={outcome.quality:.2f}")
    pipeline.close()
    return 0 if result.ok else 1


def cmd_analyze(config: AppConfig, args: argparse.Namespace) -> int:
    log = get_logger("cli.analyze")
    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    state = {"last": 0.0}

    def on_progress(update: ProgressUpdate) -> None:
        if args.quiet:
            return
        now = time.perf_counter()
        if now - state["last"] < 0.5 and not update.finished:
            return
        state["last"] = now
        print(
            f"\r  {update.percent:5.1f}% | decode {update.decode_fps:6.1f} fps | "
            f"analysis {update.analysis_fps:5.2f} fps | x{update.realtime_factor:5.2f} "
            f"realtime | GPU {update.gpu_utilization:3.0f}% "
            f"{update.gpu_memory_mb / 1024:4.1f} GB | events {update.events:4d}",
            end="", flush=True,
        )

    pipeline = AnalysisPipeline(config, PipelineCallbacks(on_progress=on_progress))
    pipeline.prepare()
    if pipeline.matcher.is_empty:
        log.warning("No actors registered - every face will be reported as Unknown")

    document = pipeline.analyze(video)
    if not args.quiet:
        print()

    if args.no_export:
        print(document.to_json(verbose=args.full))
    else:
        out_dir = Path(args.output) if args.output else (
            config.paths.resolve(config.paths.output_dir) / video.stem
        )
        paths = export_all(document, out_dir, video.stem, verbose_json=args.full)
        for kind, path in paths.items():
            print(f"  {kind:<12} {path}")

    info = document.analysis
    total = document.video.frame_count
    share = (100.0 * info.frames_analysed / total) if total else 0.0
    print(
        f"\n  Analysed in {info.elapsed_seconds:.1f}s "
        f"({info.realtime_factor:.2f}x realtime)"
    )
    print(
        f"  {info.frames_analysed}/{total} frames analysed ({share:.1f}%) - "
        f"stride {config.sampling.frame_stride}"
        + (" + scene-cut boost" if config.sampling.scene_cut_enabled else "")
    )
    print(
        f"  {info.expression_changes} expression changes | {len(document.events)} events | "
        f"{len(document.actors)} cast members | processed at "
        f"{document.video.processed_fps:.2f} fps"
    )
    for actor in document.actors:
        print(
            f"    {actor.name:<20} {actor.expression_changes:>3} changes  "
            f"{actor.screen_time:6.2f}s on screen  sim {actor.mean_similarity:.3f}"
        )
    for emotion, seconds in document.emotion_totals().items():
        print(f"      {emotion:<10} {seconds:7.2f}s")
    pipeline.close()
    return 0


def cmd_probe(config: AppConfig, args: argparse.Namespace) -> int:
    """Report the best gallery similarity reached in a video.

    Answers the most common "why is my timeline empty?" question: either the registered
    cast genuinely is not in this clip, or the similarity threshold is set too high.
    """
    from detector.scrfd import SCRFDDetector
    from pipeline.frame_source import VideoSource
    from recognition.arcface import ArcFaceEmbedder
    from recognition.matcher import GalleryMatcher
    from utils.gpu import resolve_device
    from utils.image_ops import align_face

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    db = CastDatabase(
        config.paths.resolve(config.paths.cast_db),
        config.paths.resolve(config.paths.cast_pickle),
    )
    vectors, owners, names = db.build_gallery(include_samples=True)
    if not names:
        print("  no actors registered - run 'cli.py register' first")
        db.close()
        return 1

    device = resolve_device(config.gpu.device, config.gpu.device_id)
    detector = SCRFDDetector(config, device)
    embedder = ArcFaceEmbedder(config, device)
    # Threshold 0 so every face reports its raw best score.
    matcher = GalleryMatcher(device, similarity_threshold=0.0, margin_threshold=0.0)
    matcher.set_gallery(vectors, owners, names)

    threshold = config.recognition.similarity_threshold
    best: dict[str, float] = {}
    hits: dict[str, int] = {}
    faces = 0

    source = VideoSource(video, target_long_side=config.video.analysis_long_side)
    print(f"  probing {video.name} against {len(names)} actors (threshold {threshold})")
    for index, frame in enumerate(source.frames()):
        if index % max(1, args.every):
            continue
        if index > args.max_frames:
            break
        detections = [d for d in detector.detect(frame.image) if d.landmarks is not None]
        if not detections:
            continue
        crops = [align_face(frame.image, d.landmarks, embedder.input_size) for d in detections]
        faces += len(crops)
        for result in matcher.match(embedder.embed_aligned(crops)):
            best[result.name] = max(best.get(result.name, 0.0), result.similarity)
            if result.similarity >= threshold:
                hits[result.name] = hits.get(result.name, 0) + 1
    source.close()

    print(f"  {faces} faces sampled\n")
    for name, score in sorted(best.items(), key=lambda kv: -kv[1]):
        verdict = "PRESENT" if score >= threshold else "not found"
        print(f"    {name:<24} best similarity {score:.3f}  {verdict}  "
              f"({hits.get(name, 0)} frames above threshold)")

    if not any(score >= threshold for score in best.values()):
        top = max(best.values()) if best else 0.0
        print(
            f"\n  No registered actor cleared {threshold:.2f} in this clip "
            f"(best was {top:.3f})."
        )
        print("  Either they are not in this video, or - if you expected a match -")
        print(f"  retry with:  --set recognition.similarity_threshold={max(0.25, top - 0.03):.2f}")

    detector.close()
    embedder.close()
    db.close()
    return 0


def cmd_cast(config: AppConfig, args: argparse.Namespace) -> int:
    db = CastDatabase(
        config.paths.resolve(config.paths.cast_db),
        config.paths.resolve(config.paths.cast_pickle),
    )
    try:
        if args.remove:
            actor_id = db.get_actor_id(args.remove)
            if actor_id is None:
                print(f"  no such actor: {args.remove}")
                return 1
            db.delete_actor(actor_id)
            db.export_pickle()
            print(f"  removed {args.remove}")
        if args.export_pickle:
            print(f"  written {db.export_pickle(Path(args.export_pickle))}")
        if args.import_pickle:
            print(f"  imported {db.import_pickle(Path(args.import_pickle))} actors")
        if args.list or not (args.remove or args.export_pickle or args.import_pickle):
            actors = db.list_actors()
            if not actors:
                print("  (no actors registered)")
            for actor in actors:
                print(
                    f"  #{actor.id:<3} {actor.name:<28} {actor.num_images:>3} photos  "
                    f"dim={actor.centroid.shape[0] if actor.centroid.size else 0}"
                )
            stats = db.stats()
            print(f"\n  {stats['actors']} actors, {stats['embeddings']} embeddings")
    finally:
        db.close()
    return 0


def cmd_models(config: AppConfig, args: argparse.Namespace) -> int:
    weights = config.paths.resolve(config.paths.weights_dir)
    print(f"  weights directory: {weights}\n")
    for spec in list_models():
        path = weights / spec.filename
        status = "cached" if path.exists() else "not downloaded"
        print(f"  {spec.key:<24} {spec.task:<12} {spec.filename:<28} [{status}]")
        print(f"      {spec.description}")
    return 0


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = AppConfig.load(getattr(args, "config", None))
    config = apply_overrides(config, getattr(args, "set", []) or [])
    log_level = getattr(args, "log_level", None)
    if log_level:
        config.logging.level = log_level

    setup_logging(
        config.logging.level,
        config.paths.resolve(config.paths.log_dir),
        config.logging.to_file,
    )
    log = get_logger("cli")
    log.info("Device: %s", get_device_info(config.gpu.device_id))

    handlers = {
        "register": cmd_register,
        "analyze": cmd_analyze,
        "probe": cmd_probe,
        "cast": cmd_cast,
        "models": cmd_models,
    }
    return handlers[args.command](config, args)


if __name__ == "__main__":
    raise SystemExit(main())
