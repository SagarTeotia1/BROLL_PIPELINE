"""Exporters: JSON, CSV and a Gantt-style emotion timeline PNG.

All three take the same :class:`timeline.events.TimelineDocument`, so an export never
re-derives anything - what you see in the GUI is what lands on disk.

The PNG uses matplotlib with the ``Agg`` backend so it can be rendered from a worker
thread without a display.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from emotion.hsemotion import EMOTION_COLORS  # noqa: E402
from timeline.events import TimelineDocument  # noqa: E402
from utils.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)

CSV_COLUMNS = (
    "actor",
    "actor_id",
    "emotion",
    "start",
    "end",
    "duration",
    "confidence",
    "similarity",
    "start_frame",
    "end_frame",
    "samples",
    "track_ids",
)


def export_json(document: TimelineDocument, path: Path, verbose: bool = False) -> Path:
    """Write the timeline as JSON.

    ``verbose=False`` (default) writes the result: header, analysis time, cast and
    events. ``verbose=True`` appends the profiler dump and per-event debug fields.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(document.as_dict(verbose), fh, indent=2, ensure_ascii=False)
    log.info("JSON exported: %s (%d events)", path, len(document.events))
    return path


def export_csv(document: TimelineDocument, path: Path) -> Path:
    """Write one row per event, sorted by start time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for ev in sorted(document.events, key=lambda e: (e.start, e.actor)):
            writer.writerow(
                {
                    "actor": ev.actor,
                    "actor_id": ev.actor_id,
                    "emotion": ev.emotion,
                    "start": f"{ev.start:.3f}",
                    "end": f"{ev.end:.3f}",
                    "duration": f"{ev.duration:.3f}",
                    "confidence": f"{ev.confidence:.4f}",
                    "similarity": f"{ev.similarity:.4f}",
                    "start_frame": ev.start_frame,
                    "end_frame": ev.end_frame,
                    "samples": ev.samples,
                    "track_ids": "|".join(str(t) for t in sorted(set(ev.track_ids))),
                }
            )
    log.info("CSV exported: %s", path)
    return path


def export_timeline_png(
    document: TimelineDocument,
    path: Path,
    width_inches: float = 16.0,
    row_height: float = 0.72,
    dpi: int = 130,
    title: Optional[str] = None,
) -> Path:
    """Render a Gantt-style emotion timeline, one lane per actor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    actors = [a.name for a in document.actors] or sorted({e.actor for e in document.events})
    if not actors:
        actors = ["(no actors)"]
    duration = document.video.duration or (
        max((e.end for e in document.events), default=1.0)
    )

    fig_height = max(2.6, 1.7 + row_height * len(actors))
    fig, ax = plt.subplots(figsize=(width_inches, fig_height), dpi=dpi)
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#171a21")

    lane = {name: i for i, name in enumerate(actors)}
    used_emotions: List[str] = []

    for ev in document.events:
        if ev.actor not in lane:
            continue
        y = lane[ev.actor]
        color = EMOTION_COLORS.get(ev.emotion, "#8a8f98")
        if ev.emotion not in used_emotions:
            used_emotions.append(ev.emotion)
        ax.barh(
            y=y,
            width=max(ev.duration, duration * 0.0015),
            left=ev.start,
            height=0.58,
            color=color,
            edgecolor="#0d0f13",
            linewidth=0.6,
            alpha=0.92,
        )
        # Label spans that are wide enough to hold text.
        if ev.duration > duration * 0.045:
            ax.text(
                ev.start + ev.duration / 2, y, ev.emotion,
                ha="center", va="center", fontsize=7.5, color="#0f1115", weight="bold",
            )

    ax.set_yticks(range(len(actors)))
    ax.set_yticklabels(actors, color="#e6e8eb", fontsize=10)
    ax.set_ylim(-0.6, len(actors) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, max(duration, 1.0))
    ax.set_xlabel("Time (seconds)", color="#c3c8d0", fontsize=10)
    ax.tick_params(axis="x", colors="#9aa1ab")
    for spine in ax.spines.values():
        spine.set_color("#2a2f39")
    ax.grid(axis="x", color="#242833", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)

    heading = title or f"Emotion timeline - {Path(document.video.path).name or 'video'}"
    subtitle = (
        f"{duration:.1f}s @ {document.video.fps:.2f} fps  |  "
        f"analysed at {document.video.processed_fps:.2f} fps  |  "
        f"{len(document.events)} events"
    )
    ax.set_title(f"{heading}\n{subtitle}", color="#f2f4f7", fontsize=12, pad=14)

    # Reserve a strip at the bottom of the *figure* for the legend. Anchoring it to the
    # axes instead makes it collide with the x-axis label whenever there are few lanes,
    # because the offset is a fraction of a short axes box.
    legend_band = 0.0
    if used_emotions:
        handles = [
            Patch(facecolor=EMOTION_COLORS.get(e, "#8a8f98"), edgecolor="#0d0f13", label=e)
            for e in sorted(used_emotions)
        ]
        legend_band = min(0.22, 0.42 / fig_height + 0.02)
        legend = fig.legend(
            handles=handles, loc="lower center",
            ncol=min(len(handles), 8), frameon=False, fontsize=9,
        )
        for text in legend.get_texts():
            text.set_color("#d5d9e0")

    fig.tight_layout(rect=(0.0, legend_band, 1.0, 1.0))
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Timeline PNG exported: %s", path)
    return path


def export_emotion_summary_png(document: TimelineDocument, path: Path, dpi: int = 130) -> Path:
    """Render a stacked bar of total seconds per emotion per actor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    totals: Dict[str, Dict[str, float]] = {}
    for ev in document.events:
        totals.setdefault(ev.actor, {}).setdefault(ev.emotion, 0.0)
        totals[ev.actor][ev.emotion] += ev.duration

    actors = list(totals.keys()) or ["(no actors)"]
    emotions = sorted({e for per in totals.values() for e in per})

    fig, ax = plt.subplots(figsize=(10, max(2.4, 0.7 * len(actors) + 1.6)), dpi=dpi)
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#171a21")

    left = {a: 0.0 for a in actors}
    for emotion in emotions:
        widths = [totals.get(a, {}).get(emotion, 0.0) for a in actors]
        ax.barh(
            actors, widths, left=[left[a] for a in actors],
            color=EMOTION_COLORS.get(emotion, "#8a8f98"), label=emotion,
            edgecolor="#0d0f13", linewidth=0.6,
        )
        for a, w in zip(actors, widths):
            left[a] += w

    ax.set_xlabel("Screen time (seconds)", color="#c3c8d0")
    ax.tick_params(colors="#9aa1ab")
    ax.set_yticks(range(len(actors)))
    ax.set_yticklabels(actors, color="#e6e8eb")
    for spine in ax.spines.values():
        spine.set_color("#2a2f39")
    ax.grid(axis="x", color="#242833", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title("Emotion distribution per actor", color="#f2f4f7", fontsize=12, pad=12)
    legend_band = 0.0
    if emotions:
        legend_band = min(0.22, 0.42 / max(fig.get_figheight(), 1.0) + 0.02)
        legend = fig.legend(
            loc="lower center", ncol=min(len(emotions), 8), frameon=False, fontsize=9
        )
        for text in legend.get_texts():
            text.set_color("#d5d9e0")

    fig.tight_layout(rect=(0.0, legend_band, 1.0, 1.0))
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log.info("Emotion summary PNG exported: %s", path)
    return path


def export_all(
    document: TimelineDocument,
    output_dir: Path,
    stem: str,
    verbose_json: bool = False,
) -> Dict[str, Path]:
    """Write JSON + CSV + both PNGs into ``output_dir`` and return the paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": export_json(
            document, output_dir / f"{stem}_timeline.json", verbose=verbose_json
        ),
        "csv": export_csv(document, output_dir / f"{stem}_timeline.csv"),
        "png": export_timeline_png(document, output_dir / f"{stem}_timeline.png"),
        "summary_png": export_emotion_summary_png(
            document, output_dir / f"{stem}_emotions.png"
        ),
    }
    return paths


__all__ = [
    "export_json",
    "export_csv",
    "export_timeline_png",
    "export_emotion_summary_png",
    "export_all",
    "CSV_COLUMNS",
]
