"""ASH-FS Knowledge Base Builder — fast edition.

Keyframes: single cv2 sequential pass (grab-skip), not per-frame ffmpeg.
Clips:     parallel ffmpeg workers + GPU nvenc (auto-fallback to libx264).

Usage:
    python build_knowledge_base.py --manifest output/manifest_l4.json \
                                   --video    output/podcast.mp4       \
                                   --config   config.yaml              \
                                   --output   output/knowledge_base_l4

Output layout:
    knowledge_base/
        knowledge_base.json
        knowledge_base_flat.jsonl
        chunks/
            chunk_00000/
                clip.mp4
                keyframe_0000125.jpg
                ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import yaml


# ---------------------------------------------------------------------------
# GPU / ffmpeg helpers
# ---------------------------------------------------------------------------

def _has_nvenc() -> bool:
    """Return True if ffmpeg on PATH supports h264_nvenc."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


_NVENC_AVAILABLE: bool | None = None


def _clip_worker(args: tuple) -> tuple[int, bool]:
    """Extract one clip. Runs in a thread pool."""
    global _NVENC_AVAILABLE
    i, video, start_s, end_s, out_path, use_gpu = args

    if Path(out_path).exists():
        return i, True

    duration = end_s - start_s

    if use_gpu:
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-ss", f"{start_s:.6f}",
            "-t",  f"{duration:.6f}",
            "-i", video,
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.6f}",
            "-t",  f"{duration:.6f}",
            "-i", video,
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-threads", "2",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # GPU failed — retry CPU
        if use_gpu:
            return _clip_worker((i, video, start_s, end_s, out_path, False))
        print(f"  [WARN] clip {i} ({start_s:.1f}-{end_s:.1f}s): {r.stderr.strip()[-100:]}")
        return i, False
    return i, True


# ---------------------------------------------------------------------------
# Fast single-pass keyframe extractor
# ---------------------------------------------------------------------------

def _extract_keyframes_single_pass(
    video_path: str,
    frame_to_imgpath: dict[int, Path],
    jpeg_quality: int = 95,
) -> int:
    """Extract all keyframes in ONE sequential video pass using grab()-skip.

    Far faster than per-frame ffmpeg calls: opens the video once,
    uses grab() (no decode) to skip, read() only at target frames.

    Returns count of successfully saved frames.
    """
    sorted_indices = sorted(frame_to_imgpath.keys())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path!r}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    pos = 0
    t0 = time.time()
    n = len(sorted_indices)

    for k, target in enumerate(sorted_indices):
        out_path = frame_to_imgpath[target]
        if out_path.exists():
            saved += 1
            continue

        if target >= total_frames:
            continue

        # Skip forward with grab() — no decode, very fast
        while pos < target:
            cap.grab()
            pos += 1

        ok, frame = cap.read()
        pos += 1
        if not ok or frame is None:
            continue

        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        saved += 1

        if (k + 1) % 100 == 0 or (k + 1) == n:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed if elapsed > 0 else 1
            eta = (n - k - 1) / rate
            print(f"  keyframes {k+1}/{n}  ({rate:.0f}/s  ETA {eta:.0f}s)")

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _seconds_to_hms(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def _video_id(video_path: str) -> str:
    stem = Path(video_path).stem
    digest = hashlib.md5(Path(video_path).resolve().as_posix().encode()).hexdigest()[:8]
    return f"{stem}_{digest}"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build(
    manifest_path: str,
    video_path: str,
    config_path: str,
    output_dir: str,
    extract_clips: bool = True,
    extract_keyframes: bool = True,
    jpeg_quality: int = 95,
    clip_workers: int = 0,      # 0 = auto (cpu_count // 2)
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    chunk_size: int = cfg["hierarchy"]["chunk_size_shots"]
    shots = manifest["shot_boundaries"]
    fps: float = shots[0]["fps"] if shots else 25.0

    vid_id = _video_id(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    role_map: dict[int, str] = {
        idx: role
        for idx, role in zip(
            manifest["selected_frame_indices"],
            manifest["frame_roles"],
        )
    }
    selected_set: set[int] = set(manifest["selected_frame_indices"])

    total_shots = len(shots)
    n_chunks = (total_shots + chunk_size - 1) // chunk_size

    use_gpu = _has_nvenc()
    workers = clip_workers or max(1, multiprocessing.cpu_count() // 2)

    print(f"Video     : {video_path}")
    print(f"Video ID  : {vid_id}")
    print(f"Shots     : {total_shots}  |  Chunks: {n_chunks}  |  Keyframes: {manifest['total_selected_frames']}")
    print(f"GPU nvenc : {'YES' if use_gpu else 'NO (libx264 fallback)'}  |  Clip workers: {workers}")
    print(f"Output    : {out}\n")

    # ------------------------------------------------------------------
    # Build chunk metadata + collect frame→path mappings
    # ------------------------------------------------------------------
    chunks_meta: list[dict] = []
    frame_to_imgpath: dict[int, Path] = {}

    for c_idx in range(n_chunks):
        shot_start = c_idx * chunk_size
        shot_end   = min(shot_start + chunk_size, total_shots)
        chunk_shots = shots[shot_start:shot_end]

        c_start_frame = chunk_shots[0]["start_frame"]
        c_end_frame   = chunk_shots[-1]["end_frame"]
        c_start_s     = c_start_frame / fps
        c_end_s       = (c_end_frame + 1) / fps
        c_duration_s  = c_end_s - c_start_s

        chunk_id  = f"{vid_id}_chunk_{c_idx:05d}"
        chunk_dir = out / "chunks" / f"chunk_{c_idx:05d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        shots_meta: list[dict] = []
        for s_local, shot in enumerate(chunk_shots):
            s_global = shot_start + s_local
            shots_meta.append({
                "shot_id":       f"{vid_id}_shot_{s_global:05d}",
                "shot_index":    s_global,
                "start_frame":   shot["start_frame"],
                "end_frame":     shot["end_frame"],
                "start_time_s":  round(shot["start_frame"] / fps, 3),
                "end_time_s":    round((shot["end_frame"] + 1) / fps, 3),
                "duration_s":    round(shot["duration_s"], 3),
                "start_tc":      _seconds_to_hms(shot["start_frame"] / fps),
                "end_tc":        _seconds_to_hms((shot["end_frame"] + 1) / fps),
            })

        kf_meta: list[dict] = []
        for frame_idx in manifest["selected_frame_indices"]:
            if not (c_start_frame <= frame_idx <= c_end_frame):
                continue

            owning_shot_local = None
            for s_local, shot in enumerate(chunk_shots):
                if shot["start_frame"] <= frame_idx <= shot["end_frame"]:
                    owning_shot_local = s_local
                    break

            ts_s = frame_idx / fps
            img_filename = f"keyframe_{frame_idx:07d}.jpg"
            img_path = chunk_dir / img_filename
            img_rel  = str(img_path.relative_to(out.parent))

            frame_to_imgpath[frame_idx] = img_path

            kf_meta.append({
                "keyframe_id":         f"{vid_id}_frame_{frame_idx:07d}",
                "frame_index":         frame_idx,
                "timestamp_s":         round(ts_s, 3),
                "timecode":            _seconds_to_hms(ts_s),
                "role":                role_map.get(frame_idx, "unknown"),
                "shot_index_global":   shot_start + (owning_shot_local or 0),
                "shot_index_in_chunk": owning_shot_local,
                "image_path":          img_rel,
            })

        clip_rel = str((chunk_dir / "clip.mp4").relative_to(out.parent))
        kf_tcs = ", ".join(kf["timecode"] for kf in kf_meta)
        context = (
            f"Chunk {c_idx} of {n_chunks} in video '{vid_id}'. "
            f"Time range: {_seconds_to_hms(c_start_s)} to {_seconds_to_hms(c_end_s)} "
            f"({c_duration_s:.1f}s). "
            f"Contains {len(chunk_shots)} shot(s) and {len(kf_meta)} keyframe(s). "
            f"Keyframe timecodes: [{kf_tcs}]."
        )

        chunks_meta.append({
            "chunk_id":       chunk_id,
            "chunk_index":    c_idx,
            "start_frame":    c_start_frame,
            "end_frame":      c_end_frame,
            "start_time_s":   round(c_start_s, 3),
            "end_time_s":     round(c_end_s, 3),
            "duration_s":     round(c_duration_s, 3),
            "start_tc":       _seconds_to_hms(c_start_s),
            "end_tc":         _seconds_to_hms(c_end_s),
            "shot_count":     len(chunk_shots),
            "keyframe_count": len(kf_meta),
            "shots":          shots_meta,
            "keyframes":      kf_meta,
            "clip_path":      clip_rel,
            "context":        context,
        })

    # ------------------------------------------------------------------
    # Extract keyframes — single cv2 pass (fast)
    # ------------------------------------------------------------------
    if extract_keyframes:
        total_kf = len(frame_to_imgpath)
        print(f"Extracting {total_kf} keyframes (single-pass cv2 grab-skip) …")
        t0 = time.time()
        saved = _extract_keyframes_single_pass(video_path, frame_to_imgpath, jpeg_quality)
        print(f"  Done: {saved}/{total_kf} saved in {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Extract chunk clips — parallel ffmpeg workers
    # ------------------------------------------------------------------
    if extract_clips:
        print(f"Extracting {n_chunks} chunk clips ({workers} parallel workers, GPU={'yes' if use_gpu else 'no'}) …")
        t0 = time.time()

        tasks = [
            (
                c["chunk_index"],
                video_path,
                c["start_time_s"],
                c["end_time_s"],
                str(out / "chunks" / f"chunk_{c['chunk_index']:05d}" / "clip.mp4"),
                use_gpu,
            )
            for c in chunks_meta
        ]

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_clip_worker, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                idx, ok = fut.result()
                done += 1
                if done % 20 == 0 or done == n_chunks:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 1
                    eta = (n_chunks - done) / rate
                    print(f"  clips {done}/{n_chunks}  ({rate:.1f}/s  ETA {eta:.0f}s)")

        print(f"  Done: {n_chunks} clips in {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Write JSON / JSONL
    # ------------------------------------------------------------------
    total_dur = sum(s["duration_s"] for s in shots)
    kb = {
        "schema_version": "1.0",
        "video_id":        vid_id,
        "video_path":      video_path,
        "total_duration_s": round(total_dur, 3),
        "total_duration_hms": _seconds_to_hms(total_dur),
        "fps":             fps,
        "total_shots":     total_shots,
        "total_chunks":    n_chunks,
        "total_keyframes": manifest["total_selected_frames"],
        "chunk_size_shots": chunk_size,
        "pipeline_stats": {
            "candidate_frames":  manifest["total_candidate_frames"],
            "compression_ratio": manifest["compression_stats"]["compression_ratio"],
            "nodes_expanded":    manifest["compression_stats"]["nodes_expanded"],
            "nodes_summarized":  manifest["compression_stats"]["nodes_summarized"],
        },
        "chunks": chunks_meta,
    }

    kb_json = out / "knowledge_base.json"
    kb_json.write_text(json.dumps(kb, indent=2))
    print(f"knowledge_base.json       -> {kb_json}")

    kb_jsonl = out / "knowledge_base_flat.jsonl"
    with open(kb_jsonl, "w") as fh:
        for chunk in chunks_meta:
            record = {
                "id":             chunk["chunk_id"],
                "video_id":       vid_id,
                "video_path":     video_path,
                "chunk_index":    chunk["chunk_index"],
                "start_time_s":   chunk["start_time_s"],
                "end_time_s":     chunk["end_time_s"],
                "duration_s":     chunk["duration_s"],
                "start_tc":       chunk["start_tc"],
                "end_tc":         chunk["end_tc"],
                "shot_count":     chunk["shot_count"],
                "keyframe_count": chunk["keyframe_count"],
                "keyframes":      [
                    {
                        "keyframe_id": kf["keyframe_id"],
                        "frame_index": kf["frame_index"],
                        "timestamp_s": kf["timestamp_s"],
                        "timecode":    kf["timecode"],
                        "role":        kf["role"],
                        "image_path":  kf["image_path"],
                        "shot_index":  kf["shot_index_global"],
                    }
                    for kf in chunk["keyframes"]
                ],
                "clip_path": chunk.get("clip_path", ""),
                "shots":     chunk["shots"],
                "context":   chunk["context"],
            }
            fh.write(json.dumps(record) + "\n")
    print(f"knowledge_base_flat.jsonl -> {kb_jsonl}")

    print(f"\n{'='*50}")
    print("  Knowledge Base Summary")
    print(f"{'='*50}")
    print(f"  Duration    : {_seconds_to_hms(total_dur)}")
    print(f"  Shots       : {total_shots}")
    print(f"  Chunks      : {n_chunks}")
    print(f"  Keyframes   : {manifest['total_selected_frames']}")
    print(f"  Avg KF/chunk: {manifest['total_selected_frames']/n_chunks:.1f}")
    print(f"  Output      : {out}")

    return kb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="build_knowledge_base.py")
    p.add_argument("--manifest",   default="output/manifest.json")
    p.add_argument("--video",      default=None)
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--output",     default="output/knowledge_base")
    p.add_argument("--no-clips",   dest="extract_clips",     action="store_false", default=True)
    p.add_argument("--no-keyframes", dest="extract_keyframes", action="store_false", default=True)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--clip-workers", type=int, default=0, help="Parallel ffmpeg workers (0=auto)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not Path(args.manifest).exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.video:
        video_path = args.video
    else:
        manifest = json.loads(Path(args.manifest).read_text())
        video_path = manifest["video_path"]

    if not Path(video_path).exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    build(
        manifest_path=args.manifest,
        video_path=video_path,
        config_path=args.config,
        output_dir=args.output,
        extract_clips=args.extract_clips,
        extract_keyframes=args.extract_keyframes,
        jpeg_quality=args.jpeg_quality,
        clip_workers=args.clip_workers,
    )
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
