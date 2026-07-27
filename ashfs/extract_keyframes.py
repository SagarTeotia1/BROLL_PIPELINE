"""Extract keyframe images from video using ASH-FS manifest.

Usage:
    python extract_keyframes.py --video output/podcast.mp4 --manifest output/manifest_l4.json --out output/keyframes
"""
import argparse
import json
from pathlib import Path

import cv2


def extract(video_path: str, manifest_path: str, out_dir: str, max_dim: int = 720) -> None:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    frame_indices = manifest.get("selected_frame_indices", [])
    if not frame_indices:
        print("No selected_frame_indices in manifest.")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames @ {fps:.2f}fps")
    print(f"Extracting {len(frame_indices)} keyframes → {out}/")

    frame_set = sorted(set(frame_indices))
    pos = 0
    saved = 0

    for idx in frame_set:
        if idx >= total:
            continue
        # Seek forward using grab()
        while pos < idx:
            cap.grab()
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok or frame is None:
            continue

        # Resize to max_dim on longest side
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        ts = idx / fps
        mm, ss = divmod(int(ts), 60)
        hh, mm = divmod(mm, 60)
        fname = out / f"frame_{idx:07d}_{hh:02d}h{mm:02d}m{ss:02d}s.jpg"
        cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1

        if saved % 100 == 0:
            print(f"  {saved}/{len(frame_set)} saved...")

    cap.release()
    print(f"\nDone. {saved} keyframes saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="output/keyframes")
    parser.add_argument("--max-dim", type=int, default=720)
    args = parser.parse_args()
    extract(args.video, args.manifest, args.out, args.max_dim)
