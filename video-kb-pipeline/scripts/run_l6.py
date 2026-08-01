"""Local CLI runner for Level-6 action agents (Editing Director, Color
Grading, Caption/Text Overlay, Compositing) + render.

No Modal wiring exists for L6 yet — it's user-triggered per finalized
edit_plan, not part of the automatic L1-L5 batch pipeline (see CLAUDE.md's
own STATUS note on this). Needs a local ffmpeg + the source video file —
this script downloads the video from its original URL if a local copy
isn't already at --video-path, then runs the full L6 pass.

Usage:
    python scripts/run_l6.py --edit-plan-id <id> --video-url <source-url> \\
        --output out.mp4
    # or, if you already have the video locally:
    python scripts/run_l6.py --edit-plan-id <id> --video-path input_video.mp4 \\
        --output out.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from knowledge_base.postgres.client import get_pool  # noqa: E402
from knowledge_base.postgres.queries import get_edit_plan, get_video  # noqa: E402
from pipeline.level6.updater import run_level6  # noqa: E402


def _download_video(url: str) -> str:
    suffix = ".mp4" if ".mp4" in url.lower() else ".video"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; video-kb-pipeline/1.0)"}
    )
    print(f"[l6] Downloading source video from {url} ...")
    with urllib.request.urlopen(req) as resp, open(tmp.name, "wb") as f:
        f.write(resp.read())
    print(f"[l6] Downloaded to {tmp.name}")
    return tmp.name


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Level-6 action agents + render.")
    parser.add_argument("--edit-plan-id", required=True)
    parser.add_argument("--video-path", default=None,
                         help="Local path to the source video, if already downloaded.")
    parser.add_argument("--video-url", default=None,
                         help="Source video URL to download if --video-path not given.")
    parser.add_argument("--output", required=True, help="Output rendered file path.")
    args = parser.parse_args()

    pool = await get_pool()

    plan = await get_edit_plan(pool, args.edit_plan_id)
    if plan is None:
        print(f"[l6] No edit_plans row for id={args.edit_plan_id}", file=sys.stderr)
        sys.exit(1)

    video_path = args.video_path
    if video_path is None:
        video_url = args.video_url
        if video_url is None:
            video = await get_video(pool, plan.video_id)
            if video is None:
                print(f"[l6] No videos row for id={plan.video_id}", file=sys.stderr)
                sys.exit(1)
            video_url = video.path  # original source URL, per VideoMeta.path
        video_path = _download_video(video_url)

    result = await run_level6(pool, args.edit_plan_id, video_path, args.output)
    print(json.dumps(result, indent=2, default=str))

    if not result.get("ok", True):
        print(f"\n[l6] completed with issues — check 'qa_status' above", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
