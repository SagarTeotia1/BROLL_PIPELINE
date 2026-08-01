"""Local CLI runner for Level-5 Planning.

No Modal wiring exists for L5 yet (it's CPU-only, pure Postgres + OpenRouter
API calls — no GPU needed, so it runs directly here against the same
DATABASE_URL/OPENROUTER_API_KEY in .env, same as any other local script).

Usage:
    python scripts/run_l5.py --video-id <id> --prompt "..." \\
        --duration-s 3600 --platform full_cut
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from knowledge_base.postgres.client import get_pool  # noqa: E402
from pipeline.level5.planner_runner import run_level5_planning  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Level-5 planning for one video.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--prompt", required=True, help="User editing intent, e.g. "
                         "'Edit this to a 60 min podcast, improve color grading, "
                         "zoom in/out on the important/viral sequences.'")
    parser.add_argument("--duration-s", type=float, default=None,
                         help="Target duration in seconds, e.g. 3600 for 60 min.")
    parser.add_argument("--platform", default=None,
                         help="reel | full_cut | youtube | ...")
    parser.add_argument("--must-include", nargs="*", default=None)
    parser.add_argument("--must-exclude", nargs="*", default=None)
    parser.add_argument("--pacing", default=None, help="Pacing preference, e.g. 'fast'.")
    args = parser.parse_args()

    pool = await get_pool()
    result = await run_level5_planning(
        pool,
        video_id=args.video_id,
        user_prompt=args.prompt,
        target_duration_s=args.duration_s,
        platform=args.platform,
        must_include=args.must_include,
        must_exclude=args.must_exclude,
        pacing_preference=args.pacing,
    )
    print(json.dumps(result, indent=2))

    if result.get("ok"):
        print(
            f"\n[l5] edit_plan_id: {result['edit_plan_id']}\n"
            f"  Next: python scripts/run_l6.py --edit-plan-id {result['edit_plan_id']} "
            f"--video-url <source-video-url> --output out.mp4"
        )
    else:
        print(f"\n[l5] FAILED: {result.get('reason')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
