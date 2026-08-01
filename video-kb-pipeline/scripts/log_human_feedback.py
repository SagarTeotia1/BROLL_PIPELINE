"""CLI entry point for logging holistic/qualitative human feedback
(PIPELINE ADDENDUM 3, LEVEL 8, item 8b).

Distinct from `scripts/log_l4_correction.py`: `correction_events` is
field-level ("this exact value was X, should be Y"); `human_feedback` is
for feedback that can't be pinned to one field — "this cut felt jarring",
"the pacing always feels too slow for this client's reels". `sentiment`
and `category` are closed enums (rule 28) so this is aggregatable later by
L9's `reward_signals` (not built by this script — L8 only produces the raw
signal).

Usage:
    python scripts/log_human_feedback.py \\
        --video-id <id> --sentiment negative --category pacing \\
        --text "the pacing always feels too slow for this client's reels" \\
        --edit-plan-id <edit-plan-uuid> --source client
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
from knowledge_base.postgres.queries import insert_human_feedback  # noqa: E402
from shared.types import HumanFeedbackRecord  # noqa: E402
from shared.utils import gen_id  # noqa: E402

_CATEGORIES = [
    "pacing", "color", "caption", "speaker_attribution",
    "narrative", "music_audio", "b_roll", "other",
]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log one holistic/qualitative human_feedback row."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sentiment", required=True, choices=["positive", "negative", "neutral"])
    parser.add_argument("--category", required=True, choices=_CATEGORIES)
    parser.add_argument("--text", required=True, dest="free_text")
    parser.add_argument("--edit-plan-id", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--rating", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--source", default="client", choices=["client", "internal_editor"])
    args = parser.parse_args()

    pool = await get_pool()

    feedback = HumanFeedbackRecord(
        id=gen_id(),
        video_id=args.video_id,
        sentiment=args.sentiment,
        category=args.category,
        free_text=args.free_text,
        edit_plan_id=args.edit_plan_id,
        scene_id=args.scene_id,
        rating=args.rating,
        source=args.source,
    )
    feedback_id = await insert_human_feedback(pool, feedback)

    print(json.dumps({
        "ok": True,
        "human_feedback_id": feedback_id,
        "video_id": args.video_id,
        "sentiment": args.sentiment,
        "category": args.category,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
