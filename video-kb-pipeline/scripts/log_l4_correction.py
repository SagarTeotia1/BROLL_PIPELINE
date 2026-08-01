"""CLI entry point for logging a human correction to an L4 `scenes`/
`storylines` field (PIPELINE ADDENDUM 3, LEVEL 8, item 8a).

The audit's single biggest finding: `pipeline/feedback/correction_logger.py`
already has real, correct `log_scene_correction`/`log_storyline_correction`
functions (write to `scene_overrides`/`storyline_overrides` AND
`correction_events`, per rule 25) — but they have zero callers anywhere in
the codebase. This script is the missing entry point, same CLI-first
pattern as `scripts/run_l5.py`/`scripts/run_l6.py` (this pipeline has no
admin UI yet).

This script looks up the entity's CURRENT value of --field first, so
`original_value` in the correction log is real (read from the DB), never
something the caller has to know/type themselves — the correction_logger
functions already do this internally via `get_scene_by_id`/
`get_storyline_by_id`, so this script's job is just argument parsing +
JSON-decoding --corrected-value + dispatch to the right function.

Usage:
    python scripts/log_l4_correction.py \\
        --video-id <id> --entity-type scene --entity-id <scene-uuid> \\
        --field emotional_arc --corrected-value '"A calmer arc than logged."' \\
        --reason "client felt the escalation read as too intense" \\
        --source client
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
from pipeline.feedback.correction_logger import (  # noqa: E402
    log_scene_correction,
    log_storyline_correction,
)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log a human correction to a scene/storyline field, "
        "writing both the override row L4/L5 read and a matching "
        "correction_events row."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--entity-type", required=True, choices=["scene", "storyline"])
    parser.add_argument("--entity-id", required=True, help="scenes.id or storylines.id")
    parser.add_argument("--field", required=True, help="e.g. 'summary', 'emotional_arc'")
    parser.add_argument(
        "--corrected-value",
        required=True,
        help="New value for --field, as JSON (e.g. '\"a string\"', '[\"P1\",\"P2\"]', '3.5'). "
        "Plain unquoted text is also accepted and treated as a raw string.",
    )
    parser.add_argument("--reason", default=None)
    parser.add_argument(
        "--source",
        default="internal_editor",
        choices=["client", "internal_editor", "qa_agent_flag"],
    )
    parser.add_argument("--created-by", default=None)
    args = parser.parse_args()

    try:
        corrected_value = json.loads(args.corrected_value)
    except json.JSONDecodeError:
        # Not valid JSON — treat the raw string as the value itself, so a
        # caller can pass --corrected-value "calmer arc" without having to
        # remember to quote it as JSON.
        corrected_value = args.corrected_value

    pool = await get_pool()

    try:
        if args.entity_type == "scene":
            override_id = await log_scene_correction(
                pool,
                video_id=args.video_id,
                scene_id=args.entity_id,
                field=args.field,
                new_value=corrected_value,
                correction_source=args.source,
                reason=args.reason,
                created_by=args.created_by,
            )
        else:
            override_id = await log_storyline_correction(
                pool,
                video_id=args.video_id,
                storyline_id=args.entity_id,
                field=args.field,
                new_value=corrected_value,
                correction_source=args.source,
                reason=args.reason,
                created_by=args.created_by,
            )
    except ValueError as exc:
        print(f"[log_l4_correction] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "ok": True,
        "entity_type": args.entity_type,
        "entity_id": args.entity_id,
        "field": args.field,
        "override_id": override_id,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
