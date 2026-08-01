"""One-off script (L7 7a): dump the golden video/edit-plan pair's live DB
state + ffprobe metadata of the rendered output into tests/fixtures/.

Not part of the pipeline itself — a bootstrap utility, same spirit as the
`docs/session-logs/l4_full_export_97199656.json` precedent export from
earlier in this project's history, scoped down to exactly what
`tests/integration/test_e2e_golden.py` needs.

Usage: python scripts/_bootstrap_l7_fixture.py
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import asyncpg

from shared.config import settings

VIDEO_ID = "97199656-d176-46aa-88b4-026670be4576"
EDIT_PLAN_ID = "b276be9d-c803-4cba-b86e-e3da624c479f"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUT_MP4 = Path(__file__).resolve().parent.parent / "out.mp4"


def _row_to_dict(row) -> dict:
    d = dict(row)
    for k, v in list(d.items()):
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
        else:
            try:
                json.dumps(v)
            except TypeError:
                d[k] = str(v)
    return d


async def main() -> None:
    conn = await asyncpg.connect(dsn=settings.asyncpg_url)
    try:
        video = await conn.fetchrow("SELECT * FROM videos WHERE id = $1::uuid", VIDEO_ID)
        storylines = await conn.fetch(
            "SELECT * FROM storylines WHERE video_id = $1::uuid ORDER BY version", VIDEO_ID
        )
        scenes = await conn.fetch(
            "SELECT * FROM scenes WHERE video_id = $1::uuid ORDER BY start_time", VIDEO_ID
        )
        edit_plan = await conn.fetchrow(
            "SELECT * FROM edit_plans WHERE id = $1::uuid", EDIT_PLAN_ID
        )
        qa_report = await conn.fetchrow(
            "SELECT * FROM qa_reports WHERE edit_plan_id = $1::uuid ORDER BY created_at DESC LIMIT 1",
            EDIT_PLAN_ID,
        )
        cut_items = await conn.fetch(
            "SELECT * FROM cut_list_items WHERE edit_plan_id = $1::uuid ORDER BY sequence_index",
            EDIT_PLAN_ID,
        )
    finally:
        await conn.close()

    fixture = {
        "video_id": VIDEO_ID,
        "edit_plan_id": EDIT_PLAN_ID,
        "video_meta": _row_to_dict(video) if video else None,
        "storylines": [_row_to_dict(r) for r in storylines],
        "scenes": [_row_to_dict(r) for r in scenes],
        "scene_count": len(scenes),
        "edit_plan": _row_to_dict(edit_plan) if edit_plan else None,
        "cut_list_items": [_row_to_dict(r) for r in cut_items],
        "qa_report": _row_to_dict(qa_report) if qa_report else None,
    }

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURES_DIR / "golden_97199656.json"
    fixture_path.write_text(json.dumps(fixture, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {fixture_path} ({len(scenes)} scenes, edit_plan found={edit_plan is not None})")

    # ffprobe metadata of the rendered output — expected-output baseline.
    if OUT_MP4.exists():
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_format", "-show_streams",
                "-of", "json", str(OUT_MP4),
            ],
            capture_output=True, text=True, timeout=60,
        )
        probe = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout else {"error": proc.stderr}
        probe_path = FIXTURES_DIR / "golden_97199656_ffprobe.json"
        probe_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
        print(f"Wrote {probe_path}")
    else:
        print(f"WARNING: {OUT_MP4} not found — skipping ffprobe snapshot")


if __name__ == "__main__":
    asyncio.run(main())
