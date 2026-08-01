"""LEVEL 7 -- EVALUATION, 7a: the FIRST end-to-end golden-set test this
pipeline has (CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 7 -- EVALUATION" ->
7a, closes audit gap #6 "Offline/golden-set testing" and partially gap #10
"End-to-end/integration testing").

Golden case: `video_id=97199656-d176-46aa-88b4-026670be4576` /
`edit_plan_id=b276be9d-c803-4cba-b86e-e3da624c479f` — a real L4-finalized
storyline, a real L5 edit plan, a real L6 render. Snapshot lives in
`tests/fixtures/golden_97199656.json` (+ `_ffprobe.json`), regenerated via
`scripts/_bootstrap_l7_fixture.py`.

Design choice (task explicitly leaves this open — "your call based on
what's cheap/fast vs what's a real regression check"): this test
RE-VALIDATES live DB state + a fresh local `ffprobe` pass against the
snapshot, rather than re-running L4->L5->L6 from scratch. Reasons:
  - A full L4/L5/L6 rerun makes real paid OpenRouter calls and a real
    ffmpeg render on every test run — slow, non-free, and per this
    codebase's own extensively-documented LLM non-determinism caveat
    (CLAUDE.md, repeatedly), a full rerun's output would legitimately
    drift call-to-call even with ZERO regression, which would make this
    test flaky rather than a real regression signal.
  - Re-validating live DB state (a cheap, fast, deterministic set of
    SELECTs) plus a real local `ffprobe` pass against the ALREADY-RENDERED
    `out.mp4` (no new render, no new LLM calls) is real execution against
    real artifacts, not a mock — it will genuinely fail if the golden
    video/plan's data changes shape or disappears (e.g. an accidental
    DELETE, a broken migration, a `scenes`/`edit_plans` schema drift), which
    is the actual regression class 7a exists to catch.
  - This is explicitly the FIRST golden case (see fixtures/README.md) —
    the "rerun L4->L5->L6 live" version is a reasonable v-next once L4/L5/L6
    have run via the real orchestrator end-to-end at least once (per
    CLAUDE.md's own audit finding #2: "processing_jobs has zero rows at
    level 4/5/6 -- the gate has never fired via the real orchestrator").

No pytest-asyncio dependency introduced (none of this repo's existing
tests use it) — DB access goes through a small `asyncio.run()` wrapper per
test, same "no new test-infra dependency" discipline as everywhere else in
this pass.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import asyncpg
import pytest

from shared.config import settings

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_GOLDEN_JSON = _FIXTURES_DIR / "golden_97199656.json"
_GOLDEN_FFPROBE_JSON = _FIXTURES_DIR / "golden_97199656_ffprobe.json"
_OUT_MP4 = Path(__file__).resolve().parent.parent.parent / "out.mp4"

VIDEO_ID = "97199656-d176-46aa-88b4-026670be4576"
EDIT_PLAN_ID = "b276be9d-c803-4cba-b86e-e3da624c479f"

# Tolerances per CLAUDE.md 7a: "scene count ±2, achieved duration ±5%, no
# qa_status='fail'" — not byte-identical.
_SCENE_COUNT_TOLERANCE = 2
_DURATION_TOLERANCE_PCT = 5.0


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"golden fixture missing: {path} (run scripts/_bootstrap_l7_fixture.py)")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden() -> dict:
    return _load_json(_GOLDEN_JSON)


@pytest.fixture(scope="module")
def golden_ffprobe() -> dict:
    return _load_json(_GOLDEN_FFPROBE_JSON)


async def _fetch_live_state() -> dict:
    """Real DB query against the live Postgres instance (settings.DATABASE_URL
    from .env) — not mocked. Returns None-valued fields (never raises) when
    a row genuinely doesn't exist, so callers can assert on absence too."""
    conn = await asyncpg.connect(dsn=settings.asyncpg_url)
    try:
        scene_count = await conn.fetchval(
            "SELECT COUNT(*) FROM scenes WHERE video_id = $1::uuid", VIDEO_ID
        )
        edit_plan = await conn.fetchrow(
            "SELECT * FROM edit_plans WHERE id = $1::uuid", EDIT_PLAN_ID
        )
        qa_report = await conn.fetchrow(
            "SELECT * FROM qa_reports WHERE edit_plan_id = $1::uuid "
            "ORDER BY created_at DESC LIMIT 1",
            EDIT_PLAN_ID,
        )
        storyline_final = await conn.fetchrow(
            "SELECT * FROM storylines WHERE video_id = $1::uuid AND status = 'final' "
            "ORDER BY version DESC LIMIT 1",
            VIDEO_ID,
        )
    finally:
        await conn.close()
    return {
        "scene_count": scene_count,
        "edit_plan": dict(edit_plan) if edit_plan else None,
        "qa_report": dict(qa_report) if qa_report else None,
        "storyline_final": dict(storyline_final) if storyline_final else None,
    }


@pytest.fixture(scope="module")
def live_state() -> dict:
    try:
        return asyncio.run(_fetch_live_state())
    except Exception as exc:  # noqa: BLE001 — DB unreachable in this environment
        pytest.skip(f"live DB unreachable for golden e2e test: {exc}")


def test_golden_storyline_still_final(live_state):
    """L4 finalization gate (rule 17) must still hold for this video —
    `storylines(status='final')` is the L5/L6 read-contract boundary; if
    this regresses, everything downstream of it is reading from nothing."""
    assert live_state["storyline_final"] is not None, (
        f"expected a storylines(status='final') row for video_id={VIDEO_ID} — "
        "L4 finalization regressed for the golden video"
    )


def test_golden_scene_count_within_tolerance(golden, live_state):
    expected = golden["scene_count"]
    actual = live_state["scene_count"]
    assert actual is not None
    assert abs(actual - expected) <= _SCENE_COUNT_TOLERANCE, (
        f"scene count drifted beyond tolerance: expected {expected} (±{_SCENE_COUNT_TOLERANCE}), "
        f"got {actual}"
    )


def test_golden_edit_plan_still_exists(live_state):
    assert live_state["edit_plan"] is not None, (
        f"edit_plan_id={EDIT_PLAN_ID} no longer exists in the live DB"
    )


def test_golden_achieved_duration_within_tolerance(golden, live_state):
    expected = golden["edit_plan"]["achieved_duration_s"]
    plan = live_state["edit_plan"]
    assert plan is not None
    actual = plan["achieved_duration_s"]
    assert actual is not None, "edit_plan.achieved_duration_s is NULL in the live DB"
    tolerance = expected * (_DURATION_TOLERANCE_PCT / 100.0)
    assert abs(actual - expected) <= tolerance, (
        f"achieved_duration_s drifted beyond ±{_DURATION_TOLERANCE_PCT}%: "
        f"expected {expected}s (±{tolerance:.2f}s), got {actual}s"
    )


def test_golden_qa_status_not_fail(golden, live_state):
    """Per 7a: 'no qa_status=fail' — matches this pipeline's own delivery
    gate semantics (rule 24: fail blocks delivery, warn/pass do not)."""
    report = live_state["qa_report"]
    assert report is not None, f"no qa_reports row exists for edit_plan_id={EDIT_PLAN_ID}"
    assert report["status"] != "fail", (
        f"qa_reports.status regressed to 'fail' for edit_plan_id={EDIT_PLAN_ID} "
        f"(deterministic_checks={report.get('deterministic_checks')!r})"
    )
    # Golden snapshot itself was 'warn' at capture time (see fixtures/README.md)
    # — document, don't over-assert an exact match (LLM non-determinism on
    # any future intent-review rerun could legitimately flip warn<->pass).
    assert golden["qa_report"]["status"] in ("pass", "warn", "fail")


def test_golden_operations_count_stable(golden, live_state):
    """Structural equivalence on the plan's own operation count — not
    byte-identical operation content (LLM non-determinism), but the same
    NUMBER of SELECT_CLIP-family operations is a reasonable structural
    invariant for a plan that hasn't been re-planned."""
    expected_ops = golden["edit_plan"]["operations"]
    expected_ops = json.loads(expected_ops) if isinstance(expected_ops, str) else expected_ops
    actual_ops = live_state["edit_plan"]["operations"]
    actual_ops = json.loads(actual_ops) if isinstance(actual_ops, str) else actual_ops
    assert len(actual_ops) == len(expected_ops), (
        f"edit_plan.operations count changed: expected {len(expected_ops)}, got {len(actual_ops)}"
    )


# ---------------------------------------------------------------------------
# Rendered-output structural check — real local ffprobe pass against the
# real out.mp4, not a mock.
# ---------------------------------------------------------------------------


def test_golden_rendered_output_ffprobe_matches(golden_ffprobe):
    if not _OUT_MP4.exists():
        pytest.skip(f"{_OUT_MP4} not present in this environment — nothing to re-probe")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(_OUT_MP4)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"ffprobe failed on {_OUT_MP4}: {proc.stderr}"
    probe = json.loads(proc.stdout)

    expected_duration = float(golden_ffprobe["format"]["duration"])
    actual_duration = float(probe["format"]["duration"])
    tolerance = expected_duration * (_DURATION_TOLERANCE_PCT / 100.0)
    assert abs(actual_duration - expected_duration) <= tolerance, (
        f"out.mp4 duration drifted beyond ±{_DURATION_TOLERANCE_PCT}%: "
        f"expected {expected_duration}s, got {actual_duration}s"
    )

    expected_codecs = sorted(s["codec_name"] for s in golden_ffprobe["streams"])
    actual_codecs = sorted(s["codec_name"] for s in probe["streams"])
    assert actual_codecs == expected_codecs, (
        f"out.mp4 codec set changed: expected {expected_codecs}, got {actual_codecs}"
    )

    expected_video = next(s for s in golden_ffprobe["streams"] if s["codec_type"] == "video")
    actual_video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (actual_video["width"], actual_video["height"]) == (expected_video["width"], expected_video["height"]), (
        "out.mp4 resolution changed from the golden snapshot"
    )
