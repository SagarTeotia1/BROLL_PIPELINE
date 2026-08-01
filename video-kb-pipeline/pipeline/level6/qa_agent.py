"""Level-6 QA/Validation Agent.

Implements CLAUDE.md "PIPELINE ADDENDUM 2" -> "1. QA / Validation Agent":
runs as the LAST step of `pipeline/level6/updater.py::run_level6`, after
Editing Director's render and every other L6 agent has finished. Split same
as every other L6 agent: deterministic checks first (cheap, no LLM), one
narrow LLM pass only for the one judgment call a formula can't make
(content-intent match).

Deterministic checks (no LLM, always run, pure functions over
already-rendered/already-computed data — rule 20 extended to QA):
  - `run_blackdetect` / `_parse_blackdetect_output` — FFmpeg `blackdetect`
    over the RENDERED output file (not the source — QA validates what was
    actually delivered).
  - `run_silencedetect` / `_parse_silencedetect_output` — FFmpeg
    `silencedetect` over the rendered output's audio track.
  - `run_loudness_check` / `_evaluate_loudness` — reuses
    `audio_sync.py::compute_loudnorm_measure_filter` +
    `audio_sync.py::parse_loudnorm_stats` (the exact same measurement
    filter/parser `render_direct`'s normalization step would use for a real
    two-pass `loudnorm`) rather than inventing a new loudness-measurement
    mechanism. IMPORTANT caveat, honestly flagged: `updater.py` currently
    calls `compute_loudnorm_filter(plan.platform)` with no `measured_stats`
    (single-pass dynamic mode — see that function's docstring), so there is
    no PERSISTED measure-pass output anywhere in the DB for QA to read back.
    "Reuse, don't recompute" is honored at the function level (QA calls the
    same `audio_sync` filter-string-builder and stats-parser, not a
    hand-rolled loudness regex) — but QA does run its own ffmpeg pass, this
    time against the DELIVERED render, to verify the actual output landed
    in the target loudness range. That is a different, legitimate check
    from pass-1 measurement (which measures the SOURCE before correction) —
    QA is verifying the correction actually worked on the real output, not
    re-deriving the correction itself.
  - `check_caption_drift` — pure string diff (no ffmpeg, no LLM): compares
    each already-built caption's delivered text (`caption_overlay.py`'s
    `run_caption_overlay` output, passed in by the caller — never
    re-fetched or re-run) against L3's `dialogue_subtitle` for that same
    time window, an INDEPENDENT source signal from whatever
    `caption_overlay.py` actually used (transcript_segments, with
    dialogue_subtitle only as ITS OWN fallback) — reuses
    `caption_overlay.py::_dialogue_subtitle_fallback` rather than
    duplicating the nearest-frame lookup.
  - `check_color_clipping` — pure function: reuses
    `color_grading_runner.py`'s own schema accessor (`_get_schema`) and
    clamp helper (`_clamp`) — the SAME bounds `run_color_grading` already
    clamped against at write time — to flag any `sequence_color_adjustments`
    row whose base+delta final value would exceed its schema.py bounds.
    Since `run_color_grading` already clamps before writing, this should
    normally find nothing; it exists as a genuine safety net (a future
    direct write path, a schema/bounds drift, or a manual DB edit).

One LLM pass (Groq, `settings.L6_QA_MODEL`, text-only, one call per
delivered plan — `run_intent_review`): compares `edit_plan.user_prompt`
against the real `scenes` rows actually selected into the delivered cut
(via `cut_list_items` -> op's `scene_id` -> `scenes`), flags a content-intent
mismatch. Closed-set/grounded input, same discipline as every other
reasoning stage in this pipeline. Reports pass/warn/fail + reasoning —
NEVER edits `edit_plans`/`cut_list_items`, never triggers a re-render (rule
24 — QA reports, it doesn't edit).

Aggregation (`run_qa_agent`): combines every deterministic check's own
pass/warn/fail signal with the LLM's verdict into one overall `status`,
written to `qa_reports`. Per CLAUDE.md's gate semantics: 'fail' blocks
delivery, 'warn' delivers with a flagged note, 'pass' delivers clean — but
`run_qa_agent`/`run_level6` never enforce that gate themselves (this module
only produces the status as DATA; whatever delivers the video to the client
is what actually honors the gate).

Threshold caveat, same epistemic honesty CLAUDE.md's own Addendum 2 preamble
applies to this whole agent ("a QA agent's checks and thresholds ... are
informed by what actually breaks on a real render, which doesn't exist
yet"): every numeric threshold below (`_BLACK_FRAME_FAIL_DURATION_S`,
`_SILENCE_FAIL_DURATION_S`, etc.) is a reasonable first-pass default, not
yet validated against a real client video's actual failure modes. Expect to
retune these once real renders exist to learn from.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import logging
import subprocess

import asyncpg

from knowledge_base.postgres.queries import (
    get_cut_list_items_for_edit_plan,
    get_edit_plan,
    get_frame_analyses_with_timestamps_for_video,
    get_persons_for_video,
    get_scenes_for_video,
    get_sequence_color_adjustments_for_edit_plan,
    insert_qa_report,
)
# L7 7b -- appended, not merged into the block above (this module's imports
# are otherwise untouched — see the block above for the pre-existing L6 QA
# Agent's own imports).
from knowledge_base.postgres.queries import insert_evaluation_score
from pipeline.level6.audio_sync import (
    compute_loudnorm_measure_filter,
    parse_loudnorm_stats,
)
from pipeline.level6.audio_sync import _target_lufs as _audio_target_lufs
from pipeline.level6.caption_overlay import (
    _dialogue_subtitle_fallback as _source_dialogue_subtitle,
)
from pipeline.level6.color_grading_runner import _clamp as _color_clamp
from pipeline.level6.color_grading_runner import _get_schema as _color_schema
from shared.config import settings
from shared.types import (
    CutListItemRecord,
    EditPlanRecord,
    EvaluationScoreRecord,
    QAIntentReviewOutput,
    QAReportRecord,
    RubricScoreOutput,
    SequenceColorAdjustmentRecord,
)
from shared.utils import gen_id
from prompts.qa_review import (
    L6_QA_INTENT_REVIEW_SYSTEM_PROMPT,
    L6_QA_RUBRIC_REVIEW_SYSTEM_PROMPT,
    REVIEW_INTENT_MATCH_TOOL,
    SCORE_EDIT_RUBRIC_TOOL,
)

logger = logging.getLogger(__name__)

_RUBRIC_LLM_MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Tunables (see module docstring's "Threshold caveat")
# ---------------------------------------------------------------------------

_BLACKDETECT_MIN_DURATION_S = 0.5   # ffmpeg blackdetect's own `d=` — shorter is ignored entirely
_BLACKDETECT_PIC_TH = 0.98          # fraction of pixels that must be "black" for a frame to count
_BLACKDETECT_PIX_TH = 0.10          # per-pixel luminance threshold (0-1) for "black"
_BLACK_FRAME_FAIL_DURATION_S = 3.0  # sustained black beyond this reads as a real error, not a fade/cut

_SILENCEDETECT_NOISE_DB = -30.0     # ffmpeg silencedetect's own `n=` threshold
_SILENCEDETECT_MIN_DURATION_S = 0.5
_SILENCE_FAIL_DURATION_S = 8.0      # a genuine audio drop-out, not just a natural conversational pause

_LOUDNESS_WARN_TOLERANCE_LU = 1.5
_LOUDNESS_FAIL_TOLERANCE_LU = 4.0
_LOUDNESS_MAX_LRA = 20.0            # ffmpeg loudnorm's own LRA ceiling before it's considered unusable

_CAPTION_DRIFT_WARN_RATIO = 0.75    # looser than caption_overlay's own 0.90 same-source guard — this
                                     # compares against an INDEPENDENT signal (dialogue_subtitle), so
                                     # some natural divergence from the transcript-sourced caption is
                                     # expected and not itself a bug.
_CAPTION_DRIFT_FAIL_RATIO = 0.5

_CLIPPING_FAIL_COUNT = 5            # this many out-of-bounds params in one delivered cut is a real problem

_FFMPEG_ANALYSIS_TIMEOUT_S = 180.0
_QA_LLM_MAX_TOKENS = 2048
_LLM_CALL_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# FFmpeg analysis-pass runner (shared by blackdetect/silencedetect/loudness)
# ---------------------------------------------------------------------------


def _run_ffmpeg_analysis(cmd: list[str], timeout: float = _FFMPEG_ANALYSIS_TIMEOUT_S) -> str:
    """Run an ffmpeg analysis-only pass (`-f null -`) and return stderr —
    where `blackdetect`/`silencedetect`/`loudnorm` print their stats.

    Deliberately does not raise on a non-zero exit code: an analysis-only
    `-f null -` pass can exit non-zero for reasons unrelated to whether the
    filter itself produced usable stats (muxer warnings, odd container
    metadata on a freshly rendered file) — callers parse stderr
    defensively and treat "nothing found" as the safe default, same
    tolerant-parsing precedent as `audio_sync.py::parse_loudnorm_stats`.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"qa_agent: ffmpeg analysis pass timed out after {timeout}s: {' '.join(cmd)}"
        ) from exc
    if proc.returncode != 0:
        logger.warning(
            "qa_agent: ffmpeg analysis pass exited %d (cmd=%s) — stderr tail: %s",
            proc.returncode, " ".join(cmd), (proc.stderr or "")[-500:],
        )
    return proc.stderr or ""


# ---------------------------------------------------------------------------
# Black-frame detection
# ---------------------------------------------------------------------------

_BLACKDETECT_RE = None  # compiled lazily, see below


def _parse_blackdetect_output(stderr: str) -> list[dict]:
    """Parse ffmpeg `blackdetect` stderr lines of the form
    ``[blackdetect @ 0x...] black_start:12.3 black_end:14.1 black_duration:1.8``
    into ``[{"start": 12.3, "end": 14.1, "duration": 1.8}, ...]``.

    Pure function — no ffmpeg, no I/O. Tolerant: lines that don't match are
    simply skipped, never raises on malformed/missing output.
    """
    import re

    global _BLACKDETECT_RE
    if _BLACKDETECT_RE is None:
        _BLACKDETECT_RE = re.compile(
            r"black_start:(?P<start>[\d.]+)\s+black_end:(?P<end>[\d.]+)\s+"
            r"black_duration:(?P<dur>[\d.]+)"
        )
    events: list[dict] = []
    for m in _BLACKDETECT_RE.finditer(stderr or ""):
        try:
            events.append(
                {
                    "start": float(m.group("start")),
                    "end": float(m.group("end")),
                    "duration": float(m.group("dur")),
                }
            )
        except ValueError:
            continue
    return events


def run_blackdetect(video_path: str) -> list[dict]:
    """Run ffmpeg `blackdetect` over *video_path* (the rendered output) and
    return parsed black-frame segments."""
    filt = (
        f"blackdetect=d={_BLACKDETECT_MIN_DURATION_S}:"
        f"pic_th={_BLACKDETECT_PIC_TH}:pix_th={_BLACKDETECT_PIX_TH}"
    )
    cmd = ["ffmpeg", "-i", video_path, "-vf", filt, "-an", "-f", "null", "-"]
    stderr = _run_ffmpeg_analysis(cmd)
    return _parse_blackdetect_output(stderr)


# ---------------------------------------------------------------------------
# Silence / audio drop-out detection
# ---------------------------------------------------------------------------


def _parse_silencedetect_output(stderr: str) -> list[dict]:
    """Parse ffmpeg `silencedetect` stderr — a `silence_start: <t>` line
    followed by a `silence_end: <t> | silence_duration: <d>` line per
    detected silence (a trailing silence that runs to end-of-stream may
    only ever emit `silence_start`, with no matching `silence_end` — that
    entry is returned with ``end``/``duration`` as ``None`` rather than
    dropped, since "silence ran to the end of the file" is itself a
    meaningful finding).

    Pure function — no ffmpeg, no I/O.
    """
    import re

    starts = [float(v) for v in re.findall(r"silence_start:\s*([\d.\-]+)", stderr or "")]
    ends = re.findall(r"silence_end:\s*([\d.\-]+)\s*\|\s*silence_duration:\s*([\d.]+)", stderr or "")

    events: list[dict] = []
    for i, start in enumerate(starts):
        if i < len(ends):
            end_str, dur_str = ends[i]
            events.append({"start": start, "end": float(end_str), "duration": float(dur_str)})
        else:
            events.append({"start": start, "end": None, "duration": None})
    return events


def run_silencedetect(video_path: str) -> list[dict]:
    """Run ffmpeg `silencedetect` over *video_path*'s audio track and
    return parsed silence segments."""
    filt = f"silencedetect=n={_SILENCEDETECT_NOISE_DB}dB:d={_SILENCEDETECT_MIN_DURATION_S}"
    cmd = ["ffmpeg", "-i", video_path, "-af", filt, "-vn", "-f", "null", "-"]
    stderr = _run_ffmpeg_analysis(cmd)
    return _parse_silencedetect_output(stderr)


# ---------------------------------------------------------------------------
# Loudness range check — reuses audio_sync.py's filter-builder + stats
# parser, run against the RENDERED output (see module docstring caveat).
# ---------------------------------------------------------------------------


def _evaluate_loudness(stats: dict | None, target_platform: str | None) -> dict:
    """Pure function: compare loudnorm-measured stats (from
    `audio_sync.parse_loudnorm_stats`) against the platform's target LUFS
    (`audio_sync._target_lufs` — the exact function `compute_loudnorm_filter`
    itself uses, so QA's target always matches what normalization actually
    targeted).

    Returns a dict always containing `target_i`; `measured_i`/`measured_lra`/
    `within_tolerance`/`lra_ok` are only present when *stats* was
    successfully parsed (None otherwise — "measurement unavailable" is a
    distinct, honestly-reported state from "measured and out of range").
    """
    target_i = _audio_target_lufs(target_platform)
    result: dict = {"target_i": target_i}
    if not stats:
        result["measured_i"] = None
        result["measured_lra"] = None
        result["within_tolerance"] = None
        result["lra_ok"] = None
        return result
    try:
        measured_i = float(stats["input_i"])
        measured_lra = float(stats["input_lra"])
    except (KeyError, TypeError, ValueError):
        result["measured_i"] = None
        result["measured_lra"] = None
        result["within_tolerance"] = None
        result["lra_ok"] = None
        return result

    diff = abs(measured_i - target_i)
    result["measured_i"] = measured_i
    result["measured_lra"] = measured_lra
    result["diff_lu"] = diff
    result["within_tolerance"] = diff <= _LOUDNESS_WARN_TOLERANCE_LU
    result["lra_ok"] = measured_lra <= _LOUDNESS_MAX_LRA
    return result


def run_loudness_check(video_path: str, target_platform: str | None) -> dict:
    """Run ffmpeg `loudnorm` (measurement mode, via `audio_sync.py`'s own
    filter builder) over the RENDERED output and evaluate against target
    LUFS for *target_platform* — see module docstring for why this measures
    the delivered file rather than reading back a persisted pass-1 value."""
    filt = compute_loudnorm_measure_filter(target_platform)
    cmd = ["ffmpeg", "-i", video_path, "-af", filt, "-f", "null", "-"]
    stderr = _run_ffmpeg_analysis(cmd)
    stats = parse_loudnorm_stats(stderr)
    return _evaluate_loudness(stats, target_platform)


# ---------------------------------------------------------------------------
# Caption/transcript drift check — pure string diff, no ffmpeg, no LLM.
# ---------------------------------------------------------------------------


def _text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").strip().lower(), (b or "").strip().lower()).ratio()


def _resolve_caption_window(
    op_id: str, target_op_id: str | None, operations_by_id: dict[str, dict]
) -> tuple[float, float] | None:
    """Mirrors `caption_overlay.py::run_caption_overlay`'s own time-window
    resolution (own start/end, falling back to the target SELECT_CLIP op's
    start/end) — duplicated in miniature here rather than exported from
    that module, since `run_caption_overlay`'s return contract
    (`{op_id, target_op_id, filter, style, text}`) does not carry the
    resolved window forward and QA needs it independently to look up the
    same dialogue_subtitle window."""
    op = operations_by_id.get(op_id) or {}
    start, end = op.get("start_time"), op.get("end_time")
    if start is None or end is None:
        target_op = operations_by_id.get(target_op_id) if target_op_id else None
        if target_op is not None:
            start, end = target_op.get("start_time"), target_op.get("end_time")
    if start is None or end is None or start >= end:
        return None
    return float(start), float(end)


def check_caption_drift(
    operations: list[dict], caption_results: list[dict], frame_rows: list[dict]
) -> list[dict]:
    """For each already-built caption (`caption_overlay.py::run_caption_overlay`'s
    output — never re-fetched/re-run here), diff its delivered text against
    L3's `dialogue_subtitle` for the same time window (an INDEPENDENT source
    signal from whatever `caption_overlay.py` actually grounded that
    caption's text in — see module docstring). Flags entries below
    `_CAPTION_DRIFT_WARN_RATIO`.

    Skips (does not flag) captions whose time window can't be resolved, or
    whose window has no `dialogue_subtitle` signal at all — "nothing to
    independently compare against" is not itself a drift finding.

    Pure function — no ffmpeg, no DB, no LLM.
    """
    operations_by_id = {op.get("op_id"): op for op in operations}
    flagged: list[dict] = []
    for cap in caption_results:
        op_id = cap.get("op_id")
        window = _resolve_caption_window(op_id, cap.get("target_op_id"), operations_by_id)
        if window is None:
            continue
        start, end = window
        source_subtitle = _source_dialogue_subtitle(frame_rows, start, end)
        if not source_subtitle:
            continue
        rendered_text = cap.get("text", "")
        ratio = _text_similarity(rendered_text, source_subtitle)
        if ratio < _CAPTION_DRIFT_WARN_RATIO:
            flagged.append(
                {
                    "op_id": op_id,
                    "rendered_text": rendered_text,
                    "source_dialogue_subtitle": source_subtitle,
                    "similarity": round(ratio, 3),
                }
            )
    return flagged


# ---------------------------------------------------------------------------
# Color clipping check — reuses color_grading_runner.py's own schema/clamp.
# ---------------------------------------------------------------------------


def check_color_clipping(color_adjustments: list[SequenceColorAdjustmentRecord]) -> list[dict]:
    """Flag any `sequence_color_adjustments` row whose base+delta final
    value falls outside its parameter's real `color_analyzer/analyzer/
    schema.py` bounds — reuses `color_grading_runner.py::_get_schema`/
    `_clamp` (the SAME bounds/clamp logic `run_color_grading` already
    applies at write time), never re-deriving or hardcoding new thresholds.

    Normally returns `[]` (deltas are already clamped before being
    written) — this is a safety net for drift between write-time clamping
    and whatever produced the row, not the primary defense.
    """
    param_by_name, _ = _color_schema()
    flagged: list[dict] = []
    for adj in color_adjustments:
        base_parameters = adj.base_parameters if isinstance(adj.base_parameters, dict) else {}
        for key, delta in (adj.sequence_delta or {}).items():
            param = param_by_name.get(key)
            if param is None or param.kind not in ("float", "int"):
                continue
            entry = base_parameters.get(key)
            base_val = None
            if isinstance(entry, dict):
                base_val = entry.get("recommended")
                if base_val is None:
                    base_val = entry.get("current")
            if base_val is None:
                continue
            try:
                base_f = float(base_val)
                delta_f = float(delta)
            except (TypeError, ValueError):
                continue
            final_val = base_f + delta_f
            clamped = _color_clamp(final_val, param.lo, param.hi)
            if abs(clamped - final_val) > 1e-6:
                flagged.append(
                    {
                        "cut_list_item_id": adj.cut_list_item_id,
                        "param": key,
                        "final_value": final_val,
                        "bounds": [param.lo, param.hi],
                    }
                )
    return flagged


# ---------------------------------------------------------------------------
# Status aggregation — combines every deterministic check + the LLM verdict
# into one overall gate status. See module docstring's threshold caveat.
# ---------------------------------------------------------------------------

_STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def _worse(a: str, b: str) -> str:
    return a if _STATUS_ORDER.get(a, 0) >= _STATUS_ORDER.get(b, 0) else b


def aggregate_deterministic_status(checks: dict) -> str:
    """Pure function: fold every deterministic check's own result into one
    pass/warn/fail. Any single "clearly broken" finding (sustained black,
    long silence, way-off loudness, badly drifted caption, many clipped
    params) is enough to fail outright; a smaller/borderline finding
    downgrades to warn rather than pass."""
    status = "pass"

    black_frames = checks.get("black_frames") or []
    if any((bf.get("duration") or 0) >= _BLACK_FRAME_FAIL_DURATION_S for bf in black_frames):
        return "fail"
    if black_frames:
        status = _worse(status, "warn")

    silences = checks.get("silences") or []
    if any((s.get("duration") or 0) >= _SILENCE_FAIL_DURATION_S for s in silences):
        return "fail"
    if silences:
        status = _worse(status, "warn")

    loudness = checks.get("loudness_range") or {}
    diff = loudness.get("diff_lu")
    if diff is not None:
        if diff >= _LOUDNESS_FAIL_TOLERANCE_LU:
            return "fail"
        if diff >= _LOUDNESS_WARN_TOLERANCE_LU:
            status = _worse(status, "warn")
    if loudness.get("lra_ok") is False:
        status = _worse(status, "warn")

    caption_drift = checks.get("caption_drift") or []
    if any((item.get("similarity") if item.get("similarity") is not None else 1.0) < _CAPTION_DRIFT_FAIL_RATIO for item in caption_drift):
        return "fail"
    if caption_drift:
        status = _worse(status, "warn")

    clipping = checks.get("clipping") or []
    if len(clipping) >= _CLIPPING_FAIL_COUNT:
        return "fail"
    if clipping:
        status = _worse(status, "warn")

    return status


# ---------------------------------------------------------------------------
# LLM pass — content-intent match (the one QA judgment call)
# ---------------------------------------------------------------------------


def _get_groq_client():
    """Name kept for minimal diff — actually returns an OpenRouter client
    now (see shared/llm_client.py)."""
    from shared.llm_client import get_llm_client

    return get_llm_client()


async def _call_groq_tool(
    client,
    model: str,
    system_prompt: str,
    tool_schema: dict,
    payload: dict,
    max_tokens: int,
    *,
    pool: asyncpg.Pool | None = None,
    video_id: str | None = None,
    stage: str | None = None,
) -> dict | None:
    """Same `_call_groq_tool` pattern as every other L6 module — forced
    tool call, `tool_calls[0].function.arguments` is a JSON STRING that
    must be `json.loads()`-ed, pydantic-validated by the caller."""
    import time as _time

    fn_name = tool_schema["function"]["name"]
    last_exc: Exception | None = None
    for attempt in range(1, _LLM_CALL_ATTEMPTS + 1):
        _t0 = _time.monotonic()
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": fn_name}},
                # See pipeline/level4/grounding_runner.py::_call_tool for the
                # real-video finding behind this.
                extra_body={"provider": {"require_parameters": True}},
            )
            usage = getattr(response, "usage", None)
            _latency_ms = int((_time.monotonic() - _t0) * 1000)
            if usage is not None:
                logger.info(
                    "L6 QA LLM call model=%s tool=%s prompt_tokens=%s completion_tokens=%s",
                    model, fn_name,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                )
            # L7 7c — durable cost/latency log, non-fatal.
            from shared.llm_client import log_llm_call as _log_llm_call
            await _log_llm_call(
                pool,
                video_id=video_id,
                level=6,
                stage=stage or "l6_qa",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage is not None else None,
                latency_ms=_latency_ms,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                logger.error("L6 QA LLM call returned no tool_calls (model=%s, tool=%s)", model, fn_name)
                return None
            arguments = tool_calls[0].function.arguments  # JSON string
            return json.loads(arguments)
        except Exception as exc:  # noqa: BLE001 — log & retry/return, never raise into caller loop
            last_exc = exc
            logger.warning(
                "L6 QA LLM call attempt %d/%d failed (model=%s, tool=%s): %s",
                attempt, _LLM_CALL_ATTEMPTS, model, fn_name, exc,
            )
    logger.error(
        "L6 QA LLM call exhausted %d attempts (model=%s, tool=%s): %s",
        _LLM_CALL_ATTEMPTS, model, fn_name, last_exc,
    )
    return None


async def run_intent_review(
    pool: asyncpg.Pool, plan: EditPlanRecord, cut_items: list[CutListItemRecord]
) -> QAIntentReviewOutput | None:
    """The QA Agent's one LLM pass: does the delivered cut's assembled
    scene selection match `plan.user_prompt`. Grounded ONLY in real
    `scenes` rows actually resolved from this edit_plan's own
    `cut_list_items` (via each clip's `op_id` -> `scene_id`) — never raw
    frame_analyses (rule 16, extended to QA), never invented content.

    Returns None (never a guess) when: no OPENROUTER_API_KEY configured, no
    cut_list_items, no cut_list_item resolves to a real `scenes` row, or
    the LLM call/validation failed after retries — deterministic checks
    alone still produce a `status` in that case (see `run_qa_agent`).
    """
    if not settings.OPENROUTER_API_KEY:
        # Real-video finding (this session's model-routing audit): this
        # gate previously checked GROQ_API_KEY even though _get_groq_client
        # (see above) actually constructs an OpenRouter client — the same
        # mismatch every other L4-L6 call site in this codebase already
        # avoids. Left the pass silently no-op'd (llm_status=NULL) on every
        # environment that only has OPENROUTER_API_KEY set, i.e. most of
        # them. Fixed to check the key the client actually uses.
        logger.warning(
            "run_intent_review: OPENROUTER_API_KEY not configured — skipping QA "
            "LLM intent-match pass; deterministic checks alone still gate "
            "this report's status."
        )
        return None
    if not cut_items:
        logger.info("run_intent_review: edit_plan_id=%s has no cut_list_items — nothing to review", plan.id)
        return None

    op_scene_by_op_id: dict[str, str | None] = {
        op.get("op_id"): op.get("scene_id") for op in plan.operations
    }
    scenes = await get_scenes_for_video(pool, plan.video_id)
    scenes_by_id = {s.id: s for s in scenes}
    persons = await get_persons_for_video(pool, plan.video_id)
    cast = {p.pid: (p.display_name or p.pid) for p in persons}

    selected_scenes: list[dict] = []
    for item in cut_items:
        scene_id = op_scene_by_op_id.get(item.op_id)
        scene = scenes_by_id.get(scene_id) if scene_id else None
        if scene is None:
            continue
        selected_scenes.append(
            {
                # Real-video finding: `cut_items` is already sorted by
                # sequence_index (get_cut_list_items_for_edit_plan's own
                # query), so the LIST order here is already correct
                # playback order — but the model still misread it (twice,
                # reproducibly) when asked to judge an explicit-ordering
                # request, apparently tracking JSON array position
                # unreliably. An explicit 1-based `playback_position` field
                # removes any need for the model to count list position
                # itself — it can just read the number.
                "playback_position": len(selected_scenes) + 1,
                "canonical_scene_id": scene.canonical_scene_id,
                "summary": scene.summary,
                "emotional_arc": scene.emotional_arc,
                "participants": scene.participants,
                "cut_start_time": item.source_start,
                "cut_end_time": item.source_end,
            }
        )

    if not selected_scenes:
        logger.warning(
            "run_intent_review: edit_plan_id=%s — no cut_list_items resolved to "
            "a real scenes row, nothing grounded to review the delivered content "
            "against (has L5/the Editing Director actually run for this plan?)",
            plan.id,
        )
        return None

    payload = {
        "user_prompt": plan.user_prompt,
        "target_duration_s": plan.target_duration_s,
        "platform": plan.platform,
        "selected_scenes": selected_scenes,
        "cast": cast,
    }

    client = _get_groq_client()
    raw = await _call_groq_tool(
        client, settings.L6_QA_MODEL, L6_QA_INTENT_REVIEW_SYSTEM_PROMPT,
        REVIEW_INTENT_MATCH_TOOL, payload, _QA_LLM_MAX_TOKENS,
        pool=pool, video_id=plan.video_id, stage="l6_qa_intent_review",
    )
    if raw is None:
        return None
    try:
        parsed = QAIntentReviewOutput.model_validate(raw)
    except Exception as exc:
        logger.error("run_intent_review: failed to validate LLM response: %s", exc)
        return None
    if parsed.status not in ("pass", "warn", "fail"):
        logger.warning(
            "run_intent_review: LLM returned status=%r outside the closed set "
            "{pass,warn,fail} — coercing to 'warn' rather than trusting an "
            "out-of-schema value (rule 12).",
            parsed.status,
        )
        parsed = parsed.model_copy(update={"status": "warn"})
    return parsed


# ---------------------------------------------------------------------------
# LEVEL 7 -- EVALUATION, 7b: rubric scoring (CLAUDE.md "PIPELINE ADDENDUM 3"
# -> "LEVEL 7 -- EVALUATION" -> 7b). A SECOND, independent LLM call —
# `run_intent_review` above is completely unchanged. Same grounding
# discipline (only real scenes/edit_plan/deterministic_checks data, never
# raw frame_analyses), same non-fatal contract as every other L6 LLM pass:
# a failure here never changes `qa_status` or blocks delivery (rule 27).
# ---------------------------------------------------------------------------


async def run_rubric_scoring(
    pool: asyncpg.Pool,
    plan: EditPlanRecord,
    cut_items: list[CutListItemRecord],
    deterministic_checks: dict,
) -> RubricScoreOutput | None:
    """The QA Agent's second LLM pass (7b): score the delivered edit against
    four dimensions (intent_match, narrative_coherence, pacing_consistency,
    technical_cleanliness). Grounded in the same real `scenes`/`cut_items`
    data `run_intent_review` uses, PLUS the deterministic checks already
    computed by `run_qa_agent` (never re-derived here).

    Returns None (never a guess, never a default-only RubricScoreOutput
    written to the DB) when: no OPENROUTER_API_KEY configured (real gap
    this pass is deliberately NOT reproducing — see module docstring's
    "Threshold caveat" precedent for `run_intent_review`'s own
    GROQ_API_KEY-vs-OpenRouter-client mismatch, audit finding #5; this new
    pass gates on the key the client construction ACTUALLY uses), no
    cut_list_items, no cut_list_item resolves to a real `scenes` row, or the
    LLM call/validation failed after retries. Caller (`run_qa_agent`) never
    lets a None here change `qa_status` — additive signal only (rule 27).
    """
    if not settings.OPENROUTER_API_KEY:
        logger.warning(
            "run_rubric_scoring: OPENROUTER_API_KEY not configured — skipping "
            "L7 rubric-scoring pass (additive signal only, never blocks delivery)."
        )
        return None
    if not cut_items:
        logger.info("run_rubric_scoring: edit_plan_id=%s has no cut_list_items — nothing to score", plan.id)
        return None

    op_scene_by_op_id: dict[str, str | None] = {
        op.get("op_id"): op.get("scene_id") for op in plan.operations
    }
    scenes = await get_scenes_for_video(pool, plan.video_id)
    scenes_by_id = {s.id: s for s in scenes}
    persons = await get_persons_for_video(pool, plan.video_id)
    cast = {p.pid: (p.display_name or p.pid) for p in persons}

    selected_scenes: list[dict] = []
    clip_durations: list[float] = []
    for item in cut_items:
        scene_id = op_scene_by_op_id.get(item.op_id)
        scene = scenes_by_id.get(scene_id) if scene_id else None
        clip_durations.append(round((item.source_end or 0.0) - (item.source_start or 0.0), 3))
        if scene is None:
            continue
        selected_scenes.append(
            {
                "canonical_scene_id": scene.canonical_scene_id,
                "summary": scene.summary,
                "emotional_arc": scene.emotional_arc,
                "causal_link_to_next": scene.causal_link_to_next,
                "participants": scene.participants,
                "cut_start_time": item.source_start,
                "cut_end_time": item.source_end,
            }
        )

    if not selected_scenes:
        logger.warning(
            "run_rubric_scoring: edit_plan_id=%s — no cut_list_items resolved to "
            "a real scenes row, nothing grounded to score",
            plan.id,
        )
        return None

    achieved_duration_s = round(sum(clip_durations), 3) if clip_durations else None

    payload = {
        "user_prompt": plan.user_prompt,
        "target_duration_s": plan.target_duration_s,
        "achieved_duration_s": achieved_duration_s,
        "platform": plan.platform,
        "selected_scenes": selected_scenes,
        "clip_durations": clip_durations,
        "deterministic_checks": deterministic_checks,
        "cast": cast,
    }

    client = _get_groq_client()
    raw = await _call_groq_tool(
        client, settings.L6_QA_MODEL, L6_QA_RUBRIC_REVIEW_SYSTEM_PROMPT,
        SCORE_EDIT_RUBRIC_TOOL, payload, _RUBRIC_LLM_MAX_TOKENS,
        pool=pool, video_id=plan.video_id, stage="l6_qa_rubric_scoring",
    )
    if raw is None:
        return None
    try:
        return RubricScoreOutput.model_validate(raw)
    except Exception as exc:
        logger.error("run_rubric_scoring: failed to validate LLM response: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def run_qa_agent(
    pool: asyncpg.Pool,
    edit_plan_id: str,
    rendered_output_path: str,
    caption_results: list[dict] | None = None,
    color_adjustments: list[SequenceColorAdjustmentRecord] | None = None,
) -> QAReportRecord | None:
    """Run the full QA pass for a finalized `edit_plan`'s rendered output
    and write one `qa_reports` row.

    *caption_results* / *color_adjustments* should be the SAME objects
    `pipeline/level6/updater.py::run_level6` already computed earlier in
    the same run (never re-run those agents' LLM calls just for QA — rule
    20). When omitted (e.g. a standalone QA re-run against an existing
    edit_plan), `color_adjustments` is refetched from `sequence_color_
    adjustments` (cheap, no LLM); `caption_results` cannot be reconstructed
    without re-running `run_caption_overlay`'s LLM call, so the caption-
    drift check is simply skipped (empty list, logged) in that case rather
    than triggering a fresh LLM pass QA has no business making.

    Never raises on a missing edit_plan — returns None (logged) instead,
    matching every other L6 entry point's "return a reason, don't throw"
    convention.
    """
    plan = await get_edit_plan(pool, edit_plan_id)
    if plan is None:
        logger.error("run_qa_agent: no edit_plan found for id=%s", edit_plan_id)
        return None

    cut_items = await get_cut_list_items_for_edit_plan(pool, edit_plan_id)
    if color_adjustments is None:
        color_adjustments = await get_sequence_color_adjustments_for_edit_plan(pool, edit_plan_id)

    deterministic_checks: dict = {}

    try:
        deterministic_checks["black_frames"] = run_blackdetect(rendered_output_path)
    except Exception as exc:  # noqa: BLE001 — a QA check failing to run is itself worth flagging, not fatal
        logger.error("run_qa_agent: blackdetect failed for %s: %s", rendered_output_path, exc)
        deterministic_checks["black_frames"] = []
        deterministic_checks["black_frames_error"] = str(exc)

    try:
        deterministic_checks["silences"] = run_silencedetect(rendered_output_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("run_qa_agent: silencedetect failed for %s: %s", rendered_output_path, exc)
        deterministic_checks["silences"] = []
        deterministic_checks["silences_error"] = str(exc)

    try:
        deterministic_checks["loudness_range"] = run_loudness_check(rendered_output_path, plan.platform)
    except Exception as exc:  # noqa: BLE001
        logger.error("run_qa_agent: loudness check failed for %s: %s", rendered_output_path, exc)
        deterministic_checks["loudness_range"] = {"error": str(exc)}

    if caption_results:
        frame_rows = await get_frame_analyses_with_timestamps_for_video(pool, plan.video_id)
        deterministic_checks["caption_drift"] = check_caption_drift(
            plan.operations, caption_results, frame_rows
        )
    else:
        logger.info(
            "run_qa_agent: edit_plan_id=%s — no caption_results supplied, "
            "skipping caption-drift check (not re-running the Caption Agent's "
            "LLM call just for QA).",
            edit_plan_id,
        )
        deterministic_checks["caption_drift"] = []

    deterministic_checks["clipping"] = check_color_clipping(color_adjustments)

    det_status = aggregate_deterministic_status(deterministic_checks)

    llm_review: str | None = None
    llm_status: str | None = None
    try:
        review = await run_intent_review(pool, plan, cut_items)
        if review is not None:
            llm_review = review.reasoning
            llm_status = review.status
    except Exception as exc:  # noqa: BLE001 — LLM pass failing never blocks the deterministic report
        logger.error("run_qa_agent: intent-review LLM pass raised: %s", exc)

    overall_status = _worse(det_status, llm_status) if llm_status else det_status

    report = QAReportRecord(
        id=gen_id(),
        edit_plan_id=edit_plan_id,
        video_id=plan.video_id,
        status=overall_status,
        deterministic_checks=deterministic_checks,
        llm_review=llm_review,
        llm_status=llm_status,
    )
    await insert_qa_report(pool, report)

    # L7 7b -- rubric scoring, a SECOND independent LLM call, strictly
    # additive: any failure here is caught, logged, and never changes
    # `report.status`/`overall_status` above (already written) or raised
    # into this function's return value (rule 27 — additive to qa_status,
    # never a new blocking gate).
    try:
        rubric = await run_rubric_scoring(pool, plan, cut_items, deterministic_checks)
        if rubric is not None:
            score = EvaluationScoreRecord(
                id=gen_id(),
                qa_report_id=report.id,
                edit_plan_id=edit_plan_id,
                intent_match=rubric.intent_match,
                narrative_coherence=rubric.narrative_coherence,
                pacing_consistency=rubric.pacing_consistency,
                technical_cleanliness=rubric.technical_cleanliness,
                rationale={
                    "intent_match": rubric.intent_match_rationale,
                    "narrative_coherence": rubric.narrative_coherence_rationale,
                    "pacing_consistency": rubric.pacing_consistency_rationale,
                    "technical_cleanliness": rubric.technical_cleanliness_rationale,
                },
            )
            await insert_evaluation_score(pool, score)
            logger.info(
                "run_qa_agent: edit_plan_id=%s wrote evaluation_scores row "
                "(intent_match=%.1f narrative_coherence=%.1f pacing_consistency=%.1f "
                "technical_cleanliness=%.1f)",
                edit_plan_id, rubric.intent_match, rubric.narrative_coherence,
                rubric.pacing_consistency, rubric.technical_cleanliness,
            )
    except Exception as exc:  # noqa: BLE001 — rubric scoring must never break the QA report already written
        logger.error("run_qa_agent: rubric-scoring pass raised (non-fatal, qa_status unaffected): %s", exc)

    logger.info(
        "run_qa_agent: edit_plan_id=%s status=%s (deterministic=%s llm=%s) — "
        "black_frames=%d silences=%d caption_drift=%d clipping=%d",
        edit_plan_id, overall_status, det_status, llm_status,
        len(deterministic_checks.get("black_frames") or []),
        len(deterministic_checks.get("silences") or []),
        len(deterministic_checks.get("caption_drift") or []),
        len(deterministic_checks.get("clipping") or []),
    )
    return report
