"""Unit tests for the pure/deterministic functions in
pipeline/level6/qa_agent.py: blackdetect/silencedetect stderr parsing,
loudness evaluation, the caption-drift string diff, the color-clipping
bounds check, and status aggregation.

Everything tested here takes plain data in and returns a plain value out —
no ffmpeg subprocess, no DB (`asyncpg.Pool`), no Groq call. Functions that
need those (`run_blackdetect`, `run_silencedetect`, `run_loudness_check`,
`run_intent_review`, `run_qa_agent`) are out of scope for unit tests per
CLAUDE.md B3 — same split `test_editing_director.py`/`test_audio_sync.py`
already follow.
"""
from __future__ import annotations

from pipeline.level6.qa_agent import (
    _evaluate_loudness,
    _parse_blackdetect_output,
    _parse_silencedetect_output,
    _resolve_caption_window,
    _text_similarity,
    _worse,
    aggregate_deterministic_status,
    check_caption_drift,
    check_color_clipping,
)
from shared.types import SequenceColorAdjustmentRecord

# ---------------------------------------------------------------------------
# _parse_blackdetect_output
# ---------------------------------------------------------------------------


def test_parse_blackdetect_output_extracts_events():
    stderr = (
        "[blackdetect @ 0x55d1] black_start:12.3 black_end:14.1 black_duration:1.8\n"
        "frame=  100 fps=25 q=-1.0 size=N/A time=00:00:05.00 bitrate=N/A speed=  10x\n"
        "[blackdetect @ 0x55d1] black_start:40.0 black_end:40.6 black_duration:0.6\n"
    )
    events = _parse_blackdetect_output(stderr)
    assert events == [
        {"start": 12.3, "end": 14.1, "duration": 1.8},
        {"start": 40.0, "end": 40.6, "duration": 0.6},
    ]


def test_parse_blackdetect_output_empty_when_no_matches():
    assert _parse_blackdetect_output("nothing here") == []
    assert _parse_blackdetect_output("") == []


# ---------------------------------------------------------------------------
# _parse_silencedetect_output
# ---------------------------------------------------------------------------


def test_parse_silencedetect_output_pairs_start_end():
    stderr = (
        "[silencedetect @ 0x1] silence_start: 5.2\n"
        "[silencedetect @ 0x1] silence_end: 7.4 | silence_duration: 2.2\n"
        "[silencedetect @ 0x1] silence_start: 50.0\n"
        "[silencedetect @ 0x1] silence_end: 51.5 | silence_duration: 1.5\n"
    )
    events = _parse_silencedetect_output(stderr)
    assert events == [
        {"start": 5.2, "end": 7.4, "duration": 2.2},
        {"start": 50.0, "end": 51.5, "duration": 1.5},
    ]


def test_parse_silencedetect_output_trailing_silence_has_no_end():
    stderr = "[silencedetect @ 0x1] silence_start: 90.0\n"
    events = _parse_silencedetect_output(stderr)
    assert events == [{"start": 90.0, "end": None, "duration": None}]


def test_parse_silencedetect_output_empty_when_no_matches():
    assert _parse_silencedetect_output("nothing here") == []


# ---------------------------------------------------------------------------
# _evaluate_loudness
# ---------------------------------------------------------------------------


def test_evaluate_loudness_within_tolerance():
    stats = {"input_i": "-14.2", "input_tp": "-1.5", "input_lra": "8.0", "input_thresh": "-24.0"}
    result = _evaluate_loudness(stats, "reel")
    assert result["target_i"] == -14.0
    assert result["measured_i"] == -14.2
    assert result["within_tolerance"] is True
    assert result["lra_ok"] is True


def test_evaluate_loudness_out_of_tolerance():
    stats = {"input_i": "-20.0", "input_tp": "-2.0", "input_lra": "8.0", "input_thresh": "-30.0"}
    result = _evaluate_loudness(stats, "reel")
    assert result["within_tolerance"] is False
    assert result["diff_lu"] == 6.0


def test_evaluate_loudness_high_lra_flagged():
    stats = {"input_i": "-16.0", "input_tp": "-1.5", "input_lra": "25.0", "input_thresh": "-26.0"}
    result = _evaluate_loudness(stats, "full_cut")
    assert result["lra_ok"] is False


def test_evaluate_loudness_missing_stats_reports_unavailable():
    result = _evaluate_loudness(None, "reel")
    assert result["measured_i"] is None
    assert result["within_tolerance"] is None


def test_evaluate_loudness_malformed_stats_reports_unavailable():
    result = _evaluate_loudness({"input_i": "not_a_number"}, "reel")
    assert result["measured_i"] is None


# ---------------------------------------------------------------------------
# _resolve_caption_window / check_caption_drift
# ---------------------------------------------------------------------------


def test_resolve_caption_window_uses_own_times():
    ops = {"cap1": {"op_id": "cap1", "start_time": 1.0, "end_time": 3.0}}
    assert _resolve_caption_window("cap1", None, ops) == (1.0, 3.0)


def test_resolve_caption_window_falls_back_to_target_op():
    ops = {
        "cap1": {"op_id": "cap1"},
        "clip1": {"op_id": "clip1", "start_time": 5.0, "end_time": 9.0},
    }
    assert _resolve_caption_window("cap1", "clip1", ops) == (5.0, 9.0)


def test_resolve_caption_window_none_when_unresolvable():
    ops = {"cap1": {"op_id": "cap1"}}
    assert _resolve_caption_window("cap1", None, ops) is None
    assert _resolve_caption_window("cap1", "missing", ops) is None


def test_text_similarity_identical_strings_is_1():
    assert _text_similarity("hello world", "Hello World") == 1.0


def test_text_similarity_different_strings_below_1():
    assert _text_similarity("hello world", "goodbye moon") < 0.5


def test_check_caption_drift_flags_low_similarity():
    operations = [{"op_id": "cap1", "start_time": 1.0, "end_time": 3.0}]
    caption_results = [{"op_id": "cap1", "target_op_id": None, "text": "completely different words"}]
    frame_rows = [
        {"timestamp_s": 2.0, "qwen_output": {"dialogue_subtitle": "the real transcript text here"}}
    ]
    flagged = check_caption_drift(operations, caption_results, frame_rows)
    assert len(flagged) == 1
    assert flagged[0]["op_id"] == "cap1"
    assert flagged[0]["similarity"] < 0.75


def test_check_caption_drift_no_flag_when_matching():
    operations = [{"op_id": "cap1", "start_time": 1.0, "end_time": 3.0}]
    caption_results = [{"op_id": "cap1", "target_op_id": None, "text": "the real transcript text"}]
    frame_rows = [
        {"timestamp_s": 2.0, "qwen_output": {"dialogue_subtitle": "the real transcript text"}}
    ]
    assert check_caption_drift(operations, caption_results, frame_rows) == []


def test_check_caption_drift_skips_unresolvable_window():
    operations = [{"op_id": "cap1"}]  # no start/end, no target
    caption_results = [{"op_id": "cap1", "target_op_id": None, "text": "anything"}]
    assert check_caption_drift(operations, caption_results, []) == []


def test_check_caption_drift_skips_when_no_source_subtitle():
    operations = [{"op_id": "cap1", "start_time": 1.0, "end_time": 3.0}]
    caption_results = [{"op_id": "cap1", "target_op_id": None, "text": "anything"}]
    frame_rows = [{"timestamp_s": 2.0, "qwen_output": {}}]
    assert check_caption_drift(operations, caption_results, frame_rows) == []


# ---------------------------------------------------------------------------
# check_color_clipping
# ---------------------------------------------------------------------------


def test_check_color_clipping_flags_out_of_bounds_delta():
    # white_balance.temperature bounds are [2000, 12000] per
    # color-grading/color_analyzer/analyzer/schema.py — base near the top of
    # range plus a big positive delta should clip.
    adj = SequenceColorAdjustmentRecord(
        id="adj1",
        edit_plan_id="plan1",
        cut_list_item_id="clip1",
        base_parameters={"white_balance.temperature": {"recommended": 11000.0}},
        sequence_delta={"white_balance.temperature": 5000.0},
    )
    flagged = check_color_clipping([adj])
    assert len(flagged) == 1
    assert flagged[0]["param"] == "white_balance.temperature"
    assert flagged[0]["cut_list_item_id"] == "clip1"


def test_check_color_clipping_no_flag_when_in_bounds():
    adj = SequenceColorAdjustmentRecord(
        id="adj1",
        edit_plan_id="plan1",
        cut_list_item_id="clip1",
        base_parameters={"white_balance.temperature": {"recommended": 6500.0}},
        sequence_delta={"white_balance.temperature": 100.0},
    )
    assert check_color_clipping([adj]) == []


def test_check_color_clipping_ignores_unknown_param():
    adj = SequenceColorAdjustmentRecord(
        id="adj1",
        edit_plan_id="plan1",
        cut_list_item_id="clip1",
        base_parameters={"not_a_real_param": {"recommended": 1.0}},
        sequence_delta={"not_a_real_param": 999.0},
    )
    assert check_color_clipping([adj]) == []


def test_check_color_clipping_empty_input():
    assert check_color_clipping([]) == []


# ---------------------------------------------------------------------------
# _worse / aggregate_deterministic_status
# ---------------------------------------------------------------------------


def test_worse_orders_pass_warn_fail():
    assert _worse("pass", "warn") == "warn"
    assert _worse("warn", "fail") == "fail"
    assert _worse("fail", "pass") == "fail"
    assert _worse("pass", "pass") == "pass"


def test_aggregate_status_all_clean_is_pass():
    checks = {
        "black_frames": [],
        "silences": [],
        "loudness_range": {"diff_lu": 0.2, "lra_ok": True},
        "caption_drift": [],
        "clipping": [],
    }
    assert aggregate_deterministic_status(checks) == "pass"


def test_aggregate_status_short_black_frame_is_warn():
    checks = {
        "black_frames": [{"start": 1.0, "end": 1.6, "duration": 0.6}],
        "silences": [],
        "loudness_range": {},
        "caption_drift": [],
        "clipping": [],
    }
    assert aggregate_deterministic_status(checks) == "warn"


def test_aggregate_status_sustained_black_frame_is_fail():
    checks = {
        "black_frames": [{"start": 1.0, "end": 5.0, "duration": 4.0}],
        "silences": [],
        "loudness_range": {},
        "caption_drift": [],
        "clipping": [],
    }
    assert aggregate_deterministic_status(checks) == "fail"


def test_aggregate_status_long_silence_is_fail():
    checks = {
        "black_frames": [],
        "silences": [{"start": 1.0, "end": 12.0, "duration": 11.0}],
        "loudness_range": {},
        "caption_drift": [],
        "clipping": [],
    }
    assert aggregate_deterministic_status(checks) == "fail"


def test_aggregate_status_far_off_loudness_is_fail():
    checks = {
        "black_frames": [],
        "silences": [],
        "loudness_range": {"diff_lu": 5.0, "lra_ok": True},
        "caption_drift": [],
        "clipping": [],
    }
    assert aggregate_deterministic_status(checks) == "fail"


def test_aggregate_status_slightly_off_loudness_is_warn():
    checks = {
        "black_frames": [],
        "silences": [],
        "loudness_range": {"diff_lu": 2.0, "lra_ok": True},
        "caption_drift": [],
        "clipping": [],
    }
    assert aggregate_deterministic_status(checks) == "warn"


def test_aggregate_status_badly_drifted_caption_is_fail():
    checks = {
        "black_frames": [],
        "silences": [],
        "loudness_range": {},
        "caption_drift": [{"op_id": "cap1", "similarity": 0.2}],
        "clipping": [],
    }
    assert aggregate_deterministic_status(checks) == "fail"


def test_aggregate_status_many_clipped_params_is_fail():
    checks = {
        "black_frames": [],
        "silences": [],
        "loudness_range": {},
        "caption_drift": [],
        "clipping": [{"param": f"p{i}"} for i in range(6)],
    }
    assert aggregate_deterministic_status(checks) == "fail"


def test_aggregate_status_few_clipped_params_is_warn():
    checks = {
        "black_frames": [],
        "silences": [],
        "loudness_range": {},
        "caption_drift": [],
        "clipping": [{"param": "p1"}],
    }
    assert aggregate_deterministic_status(checks) == "warn"
