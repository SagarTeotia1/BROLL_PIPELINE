"""Unit tests for pipeline/level6/audio_sync.py.

Per the module's own docstring, this is pure deterministic DSP filter-string
building with explicitly no LLM/DB/network call anywhere in it — exactly the
"deterministic where possible" functions CLAUDE.md B3 calls out for unit
testing.
"""
from __future__ import annotations

import pytest

from pipeline.level6.audio_sync import (
    apply_multicam_sync,
    compute_ducking_filter,
    compute_loudnorm_filter,
    compute_loudnorm_measure_filter,
    parse_loudnorm_stats,
)

# ---------------------------------------------------------------------------
# Platform -> target LUFS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", ["reel", "youtube", "REEL", " YouTube "])
def test_streaming_platforms_target_minus_14_lufs(platform):
    filt = compute_loudnorm_measure_filter(platform)
    assert "I=-14.0" in filt


@pytest.mark.parametrize("platform", ["full_cut", None, "", "podcast", "unknown_platform"])
def test_non_streaming_platforms_default_to_minus_16_lufs(platform):
    filt = compute_loudnorm_measure_filter(platform)
    assert "I=-16.0" in filt


def test_measure_filter_has_json_print_format():
    filt = compute_loudnorm_measure_filter("reel")
    assert filt.startswith("loudnorm=")
    assert "print_format=json" in filt
    assert "TP=-1.5" in filt
    assert "LRA=11.0" in filt


# ---------------------------------------------------------------------------
# parse_loudnorm_stats
# ---------------------------------------------------------------------------


def test_parse_loudnorm_stats_extracts_json_block_from_noisy_stderr():
    stderr = """
    [Parsed_loudnorm_0 @ 0x555] Some noise before
    [Parsed_loudnorm_0 @ 0x555]
    {
        "input_i" : "-23.45",
        "input_tp" : "-2.30",
        "input_lra" : "5.10",
        "input_thresh" : "-33.50",
        "output_i" : "-16.00",
        "target_offset" : "0.20"
    }
    more noise after
    """
    stats = parse_loudnorm_stats(stderr)
    assert stats is not None
    assert stats["input_i"] == "-23.45"
    assert stats["target_offset"] == "0.20"


def test_parse_loudnorm_stats_returns_none_when_absent():
    assert parse_loudnorm_stats("no json here at all") is None
    assert parse_loudnorm_stats("") is None
    assert parse_loudnorm_stats(None) is None  # type: ignore[arg-type]


def test_parse_loudnorm_stats_returns_none_on_malformed_json():
    # Looks like a candidate block (has "input_i") but isn't valid JSON.
    stderr = '{"input_i": -23.45, this is not valid json}'
    assert parse_loudnorm_stats(stderr) is None


# ---------------------------------------------------------------------------
# compute_loudnorm_filter — single-pass vs two-pass
# ---------------------------------------------------------------------------


def test_compute_loudnorm_filter_single_pass_when_no_stats():
    filt = compute_loudnorm_filter("reel", measured_stats=None)
    assert filt == "loudnorm=I=-14.0:TP=-1.5:LRA=11.0"
    assert "measured_I" not in filt
    assert "linear=true" not in filt


def test_compute_loudnorm_filter_two_pass_with_measured_stats():
    stats = {
        "input_i": "-23.45",
        "input_tp": "-2.30",
        "input_lra": "5.10",
        "input_thresh": "-33.50",
        "target_offset": "0.20",
    }
    filt = compute_loudnorm_filter("full_cut", measured_stats=stats)
    assert "I=-16.0" in filt
    assert "measured_I=-23.45" in filt
    assert "measured_TP=-2.3" in filt
    assert "measured_LRA=5.1" in filt
    assert "measured_thresh=-33.5" in filt
    assert "linear=true" in filt
    assert "offset=0.2" in filt


def test_compute_loudnorm_filter_two_pass_without_offset_key():
    stats = {
        "input_i": "-20.0", "input_tp": "-1.0",
        "input_lra": "4.0", "input_thresh": "-30.0",
    }
    filt = compute_loudnorm_filter("reel", measured_stats=stats)
    assert "linear=true" in filt
    assert "offset=" not in filt


def test_compute_loudnorm_filter_falls_back_on_malformed_stats():
    """Missing/invalid required keys in measured_stats -> falls back to
    single-pass dynamic loudnorm rather than crashing or emitting a broken
    filter string."""
    bad_stats = {"input_i": "not-a-number", "input_tp": "-2.0",
                 "input_lra": "5.0", "input_thresh": "-30.0"}
    filt = compute_loudnorm_filter("reel", measured_stats=bad_stats)
    assert filt == "loudnorm=I=-14.0:TP=-1.5:LRA=11.0"


def test_compute_loudnorm_filter_falls_back_when_stats_missing_key():
    incomplete_stats = {"input_i": "-20.0"}  # missing tp/lra/thresh
    filt = compute_loudnorm_filter("full_cut", measured_stats=incomplete_stats)
    assert filt == "loudnorm=I=-16.0:TP=-1.5:LRA=11.0"


def test_compute_loudnorm_filter_empty_dict_stats_treated_as_none():
    filt = compute_loudnorm_filter("reel", measured_stats={})
    assert filt == "loudnorm=I=-14.0:TP=-1.5:LRA=11.0"


# ---------------------------------------------------------------------------
# Documented no-ops — ducking, multicam sync
# ---------------------------------------------------------------------------


def test_compute_ducking_filter_always_returns_none():
    assert compute_ducking_filter() is None
    assert compute_ducking_filter([]) is None
    assert compute_ducking_filter([{"type": "AUDIO_DUCK_REQUEST", "op_id": "op_1"}]) is None
    assert compute_ducking_filter([{"type": "SELECT_CLIP", "op_id": "op_1"}]) is None


def test_apply_multicam_sync_is_a_passthrough():
    items = [{"id": "a"}, {"id": "b"}]
    result = apply_multicam_sync(items)
    assert result is items  # identity passthrough, not a copy
    assert apply_multicam_sync([]) == []
