"""Unit tests for the pure/deterministic functions in
pipeline/level6/editing_director.py: cut-point snapping (the module's own
"one judgment call, small deterministic search over an existing signal") and
the A4/A5 filter-builder mechanics (`build_zoom_emphasis_filter`,
`build_highlight_callout_filter`, `build_layer_composite_filter_template`,
`build_pip_overlay_filter_template`, plus their small deterministic helpers).

Everything tested here takes plain data in and returns a plain value/string
out — no DB (`asyncpg.Pool`), no ffmpeg subprocess, no filesystem. Functions
that need those (`snap_cut_points`, `write_cut_list`,
`materialize_layer_composite_ops`, `render_direct`, ...) are out of scope for
unit tests per CLAUDE.md B3 — that's what tests/integration/ (or a future
live-DB test) is for.
"""
from __future__ import annotations

import pytest

from pipeline.level6.editing_director import (
    _build_highlight_callout_mechanics,
    _build_zoom_emphasis_mechanics,
    _chain_filter,
    _clamp_rect,
    _flatten_words,
    _shot_for_window,
    _snap_boundary,
    build_highlight_callout_filter,
    build_layer_composite_filter_template,
    build_pip_overlay_filter_template,
    build_zoom_emphasis_filter,
)
from shared.types import (
    CutListItemRecord,
    EmphasisEffectRecord,
    HighlightCalloutMechanics,
    LayerCompositeRecord,
    ShotRecord,
    ZoomEmphasisMechanics,
)


class _FakeSegment:
    """Minimal stand-in for a TranscriptSegment — _flatten_words only
    touches `.words`."""

    def __init__(self, words):
        self.words = words


# ---------------------------------------------------------------------------
# _flatten_words
# ---------------------------------------------------------------------------


def test_flatten_words_sorts_across_segments():
    segments = [
        _FakeSegment([{"word": "b", "start": 5.0, "end": 5.5}]),
        _FakeSegment([{"word": "a", "start": 1.0, "end": 1.5}]),
    ]
    words = _flatten_words(segments)
    assert words == [(1.0, 1.5), (5.0, 5.5)]


def test_flatten_words_skips_malformed_entries():
    segments = [
        _FakeSegment([
            {"word": "ok", "start": 1.0, "end": 1.5},
            {"word": "bad_type", "start": "oops", "end": 2.5},
            {"word": "missing_end", "start": 3.0},
            {"word": "inverted", "start": 5.0, "end": 4.0},  # end < start, dropped
        ]),
    ]
    words = _flatten_words(segments)
    assert words == [(1.0, 1.5)]


def test_flatten_words_handles_none_words_field():
    segments = [_FakeSegment(None)]
    assert _flatten_words(segments) == []


def test_flatten_words_empty_input():
    assert _flatten_words([]) == []


# ---------------------------------------------------------------------------
# _snap_boundary
# ---------------------------------------------------------------------------


def test_snap_boundary_snaps_to_largest_nearby_gap_midpoint():
    # Gap between (1.0,1.2) and (1.9,2.0) is 0.7s, centered at 1.55 — the
    # largest gap within +/-0.5s of boundary=1.5.
    words = [(0.0, 1.0), (1.0, 1.2), (1.9, 2.0), (2.0, 3.0)]
    snapped = _snap_boundary(words, boundary=1.5, window=0.5)
    assert snapped == pytest.approx(1.55)


def test_snap_boundary_unchanged_when_no_gap_in_window():
    # Words are contiguous (no silence) near the boundary.
    words = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    snapped = _snap_boundary(words, boundary=1.5, window=0.3)
    assert snapped == 1.5


def test_snap_boundary_unchanged_with_fewer_than_two_words():
    assert _snap_boundary([], boundary=2.0) == 2.0
    assert _snap_boundary([(0.0, 1.0)], boundary=2.0) == 2.0


def test_snap_boundary_ignores_gaps_outside_window():
    # Words are contiguous near the boundary (no gap overlaps [0.55, 1.55]);
    # a real gap exists further out but its interval never touches the
    # window, so it must not be picked.
    words = [(0.0, 1.0), (1.0, 1.1), (1.1, 2.0), (20.0, 20.1), (30.0, 31.0)]
    snapped = _snap_boundary(words, boundary=1.05, window=0.5)
    assert snapped == 1.05


def test_snap_boundary_picks_largest_of_multiple_gaps_in_window():
    # Two gaps within the window: (1.0-1.1) size .1, (1.3-1.9) size .6 -> pick larger.
    words = [(0.5, 1.0), (1.1, 1.3), (1.9, 2.0)]
    snapped = _snap_boundary(words, boundary=1.4, window=0.5)
    assert snapped == pytest.approx(1.6)  # midpoint of (1.3, 1.9)


# ---------------------------------------------------------------------------
# _chain_filter
# ---------------------------------------------------------------------------


def test_chain_filter_appends_with_comma_when_existing():
    assert _chain_filter("crop=100:100", "scale=50:50") == "crop=100:100,scale=50:50"


def test_chain_filter_returns_addition_alone_when_no_existing():
    assert _chain_filter(None, "scale=50:50") == "scale=50:50"
    assert _chain_filter("", "scale=50:50") == "scale=50:50"


# ---------------------------------------------------------------------------
# _clamp_rect
# ---------------------------------------------------------------------------


def test_clamp_rect_leaves_in_bounds_rect_unchanged():
    rect = {"x": 10.0, "y": 20.0, "w": 100.0, "h": 50.0}
    clamped = _clamp_rect(rect, width=1920, height=1080)
    assert clamped == {"x": 10.0, "y": 20.0, "w": 100.0, "h": 50.0}


def test_clamp_rect_clamps_negative_origin_to_zero():
    rect = {"x": -50.0, "y": -30.0, "w": 100.0, "h": 100.0}
    clamped = _clamp_rect(rect, width=1920, height=1080)
    assert clamped["x"] == 0.0
    assert clamped["y"] == 0.0


def test_clamp_rect_clamps_oversized_dimensions_to_frame():
    rect = {"x": 0.0, "y": 0.0, "w": 5000.0, "h": 5000.0}
    clamped = _clamp_rect(rect, width=1920, height=1080)
    assert clamped["w"] == 1920
    assert clamped["h"] == 1080


def test_clamp_rect_handles_missing_keys_with_defaults():
    clamped = _clamp_rect({}, width=1920, height=1080)
    assert clamped == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def test_clamp_rect_width_height_never_below_one():
    # x sits exactly at frame width -> remaining width is 0, must clamp to 1.
    rect = {"x": 1920.0, "y": 0.0, "w": 100.0, "h": 100.0}
    clamped = _clamp_rect(rect, width=1920, height=1080)
    assert clamped["w"] >= 1.0


# ---------------------------------------------------------------------------
# _shot_for_window
# ---------------------------------------------------------------------------


def _shot(id_, start, end):
    return ShotRecord(
        id=id_, chunk_id="c1", video_id="v1", shot_index=0,
        start_frame=None, end_frame=None,
        start_time=start, end_time=end, shot_type=None, complexity=None,
    )


def test_shot_for_window_finds_containing_shot():
    shots = [_shot("s1", 0.0, 5.0), _shot("s2", 5.0, 10.0), _shot("s3", 10.0, 15.0)]
    result = _shot_for_window(shots, start=6.0, end=8.0)  # midpoint 7.0 -> s2
    assert result.id == "s2"


def test_shot_for_window_falls_back_to_nearest_when_in_a_gap():
    shots = [_shot("s1", 0.0, 5.0), _shot("s2", 8.0, 12.0)]
    # midpoint 6.5 falls in the gap between s1 and s2 -> nearest by start_time distance
    result = _shot_for_window(shots, start=6.0, end=7.0)
    assert result.id == "s2"  # |8.0 - 6.5| = 1.5 < |0.0 - 6.5| = 6.5


def test_shot_for_window_returns_none_when_no_shots_have_time():
    shots = [
        ShotRecord(id="s1", chunk_id="c1", video_id="v1", shot_index=0,
                   start_frame=None, end_frame=None, start_time=None,
                   end_time=None, shot_type=None, complexity=None)
    ]
    assert _shot_for_window(shots, 1.0, 2.0) is None


def test_shot_for_window_empty_list():
    assert _shot_for_window([], 1.0, 2.0) is None


# ---------------------------------------------------------------------------
# A5 — zoom emphasis mechanics + filter
# ---------------------------------------------------------------------------


def _cut_item(start=0.0, end=5.0):
    return CutListItemRecord(
        id="cli-1", edit_plan_id="ep-1", op_id="op_1", sequence_index=0,
        source_start=start, source_end=end,
    )


def test_build_zoom_emphasis_mechanics_returns_none_without_target_bbox():
    effect = EmphasisEffectRecord(
        id="e1", cut_list_item_id="cli-1", effect_type="zoom", parameters={},
    )
    result = _build_zoom_emphasis_mechanics(effect, _cut_item(), 1920, 1080)
    assert result is None


def test_build_zoom_emphasis_mechanics_computes_padded_end_rect():
    effect = EmphasisEffectRecord(
        id="e1", cut_list_item_id="cli-1", effect_type="zoom",
        parameters={"target_bbox": {"x": 800.0, "y": 400.0, "w": 100.0, "h": 100.0}},
    )
    item = _cut_item(start=10.0, end=14.0)
    mech = _build_zoom_emphasis_mechanics(effect, item, video_width=1920, video_height=1080)
    assert mech is not None
    assert mech.cut_list_item_id == "cli-1"
    assert mech.start_rect == {"x": 0.0, "y": 0.0, "w": 1920.0, "h": 1080.0}
    assert mech.duration == pytest.approx(4.0)
    # Padded 1.6x around center (850, 450): w=h=160, centered.
    assert mech.end_rect["w"] == pytest.approx(160.0)
    assert mech.end_rect["h"] == pytest.approx(160.0)
    assert mech.end_rect["x"] == pytest.approx(770.0)
    assert mech.end_rect["y"] == pytest.approx(370.0)
    assert mech.easing == "ease_in_out"


def test_build_zoom_emphasis_filter_produces_crop_and_scale():
    mech = ZoomEmphasisMechanics(
        cut_list_item_id="cli-1",
        start_rect={"x": 0.0, "y": 0.0, "w": 1920.0, "h": 1080.0},
        end_rect={"x": 800.0, "y": 400.0, "w": 200.0, "h": 200.0},
        easing="ease_in_out",
        duration=4.0,
    )
    filt = build_zoom_emphasis_filter(mech, video_width=1920, video_height=1080)
    assert filt.startswith("crop=")
    assert ",scale=1920:1080" in filt
    assert "w='" in filt and "h='" in filt and "x='" in filt and "y='" in filt


def test_build_zoom_emphasis_filter_handles_zero_duration_without_crash():
    mech = ZoomEmphasisMechanics(
        cut_list_item_id="cli-1",
        start_rect={"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
        end_rect={"x": 0.0, "y": 0.0, "w": 50.0, "h": 50.0},
        easing="ease_in_out",
        duration=0.0,
    )
    filt = build_zoom_emphasis_filter(mech, video_width=100, video_height=100)
    assert "crop=" in filt  # doesn't raise ZeroDivisionError


# ---------------------------------------------------------------------------
# A5 — highlight callout mechanics + filter
# ---------------------------------------------------------------------------


def test_build_highlight_callout_mechanics_returns_none_without_target_bbox():
    effect = EmphasisEffectRecord(
        id="e1", cut_list_item_id="cli-1", effect_type="highlight", parameters={},
    )
    assert _build_highlight_callout_mechanics(effect, _cut_item()) is None


def test_build_highlight_callout_mechanics_defaults_to_circle_shape():
    effect = EmphasisEffectRecord(
        id="e1", cut_list_item_id="cli-1", effect_type="highlight",
        parameters={"target_bbox": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}},
    )
    item = _cut_item(start=0.0, end=6.0)
    mech = _build_highlight_callout_mechanics(effect, item)
    assert mech.shape == "circle"
    assert mech.target_bbox == {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
    assert mech.duration == pytest.approx(6.0)
    assert mech.start_time == 0.0


@pytest.mark.parametrize("shape,expected_prefix", [
    ("underline", "drawbox="),
    ("arrow", "drawtext="),
    ("circle", "drawbox="),
    ("something_unrecognized", "drawbox="),  # falls back to circle-style drawbox
])
def test_build_highlight_callout_filter_shape_dispatch(shape, expected_prefix):
    mech = HighlightCalloutMechanics(
        cut_list_item_id="cli-1", shape=shape,
        target_bbox={"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
        start_time=1.0, duration=2.0,
    )
    filt = build_highlight_callout_filter(mech)
    assert filt.startswith(expected_prefix)
    assert "enable='between(t,1.000,3.000)'" in filt


def test_build_highlight_callout_filter_underline_uses_bbox_bottom():
    mech = HighlightCalloutMechanics(
        cut_list_item_id="cli-1", shape="underline",
        target_bbox={"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
        start_time=0.0, duration=1.0,
    )
    filt = build_highlight_callout_filter(mech)
    assert "y=60.0" in filt  # y + h = 20 + 40


# ---------------------------------------------------------------------------
# A4 — layer composite filter templates
# ---------------------------------------------------------------------------


def test_build_layer_composite_filter_template_contains_placeholders():
    layer = LayerCompositeRecord(
        id="lc1", cut_list_item_id="cli-1", layer_type="background_swap",
        source_ref="bg-asset-1", position=None, opacity=0.8, z_index=0,
    )
    tmpl = build_layer_composite_filter_template(layer, video_width=1920, video_height=1080)
    assert "{V}" in tmpl
    assert "{OUT}" in tmpl
    assert "{IN_matte}" in tmpl
    assert "{IN_bg}" in tmpl
    assert "aa=0.8" in tmpl
    assert "scale=1920:1080" in tmpl


def test_build_layer_composite_filter_template_clamps_opacity():
    layer = LayerCompositeRecord(
        id="lc1", cut_list_item_id="cli-1", layer_type="background_swap",
        source_ref="bg-asset-1", position=None, opacity=5.0, z_index=0,
    )
    tmpl = build_layer_composite_filter_template(layer, 1280, 720)
    assert "aa=1.0" in tmpl  # clamped to [0,1]


def test_build_pip_overlay_filter_template_trims_other_item_and_overlays():
    layer = LayerCompositeRecord(
        id="lc2", cut_list_item_id="cli-1", layer_type="pip",
        source_ref="cli-2", position={"x": 100, "y": 50, "w": 400, "h": 300},
        opacity=1.0, z_index=1,
    )
    other_item = CutListItemRecord(
        id="cli-2-abcd1234", edit_plan_id="ep-1", op_id="op_2",
        sequence_index=1, source_start=20.0, source_end=25.0,
    )
    tmpl = build_pip_overlay_filter_template(layer, other_item, video_width=1920, video_height=1080)
    assert "trim=start=20.0:end=25.0" in tmpl
    assert "scale=400:300" in tmpl
    assert "overlay=100:50" in tmpl
    assert "{V}" in tmpl and "{OUT}" in tmpl


def test_build_pip_overlay_filter_template_defaults_position_when_missing():
    layer = LayerCompositeRecord(
        id="lc2", cut_list_item_id="cli-1", layer_type="pip",
        source_ref="cli-2", position=None, opacity=1.0, z_index=1,
    )
    other_item = CutListItemRecord(
        id="cli-2", edit_plan_id="ep-1", op_id="op_2",
        sequence_index=1, source_start=0.0, source_end=1.0,
    )
    tmpl = build_pip_overlay_filter_template(layer, other_item, video_width=1000, video_height=1000)
    # Default pw/ph = 30% of frame dims.
    assert "scale=300:300" in tmpl
