"""Schema-drift tests for Level-6 structured tool-call output models
(`shared/types.py`): Color Grading (`compute_sequence_deltas`), Caption
Agent (`choose_caption_style`), and Compositing Agent
(`select_background` / `decide_emphasis`).

Same discipline as L4/L5: every Groq tool-call response is validated with
these models before use, never written as raw `response["field"]` text
(rule 12). Per each model's own module-docstring in shared/types.py, these
models enforce *shape* only — CLOSED-SET membership (asset_id must be an
actually-offered candidate, target_pid must be an actually-observed face)
is enforced by the runner AFTER pydantic validation, not here; tests below
document that boundary rather than asserting something pydantic cannot
check.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from shared.types import (
    BackgroundPickItem,
    CaptionStyleItem,
    ChooseCaptionStyleOutput,
    ComputeSequenceDeltaOutput,
    DecideEmphasisOutput,
    EmphasisDecisionItem,
    SelectBackgroundOutput,
    SequenceDeltaItem,
)

# ---------------------------------------------------------------------------
# SequenceDeltaItem / ComputeSequenceDeltaOutput (Color Grading Agent)
# ---------------------------------------------------------------------------


def test_sequence_delta_item_valid_parses():
    payload = {
        "cut_list_item_id": "cli-1",
        "sequence_delta": {"white_balance.temperature": -200.0, "primary.exposure": 0.15},
        "rationale": "pull warmer to match neighbors",
    }
    parsed = SequenceDeltaItem.model_validate(payload)
    assert parsed.sequence_delta["primary.exposure"] == 0.15


def test_sequence_delta_item_missing_cut_list_item_id_rejected():
    with pytest.raises(ValidationError):
        SequenceDeltaItem.model_validate({"sequence_delta": {}})


def test_sequence_delta_item_defaults_empty_delta_and_rationale():
    parsed = SequenceDeltaItem.model_validate({"cut_list_item_id": "cli-1"})
    assert parsed.sequence_delta == {}
    assert parsed.rationale == ""


def test_sequence_delta_item_rejects_non_numeric_delta_value():
    payload = {"cut_list_item_id": "cli-1", "sequence_delta": {"primary.exposure": "brighter"}}
    with pytest.raises(ValidationError):
        SequenceDeltaItem.model_validate(payload)


def test_compute_sequence_delta_output_valid_parses():
    payload = {"adjustments": [{"cut_list_item_id": "cli-1", "sequence_delta": {"x": 1.0}}]}
    parsed = ComputeSequenceDeltaOutput.model_validate(payload)
    assert len(parsed.adjustments) == 1


def test_compute_sequence_delta_output_defaults_empty():
    assert ComputeSequenceDeltaOutput.model_validate({}).adjustments == []


# ---------------------------------------------------------------------------
# CaptionStyleItem / ChooseCaptionStyleOutput (Caption Agent)
# ---------------------------------------------------------------------------

VALID_CAPTION_STYLE = {
    "op_id": "op_014",
    "text": "So tell me about the match.",
    "font_size_tier": "large",
    "position": "top",
    "timing_mode": "karaoke_word_by_word",
    "emphasis_words": ["match"],
    "rationale": "reel format, punchy word-by-word",
}


def test_caption_style_item_valid_parses():
    parsed = CaptionStyleItem.model_validate(VALID_CAPTION_STYLE)
    assert parsed.timing_mode == "karaoke_word_by_word"


def test_caption_style_item_missing_text_rejected():
    payload = copy.deepcopy(VALID_CAPTION_STYLE)
    del payload["text"]
    with pytest.raises(ValidationError):
        CaptionStyleItem.model_validate(payload)


def test_caption_style_item_applies_documented_defaults():
    parsed = CaptionStyleItem.model_validate({"op_id": "op_1", "text": "hi"})
    assert parsed.font_size_tier == "medium"
    assert parsed.position == "lower_third"
    assert parsed.timing_mode == "static_block"
    assert parsed.emphasis_words == []


def test_caption_style_item_wrong_type_emphasis_words_rejected():
    payload = copy.deepcopy(VALID_CAPTION_STYLE)
    payload["emphasis_words"] = "match"  # should be a list, not a bare string
    with pytest.raises(ValidationError):
        CaptionStyleItem.model_validate(payload)


def test_choose_caption_style_output_valid_parses():
    payload = {"styles": [VALID_CAPTION_STYLE]}
    parsed = ChooseCaptionStyleOutput.model_validate(payload)
    assert len(parsed.styles) == 1


def test_choose_caption_style_output_rejects_malformed_nested_style():
    payload = {"styles": [{"op_id": "op_1"}]}  # missing required `text`
    with pytest.raises(ValidationError):
        ChooseCaptionStyleOutput.model_validate(payload)


# ---------------------------------------------------------------------------
# BackgroundPickItem / SelectBackgroundOutput (Compositing Agent A3)
# ---------------------------------------------------------------------------


def test_background_pick_item_valid_parses():
    payload = {
        "scene_id": "scene-1", "asset_id": "stock-asset-42",
        "start_offset": 3.5, "loop": True, "rationale": "matches mood",
    }
    parsed = BackgroundPickItem.model_validate(payload)
    assert parsed.asset_id == "stock-asset-42"
    assert parsed.loop is True


def test_background_pick_item_null_asset_id_is_valid_no_fit_signal():
    """`asset_id=None` is a documented valid state — 'no candidate was a
    good fit, skip' — must parse cleanly, not be treated as drift."""
    parsed = BackgroundPickItem.model_validate({"scene_id": "scene-1", "asset_id": None})
    assert parsed.asset_id is None


def test_background_pick_item_missing_scene_id_rejected():
    with pytest.raises(ValidationError):
        BackgroundPickItem.model_validate({"asset_id": "stock-1"})


def test_background_pick_item_wrong_type_loop_rejected():
    payload = {"scene_id": "scene-1", "loop": "yes-please"}
    with pytest.raises(ValidationError):
        BackgroundPickItem.model_validate(payload)


def test_select_background_output_valid_parses():
    payload = {"picks": [{"scene_id": "scene-1", "asset_id": "a1"}]}
    parsed = SelectBackgroundOutput.model_validate(payload)
    assert len(parsed.picks) == 1


def test_select_background_output_defaults_empty():
    assert SelectBackgroundOutput.model_validate({}).picks == []


# ---------------------------------------------------------------------------
# EmphasisDecisionItem / DecideEmphasisOutput (Compositing Agent A5-decision)
# ---------------------------------------------------------------------------


def test_emphasis_decision_item_valid_parses():
    payload = {
        "cut_list_item_id": "cli-1", "effect_type": "zoom",
        "target_pid": "P1", "rationale": "P1 is the focal speaker here",
    }
    parsed = EmphasisDecisionItem.model_validate(payload)
    assert parsed.effect_type == "zoom"


def test_emphasis_decision_item_defaults_to_none_effect_type():
    parsed = EmphasisDecisionItem.model_validate({"cut_list_item_id": "cli-1"})
    assert parsed.effect_type == "none"
    assert parsed.target_pid is None


def test_emphasis_decision_item_missing_cut_list_item_id_rejected():
    with pytest.raises(ValidationError):
        EmphasisDecisionItem.model_validate({"effect_type": "zoom"})


def test_decide_emphasis_output_valid_parses():
    """Regression test for a real bug this test-writing pass found: prior
    to this fix, `DecideEmphasisOutput` in shared/types.py declared NO
    fields at all (just `model_config`), while
    `pipeline/level6/emphasis_selector.py::run_emphasis_selection` reads
    `parsed.decisions` unconditionally right after validating — every real
    call would have raised AttributeError. Fixed by adding the missing
    `decisions: list[EmphasisDecisionItem] = []` field, matching every
    sibling *Output model's established shape (resolutions/results/
    scene_beats/candidates/operations/diff_operations/adjustments/styles/
    picks all follow this exact one-list-field pattern)."""
    payload = {"decisions": [{"cut_list_item_id": "cli-1", "effect_type": "highlight"}]}
    parsed = DecideEmphasisOutput.model_validate(payload)
    assert len(parsed.decisions) == 1
    assert parsed.decisions[0].effect_type == "highlight"


def test_decide_emphasis_output_defaults_empty_when_missing_key():
    assert DecideEmphasisOutput.model_validate({}).decisions == []
