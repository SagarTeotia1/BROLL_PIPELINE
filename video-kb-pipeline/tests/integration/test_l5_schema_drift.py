"""Schema-drift tests for Level-5 Planning (Selection & Scoring / Sequencing
& Pacing) structured tool-call output models (`shared/types.py`).

Same discipline as the L4 tests: every Groq tool-call response is validated
with these models before any DB write (CLAUDE.md L5 "Validation before a
plan is usable" + rule 12), never written as raw `response["field"]` text.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from shared.types import (
    ApplyPlanRevisionOutput,
    BuildEditPlanOutput,
    CandidateSceneScore,
    EditOperationItem,
    PlanRevisionDiffItem,
    ScoreCandidateScenesOutput,
)

# ---------------------------------------------------------------------------
# CandidateSceneScore / ScoreCandidateScenesOutput (Pass A)
# ---------------------------------------------------------------------------


def test_score_candidate_scenes_output_valid_parses():
    payload = {
        "candidates": [
            {"scene_id": "scene-1", "relevance_score": 0.87, "rationale": "key stat callout"},
            {"scene_id": "scene-2", "relevance_score": 0.12},
        ]
    }
    parsed = ScoreCandidateScenesOutput.model_validate(payload)
    assert len(parsed.candidates) == 2
    assert parsed.candidates[1].rationale == ""


def test_candidate_scene_score_missing_relevance_score_rejected():
    with pytest.raises(ValidationError):
        CandidateSceneScore.model_validate({"scene_id": "scene-1"})


def test_candidate_scene_score_wrong_type_relevance_score_rejected():
    with pytest.raises(ValidationError):
        CandidateSceneScore.model_validate({"scene_id": "scene-1", "relevance_score": "very relevant"})


def test_score_candidate_scenes_output_defaults_empty_when_missing_key():
    assert ScoreCandidateScenesOutput.model_validate({}).candidates == []


# ---------------------------------------------------------------------------
# EditOperationItem / BuildEditPlanOutput (Pass B)
# ---------------------------------------------------------------------------

VALID_OP = {
    "op_id": "op_001",
    "type": "SELECT_CLIP",
    "scene_id": "scene-uuid-1",
    "start_time": 142.3,
    "end_time": 158.9,
    "sequence_index": 0,
    "rationale": "beat_id:xyz — key stat callout, high relevance",
    "transition_in": "cut",
    "downstream_ops": ["TEXT_OVERLAY:op_014"],
}


def test_edit_operation_item_valid_parses():
    parsed = EditOperationItem.model_validate(VALID_OP)
    assert parsed.op_id == "op_001"
    assert parsed.type == "SELECT_CLIP"


def test_edit_operation_item_dispatch_request_type_allows_null_scene_and_time():
    """COLOR_MATCH_REQUEST/TEXT_OVERLAY_REQUEST/etc. don't necessarily carry
    their own scene/time range per CLAUDE.md's EditOperationItem docstring
    — scene_id/start_time/end_time are Optional, must not be rejected when
    absent."""
    payload = {
        "op_id": "op_014", "type": "TEXT_OVERLAY_REQUEST",
        "sequence_index": 1, "rationale": "caption this clip",
    }
    parsed = EditOperationItem.model_validate(payload)
    assert parsed.scene_id is None
    assert parsed.start_time is None
    assert parsed.end_time is None
    assert parsed.transition_in == "cut"  # default
    assert parsed.downstream_ops == []  # default


def test_edit_operation_item_missing_required_field_rejected():
    payload = copy.deepcopy(VALID_OP)
    del payload["rationale"]
    with pytest.raises(ValidationError):
        EditOperationItem.model_validate(payload)


def test_edit_operation_item_missing_sequence_index_allowed_at_schema_level():
    """`sequence_index` is intentionally OPTIONAL on EditOperationItem —
    real-video finding, first live L5 run: the prompt tells the model
    dispatch-request ops (COLOR_MATCH_REQUEST etc.) "do not participate in
    sequence_index ordering," so the model reasonably omits it on those —
    making it schema-required meant one such op sank an otherwise-valid
    18-op plan (pydantic validation is all-or-nothing on the whole array).
    Enforcement moved to `pipeline/level5/planner_runner.py::validate_plan`
    (DB-dependent, not covered by this schema-only test file), which DOES
    still reject a missing `sequence_index` — but only for SELECT_CLIP ops
    specifically, since only those actually need one."""
    payload = copy.deepcopy(VALID_OP)
    del payload["sequence_index"]
    parsed = EditOperationItem.model_validate(payload)
    assert parsed.sequence_index is None


def test_edit_operation_item_wrong_type_sequence_index_rejected():
    payload = copy.deepcopy(VALID_OP)
    payload["sequence_index"] = "zero"
    with pytest.raises(ValidationError):
        EditOperationItem.model_validate(payload)


def test_build_edit_plan_output_valid_parses():
    payload = {"operations": [VALID_OP]}
    parsed = BuildEditPlanOutput.model_validate(payload)
    assert len(parsed.operations) == 1


def test_build_edit_plan_output_rejects_op_missing_op_id():
    payload = {"operations": [{"type": "SELECT_CLIP", "sequence_index": 0, "rationale": "x"}]}
    with pytest.raises(ValidationError):
        BuildEditPlanOutput.model_validate(payload)


# ---------------------------------------------------------------------------
# PlanRevisionDiffItem / ApplyPlanRevisionOutput
# ---------------------------------------------------------------------------


def test_plan_revision_diff_item_inherits_edit_operation_fields_plus_action():
    payload = copy.deepcopy(VALID_OP)
    payload["action"] = "add"
    parsed = PlanRevisionDiffItem.model_validate(payload)
    assert parsed.action == "add"
    assert parsed.op_id == "op_001"


def test_plan_revision_diff_item_action_defaults_to_modify():
    parsed = PlanRevisionDiffItem.model_validate(VALID_OP)
    assert parsed.action == "modify"


def test_plan_revision_diff_item_still_requires_base_op_fields():
    payload = {"action": "remove", "op_id": "op_001"}  # missing type/sequence_index/rationale
    with pytest.raises(ValidationError):
        PlanRevisionDiffItem.model_validate(payload)


def test_apply_plan_revision_output_valid_parses():
    payload = {"diff_operations": [{**VALID_OP, "action": "remove"}]}
    parsed = ApplyPlanRevisionOutput.model_validate(payload)
    assert parsed.diff_operations[0].action == "remove"


def test_apply_plan_revision_output_defaults_empty_when_missing_key():
    assert ApplyPlanRevisionOutput.model_validate({}).diff_operations == []
