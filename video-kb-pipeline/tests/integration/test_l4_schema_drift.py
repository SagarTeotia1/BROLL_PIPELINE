"""Schema-drift tests for Level-4 Grounding Agent / Story Architect Agent
structured tool-call output models (`shared/types.py`).

Per CLAUDE.md rule 12 ("Closed-set outputs only ... Validate every LLM
response with pydantic before any write; reject and log on schema mismatch,
never write raw text to a typed column") and B3's "response-shape drift is
an already-documented expected failure mode for L3/L4/L5/L6". These tests
feed each model both valid fixture JSON (must parse) and deliberately
drifted JSON (missing field / wrong type / extra field) and assert the
expected outcome.

Note: `person_id` in `SpeakerResolutionItem` and `canonical_relation` in
`RelationClusterResult` are validated as PLAIN STRINGS at the pydantic
layer — the actual closed-set membership check (is this pid really in the
video's cast? is this canonical_relation really in the ontology?) happens
in `pipeline/level4/grounding_runner.py` AFTER pydantic validation (see its
module docstring), because the valid set is per-call, not knowable at
model-definition time. These tests document that boundary rather than
pretending pydantic enforces something it structurally cannot.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from shared.types import (
    RelationCanonicalizationBatch,
    RelationClusterResult,
    SceneBeatItem,
    SpeakerResolutionBatch,
    SpeakerResolutionItem,
    StorylineSummaryOutput,
    WriteSceneBeatsOutput,
)

# ---------------------------------------------------------------------------
# SpeakerResolutionItem / SpeakerResolutionBatch (1a)
# ---------------------------------------------------------------------------

VALID_SPEAKER_RESOLUTION = {
    "resolutions": [
        {"turn_id": "turn-1", "person_id": "P1", "confidence": 0.92, "reasoning": "self-reference"},
        {"turn_id": "turn-2", "person_id": None, "confidence": 0.1, "reasoning": "no signal"},
    ]
}


def test_speaker_resolution_batch_valid_parses():
    parsed = SpeakerResolutionBatch.model_validate(VALID_SPEAKER_RESOLUTION)
    assert len(parsed.resolutions) == 2
    assert parsed.resolutions[0].person_id == "P1"
    assert parsed.resolutions[1].person_id is None


def test_speaker_resolution_item_missing_required_field_rejected():
    payload = copy.deepcopy(VALID_SPEAKER_RESOLUTION["resolutions"][0])
    del payload["turn_id"]
    with pytest.raises(ValidationError):
        SpeakerResolutionItem.model_validate(payload)


def test_speaker_resolution_item_missing_confidence_rejected():
    # confidence has no default -> required.
    payload = {"turn_id": "turn-1", "person_id": "P1"}
    with pytest.raises(ValidationError):
        SpeakerResolutionItem.model_validate(payload)


def test_speaker_resolution_item_wrong_type_confidence_rejected():
    payload = {"turn_id": "turn-1", "person_id": "P1", "confidence": "very high"}
    with pytest.raises(ValidationError):
        SpeakerResolutionItem.model_validate(payload)


def test_speaker_resolution_item_extra_field_ignored():
    payload = {
        "turn_id": "turn-1", "person_id": "P1", "confidence": 0.9,
        "unexpected_field_from_drift": "surprise",
    }
    parsed = SpeakerResolutionItem.model_validate(payload)
    assert not hasattr(parsed, "unexpected_field_from_drift")


def test_speaker_resolution_batch_defaults_to_empty_list_when_key_missing():
    parsed = SpeakerResolutionBatch.model_validate({})
    assert parsed.resolutions == []


def test_speaker_resolution_batch_rejects_malformed_item_in_list():
    payload = {"resolutions": [{"turn_id": "t1"}]}  # missing confidence
    with pytest.raises(ValidationError):
        SpeakerResolutionBatch.model_validate(payload)


# ---------------------------------------------------------------------------
# RelationClusterResult / RelationCanonicalizationBatch (1b)
# ---------------------------------------------------------------------------


def test_relation_canonicalization_batch_valid_parses():
    payload = {
        "results": [
            {"cluster_id": "c0", "canonical_relation": "SPEAKS_TO", "reasoning": "..."},
            {"cluster_id": "c1", "canonical_relation": "OTHER"},
        ]
    }
    parsed = RelationCanonicalizationBatch.model_validate(payload)
    assert len(parsed.results) == 2
    assert parsed.results[1].reasoning == ""  # default applied


def test_relation_cluster_result_missing_canonical_relation_rejected():
    with pytest.raises(ValidationError):
        RelationClusterResult.model_validate({"cluster_id": "c0"})


def test_relation_cluster_result_wrong_type_cluster_id_rejected():
    # cluster_id typed as str; a dict is not string-coercible.
    with pytest.raises(ValidationError):
        RelationClusterResult.model_validate({"cluster_id": {"nested": True}, "canonical_relation": "OTHER"})


# ---------------------------------------------------------------------------
# SceneBeatItem / WriteSceneBeatsOutput (Story Architect)
# ---------------------------------------------------------------------------

VALID_SCENE_BEAT = {
    # scene_index required — real-video finding (see shared/types.py::
    # SceneBeatItem / CLAUDE.md L4 Story Architect "scene_index added after
    # a real-video finding"): the model can return fewer beats than input
    # scenes, so each input scene's 0-based batch position must be echoed
    # back to match beats to scenes by index, not list position.
    "scene_index": 0,
    "canonical_scene_id": "interview_studio_01",
    "discarded_aliases": ["studio_podcast_01", "scene_001"],
    "participants": ["P1", "P2"],
    "summary": "Host interviews guest about the match.",
    "emotional_arc": "calm -> tense",
    "causal_link_to_next": "guest storms off",
}


def test_scene_beat_item_valid_parses():
    parsed = SceneBeatItem.model_validate(VALID_SCENE_BEAT)
    assert parsed.canonical_scene_id == "interview_studio_01"
    assert parsed.causal_link_to_next == "guest storms off"


def test_scene_beat_item_causal_link_optional_null():
    payload = copy.deepcopy(VALID_SCENE_BEAT)
    payload["causal_link_to_next"] = None
    parsed = SceneBeatItem.model_validate(payload)
    assert parsed.causal_link_to_next is None


def test_scene_beat_item_missing_required_field_rejected():
    payload = copy.deepcopy(VALID_SCENE_BEAT)
    del payload["summary"]
    with pytest.raises(ValidationError):
        SceneBeatItem.model_validate(payload)


def test_scene_beat_item_wrong_type_canonical_scene_id_rejected():
    payload = copy.deepcopy(VALID_SCENE_BEAT)
    payload["canonical_scene_id"] = 12345
    with pytest.raises(ValidationError):
        SceneBeatItem.model_validate(payload)


def test_scene_beat_item_start_end_time_no_longer_fields():
    """Real-video finding (see shared/types.py::SceneBeatItem): start_time/
    end_time were removed from this schema entirely — write-back always uses
    ground-truth frame timestamps, never the LLM's. Confirms they're quietly
    ignored (extra="ignore") rather than silently expected somewhere."""
    payload = copy.deepcopy(VALID_SCENE_BEAT)
    payload["start_time"] = "not even a number"
    payload["end_time"] = None
    parsed = SceneBeatItem.model_validate(payload)
    assert not hasattr(parsed, "start_time")
    assert not hasattr(parsed, "end_time")


def test_scene_beat_item_participants_defaults_to_empty_list():
    payload = copy.deepcopy(VALID_SCENE_BEAT)
    del payload["participants"]
    parsed = SceneBeatItem.model_validate(payload)
    assert parsed.participants == []


def test_write_scene_beats_output_valid_parses():
    payload = {"scene_beats": [VALID_SCENE_BEAT], "updated_rolling_summary": "So far: intro done."}
    parsed = WriteSceneBeatsOutput.model_validate(payload)
    assert len(parsed.scene_beats) == 1
    assert parsed.updated_rolling_summary == "So far: intro done."


def test_write_scene_beats_output_rejects_malformed_nested_beat():
    payload = {"scene_beats": [{"canonical_scene_id": "x"}], "updated_rolling_summary": ""}
    with pytest.raises(ValidationError):
        WriteSceneBeatsOutput.model_validate(payload)


# ---------------------------------------------------------------------------
# StorylineSummaryOutput
# ---------------------------------------------------------------------------


def test_storyline_summary_output_valid_parses():
    parsed = StorylineSummaryOutput.model_validate(
        {"title": "The Interview", "synopsis": "A host interviews a guest about a match."}
    )
    assert parsed.title == "The Interview"


def test_storyline_summary_output_missing_synopsis_rejected():
    with pytest.raises(ValidationError):
        StorylineSummaryOutput.model_validate({"title": "The Interview"})


def test_storyline_summary_output_extra_field_ignored():
    parsed = StorylineSummaryOutput.model_validate(
        {"title": "T", "synopsis": "S", "drifted_extra_key": 123}
    )
    assert not hasattr(parsed, "drifted_extra_key")
