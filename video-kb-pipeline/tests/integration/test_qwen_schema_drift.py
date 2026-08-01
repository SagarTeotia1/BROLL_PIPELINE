"""Schema-drift tests for Level-3 Qwen structured-output pydantic models
(`shared/types.py`'s `QwenFrameOutput` and its nested models).

Per CLAUDE.md B8: "L3 | Qwen response shape drift | No — per-frame | Reject
+ log, frame skipped". These tests exercise both directions of that
contract: a valid frame parses, and drifted/malformed responses are
rejected (or safely coerced to a documented fallback — see
`QwenComposition`) rather than crashing the pipeline or writing raw text to
a typed column (rule 12).

Filed under tests/integration/ per CLAUDE.md B3 ("pydantic schema-drift
tests for Qwen/Groq responses") even though no network/DB call happens here
— the label in CLAUDE.md groups these with the "LLM response shape" test
family, not the "pure function" family in tests/unit/.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from shared.types import QwenComposition, QwenFrameOutput, QwenPersonEntry, QwenPersonInteraction

VALID_FRAME = {
    "id": "frame_0001",
    "t": 12.5,
    "scene_change": True,
    "scene_id": "interview_studio_01",
    "location_id": "studio_desk",
    "dominant_person": "P1",
    "caption": "Two people talking at a desk.",
    "setting": {
        "location_type": "studio", "specific_location": "desk",
        "time_of_day": "day", "indoor_outdoor": "indoor",
    },
    "composition": {
        "shot": "medium", "angle": "eye-level", "movement": "static",
        "framing": "centered", "depth": "shallow",
    },
    "lighting_color": {
        "lighting": "soft key light", "palette": ["blue", "white"],
        "grade_mood": "neutral",
    },
    "objects": [{"name": "microphone", "position": "foreground", "story_relevance": "prop"}],
    "people": [
        {
            "pid": "P1", "pose": "seated", "action": "talking", "gaze": "at camera",
            "clothing": "suit", "emotion_inferred": "neutral",
            "emotion_source": "visual_read", "story_role": "host",
            "apparent_goal": "explain a point",
        },
    ],
    "interactions": [{"p1": "P1", "p2": "P2", "type": "address", "notes": "host addresses guest"}],
    "relations": [{"subject": "P1", "predicate": "sits_near", "object": "P2", "confidence": 0.9}],
    "text_detected": [{"text": "LIVE", "location": "top_left"}],
    "dialogue_subtitle": "So tell me about the match.",
    "story_beat": {
        "beat_type": "setup", "act_position": "opening",
        "conflict_present": False, "stakes_signal": "low",
    },
    "causality": {
        "why_this_happens": "interview opening", "likely_trigger": "show start",
        "likely_consequence": "guest responds",
    },
    "continuity": {
        "connects_to_previous": "cold open", "sets_up_next": "guest reply",
        "recurring_object": None, "recurring_motif": None,
    },
    "emotional_tone": {"scene_mood": "calm", "tension_level": "low", "note": ""},
    "themes_symbols": ["sports"],
    "searchable_facts": ["P1 interviews P2 about a football match."],
    "tags": ["interview", "studio"],
}


def test_valid_frame_parses_cleanly():
    parsed = QwenFrameOutput.model_validate(VALID_FRAME)
    assert parsed.id == "frame_0001"
    assert parsed.composition.shot == "medium"
    assert parsed.people[0].pid == "P1"
    assert len(parsed.get_relations_structured()) == 1


def test_relations_accepts_legacy_list_of_lists_format():
    """QwenFrameOutput.relations explicitly accepts both the structured
    QwenRelation shape AND the older `[subject, predicate, object]` list
    format — a documented dual-format tolerance, not drift."""
    frame = copy.deepcopy(VALID_FRAME)
    frame["relations"] = [["P1", "addresses", "P2"]]
    parsed = QwenFrameOutput.model_validate(frame)
    structured = parsed.get_relations_structured()
    assert len(structured) == 1
    assert structured[0].subject == "P1"
    assert structured[0].predicate == "addresses"


def test_missing_required_top_level_field_is_rejected():
    frame = copy.deepcopy(VALID_FRAME)
    del frame["caption"]
    with pytest.raises(ValidationError):
        QwenFrameOutput.model_validate(frame)


def test_missing_required_nested_setting_field_is_rejected():
    frame = copy.deepcopy(VALID_FRAME)
    del frame["setting"]["time_of_day"]
    with pytest.raises(ValidationError):
        QwenFrameOutput.model_validate(frame)


def test_extra_unexpected_top_level_field_is_ignored_not_rejected():
    """`model_config = ConfigDict(extra="ignore")` is a deliberate design
    choice throughout shared/types.py — Qwen adding a new field the schema
    doesn't know about yet must not crash the pipeline. Verifies that
    contract holds, not that extra fields are rejected."""
    frame = copy.deepcopy(VALID_FRAME)
    frame["a_field_qwen_invented_this_run"] = {"nested": "junk"}
    parsed = QwenFrameOutput.model_validate(frame)
    assert not hasattr(parsed, "a_field_qwen_invented_this_run")


def test_wrong_type_for_scalar_field_is_rejected():
    frame = copy.deepcopy(VALID_FRAME)
    frame["t"] = "not-a-number"
    with pytest.raises(ValidationError):
        QwenFrameOutput.model_validate(frame)


def test_wrong_type_for_scene_change_is_rejected():
    frame = copy.deepcopy(VALID_FRAME)
    # pydantic v2's bool coercion is lenient about common string spellings
    # ("yes"/"no"/"true"/"1"/...), so use a shape it can never coerce
    # (a dict) to actually exercise rejection of a genuinely wrong type.
    frame["scene_change"] = {"maybe": True}
    with pytest.raises(ValidationError):
        QwenFrameOutput.model_validate(frame)


# ---------------------------------------------------------------------------
# QwenComposition — closed-set coercion (not rejection) for shot/angle/movement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,bad_value,fallback", [
    ("shot", "extreme_wide_panoramic_drone", "unknown"),
    ("angle", "bird's eye deluxe", "unknown"),
    ("movement", "dolly_zoom_supreme", "static"),
])
def test_composition_coerces_unknown_enum_values_to_fallback(field, bad_value, fallback):
    """Recent hardening (git history: 'coerce unknown Qwen composition
    shot/angle/movement types to fallback instead of crashing') — these
    fields must NEVER raise on an out-of-vocabulary value; they silently
    fall back to a safe default instead."""
    payload = {"framing": "centered", "depth": "shallow", field: bad_value}
    parsed = QwenComposition.model_validate(payload)
    assert getattr(parsed, field) == fallback


def test_composition_accepts_valid_enum_values_unchanged():
    payload = {"shot": "close", "angle": "low", "movement": "handheld",
               "framing": "off-center", "depth": "deep"}
    parsed = QwenComposition.model_validate(payload)
    assert parsed.shot == "close"
    assert parsed.angle == "low"
    assert parsed.movement == "handheld"


def test_composition_coerces_wrong_type_for_enum_field():
    """A non-string value (e.g. Qwen emits a number) for a coerced enum
    field must not crash — falls back same as an out-of-vocabulary string."""
    payload = {"framing": "centered", "depth": "shallow", "shot": 42}
    parsed = QwenComposition.model_validate(payload)
    assert parsed.shot == "unknown"


# ---------------------------------------------------------------------------
# QwenPersonEntry.emotion_source — a real closed-set Literal (must reject)
# ---------------------------------------------------------------------------


def _person_entry(emotion_source):
    return {
        "pid": "P1", "pose": "seated", "action": "talking", "gaze": "at camera",
        "clothing": "suit", "emotion_inferred": "neutral",
        "emotion_source": emotion_source, "story_role": "host",
        "apparent_goal": "explain",
    }


def test_person_entry_rejects_value_outside_emotion_source_literal():
    with pytest.raises(ValidationError):
        QwenPersonEntry.model_validate(_person_entry("guessed_from_vibes"))


@pytest.mark.parametrize("valid_source", [
    "visual_read", "transcript_tone", "fallback_automated_guess", "not_determinable",
])
def test_person_entry_accepts_every_valid_emotion_source(valid_source):
    parsed = QwenPersonEntry.model_validate(_person_entry(valid_source))
    assert parsed.emotion_source == valid_source


# ---------------------------------------------------------------------------
# QwenPersonInteraction — coerced closed-set `type`, plus `is_valid` property
# ---------------------------------------------------------------------------


def test_person_interaction_coerces_unknown_type_to_observe():
    parsed = QwenPersonInteraction.model_validate(
        {"p1": "P1", "p2": "P2", "type": "comforts", "notes": "n/a"}
    )
    assert parsed.type == "observe"


def test_person_interaction_is_valid_requires_two_distinct_people():
    assert QwenPersonInteraction.model_validate({"p1": "P1", "p2": "P2"}).is_valid is True
    assert QwenPersonInteraction.model_validate({"p1": "P1", "p2": "P1"}).is_valid is False
    assert QwenPersonInteraction.model_validate({"p1": None, "p2": "P2"}).is_valid is False
    assert QwenPersonInteraction.model_validate({}).is_valid is False
