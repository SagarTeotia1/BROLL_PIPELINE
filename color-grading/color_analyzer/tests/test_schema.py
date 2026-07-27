"""Contract tests for the 45-parameter grade document.

These guard the *shape* of the output — the promise downstream consumers rely
on.  Behaviour (does the engine recommend the right thing) lives in
``test_decision_engine.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from color_analyzer import ColorGradingEngine
from color_analyzer.analyzer import schema
from color_analyzer.analyzer.decision_engine import to_executor_decision


def _image(seed: int = 0) -> np.ndarray:
    """A frame with skin-ish left half, cool right half and a little noise."""
    rng = np.random.default_rng(seed)
    x = np.full((120, 180, 3), 0.5, np.float32)
    x[:, :90] = (0.6, 0.48, 0.42)
    x[:, 90:] = (0.3, 0.42, 0.5)
    return np.clip(x + rng.random((120, 180, 3)).astype(np.float32) * 0.03, 0, 1)


@pytest.fixture(scope="module")
def document():
    return ColorGradingEngine().grade(_image())


# -- the contract itself ----------------------------------------------------
def test_schema_declares_exactly_45_parameters():
    assert len(schema.PARAMS) == 45
    assert len(set(schema.PARAM_NAMES)) == 45, "parameter names must be unique"


def test_every_parameter_is_dotted_group_field():
    for param in schema.PARAMS:
        assert "." in param.name, param.name
        assert param.group and param.field


def test_document_has_exactly_the_declared_parameters(document):
    assert tuple(document["grade"]) == schema.PARAM_NAMES


def test_document_validates(document):
    schema.validate(document)  # raises on any contract violation


def test_top_level_keys(document):
    assert set(document) == {"meta", "style", "grade", "notes", "palette"}


# -- read-only vs adjustable ------------------------------------------------
def test_readonly_parameters_have_no_recommendation(document):
    for param in schema.READONLY:
        entry = document["grade"][param.name]
        assert "recommended" not in entry, param.name
        assert "delta" not in entry, param.name
        assert "current" in entry, param.name


def test_adjustable_parameters_carry_current_and_recommended(document):
    for param in schema.ADJUSTABLE:
        entry = document["grade"][param.name]
        assert entry.get("current") is not None, param.name
        assert entry.get("recommended") is not None, param.name


def test_delta_is_recommended_minus_current(document):
    for param in schema.ADJUSTABLE:
        if param.kind not in ("float", "int"):
            continue
        entry = document["grade"][param.name]
        expected = float(entry["recommended"]) - float(entry["current"])
        assert entry["delta"] == pytest.approx(expected, abs=1e-2), param.name


# -- ranges and types -------------------------------------------------------
def test_values_stay_inside_declared_ranges(document):
    for param in schema.PARAMS:
        entry = document["grade"][param.name]
        for key in ("current", "recommended"):
            if key not in entry or entry[key] is None:
                continue
            value = entry[key]
            if param.kind in ("float", "int"):
                assert param.lo <= float(value) <= param.hi, f"{param.name}.{key}={value}"
            elif param.kind == "enum":
                assert value in param.choices, f"{param.name}.{key}={value}"
            elif param.kind == "bool":
                assert isinstance(value, bool), f"{param.name}.{key}={value}"


def test_coerce_clamps_rather_than_raising():
    param = schema.PARAM_BY_NAME["primary.contrast"]
    assert schema.coerce(param, 5000.0) == param.hi
    assert schema.coerce(param, -5000.0) == param.lo


def test_coerce_falls_back_for_unknown_enum():
    param = schema.PARAM_BY_NAME["tone_curve.curve_type"]
    assert schema.coerce(param, "nonsense") == param.choices[0]


def test_kelvin_and_hue_are_whole_numbers(document):
    for name in ("white_balance.temperature", "split_toning.shadow_hue"):
        entry = document["grade"][name]
        assert isinstance(entry["current"], int), name
        assert isinstance(entry["recommended"], int), name


# -- assemble rejects producer bugs ----------------------------------------
def test_assemble_rejects_unknown_parameter():
    with pytest.raises(KeyError, match="not in the schema"):
        schema.assemble(current={"made.up": 1.0}, recommended={})


def test_assemble_emits_none_for_missing_parameter():
    doc = schema.assemble(current={}, recommended={})
    assert tuple(doc["grade"]) == schema.PARAM_NAMES
    assert doc["grade"]["primary.contrast"]["current"] is None


def test_validate_catches_a_missing_parameter(document):
    broken = json.loads(json.dumps(document))
    broken["grade"].pop("primary.contrast")
    with pytest.raises(ValueError, match="key mismatch"):
        schema.validate(broken)


# -- flatten ----------------------------------------------------------------
def test_flatten_recommended_covers_every_adjustable(document):
    flat = schema.flatten(document, "recommended")
    assert set(flat) == {p.name for p in schema.ADJUSTABLE}


def test_flatten_current_covers_every_parameter(document):
    flat = schema.flatten(document, "current")
    assert set(flat) == set(schema.PARAM_NAMES)


# -- serialisation ----------------------------------------------------------
def test_document_is_json_serialisable_without_nan(document):
    text = json.dumps(document)
    assert "NaN" not in text and "Infinity" not in text


def test_executor_decision_has_the_sections_the_renderer_reads(document):
    plan = to_executor_decision(document)
    assert {"white_balance", "primary_corrections", "tone_curve", "color_wheels",
            "hsl_adjustments", "presence"}.issubset(plan)
