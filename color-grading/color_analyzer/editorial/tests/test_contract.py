"""The output contract: what the document contains, and what it must never contain."""

from __future__ import annotations

import json

import numpy as np
import pytest

from color_analyzer.editorial import EditorialAnalyzer, SchemaError, validate
from color_analyzer.editorial import schema

from .conftest import portrait_frame


@pytest.fixture(scope="module")
def document():
    return EditorialAnalyzer().analyze_rgb(portrait_frame())


# -- sections ---------------------------------------------------------------
def test_document_has_exactly_the_declared_sections(document):
    assert set(document) == set(schema.SECTIONS)


def test_requested_controls_are_all_present(document):
    """Every control named in the specification has a home in the output."""
    assert {"overall_look", "brightness", "contrast", "colorfulness", "mood"} <= set(document["look"])
    assert {"temperature", "tint"} <= set(document["white_balance"])
    assert {"exposure", "gamma", "black_point", "white_point"} <= set(document["tone"])
    assert {"saturation", "vibrance"} <= set(document["color"])
    assert set(document["wheels"]) == {"lift", "gamma", "gain"}
    assert {"shadows", "highlights"} <= set(document["split_toning"])
    assert document["palette"] and document["skin_tone"]
    assert set(document["hsl"]) == {
        "red", "orange", "yellow", "green", "cyan", "blue", "purple"
    }


def test_hsl_bands_are_in_panel_order(document):
    assert tuple(document["hsl"]) == schema.HSL_BANDS


def test_wheels_expose_channel_and_luma_controls(document):
    for wheel in ("lift", "gamma", "gain"):
        assert {"red", "green", "blue", "luma"} <= set(document["wheels"][wheel])


# -- the ban ----------------------------------------------------------------
def test_no_forbidden_key_anywhere(document):
    validate(document)  # raises on any forbidden key at any depth


@pytest.mark.parametrize("banned", [
    "histogram", "rgb_histogram", "entropy", "variance", "kurtosis",
    "skewness", "feature_vector", "histogram_peaks", "histogram_valleys",
    "lab", "lab_mean", "xyz", "ycrcb", "moments", "std", "percentile",
])
def test_validate_rejects_a_forbidden_key(document, banned):
    broken = json.loads(json.dumps(document))
    broken["tone"][banned] = 1.0
    with pytest.raises(SchemaError, match="forbidden key"):
        validate(broken)


def test_validate_allows_innocent_keys_containing_banned_substrings(document):
    """Short bans are matched on token boundaries, not as raw substrings."""
    broken = json.loads(json.dumps(document))
    broken["look"]["label"] = "warm"          # contains "lab"
    broken["tone"]["standard_range"] = True   # contains "std"
    broken["color"]["variant"] = "a"          # contains "var"
    validate(broken)


def test_validate_rejects_a_long_array(document):
    broken = json.loads(json.dumps(document))
    broken["tone"]["curve"] = list(range(256))
    with pytest.raises(SchemaError, match="statistical data"):
        validate(broken)


def test_no_array_in_the_document_is_a_distribution(document):
    """Nothing an editor adjusts is hundreds of numbers long."""
    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            assert len(node) <= schema.MAX_ARRAY_LENGTH
            for value in node:
                walk(value)

    walk(document)


def test_validate_rejects_a_missing_section(document):
    broken = json.loads(json.dumps(document))
    del broken["hsl"]
    with pytest.raises(SchemaError, match="missing sections"):
        validate(broken)


# -- serialisation ----------------------------------------------------------
def test_document_is_json_serialisable_without_nan(document):
    text = json.dumps(document)
    assert "NaN" not in text and "Infinity" not in text


def test_document_contains_no_numpy_scalars(document):
    """Plain Python types only, so the JSON survives any serialiser."""
    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        else:
            assert not isinstance(node, np.generic), f"numpy scalar leaked: {node!r}"

    walk(document)
