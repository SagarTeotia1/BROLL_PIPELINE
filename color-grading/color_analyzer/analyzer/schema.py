"""The 45-parameter grading contract.

This module is the **single source of truth** for what the engine emits.  Both
the decision engine and the report layer iterate :data:`PARAMS`, so the schema
cannot drift between them: adding a parameter here is the only way to add one to
the output, and :func:`assemble` refuses to emit a key that is not declared.

Why 45 and not the previous ~70+
--------------------------------
The old output modelled a *professional* panel — a full six-band HSL, four
colour wheels with tint axes, eight creative-style scores.  Most of it was not
actually measured: four of the six HSL bands were hard-coded constants, both
wheel tint axes were always zero, and ``luxury``/``editorial`` were linear
combinations of scores already present.  This schema keeps the parameters an
*intermediate* colourist actually reaches for, and every one of them is derived
from the image.

Read-only parameters
--------------------
Some parameters describe the frame rather than prescribe a change: style scores,
quality flags, whether skin is present.  These are marked ``readonly`` and carry
only a ``current`` value — asking for a "recommended" ``shadow_crush`` is
meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# Curve shapes the tone-curve parameter may take.
CURVE_TYPES: Tuple[str, ...] = ("linear", "soft_s", "gentle_s", "strong_s", "filmic", "faded", "lifted")
# Noise-risk buckets.
NOISE_LEVELS: Tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class Param:
    """One entry of the grading contract.

    Attributes
    ----------
    name:
        Dotted ``group.field`` key as it appears in the output.
    kind:
        ``"float"``, ``"int"``, ``"bool"`` or ``"enum"``.  Drives coercion and
        range validation in :func:`coerce`.
    lo, hi:
        Inclusive bounds for numeric parameters.  Values outside are clamped,
        not rejected — a heuristic that overshoots should still emit a legal
        grade rather than crash a video pipeline mid-render.
    readonly:
        Measured-only; carries ``current`` but never ``recommended``.
    choices:
        Allowed values for ``kind="enum"``.
    """

    name: str
    kind: str
    lo: Optional[float] = None
    hi: Optional[float] = None
    readonly: bool = False
    unit: str = ""
    choices: Tuple[str, ...] = ()

    @property
    def group(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def field(self) -> str:
        return self.name.split(".", 1)[1]


def _f(name: str, lo: float, hi: float, unit: str = "") -> Param:
    return Param(name, "float", lo, hi, unit=unit)


def _ro(name: str, kind: str, lo: Optional[float] = None, hi: Optional[float] = None,
        unit: str = "", choices: Tuple[str, ...] = ()) -> Param:
    return Param(name, kind, lo, hi, readonly=True, unit=unit, choices=choices)


# ---------------------------------------------------------------------------
# The contract.  Order is the emission order; do not sort it downstream.
# ---------------------------------------------------------------------------
PARAMS: Tuple[Param, ...] = (
    # -- white balance (2) --------------------------------------------------
    _f("white_balance.temperature", 2000.0, 12000.0, unit="K"),
    _f("white_balance.tint", -100.0, 100.0),

    # -- primary corrections (7) -------------------------------------------
    _f("primary.exposure", -3.0, 3.0, unit="stops"),
    _f("primary.contrast", -100.0, 100.0),
    _f("primary.highlights", -100.0, 100.0),
    _f("primary.shadows", -100.0, 100.0),
    _f("primary.whites", -100.0, 100.0),
    _f("primary.blacks", -100.0, 100.0),
    _f("primary.gamma", 0.5, 2.0),

    # -- presence (5) -------------------------------------------------------
    _f("presence.vibrance", -100.0, 100.0),
    _f("presence.saturation", -100.0, 100.0),
    _f("presence.clarity", -100.0, 100.0),
    _f("presence.texture", -100.0, 100.0),
    _f("presence.dehaze", -100.0, 100.0),

    # -- tone curve (5) -----------------------------------------------------
    Param("tone_curve.curve_type", "enum", choices=CURVE_TYPES),
    _f("tone_curve.shadow_lift", 0.0, 1.0),
    _f("tone_curve.midtone", -1.0, 1.0),
    _f("tone_curve.highlight_rolloff", 0.0, 1.0),
    _f("tone_curve.contrast_strength", 0.0, 1.0),

    # -- colour wheels (6) --------------------------------------------------
    # temp: -100 (cool/teal) .. 100 (warm/orange); strength: 0..1
    _f("color_wheels.lift_temp", -100.0, 100.0),
    _f("color_wheels.lift_strength", 0.0, 1.0),
    _f("color_wheels.gamma_temp", -100.0, 100.0),
    _f("color_wheels.gamma_strength", 0.0, 1.0),
    _f("color_wheels.gain_temp", -100.0, 100.0),
    _f("color_wheels.gain_strength", 0.0, 1.0),

    # -- split toning (4) ---------------------------------------------------
    _f("split_toning.shadow_hue", 0.0, 360.0, unit="deg"),
    _f("split_toning.shadow_sat", 0.0, 100.0),
    _f("split_toning.highlight_hue", 0.0, 360.0, unit="deg"),
    _f("split_toning.highlight_sat", 0.0, 100.0),

    # -- HSL: only the two bands that are actually measured (4) -------------
    # orange == skin, blue == background separation.
    _f("hsl.orange_sat", -100.0, 100.0),
    _f("hsl.orange_lum", -100.0, 100.0),
    _f("hsl.blue_sat", -100.0, 100.0),
    _f("hsl.blue_lum", -100.0, 100.0),

    # -- subject (4) --------------------------------------------------------
    _ro("subject.skin_present", "bool"),
    _f("subject.skin_warmth", 0.0, 1.0),
    _f("subject.skin_luminance", 0.0, 1.0),
    _f("subject.subject_pop", 0.0, 1.0),

    # -- creative style, measured only (4) ----------------------------------
    _ro("creative_style.cinematic", "float", 0.0, 1.0),
    _ro("creative_style.commercial", "float", 0.0, 1.0),
    _ro("creative_style.natural", "float", 0.0, 1.0),
    _ro("creative_style.moody", "float", 0.0, 1.0),

    # -- quality, measured only (4) -----------------------------------------
    _ro("quality.highlight_clipping", "bool"),
    _ro("quality.shadow_crush", "bool"),
    _ro("quality.dynamic_range", "float", 0.0, 20.0, unit="stops"),
    _ro("quality.noise_risk", "enum", choices=NOISE_LEVELS),
)

PARAM_BY_NAME: Dict[str, Param] = {p.name: p for p in PARAMS}
PARAM_NAMES: Tuple[str, ...] = tuple(p.name for p in PARAMS)
ADJUSTABLE: Tuple[Param, ...] = tuple(p for p in PARAMS if not p.readonly)
READONLY: Tuple[Param, ...] = tuple(p for p in PARAMS if p.readonly)

# Emission order of the groups.
GROUPS: Tuple[str, ...] = tuple(dict.fromkeys(p.group for p in PARAMS))


def coerce(param: Param, value: Any) -> Any:
    """Coerce ``value`` to ``param``'s declared type and range.

    Numeric values are clamped into ``[lo, hi]`` rather than rejected; an enum
    value outside ``choices`` falls back to the first choice.  A ``None`` stays
    ``None`` so callers can distinguish "not applicable" from "zero".
    """
    if value is None:
        return None
    if param.kind == "bool":
        return bool(value)
    if param.kind == "enum":
        text = str(value)
        return text if text in param.choices else param.choices[0]
    number = float(value)
    if param.lo is not None:
        number = max(param.lo, number)
    if param.hi is not None:
        number = min(param.hi, number)
    if param.kind == "int":
        return int(round(number))
    # Kelvin and hue are read as whole numbers by every grading UI.
    digits = 0 if param.unit in ("K", "deg") else 3
    rounded = round(number, digits)
    return int(rounded) if digits == 0 else (0.0 if rounded == 0 else rounded)


def assemble(
    current: Mapping[str, Any],
    recommended: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
    style: Optional[Mapping[str, Any]] = None,
    notes: Optional[Sequence[str]] = None,
    palette: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build the unified grade document from two flat ``{param_name: value}`` maps.

    Every declared parameter appears in the output exactly once, in
    :data:`PARAMS` order.  A parameter missing from ``current`` emits ``None``
    rather than being dropped, so consumers can rely on a fixed key set.

    Raises
    ------
    KeyError
        If either mapping contains a name that is not in the contract — that is
        a bug in the producer, not a value to silently ignore.
    """
    for source, label in ((current, "current"), (recommended, "recommended")):
        unknown = set(source) - set(PARAM_BY_NAME)
        if unknown:
            raise KeyError(f"{label} contains parameters not in the schema: {sorted(unknown)}")

    grade: Dict[str, Dict[str, Any]] = {}
    for param in PARAMS:
        entry: Dict[str, Any] = {"current": coerce(param, current.get(param.name))}
        if not param.readonly:
            rec = coerce(param, recommended.get(param.name))
            entry["recommended"] = rec
            # A delta is only meaningful for numbers present on both sides. It is
            # deliberately NOT run through coerce(): a delta lives on a different
            # scale than the parameter, so clamping it to [lo, hi] would turn a
            # -1150 K correction into the 2000 K floor.
            if param.kind in ("float", "int") and entry["current"] is not None and rec is not None:
                digits = 0 if param.unit in ("K", "deg") else 3
                delta = round(float(rec) - float(entry["current"]), digits)
                entry["delta"] = int(delta) if digits == 0 else (0.0 if delta == 0 else delta)
        grade[param.name] = entry

    return {
        "meta": dict(meta or {}),
        "style": dict(style or {}),
        "grade": grade,
        "notes": list(notes or []),
        "palette": list(palette or []),
    }


def flatten(document: Mapping[str, Any], field: str = "recommended") -> Dict[str, Any]:
    """Flatten a grade document to ``{param_name: value}`` for a renderer.

    ``field`` selects ``"recommended"`` (what to apply) or ``"current"`` (what
    was measured).  Read-only parameters have no ``recommended`` value and are
    skipped when that field is requested.
    """
    grade = document.get("grade", {})
    out: Dict[str, Any] = {}
    for name in PARAM_NAMES:
        entry = grade.get(name)
        if isinstance(entry, Mapping) and field in entry and entry[field] is not None:
            out[name] = entry[field]
    return out


def validate(document: Mapping[str, Any]) -> None:
    """Assert that ``document`` matches the contract exactly.

    Used by the tests and cheap enough to call in a pipeline's smoke check.
    """
    grade = document.get("grade")
    if not isinstance(grade, Mapping):
        raise ValueError("document has no 'grade' mapping")
    if tuple(grade) != PARAM_NAMES:
        missing = set(PARAM_NAMES) - set(grade)
        extra = set(grade) - set(PARAM_NAMES)
        raise ValueError(f"grade key mismatch (missing={sorted(missing)}, extra={sorted(extra)})")

    for param in PARAMS:
        entry = grade[param.name]
        if param.readonly and "recommended" in entry:
            raise ValueError(f"{param.name} is read-only but carries a 'recommended' value")
        if not param.readonly and "recommended" not in entry:
            raise ValueError(f"{param.name} is adjustable but has no 'recommended' value")
        for key in ("current", "recommended"):
            value = entry.get(key)
            if value is None or key not in entry:
                continue
            if param.kind == "enum" and value not in param.choices:
                raise ValueError(f"{param.name}.{key}={value!r} not in {param.choices}")
            if param.kind in ("float", "int"):
                if param.lo is not None and float(value) < param.lo - 1e-6:
                    raise ValueError(f"{param.name}.{key}={value} below {param.lo}")
                if param.hi is not None and float(value) > param.hi + 1e-6:
                    raise ValueError(f"{param.name}.{key}={value} above {param.hi}")
