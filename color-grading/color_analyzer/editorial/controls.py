"""The editable surface: what a model is allowed to change, and how.

:mod:`.schema` declares what the analyzer *emits*; this declares the subset a
grading model may *set*.  The two are deliberately separate — most of the
analysis output is context (how much skin is in frame, what the palette is, how
confident the white-balance reading was) that informs a decision without being
a knob.

Everything here is a **target state**, not an adjustment.  A control reading
``temperature: 5200`` means the frame currently sits at 5200 K; returning
``4200`` asks for a warmer result.  :mod:`.apply` works out the difference.

Robustness is a feature
-----------------------
The payloads this parses come back from a language model, so they arrive
malformed on a regular basis: numbers as strings, invented keys, read-only
fields helpfully "updated", values an order of magnitude out of range, whole
sections missing.  :func:`sanitise` never raises.  A single bad field must not
cost the caller the other forty-six.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .hsl import BAND_CENTRES
from .scales import clamp

#: The three colour wheels and the four axes each exposes.
WHEELS: Tuple[str, ...] = ("lift", "gamma", "gain")
WHEEL_AXES: Tuple[str, ...] = ("red", "green", "blue", "luma")

#: The two ends of a split tone.
SPLIT_ZONES: Tuple[str, ...] = ("shadows", "highlights")

#: The seven hue bands, taken from the analyzer so the two cannot drift apart.
HSL_BANDS: Tuple[str, ...] = tuple(BAND_CENTRES)


@dataclass(frozen=True)
class Control:
    """One editable parameter.

    ``path`` is the dotted location in the analysis document, so a control maps
    to exactly one reading and back again.
    """

    path: str
    kind: str          # "float" | "int"
    lo: float
    hi: float
    unit: str = ""
    description: str = ""

    @property
    def parts(self) -> Tuple[str, ...]:
        return tuple(self.path.split("."))


def _c(path: str, lo: float, hi: float, description: str,
       kind: str = "float", unit: str = "") -> Control:
    return Control(path, kind, lo, hi, unit, description)


def _build_controls() -> Tuple[Control, ...]:
    """Assemble the control list in the order a colourist works."""
    controls: List[Control] = [
        # -- white balance --------------------------------------------------
        _c("white_balance.temperature", 2000, 12000,
           "Colour temperature of the light. Lower is warmer/oranger, higher is "
           "cooler/bluer.", kind="int", unit="K"),
        _c("white_balance.tint", -100, 100,
           "Green to magenta balance. Negative is greener, positive is more magenta."),

        # -- tone ------------------------------------------------------------
        _c("tone.exposure", -3, 3,
           "Overall brightness in photographic stops. -1 halves the light, +1 doubles it.",
           unit="stops"),
        _c("tone.brightness", -100, 100, "Overall lightness offset."),
        _c("tone.contrast", -100, 100,
           "Separation between light and dark. Negative flattens, positive adds punch."),
        _c("tone.gamma", 0.3, 3.0,
           "Midtone curve. Above 1 pushes midtones darker, below 1 lifts them."),
        # Both span the full signal range, not the "sensible" part of it. A
        # control's bounds have to cover everything the analyzer can report:
        # a dark frame legitimately measures a white point of 0.23, and clamping
        # that to a tidier floor turns an unedited value into a change the
        # caller never asked for, silently grading the frame on a no-op payload.
        _c("tone.black_point", 0.0, 1.0,
           "Signal level the darkest tones sit at. Raise it for lifted, milky blacks; "
           "lower it for deep crushed blacks."),
        _c("tone.white_point", 0.0, 1.0,
           "Signal level the brightest tones reach. Lower it to hold highlights back."),

        # -- colour ----------------------------------------------------------
        _c("color.saturation", -100, 100, "Overall colour intensity."),
        _c("color.vibrance", -100, 100,
           "Intensity of the muted colours only, leaving already-saturated ones alone."),
    ]

    # -- colour wheels -------------------------------------------------------
    zone_of = {"lift": "shadows", "gamma": "midtones", "gain": "highlights"}
    for wheel in WHEELS:
        zone = zone_of[wheel]
        for axis in WHEEL_AXES:
            if axis == "luma":
                text = f"Brightness of the {zone}."
            else:
                text = f"How much {axis} sits in the {zone}, relative to the other channels."
            controls.append(_c(f"wheels.{wheel}.{axis}", -100, 100, text))

    # -- split toning --------------------------------------------------------
    for zone in SPLIT_ZONES:
        controls.append(_c(f"split_toning.{zone}.hue", 0, 360,
                           f"Hue of the tint in the {zone}. 0 red, 60 yellow, 120 green, "
                           f"210 teal, 240 blue.", kind="int", unit="deg"))
        controls.append(_c(f"split_toning.{zone}.saturation", 0, 100,
                           f"Strength of the {zone} tint."))

    # -- HSL bands -----------------------------------------------------------
    for band in HSL_BANDS:
        controls.append(_c(f"hsl.{band}.hue", -100, 100,
                           f"Shifts {band} tones toward a neighbouring hue."))
        controls.append(_c(f"hsl.{band}.saturation", 0, 100,
                           f"Colour intensity of {band} tones only. 0 is grey, "
                           f"100 is fully saturated."))
        controls.append(_c(f"hsl.{band}.luminance", 0, 100,
                           f"Brightness of {band} tones only. 0 is black, 100 is white."))

    return tuple(controls)


CONTROLS: Tuple[Control, ...] = _build_controls()
CONTROL_BY_PATH: Dict[str, Control] = {c.path: c for c in CONTROLS}
CONTROL_PATHS: Tuple[str, ...] = tuple(c.path for c in CONTROLS)

#: Sections of the analysis document that hold no editable control at all.
CONTEXT_SECTIONS: Tuple[str, ...] = ("meta", "look", "palette", "skin_tone")


# ---------------------------------------------------------------------------
# reading controls out of an analysis document
# ---------------------------------------------------------------------------
def _dig(node: Any, parts: Tuple[str, ...]) -> Any:
    """Follow a dotted path through nested mappings, or ``None``."""
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def extract(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Pull the editable values out of an analysis document, nested by group.

    The shape mirrors the analysis document so a model can read one and return
    the other without transposing anything.
    """
    out: Dict[str, Any] = {}
    for control in CONTROLS:
        value = _dig(state, control.parts)
        if value is None:
            continue
        node = out
        for part in control.parts[:-1]:
            node = node.setdefault(part, {})
        node[control.parts[-1]] = value
    return out


def flatten(controls: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a nested control payload to ``{dotted_path: value}``."""
    out: Dict[str, Any] = {}
    for control in CONTROLS:
        value = _dig(controls, control.parts)
        if value is not None:
            out[control.path] = value
    return out


# ---------------------------------------------------------------------------
# sanitising a model's reply
# ---------------------------------------------------------------------------
@dataclass
class Sanitised:
    """Result of cleaning a model-supplied control payload.

    Attributes
    ----------
    values:
        ``{dotted_path: value}`` for controls that survived, coerced and clamped.
    ignored:
        Human-readable notes about keys that were dropped.
    clamped:
        Notes about values pulled back into range.
    """

    values: Dict[str, float] = field(default_factory=dict)
    ignored: List[str] = field(default_factory=list)
    clamped: List[str] = field(default_factory=list)


_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


def _coerce(value: Any) -> Optional[float]:
    """Best-effort number from whatever the model returned.

    Handles the shapes a language model actually emits: bare numbers, numeric
    strings, signed strings, and values with a unit stuck on the end
    (``"4200K"``, ``"+15 units"``). Anything with no number in it is rejected.
    """
    if isinstance(value, bool):
        return None  # a boolean is not a slider position
    if isinstance(value, (int, float)):
        return float(value) if _finite(float(value)) else None
    if isinstance(value, str):
        match = _NUMBER.search(value.replace(",", ""))
        if match:
            try:
                number = float(match.group())
            except ValueError:
                return None
            return number if _finite(number) else None
    return None


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def sanitise(payload: Any) -> Sanitised:
    """Clean a model-supplied control payload. Never raises.

    Accepts the nested group shape produced by :func:`extract`, a flat
    ``{"tone.contrast": 20}`` mapping, or a mixture. Unknown keys, read-only
    fields and unparseable values are dropped and reported rather than being
    allowed to fail the whole grade.
    """
    result = Sanitised()
    if not isinstance(payload, Mapping):
        result.ignored.append(
            f"payload is {type(payload).__name__}, expected an object; nothing applied"
        )
        return result

    for path, value in _walk(payload):
        control = CONTROL_BY_PATH.get(path)
        if control is None:
            result.ignored.append(_explain_unknown(path))
            continue

        number = _coerce(value)
        if number is None:
            result.ignored.append(f"{path}: {value!r} is not a number")
            continue

        limited = clamp(number, control.lo, control.hi)
        if limited != number:
            result.clamped.append(
                f"{path}: {number:g} out of range [{control.lo:g}, {control.hi:g}], "
                f"using {limited:g}"
            )
        result.values[path] = int(round(limited)) if control.kind == "int" else limited

    return result


def _walk(node: Mapping[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """Flatten arbitrary nesting to ``(dotted_path, value)`` leaves.

    Lists are not descended into: no control is a list, so a list is by
    definition something the caller should not have sent.
    """
    found: List[Tuple[str, Any]] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            found.extend(_walk(value, path))
        else:
            found.append((path, value))
    return found


def _explain_unknown(path: str) -> str:
    """Say *why* a key was dropped — the distinction matters when debugging."""
    head = path.split(".", 1)[0]
    if head in CONTEXT_SECTIONS:
        return f"{path}: '{head}' is read-only context, not a control"
    if path in {"palette", "look", "meta"}:
        return f"{path}: read-only"
    return f"{path}: not a known control"
