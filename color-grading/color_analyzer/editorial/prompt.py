"""Build the payload a grading model is asked to edit.

A model handed the raw analysis has no way to know that ``wheels.lift.red: 63``
means the shadows carry a red cast, that temperature is in Kelvin and *lower* is
warmer, or that it must return numbers rather than prose. This module assembles
the three things it needs: the controls it may change, enough read-only context
to make a sensible decision, and instructions describing both.

The payload is deliberately small — the controls alone are under 2 KB — so it
fits comfortably in a prompt alongside the instruction and leaves room for the
reply.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from .controls import CONTROLS, extract

#: Sections summarised as context. The model reads these to decide; it cannot
#: set them, and any attempt to is dropped by :func:`~.controls.sanitise`.
CONTEXT_KEYS = ("look", "palette", "skin_tone")


def build_payload(state: Mapping[str, Any], instruction: str) -> Dict[str, Any]:
    """Assemble the object to send to a grading model.

    Parameters
    ----------
    state:
        An analysis document from :class:`~.analyzer.EditorialAnalyzer`.
    instruction:
        What the grade should achieve, in plain language — "dark cinematic",
        "warm and nostalgic", "clean commercial".
    """
    return {
        "instruction": instruction,
        "how_to_respond": _response_rules(),
        "context": _context(state),
        "controls": extract(state),
        "control_reference": _reference(),
    }


def build_prompt(state: Mapping[str, Any], instruction: str) -> str:
    """The payload as a single prompt string, ready to send."""
    payload = build_payload(state, instruction)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Grade this frame: {instruction}\n\n"
        f"{json.dumps({k: v for k, v in payload.items() if k != 'how_to_respond'}, indent=2)}"
    )


SYSTEM_PROMPT = """\
You are a colourist. You will be given the measured colour state of a single \
video frame and a description of the look it should have. Return the same \
`controls` object with the values changed to what they should become.

The numbers are a *state*, not adjustments. `temperature: 5200` means the frame \
currently sits at 5200 K; return 4200 to make it warmer, 7000 to make it cooler. \
The renderer works out the difference.

Leave a control unchanged if the look does not call for moving it. Do not invent \
controls, do not change anything under `context`, and return JSON only — no \
explanation, no markdown fence.\
"""


def _response_rules() -> List[str]:
    return [
        "Return only the `controls` object, with the same shape, as raw JSON.",
        "Values are target states, not offsets: return what the reading should become.",
        "Leave controls you do not want to move at their current value.",
        "Stay inside each control's stated range.",
        "Do not add controls, and do not return anything from `context`.",
    ]


def _context(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The read-only facts worth knowing before deciding on a grade."""
    context: Dict[str, Any] = {
        key: state[key] for key in CONTEXT_KEYS if key in state
    }
    context["notes"] = _caveats(state)
    return context


def _caveats(state: Mapping[str, Any]) -> List[str]:
    """Warnings about this specific frame that change what a grade can achieve.

    Without these a model asks for things the renderer cannot deliver and the
    result looks like a bug rather than a limit of the material.
    """
    notes: List[str] = [
        "Lowering exposure also lowers the measured contrast, because contrast is "
        "the absolute tonal spread. For a darker AND punchier look, raise contrast "
        "considerably more than the arithmetic suggests.",
    ]

    white_balance = state.get("white_balance", {})
    if float(white_balance.get("confidence", 1.0)) < 0.4:
        notes.append(
            f"White balance confidence is only {white_balance.get('confidence')}: this "
            f"frame has little neutral content, so its temperature reading is "
            f"unreliable and hard to steer. Prefer the colour wheels for a cast."
        )

    skin = state.get("skin_tone", {})
    if skin.get("detected"):
        notes.append(
            f"Skin covers {skin.get('coverage')} of the frame ({skin.get('tone')}). "
            f"Global moves are held back on skin to keep faces natural, so orange "
            f"and red controls will move less than requested."
        )

    absent = [band for band, values in state.get("hsl", {}).items()
              if float(values.get("presence", 0.0)) <= 0.0]
    if absent:
        notes.append(
            f"These hue bands are not present in the frame and cannot be graded: "
            f"{', '.join(absent)}."
        )

    return notes


def _reference() -> Dict[str, Dict[str, Any]]:
    """What each control means and the range it accepts.

    Grouped by prefix so the reference reads in the same shape as the controls,
    and so seven near-identical HSL bands are described once rather than
    twenty-one times.
    """
    reference: Dict[str, Dict[str, Any]] = {}
    seen_hsl = False

    for control in CONTROLS:
        group, field = control.path.split(".", 1)

        if group == "hsl":
            if seen_hsl:
                continue
            seen_hsl = True
            reference["hsl.<band>"] = {
                "hue": "Shifts that band toward a neighbouring hue. -100..100.",
                "saturation": "Colour intensity of that band alone. 0..100.",
                "luminance": "Brightness of that band alone. 0..100.",
                "bands": "red, orange, yellow, green, cyan, blue, purple",
                "note": "A band with presence 0 is not in the frame and will be ignored.",
            }
            continue

        entry = reference.setdefault(group, {})
        span = f"{control.lo:g}..{control.hi:g}"
        unit = f" {control.unit}" if control.unit else ""
        entry[field] = f"{control.description} Range {span}{unit}."

    return reference
