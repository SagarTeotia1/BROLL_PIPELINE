"""Look stage: the one-line summary and the mood word.

Everything here is derived from readings the other stages already produced — no
pixels are touched.  The point is to give a downstream model, or a human
skimming the JSON, a handle on the frame before it reads 40 numbers.

``mood`` is scored rather than branched.  A cascade of ``if`` statements makes
the first matching rule win, so a frame that is both dark and warm gets whichever
test happened to be written first.  Scoring every candidate and taking the best
makes the choice explicit, and lets ``confidence`` report how clear-cut it was.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .color import describe_colorfulness
from .scales import clamp, ratio
from .tone import describe_brightness, describe_contrast
from .white_balance import describe_temperature

#: Mood name -> the conditions that support it. Each entry is
#: ``(reading, target, tolerance, weight)`` where reading is a normalised
#: -1..1 axis extracted below.
_MOOD_PROFILES: Dict[str, Tuple[Tuple[str, float, float, float], ...]] = {
    "moody": (("brightness", -0.5, 1.0, 1.0), ("saturation", -0.3, 1.0, 0.7),
              ("warmth", -0.3, 1.2, 0.5)),
    "warm": (("warmth", 0.6, 0.9, 1.2), ("brightness", 0.1, 1.2, 0.4)),
    "cool": (("warmth", -0.6, 0.9, 1.2), ("brightness", 0.0, 1.2, 0.4)),
    "vibrant": (("saturation", 0.6, 0.8, 1.2), ("contrast", 0.3, 1.0, 0.6)),
    "muted": (("saturation", -0.6, 0.8, 1.2), ("contrast", -0.2, 1.0, 0.5)),
    "bright": (("brightness", 0.6, 0.9, 1.2), ("contrast", -0.1, 1.2, 0.4)),
    "neutral": (("brightness", 0.0, 0.7, 1.0), ("saturation", 0.0, 0.7, 1.0),
                ("warmth", 0.0, 0.7, 1.0), ("contrast", 0.0, 0.8, 0.6)),
    # The two contrast-led moods also require the frame to be unremarkable in
    # colour. Without that they are single-condition profiles that score a
    # perfect 1.0 on any flat frame and hijack one that is flat *and* strongly
    # warm — where "warm" is the more useful thing to say.
    "high-contrast": (("contrast", 0.8, 0.7, 1.4), ("warmth", 0.0, 1.0, 0.6),
                      ("saturation", 0.0, 1.2, 0.4)),
    "flat": (("contrast", -0.8, 0.7, 1.4), ("warmth", 0.0, 1.0, 0.6),
             ("saturation", 0.0, 1.2, 0.4)),
}


def analyze_look(tone: Dict[str, Any], white_balance: Dict[str, Any],
                 color: Dict[str, Any], split: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise the frame from the other stages' readings."""
    axes = _axes(tone, white_balance, color)
    mood, confidence = _score_mood(axes)

    brightness = describe_brightness(tone)
    contrast = describe_contrast(tone)
    colorfulness = describe_colorfulness(color)
    temperature = describe_temperature(white_balance)

    return {
        "overall_look": _phrase(temperature, contrast, colorfulness, split),
        "mood": mood,
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "temperature": temperature,
        "confidence": ratio(confidence, digits=2),
    }


def _axes(tone: Dict[str, Any], white_balance: Dict[str, Any],
          color: Dict[str, Any]) -> Dict[str, float]:
    """Normalise the readings the mood model uses onto -1..1 axes."""
    return {
        "brightness": clamp(tone["brightness"] / 100.0, -1.0, 1.0),
        "contrast": clamp(tone["contrast"] / 100.0, -1.0, 1.0),
        "saturation": clamp(color["saturation"] / 100.0, -1.0, 1.0),
        # Warm light has a *low* Kelvin number, so the axis is inverted to make
        # positive mean warm — which is what every other reading here does.
        "warmth": clamp((6500.0 - white_balance["temperature"]) / 2500.0, -1.0, 1.0),
    }


def _score_mood(axes: Dict[str, float]) -> Tuple[str, float]:
    """Best-scoring mood and how decisively it won.

    Each condition contributes a Gaussian-ish score on how close the reading is
    to its target; the mood with the highest weighted mean wins.  Confidence is
    the margin over the runner-up, so a frame that is equally "warm" and "moody"
    reports low confidence rather than an arbitrary pick.
    """
    scores: Dict[str, float] = {}
    for mood, conditions in _MOOD_PROFILES.items():
        total = 0.0
        weight_sum = 0.0
        for axis, target, tolerance, weight in conditions:
            distance = abs(axes[axis] - target) / tolerance
            total += weight * max(0.0, 1.0 - distance * distance)
            weight_sum += weight
        scores[mood] = total / weight_sum if weight_sum else 0.0

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = clamp(best_score * (0.5 + 0.5 * (best_score - runner_up) * 2.0), 0.0, 1.0)
    return best, confidence


def _phrase(temperature: str, contrast: str, colorfulness: str,
            split: Dict[str, Any]) -> str:
    """Human sentence fragment describing the look."""
    parts = [temperature, f"{contrast} contrast", colorfulness]
    if split.get("strength", 0) >= 25:
        parts.append("split-toned")
    return ", ".join(parts)
