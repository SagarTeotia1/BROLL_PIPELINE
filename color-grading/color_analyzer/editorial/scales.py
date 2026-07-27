"""Slider scales, quantisation and the labelling vocabulary.

Every number this engine emits is on a scale an editor already knows — the
-100..100 of a Lumetri or Lightroom slider, Kelvin for temperature, degrees for
hue — so a downstream model can move a value without first learning what a
"Hasler-Susstrunk index" is.

Quantisation is a correctness feature, not cosmetics
----------------------------------------------------
The engine promises stable output across similar frames.  Two consecutive
frames of the same shot differ by sensor noise, so a raw measurement wobbles in
the third decimal; emitted unrounded, that wobble becomes a different grade on
every frame and the result flickers.  Each quantity is therefore snapped to the
coarsest step that still carries editorial meaning: whole slider units, 50 K of
colour temperature, one degree of hue.
"""

from __future__ import annotations

from typing import Sequence, Tuple

# Quantisation steps, applied by the helpers below.
SLIDER_STEP = 1.0        # -100..100 controls
TEMPERATURE_STEP = 50.0  # Kelvin
HUE_STEP = 1.0           # degrees


def clamp(value: float, low: float, high: float) -> float:
    """Clamp to ``[low, high]``."""
    return low if value < low else high if value > high else value


def slider(value: float, low: float = -100.0, high: float = 100.0) -> int:
    """Quantise to a whole slider unit within ``[low, high]``."""
    return int(round(clamp(float(value), low, high) / SLIDER_STEP) * SLIDER_STEP)


def ratio(value: float, digits: int = 3, low: float = 0.0, high: float = 1.0) -> float:
    """Quantise a ``[low, high]`` ratio (coverage, black point, ...)."""
    out = round(clamp(float(value), low, high), digits)
    return 0.0 if out == 0 else out


def stops(value: float, low: float = -5.0, high: float = 5.0) -> float:
    """Quantise an exposure value in stops."""
    out = round(clamp(float(value), low, high), 2)
    return 0.0 if out == 0 else out


def kelvin(value: float, low: float = 2000.0, high: float = 12000.0) -> int:
    """Quantise a colour temperature to the nearest :data:`TEMPERATURE_STEP`."""
    snapped = round(clamp(float(value), low, high) / TEMPERATURE_STEP) * TEMPERATURE_STEP
    return int(snapped)


def degrees(value: float) -> int:
    """Quantise a hue angle to whole degrees in ``[0, 360)``."""
    return int(round(float(value) % 360.0)) % 360


def centred(value: float, neutral: float, span: float,
            low: float = -100.0, high: float = 100.0) -> int:
    """Map a measurement onto a slider centred on ``neutral``.

    ``span`` is the measurement distance that corresponds to a full-scale
    deflection, so ``value == neutral + span`` reports ``+100``.
    """
    if span <= 0:
        return 0
    return slider((float(value) - neutral) / span * 100.0, low, high)


def label(value: float, thresholds: Sequence[Tuple[float, str]], default: str) -> str:
    """Map ``value`` to a word using ascending ``(upper_bound, label)`` pairs."""
    for upper, text in thresholds:
        if value < upper:
            return text
    return default


def hex_color(rgb01: Sequence[float]) -> str:
    """``#rrggbb`` for an RGB triple in ``[0,1]``."""
    r, g, b = (int(clamp(float(c), 0.0, 1.0) * 255 + 0.5) for c in rgb01)
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_255(rgb01: Sequence[float]) -> list:
    """8-bit RGB triple for an RGB triple in ``[0,1]``."""
    return [int(clamp(float(c), 0.0, 1.0) * 255 + 0.5) for c in rgb01]
