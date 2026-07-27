"""Colour-grading *application* (the inverse of analysis).

While the rest of the package **measures** a grade, this module **applies** one:
given a compact set of grading controls (the kind a colourist sets in a video
editor — temperature, contrast, lift/gamma/gain, split toning, …) it renders a
graded image.  It is used by the app's "Grade" panel, where an uploaded JSON of
these controls auto-populates the sliders and the preview updates live.

The pipeline is fully vectorised against the ``xp`` backend, so it runs on GPU
(CuPy) or CPU (NumPy) with a single implementation.  Operations are applied in a
conventional order (white balance → exposure → levels → tone → contrast →
saturation → split toning) and the result is clamped to ``[0, 1]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

import numpy as np

from .utils import Backend, rgb_to_luminance


@dataclass
class GradingParams:
    """Colour-grading controls.  Every default is the identity (no change).

    Ranges (for UI sliders) are documented per field; values outside are
    clamped on load.  ``to_dict``/``from_dict`` make this trivially JSON round-
    trippable so a grade can be saved, shared and re-applied.
    """

    temperature: float = 0.0      # [-100, 100]  warm(+) / cool(-)
    tint: float = 0.0             # [-100, 100]  magenta(+) / green(-)
    exposure: float = 0.0         # [-3, 3] stops
    contrast: float = 1.0         # [0, 2]   1 = neutral
    saturation: float = 1.0       # [0, 2]   1 = neutral
    vibrance: float = 0.0         # [-1, 1]  boosts low-saturation colours
    highlights: float = 0.0       # [-1, 1]  lift/lower highlights
    shadows: float = 0.0          # [-1, 1]  lift/lower shadows
    gamma: float = 1.0            # [0.2, 3] midtone gamma (>1 brightens)
    black_point: float = 0.0      # [0, 0.5] input black level
    white_point: float = 1.0      # [0.5, 1] input white level
    split_shadow_hue: float = 210.0       # [0, 360] deg
    split_shadow_strength: float = 0.0    # [0, 1]
    split_highlight_hue: float = 45.0     # [0, 360] deg
    split_highlight_strength: float = 0.0  # [0, 1]

    # inclusive (min, max) ranges used for clamping + UI slider bounds.
    _RANGES: Dict[str, tuple] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.pop("_RANGES", None)
        return {k: float(v) for k, v in d.items()}

    @classmethod
    def ranges(cls) -> Dict[str, tuple]:
        return {
            "temperature": (-100.0, 100.0),
            "tint": (-100.0, 100.0),
            "exposure": (-3.0, 3.0),
            "contrast": (0.0, 2.0),
            "saturation": (0.0, 2.0),
            "vibrance": (-1.0, 1.0),
            "highlights": (-1.0, 1.0),
            "shadows": (-1.0, 1.0),
            "gamma": (0.2, 3.0),
            "black_point": (0.0, 0.5),
            "white_point": (0.5, 1.0),
            "split_shadow_hue": (0.0, 360.0),
            "split_shadow_strength": (0.0, 1.0),
            "split_highlight_hue": (0.0, 360.0),
            "split_highlight_strength": (0.0, 1.0),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradingParams":
        """Build from a (possibly partial) dict, ignoring unknown keys.

        Accepts either a flat ``{field: value}`` mapping or a wrapper dict with
        a ``"grading"`` sub-object.  Values are clamped to their valid range.
        """
        if "grading" in data and isinstance(data["grading"], dict):
            data = data["grading"]
        ranges = cls.ranges()
        valid = {f.name for f in fields(cls) if not f.name.startswith("_")}
        kwargs: Dict[str, float] = {}
        for key, value in data.items():
            if key in valid and isinstance(value, (int, float)):
                lo, hi = ranges[key]
                kwargs[key] = float(min(max(float(value), lo), hi))
        return cls(**kwargs)

    @classmethod
    def is_grading_dict(cls, data: Dict[str, Any]) -> bool:
        """True if ``data`` looks like grading controls (vs an analysis report)."""
        if "grading" in data:
            return True
        valid = {f.name for f in fields(cls) if not f.name.startswith("_")}
        return len(valid.intersection(data.keys())) >= 3

    # -- derive a grade from an analysis report -----------------------------
    @classmethod
    def from_analysis(cls, report: Dict[str, Any]) -> "GradingParams":
        """Approximate grading controls that reproduce an analysed *look*.

        Lets a user drop in a ``report.json`` (produced by the analysis engine
        for a reference frame) and get a plausible preset that pushes an image
        toward that reference's temperature, contrast, saturation, tone and
        split toning.  It is a best-effort look-transfer, not an exact inverse.
        """
        fv = report.get("feature_vector", report)

        def g(key: str, default: float) -> float:
            v = fv.get(key, default)
            return float(v) if isinstance(v, (int, float)) else default

        cct = g("white_balance.color_temperature", 6500.0)
        # Warmer reference (lower K) -> positive temperature control.
        temperature = max(-100.0, min(100.0, (6500.0 - cct) / 3500.0 * 100.0))
        tint = max(-100.0, min(100.0, g("white_balance.tint", 0.0) * 100.0))

        global_contrast = g("contrast.global_contrast", 0.3)
        contrast = max(0.5, min(1.8, global_contrast / 0.3))
        richness = g("colorfulness.color_richness", 0.4)
        saturation = max(0.4, min(1.8, 0.6 + richness))

        params = cls(
            temperature=temperature,
            tint=tint,
            contrast=contrast,
            saturation=saturation,
            gamma=max(0.5, min(2.0, g("tone_curve.gamma", 1.0))),
            black_point=max(0.0, min(0.4, g("tone_curve.black_point", 0.0))),
            white_point=max(0.6, min(1.0, g("tone_curve.white_point", 1.0))),
        )
        # Transfer split toning only if the reference showed a confident split.
        if g("split_toning.split_tone_confidence", 0.0) > 0.35:
            params.split_shadow_hue = g("split_toning.shadows.hue", 210.0)
            params.split_shadow_strength = min(0.6, g("split_toning.shadows.saturation", 0.0))
            params.split_highlight_hue = g("split_toning.highlights.hue", 45.0)
            params.split_highlight_strength = min(0.6, g("split_toning.highlights.saturation", 0.0))
        return params


def _hue_to_rgb(hue_deg: float) -> tuple:
    """Fully saturated RGB (``[0,1]``) for a hue angle — used for split tints."""
    h = (hue_deg % 360.0) / 60.0
    c = 1.0
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    if h < 1:
        r, g, b = c, x, 0.0
    elif h < 2:
        r, g, b = x, c, 0.0
    elif h < 3:
        r, g, b = 0.0, c, x
    elif h < 4:
        r, g, b = 0.0, x, c
    elif h < 5:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return (r, g, b)


class ColorGrader:
    """Applies :class:`GradingParams` to an RGB image (vectorised, GPU/CPU)."""

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = backend or Backend()

    def apply(self, rgb01: np.ndarray, params: GradingParams) -> np.ndarray:
        """Return a graded copy of ``rgb01`` (float ``[0,1]``, shape ``(H,W,3)``).

        The output is always a host NumPy array so callers (e.g. the UI) can
        display/encode it directly.
        """
        xp = self.backend.xp
        x = xp.asarray(rgb01, dtype=xp.float32).copy()
        eps = 1e-6

        # 1) White balance: temperature tilts R vs B; tint tilts G.
        t = params.temperature / 100.0
        x[..., 0] *= 1.0 + 0.35 * t
        x[..., 2] *= 1.0 - 0.35 * t
        tint = params.tint / 100.0
        x[..., 1] *= 1.0 - 0.25 * tint

        # 2) Exposure in stops: out = in * 2^exposure.
        if params.exposure != 0.0:
            x *= float(2.0 ** params.exposure)

        # 3) Input levels (black/white point) remap to [0,1].
        span = max(params.white_point - params.black_point, eps)
        x = (x - params.black_point) / span

        # 4) Shadow/highlight lift using luminance-weighted masks.
        luma = rgb_to_luminance(xp, xp.clip(x, 0.0, 1.0))[..., None]
        shadow_w = xp.clip(1.0 - luma * 2.0, 0.0, 1.0)      # strong in darks
        highlight_w = xp.clip(luma * 2.0 - 1.0, 0.0, 1.0)   # strong in brights
        x = x + params.shadows * 0.5 * shadow_w
        x = x + params.highlights * 0.5 * highlight_w

        # 5) Midtone gamma (operate on non-negative values).
        x = xp.clip(x, 0.0, None)
        if params.gamma != 1.0:
            x = x ** (1.0 / params.gamma)

        # 6) Contrast around the 0.5 pivot.
        if params.contrast != 1.0:
            x = (x - 0.5) * params.contrast + 0.5

        # 7) Saturation + vibrance relative to luminance.
        luma2 = rgb_to_luminance(xp, xp.clip(x, 0.0, 1.0))[..., None]
        if params.saturation != 1.0:
            x = luma2 + (x - luma2) * params.saturation
        if params.vibrance != 0.0:
            # Vibrance boosts less-saturated pixels more than already-vivid ones.
            chroma = xp.abs(x - luma2).max(axis=-1, keepdims=True)
            gain = 1.0 + params.vibrance * (1.0 - xp.clip(chroma * 2.0, 0.0, 1.0))
            x = luma2 + (x - luma2) * gain

        # 8) Split toning: tint shadows & highlights toward chosen hues,
        #    luminance-preserving (centred offsets), weighted by tonal masks.
        x = self._split_tone(xp, x, params)

        x = xp.clip(x, 0.0, 1.0)
        return self.backend.to_numpy(x).astype(np.float32)

    @staticmethod
    def _split_tone(xp: Any, x: Any, p: GradingParams) -> Any:
        """Add hue tints to shadows/highlights based on luminance masks."""
        if p.split_shadow_strength <= 0.0 and p.split_highlight_strength <= 0.0:
            return x
        luma = rgb_to_luminance(xp, xp.clip(x, 0.0, 1.0))[..., None]
        shadow_w = xp.clip(1.0 - luma * 2.0, 0.0, 1.0)
        highlight_w = xp.clip(luma * 2.0 - 1.0, 0.0, 1.0)

        if p.split_shadow_strength > 0.0:
            tint = xp.asarray(_hue_to_rgb(p.split_shadow_hue)) - 0.5
            x = x + shadow_w * (p.split_shadow_strength * 0.5) * tint
        if p.split_highlight_strength > 0.0:
            tint = xp.asarray(_hue_to_rgb(p.split_highlight_hue)) - 0.5
            x = x + highlight_w * (p.split_highlight_strength * 0.5) * tint
        return x
