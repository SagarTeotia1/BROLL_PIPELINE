"""Render a target colour state onto a frame.

The analyzer reports where a frame *is*; a grading model returns where it should
*be*. This module moves it, by working out the difference and applying it.

Why it needs both states
------------------------
Nothing here can "set" a reading. A control saying ``temperature: 4200`` on a
frame measured at 5200 K is a request to warm it by 1000 K — the absolute number
is only meaningful against the measurement it came from. So every operation is
driven by ``target - source``, and :func:`apply_controls` measures the frame
itself when the caller does not supply the source state.

The masks come from the analyzer
--------------------------------
Every zone weight and hue membership used here is the *same object* the analyzer
measured with: :meth:`Frame.zone_mask`, :attr:`Frame.chroma_gate`, the HSL band
table. That is not a tidiness point. If the analyzer called a pixel a shadow and
the renderer disagreed, the measurement and the correction would describe
different populations, the reported state after grading would not match the
target, and repeated passes would never converge.

Masks are computed once, from the source frame, and reused for every operation
rather than recomputed as the image changes. This is how grading tools normally
behave — the qualifier reads the input — and it is much faster. The cost is that
after a very large tonal move the zone boundaries no longer describe the result
exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import controls as controls_module
from .controls import Sanitised, sanitise
from .frame import Frame
from .hsl import BAND_CENTRES, PRESENCE_FLOOR, _BAND_TABLE, _TABLE_STEPS_PER_DEGREE
from .skin import skin_mask

#: Neutral reference the temperature control is measured against.
NEUTRAL_KELVIN = 6500.0

#: Pivot the contrast control scales around.
CONTRAST_PIVOT = 0.5

#: How much of a global operation is held back on skin when protection is on.
#: Not 1.0: skin should stay recognisable through a grade, not be cut out of it.
SKIN_HOLDBACK = 0.6

#: Operations that are moderated on skin. White balance and exposure are absent
#: deliberately — they move skin the way skin should move, and excluding them
#: would leave faces lit differently from the scene around them.
SKIN_SENSITIVE = ("contrast", "gamma", "wheels", "hsl", "split_toning", "saturation")

# ---------------------------------------------------------------------------
# Response compensation
# ---------------------------------------------------------------------------
# An operation rarely moves its own reading one-for-one. Some of the loss is
# structural: a colour wheel writes through a soft zone weight but the analyzer
# measures through the same weight, so the round trip picks up a factor of
# E[w^2]/E[w]; an HSL band boost lifts the frame it is measured against.
#
# These factors are **measured, not guessed**: each control was driven by a
# known amount on real frames, the resulting reading change recorded, and the
# factor set to the reciprocal of the achieved fraction. Re-measure with
# `benchmarks/`-style probes if an operation changes.
#
# Measured with skin protection off. With it on, controls that mostly touch skin
# — orange especially — deliberately move less, and that is the point of it.
RESPONSE: Dict[str, float] = {
    "white_balance.temperature": 1.40,
    "tone.exposure": 1.13,
    "tone.contrast": 1.12,
    "tone.brightness": 1.11,
    "tone.gamma": 2.23,
    "color.saturation": 1.11,
    "color.vibrance": 2.82,
    "wheels.channel": 1.85,
    "wheels.luma": 2.60,
    "hsl.saturation": 2.08,
    "hsl.luminance": 2.30,
    "split_toning": 1.14,
}


@dataclass
class GradeResult:
    """A graded image plus an account of what happened.

    Attributes
    ----------
    image:
        Graded RGB, float32 in ``[0,1]``.
    applied:
        Operations that actually ran; an operation whose controls were all
        unchanged is skipped.
    ignored, clamped:
        Passed through from :func:`~.controls.sanitise` — what the caller sent
        that could not be used, and what had to be pulled into range.
    """

    image: np.ndarray
    applied: List[str] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)
    clamped: List[str] = field(default_factory=list)

    @property
    def crushed_shadows(self) -> float:
        """Fraction of the result sitting at black, where detail is unrecoverable."""
        return float((self.image.max(axis=2) <= 0.004).mean())

    @property
    def blown_highlights(self) -> float:
        """Fraction of the result sitting at white."""
        return float((self.image.min(axis=2) >= 0.996).mean())

    def warnings(self) -> List[str]:
        """Detail the grade destroyed, if any.

        Worth surfacing separately from the controls: an aggressive combined
        grade — less exposure, more contrast and a raised black point together —
        will happily crush a third of the frame to black, and nothing in the
        control values says so. The render is doing what it was told; the caller
        needs to know what it cost.
        """
        notes: List[str] = []
        if self.crushed_shadows > 0.02:
            notes.append(
                f"{self.crushed_shadows:.1%} of the frame is crushed to black; "
                f"raise the black point or ease off exposure and contrast"
            )
        if self.blown_highlights > 0.02:
            notes.append(
                f"{self.blown_highlights:.1%} of the frame is blown to white; "
                f"lower the white point or ease off exposure"
            )
        return notes

    def summary(self) -> str:
        """One-line description, for a CLI or a log."""
        parts = [f"applied: {', '.join(self.applied) or 'nothing'}"]
        if self.clamped:
            parts.append(f"{len(self.clamped)} clamped")
        if self.ignored:
            parts.append(f"{len(self.ignored)} ignored")
        for note in self.warnings():
            parts.append(note)
        return " | ".join(parts)


def apply_controls(
    rgb01: np.ndarray,
    target: Mapping[str, Any],
    source: Optional[Mapping[str, Any]] = None,
    protect_skin: bool = True,
    max_side: Optional[int] = None,
) -> GradeResult:
    """Grade ``rgb01`` from its current state toward ``target``.

    Parameters
    ----------
    rgb01:
        RGB image, uint8 or float. Graded at its native resolution.
    target:
        Control values from a grading model. Cleaned by
        :func:`~.controls.sanitise`, so malformed content is reported rather
        than raised.
    source:
        The frame's measured state. Analysed from the image when omitted.
    protect_skin:
        Hold back globally destructive operations on skin, so an aggressive
        grade does not turn faces grey or green.
    max_side:
        Resolution cap for *measuring* the source state. The grade is always
        applied at full resolution; this only bounds the analysis pass.

    Notes
    -----
    This is a **single open-loop pass**: every operation is calibrated against
    the original measurement. One control at a time lands accurately, but a
    large combined grade does not — ask for less exposure, more contrast and a
    new black point together and all three fire at full strength against a frame
    the other two are also changing.

    Iterating — measure the result, apply what is left, repeat — was tried and
    removed. The readings are coupled (moving exposure changes the contrast
    reading, moving contrast changes the exposure reading), so per-control
    feedback on them oscillates rather than converging: a requested contrast of
    94 landed at -16, then 33, then -58 across three passes, and damping the
    correction to 45% only slowed the swing. Converging properly needs a solver
    that accounts for the coupling, not a repeat of the same open-loop step.
    """
    from .analyzer import EditorialAnalyzer  # local: analyzer imports this module

    cleaned = sanitise(target)
    if source is None:
        source = EditorialAnalyzer(max_side=max_side).analyze_rgb(rgb01)

    frame = Frame.from_rgb(rgb01, max_side=None)  # grade at native resolution
    source_values = controls_module.flatten(source)

    image = frame.rgb.astype(np.float32).copy()
    result = GradeResult(image=image, ignored=cleaned.ignored, clamped=cleaned.clamped)

    deltas = _deltas(cleaned, source_values, source)
    if not deltas:
        return result  # nothing asked for; hand the frame back untouched

    context = _Context(frame=frame, source=source_values, cleaned=cleaned,
                       deltas=deltas, skin=skin_mask(frame) if protect_skin else None,
                       result=result)

    # Order matters: neutralise the light, set the tonal range, shape the
    # midtones, then colour, then saturation last so it acts on the final hues.
    for step in (_white_balance, _levels, _exposure, _brightness, _contrast,
                 _gamma, _wheels, _hsl, _split_toning, _saturation):
        image = step(image, context)

    result.image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return result



# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
@dataclass
class _Context:
    """Everything the operations share."""

    frame: Frame
    source: Dict[str, Any]
    cleaned: Sanitised
    deltas: Dict[str, float]
    skin: Optional[np.ndarray]
    result: GradeResult

    def delta(self, path: str, default: float = 0.0) -> float:
        return float(self.deltas.get(path, default))

    def target(self, path: str, default: float) -> float:
        value = self.cleaned.values.get(path)
        return float(value) if value is not None else float(default)

    def source_value(self, path: str, default: float) -> float:
        value = self.source.get(path)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def any_delta(self, *paths: str, threshold: float = 1e-6) -> bool:
        return any(abs(self.delta(p)) > threshold for p in paths)

    def note(self, operation: str) -> None:
        if operation not in self.result.applied:
            self.result.applied.append(operation)

    def weight(self, operation: str, image: np.ndarray, channels: bool = True) -> Any:
        """Per-pixel strength for an operation, accounting for skin protection.

        Returns a scalar 1.0 when nothing is held back, so callers that do not
        need a mask pay nothing. The skin mask is stored flat, so it is reshaped
        against the image being graded rather than assumed.
        """
        if self.skin is None or operation not in SKIN_SENSITIVE:
            return 1.0
        height, width = image.shape[:2]
        held = (1.0 - SKIN_HOLDBACK * self.skin).astype(np.float32)
        return held.reshape(height, width, 1) if channels else held.reshape(height, width)


def _deltas(cleaned: Sanitised, source: Mapping[str, Any],
            state: Mapping[str, Any]) -> Dict[str, float]:
    """``target - source`` for every control the caller actually set.

    Controls whose source reading is a *sentinel* rather than a measurement are
    skipped — see :func:`_is_measured`.
    """
    out: Dict[str, float] = {}
    for path, target in cleaned.values.items():
        try:
            current = float(source[path])
        except (KeyError, TypeError, ValueError):
            # No measurement to difference against: treat the request as
            # already satisfied rather than guessing at an origin.
            continue
        if not _is_measured(path, state):
            continue
        delta = float(target) - current
        if abs(delta) > 1e-9:
            out[path] = delta
    return out


def _is_measured(path: str, state: Mapping[str, Any]) -> bool:
    """Whether a control's source reading is a real measurement.

    The analyzer reports ``0`` for an HSL band that is not present in the frame,
    and that zero means "no such colour here", not "this colour is exactly
    neutral". Differencing a target against it produces a correction with no
    basis: asking for ``hsl.blue.saturation: 40`` on a frame containing almost
    no blue was measured pushing the band the *wrong* way, because the applied
    change was large enough to lift the band over the presence floor and the
    reading then came back as a genuine — and quite different — number.

    So a band below its presence floor is left alone. A grading model cannot
    conjure a colour that is not in the frame, and pretending otherwise
    produces artefacts rather than a grade.
    """
    parts = path.split(".")
    if parts[0] != "hsl":
        return True

    band = state.get("hsl", {}).get(parts[1])
    if not isinstance(band, Mapping):
        return False
    return float(band.get("presence", 0.0)) >= PRESENCE_FLOOR


def _luma(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------
def _white_balance(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Per-channel gains from a colour-temperature and tint difference.

    The gain is **derived, not tuned.** The analyzer reads temperature by
    looking a frame's red/blue ratio up in a daylight-locus table; inverting
    that table gives the ratio each temperature corresponds to, so the gain
    needed to move from one to the other is exactly the quotient of the two
    entries. Splitting it symmetrically between red and blue changes the ratio
    without changing overall level.

    A tuned coefficient was tried first and closed only half the requested
    shift, because the relationship between Kelvin and channel ratio is
    markedly non-linear across the range.
    """
    if not ctx.any_delta("white_balance.temperature", "white_balance.tint"):
        return image

    from .white_balance import kelvin_to_ratio

    graded = image.copy()

    source_k = ctx.source_value("white_balance.temperature", NEUTRAL_KELVIN)
    target_k = ctx.target("white_balance.temperature", source_k)
    if abs(target_k - source_k) > 1e-6 and source_k > 0 and target_k > 0:
        source_ratio = kelvin_to_ratio(source_k)
        target_ratio = kelvin_to_ratio(target_k)
        raw = target_ratio / max(source_ratio, 1e-6)
        gain = float(np.clip(raw ** RESPONSE["white_balance.temperature"], 0.2, 5.0))
        root = math.sqrt(gain)
        graded[..., 0] *= root        # red up, blue down => warmer
        graded[..., 2] /= root

    # Tint reads as (R+B)/2 - G over the frame level, scaled by 250, so a slider
    # unit is 0.4% of level on green.
    tint_delta = ctx.delta("white_balance.tint")
    if abs(tint_delta) > 1e-9:
        graded[..., 1] *= float(np.clip(1.0 - tint_delta / 250.0, 0.3, 3.0))

    ctx.note("white_balance")
    return graded


def _levels(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Remap the tonal range onto the requested black and white points.

    The only operation here that hits its target exactly: black and white point
    are levels, so a linear remap puts them where they were asked to go.
    """
    source_black = ctx.source_value("tone.black_point", 0.0)
    source_white = ctx.source_value("tone.white_point", 1.0)
    target_black = ctx.target("tone.black_point", source_black)
    target_white = ctx.target("tone.white_point", source_white)

    if abs(target_black - source_black) < 1e-6 and abs(target_white - source_white) < 1e-6:
        return image

    source_span = max(source_white - source_black, 1e-3)
    target_span = max(target_white - target_black, 1e-3)

    graded = (image - source_black) / source_span * target_span + target_black
    ctx.note("levels")
    return graded


def _exposure(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Scale light by a difference in stops."""
    delta = ctx.delta("tone.exposure")
    if abs(delta) < 1e-6:
        return image
    ctx.note("exposure")
    return image * float(2.0 ** (delta * RESPONSE["tone.exposure"]))


def _brightness(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Additive lightness offset."""
    delta = ctx.delta("tone.brightness")
    if abs(delta) < 1e-6:
        return image
    ctx.note("brightness")
    # The reading is the frame mean mapped to +/-100 over a 0.5 span, so a
    # slider unit is 0.005 of signal. Inverting that keeps the two consistent.
    return image + delta * 0.005 * RESPONSE["tone.brightness"]


def _contrast(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Expand or compress tones around a mid pivot."""
    delta = ctx.delta("tone.contrast")
    if abs(delta) < 1e-6:
        return image

    # The reading spans +/-100 over a 0.45 change in the p95-p5 spread; a slider
    # unit is therefore a 0.45% change in that spread.
    strength = delta / 100.0 * 0.9 * RESPONSE["tone.contrast"]
    weight = ctx.weight("contrast", image)
    scaled = (image - CONTRAST_PIVOT) * (1.0 + strength) + CONTRAST_PIVOT

    ctx.note("contrast")
    return image + (scaled - image) * weight


def _gamma(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Reshape the midtones toward the requested transfer exponent."""
    source_gamma = ctx.source_value("tone.gamma", 1.0)
    target_gamma = ctx.target("tone.gamma", source_gamma)
    if abs(target_gamma - source_gamma) < 1e-6 or source_gamma <= 0 or target_gamma <= 0:
        return image

    # The measured gamma is compressed toward 1.0 by the analyzer's range
    # normalisation (documented in tone.py), so the exponent needed to move
    # between two readings is damped to match rather than overshooting.
    damping = 0.7 * RESPONSE["tone.gamma"]
    exponent = float(np.clip((target_gamma / source_gamma) ** damping, 0.25, 4.0))
    weight = ctx.weight("gamma", image)
    shaped = np.clip(image, 1e-4, None) ** exponent

    ctx.note("gamma")
    return image + (shaped - image) * weight


def _wheels(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Lift, gamma and gain: per-zone colour balance and brightness.

    Zone weights come from :meth:`Frame.zone_mask`, the same soft, overlapping
    weighting the analyzer measured each zone with.
    """
    zone_of = {"lift": "shadows", "gamma": "midtones", "gain": "highlights"}
    paths = [f"wheels.{w}.{a}" for w in zone_of for a in ("red", "green", "blue", "luma")]
    if not ctx.any_delta(*paths):
        return image

    height, width = image.shape[:2]
    protection = ctx.weight("wheels", image)
    graded = image

    for wheel, zone in zone_of.items():
        weight = ctx.frame.zone_mask(zone).reshape(height, width, 1)

        # Channel offsets. The reading is a channel's deviation from the zone
        # level as a fraction of it, scaled so 100 units is a 50% imbalance.
        offsets = np.array(
            [ctx.delta(f"wheels.{wheel}.{axis}") for axis in ("red", "green", "blue")],
            dtype=np.float32,
        ) / 100.0 * 0.5 * RESPONSE["wheels.channel"]
        if np.any(offsets):
            zone_level = float(np.mean(_luma(image) * weight[..., 0]) /
                               max(float(weight.mean()), 1e-6))
            graded = graded + weight * protection * offsets * max(zone_level, 0.05)

        # Brightness of the zone; the reading spans +/-100 over one zone
        # standard deviation, about 0.12 of signal.
        luma_delta = ctx.delta(f"wheels.{wheel}.luma") / 100.0 * 0.12 * RESPONSE["wheels.luma"]
        if abs(luma_delta) > 1e-9:
            graded = graded + weight * protection * luma_delta

    ctx.note("wheels")
    return graded


def _hsl(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Selective hue, saturation and luminance per hue band.

    Membership comes from the analyzer's band table, so "orange" means the same
    pixels here as it did in the measurement.
    """
    paths = [f"hsl.{b}.{f}" for b in BAND_CENTRES for f in ("hue", "saturation", "luminance")]
    if not ctx.any_delta(*paths):
        return image

    import cv2

    hsv = cv2.cvtColor(np.clip(image, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Membership is taken from the *current* hues, and gated on chroma exactly
    # as the analyzer gates it, so achromatic pixels stay out of every band.
    indices = np.minimum((hue * _TABLE_STEPS_PER_DEGREE).astype(np.int32),
                         _BAND_TABLE.shape[0] - 1)
    gate = _chroma_gate(sat, val)
    protection = ctx.weight("hsl", image, channels=False)

    hue_out, sat_out, val_out = hue.copy(), sat.copy(), val.copy()
    for index, band in enumerate(BAND_CENTRES):
        weight = _BAND_TABLE[indices, index] * gate
        if isinstance(protection, np.ndarray):
            weight = weight * protection
        if not weight.any():
            continue

        # Hue: the reading is an offset from the band centre scaled to +/-100
        # over the band's half-width, so a unit is that fraction of a degree.
        half_width = _band_half_width(band)
        hue_delta = ctx.delta(f"hsl.{band}.hue") / 100.0 * half_width
        if abs(hue_delta) > 1e-9:
            hue_out = hue_out + weight * hue_delta

        sat_delta = ctx.delta(f"hsl.{band}.saturation") / 100.0 * RESPONSE["hsl.saturation"]
        if abs(sat_delta) > 1e-9:
            sat_out = sat_out + weight * sat_delta

        lum_delta = ctx.delta(f"hsl.{band}.luminance") / 100.0 * RESPONSE["hsl.luminance"]
        if abs(lum_delta) > 1e-9:
            val_out = val_out + weight * lum_delta

    shifted = np.stack([
        hue_out % 360.0,
        np.clip(sat_out, 0.0, 1.0),
        np.clip(val_out, 0.0, 1.0),
    ], axis=-1).astype(np.float32)

    ctx.note("hsl")
    return cv2.cvtColor(shifted, cv2.COLOR_HSV2RGB)


def _band_half_width(band: str) -> float:
    from .hsl import BAND_HALF_WIDTH

    return float(BAND_HALF_WIDTH[band])


def _chroma_gate(sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    """The analyzer's chroma/value gate, evaluated on a mid-grade image."""
    from .frame import CHROMA_GATE, VALUE_GATE, _smoothstep

    return (_smoothstep(sat, *CHROMA_GATE) * _smoothstep(val, *VALUE_GATE)).astype(np.float32)


def _split_toning(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Set the hue and strength of the shadow and highlight tints.

    Worked in HSV, zone-weighted, because that is how the analyzer measures it:
    the shadow tint's "saturation" is the mean HSV saturation of the shadow
    zone, and its "hue" is the saturation-weighted circular mean hue there. So
    saturation is moved by changing S, and hue by rotating H.

    The obvious alternative — add a hue-coloured vector in RGB, scaled by the
    requested strength — is wrong in a way that is easy to miss. Reducing a
    tint then means *subtracting* its colour, and subtracting orange does not
    produce neutral shadows, it produces cyan ones. Measured: asking to drop the
    shadow tint from 89 to 60 pushed the hue from 354 to 161 and the saturation
    up to 100.
    """
    zones = ("shadows", "highlights")
    paths = [f"split_toning.{z}.{f}" for z in zones for f in ("hue", "saturation")]
    if not ctx.any_delta(*paths):
        return image

    import cv2

    hsv = cv2.cvtColor(np.clip(image, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    height, width = image.shape[:2]
    protection = ctx.weight("split_toning", image, channels=False)

    hue_out, sat_out = hue.copy(), sat.copy()
    for zone in zones:
        weight = ctx.frame.zone_mask(zone).reshape(height, width)
        if isinstance(protection, np.ndarray):
            weight = weight * protection

        # Saturation: the reading treats 0.40 as full scale.
        sat_delta = (ctx.delta(f"split_toning.{zone}.saturation") / 100.0 * 0.40
                     * RESPONSE["split_toning"])
        if abs(sat_delta) > 1e-9:
            sat_out = sat_out + weight * sat_delta

        # Hue: rotate along the shorter arc, so 350 -> 10 is +20 degrees rather
        # than a 340-degree sweep through every other colour.
        hue_delta = ctx.delta(f"split_toning.{zone}.hue")
        if abs(hue_delta) > 1e-9:
            shortest = (hue_delta + 180.0) % 360.0 - 180.0
            hue_out = hue_out + weight * shortest

    shifted = np.stack([
        hue_out % 360.0,
        np.clip(sat_out, 0.0, 1.0),
        hsv[..., 2],
    ], axis=-1).astype(np.float32)

    ctx.note("split_toning")
    return cv2.cvtColor(shifted, cv2.COLOR_HSV2RGB)


def _saturation(image: np.ndarray, ctx: _Context) -> np.ndarray:
    """Overall saturation and vibrance, applied last so it acts on final hues."""
    if not ctx.any_delta("color.saturation", "color.vibrance"):
        return image

    import cv2

    hsv = cv2.cvtColor(np.clip(image, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    # Readings span +/-100 over 0.35 of saturation.
    sat_delta = ctx.delta("color.saturation") / 100.0 * 0.35 * RESPONSE["color.saturation"]
    vib_delta = ctx.delta("color.vibrance") / 100.0 * 0.35 * RESPONSE["color.vibrance"]

    shifted = sat + sat_delta
    if abs(vib_delta) > 1e-9:
        # Vibrance acts on the muted end: weight by how far a pixel is from
        # fully saturated, which is what leaves strong colours alone.
        #
        # Gated on chroma, and that gate is not optional. Vibrance is measured as
        # a low percentile of the saturation of *chromatic* pixels; lifting
        # near-neutral pixels along with everything else pushes them over the
        # gate and into that population, where — still being the least saturated
        # things in frame — they drag the percentile down. Asking for more
        # vibrance measured as delivering less: 0.50 to 0.20 on a test frame.
        # Excluding greys is also simply what a vibrance control does. It
        # deepens colours that are already there; it does not colourise grey.
        shifted = shifted + vib_delta * (1.0 - sat) * ctx.frame.chroma_gate.reshape(sat.shape)

    protection = ctx.weight("saturation", image, channels=False)
    if isinstance(protection, np.ndarray):
        shifted = sat + (shifted - sat) * protection

    hsv[..., 1] = np.clip(shifted, 0.0, 1.0)
    ctx.note("saturation")
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------
def apply_frames(
    frames: Sequence[np.ndarray],
    target: Mapping[str, Any],
    source: Optional[Mapping[str, Any]] = None,
    protect_skin: bool = True,
    is_bgr: bool = True,
):
    """Grade a sequence of frames against one target state.

    ``source`` is measured from the first frame and reused for the rest unless
    supplied. That is deliberate: measuring each frame separately would let the
    correction drift shot-internally, and a grade is meant to be constant across
    a shot.
    """
    import cv2

    measured = source
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if is_bgr else frame
        result = apply_controls(rgb, target, measured, protect_skin=protect_skin)
        if measured is None:
            from .analyzer import EditorialAnalyzer

            measured = EditorialAnalyzer().analyze_rgb(rgb)
        yield result
