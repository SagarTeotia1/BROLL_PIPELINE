"""Decision engine — turns the analysis into the 45-parameter grade.

Emits a single document (see :mod:`color_analyzer.analyzer.schema`) in which
every parameter carries both what was **measured** and what is **recommended**:

    Image → analyzers → measure()  ─┐
                                    ├─► schema.assemble() → grade document
                        recommend() ┘

Semantics — the thing to get right
----------------------------------
``current`` and ``recommended`` are both **states on the same scale**, and
``delta`` is the adjustment that moves one to the other.  So for a frame
measured at 7250 K with a cinematic target of 6200 K you get
``current=7250, recommended=6200, delta=-1050`` — and ``delta`` is precisely
what a renderer applies.

This matters because the previous engine mixed the two: ``report.json`` reported
*states* while ``grade.json`` reported *adjustments*, on overlapping key names,
with no way to tell which was which.  A renderer that read the wrong one warmed
an already-warm frame.

The rules are heuristic and tunable (documented inline) — *"cool image with
natural skin ⇒ warm it"*, *"high commercial score ⇒ add clarity & vibrance"*,
*"teal-heavy / orange-light ⇒ push orange luminance"*.  They are **not**
validated against professionally graded ground truth; see the README.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import schema
from .engine import EngineResult
from .schema import flatten
from .utils import to_serializable

# --------------------------------------------------------------------------
# Style targets — the "look" each recognised style pulls the grade toward.
# --------------------------------------------------------------------------
# temp        : desired display colour temperature (K) for the graded result
# contrast    : desired global luminance contrast (p95-p5 of luma)
# vibrance    : baseline vibrance push (-100..100 scale)
# clarity     : baseline clarity (local contrast) push
# teal_orange : desired teal-orange separation strength [0,1]
# shadow_lift : desired lifted-shadow amount [0,1]
STYLE_TARGETS: Dict[str, Dict[str, float]] = {
    "commercial": {"temp": 5900, "contrast": 0.36, "vibrance": 16, "clarity": 12,
                   "teal_orange": 0.45, "shadow_lift": 0.08, "tint": 0},
    "cinematic": {"temp": 6200, "contrast": 0.30, "vibrance": 6, "clarity": 6,
                  "teal_orange": 0.80, "shadow_lift": 0.18, "tint": 0},
    "natural": {"temp": 6300, "contrast": 0.28, "vibrance": 5, "clarity": 4,
                "teal_orange": 0.15, "shadow_lift": 0.05, "tint": 0},
    "moody": {"temp": 6700, "contrast": 0.33, "vibrance": -4, "clarity": 7,
              "teal_orange": 0.55, "shadow_lift": 0.22, "tint": 0},
    "vintage": {"temp": 5600, "contrast": 0.24, "vibrance": -6, "clarity": 2,
                "teal_orange": 0.35, "shadow_lift": 0.26, "tint": 4},
}

_WB_NEUTRAL_K = 6500.0

# Curve shape each style aims for.
_STYLE_CURVE: Dict[str, str] = {
    "commercial": "gentle_s", "cinematic": "filmic", "moody": "filmic",
    "vintage": "faded", "natural": "linear",
}

# presence.dehaze is measured as haze = (0.24 - global_contrast) * 250 and the
# recommended correction is computed as (0.24 - global_contrast) * 45 on a 0..12
# slider.  Applying the full correction therefore clears this much measured haze.
_DEHAZE_SLIDER_TO_HAZE = 250.0 / 45.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _r(x: float, n: int = 3) -> float:
    v = round(float(x), n)
    return 0.0 if v == 0 else v  # normalise -0.0 -> 0.0


def _tone_curve_type(s_curve: float, black_point: float) -> str:
    """Classify the measured tonal transfer into a colourist curve label."""
    if s_curve > 0.25:
        return "strong_s"
    if s_curve > 0.08:
        return "soft_s"
    if s_curve < -0.15:
        return "faded"
    if black_point > 0.06:
        return "lifted"
    return "linear"


def choose_style(result: EngineResult) -> Tuple[str, Dict[str, float], float]:
    """Pick the target style from the measured cinematic tendencies.

    Returns ``(name, targets, confidence)``.  When nothing stands out the safe
    default is ``natural`` — pushing a frame toward a look it does not already
    have is how heuristic graders produce garish results.
    """
    cine = result.cinematic
    scores = {
        "commercial": cine.commercial_score,
        "cinematic": max(cine.teal_orange_score, cine.film_look_score),
        "natural": cine.natural_score,
        "moody": cine.moody_score,
        "vintage": cine.vintage_score,
    }
    name = max(scores, key=scores.get)
    confidence = float(scores[name])
    if confidence < 0.25:
        name = "natural"
    return name, STYLE_TARGETS[name], confidence


# ==========================================================================
# measure — what the frame already is
# ==========================================================================
def measure(result: EngineResult) -> Dict[str, Any]:
    """Measured state of the frame, keyed by :data:`schema.PARAM_NAMES`."""
    wb, exp, con = result.white_balance, result.exposure, result.contrast
    hsv, col, skin = result.hsv, result.colorfulness, result.skin
    cine, tc, split = result.cinematic, result.tone_curve, result.split_toning

    return {
        # -- white balance -------------------------------------------------
        "white_balance.temperature": wb.color_temperature,
        "white_balance.tint": wb.tint * 50.0,           # [-1,1] -> [-50,50]

        # -- primary: colourist scales relative to a neutral reference ------
        "primary.exposure": math.log2((exp.mean_brightness + 1e-3) / 0.45),
        "primary.contrast": (con.global_contrast - 0.30) / 0.30 * 50.0,
        "primary.highlights": (exp.highlight_percentage - 0.20) * 150.0
                              - tc.highlight_rolloff * 30.0,
        "primary.shadows": (tc.black_point - 0.03) * 300.0
                           + (0.30 - exp.shadow_percentage) * 40.0,
        "primary.whites": (tc.white_point - 0.95) * 300.0,
        "primary.blacks": (tc.black_point - 0.02) * 300.0,
        "primary.gamma": tc.gamma,

        # -- presence --------------------------------------------------------
        "presence.vibrance": (col.color_richness - 0.40) * 100.0,
        "presence.saturation": (hsv.saturation.mean - 0.35) * 100.0,
        "presence.clarity": con.local_contrast * 200.0,
        "presence.texture": con.laplacian_variance * 500.0,
        # Haze present in the frame: flat global contrast reads as atmosphere.
        "presence.dehaze": (0.24 - con.global_contrast) * 250.0,

        # -- tone curve ------------------------------------------------------
        "tone_curve.curve_type": _tone_curve_type(tc.s_curve_strength, tc.black_point),
        "tone_curve.shadow_lift": tc.lifted_shadows,
        "tone_curve.midtone": exp.median_brightness - 0.5,
        "tone_curve.highlight_rolloff": tc.highlight_rolloff,
        "tone_curve.contrast_strength": con.global_contrast / 0.5,

        # -- colour wheels, read off the measured split-tone zones ----------
        # Lab b* runs roughly +-25 over a graded frame; scale it to the wheel's
        # -100 (cool) .. 100 (warm) axis.
        "color_wheels.lift_temp": split.shadows.warmth / 25.0 * 100.0,
        "color_wheels.lift_strength": split.shadows.saturation,
        "color_wheels.gamma_temp": split.midtones.warmth / 25.0 * 100.0,
        "color_wheels.gamma_strength": split.midtones.saturation,
        "color_wheels.gain_temp": split.highlights.warmth / 25.0 * 100.0,
        "color_wheels.gain_strength": split.highlights.saturation,

        # -- split toning ----------------------------------------------------
        "split_toning.shadow_hue": split.shadows.hue,
        "split_toning.shadow_sat": split.shadows.saturation * 100.0,
        "split_toning.highlight_hue": split.highlights.hue,
        "split_toning.highlight_sat": split.highlights.saturation * 100.0,

        # -- HSL: measured band deviations from the frame mean --------------
        "hsl.orange_sat": cine.orange_saturation,
        "hsl.orange_lum": cine.orange_luminance,
        "hsl.blue_sat": cine.blue_saturation,
        "hsl.blue_lum": cine.blue_luminance,

        # -- subject ---------------------------------------------------------
        "subject.skin_present": skin.detected,
        "subject.skin_warmth": _clamp((40.0 - skin.skin_hue) / 60.0 + 0.5, 0.0, 1.0)
                               if skin.detected else 0.5,
        "subject.skin_luminance": skin.skin_exposure if skin.detected else 0.5,
        "subject.subject_pop": _clamp(con.local_contrast * 10.0, 0.0, 1.0),

        # -- creative style (read-only) --------------------------------------
        "creative_style.cinematic": max(cine.teal_orange_score, cine.film_look_score),
        "creative_style.commercial": cine.commercial_score,
        "creative_style.natural": cine.natural_score,
        "creative_style.moody": cine.moody_score,

        # -- quality (read-only) ---------------------------------------------
        "quality.highlight_clipping": exp.highlight_clipping > 0.01,
        "quality.shadow_crush": tc.crushed_blacks > 0.02,
        "quality.dynamic_range": con.dynamic_range,
        "quality.noise_risk": ("high" if exp.mean_brightness < 0.20
                               else "medium" if exp.mean_brightness < 0.32 else "low"),
    }


# ==========================================================================
# recommend — the state to grade toward
# ==========================================================================
def recommend(
    result: EngineResult,
    current: Mapping[str, Any],
    style_name: str,
    style: Mapping[str, float],
) -> Tuple[Dict[str, Any], List[str]]:
    """Target state for every adjustable parameter, plus editor-facing notes."""
    wb, exp, con = result.white_balance, result.exposure, result.contrast
    hsv, col, skin = result.hsv, result.colorfulness, result.skin
    cine, tc = result.cinematic, result.tone_curve

    notes: List[str] = []
    rec: Dict[str, Any] = {}

    # -- white balance ------------------------------------------------------
    desired_k = float(style["temp"])
    if skin.detected and skin.skin_naturalness > 0.8:
        desired_k -= 150.0  # flatter skin with a touch more warmth
    rec["white_balance.temperature"] = desired_k
    shift = wb.color_temperature - desired_k
    if shift > 400:
        notes.append("Increase warmth")
    elif shift < -400:
        notes.append("Reduce warmth")
    # Pull the tint 60% of the way to neutral, then bias by the style.
    rec["white_balance.tint"] = current["white_balance.tint"] * 0.4 + float(style.get("tint", 0.0))
    if abs(rec["white_balance.tint"] - current["white_balance.tint"]) > 6:
        notes.append("Neutralise tint")

    # -- primary ------------------------------------------------------------
    # Exposure targets mid-grey, nudged up slightly when skin is present.
    target_mean = 0.47 if skin.detected else 0.45
    exposure_adj = _clamp(
        math.log2((target_mean + 1e-3) / (exp.mean_brightness + 1e-3)), -0.6, 0.6
    )
    rec["primary.exposure"] = current["primary.exposure"] + exposure_adj

    contrast_adj = _clamp((style["contrast"] - con.global_contrast) * 180.0, -30.0, 35.0)
    rec["primary.contrast"] = current["primary.contrast"] + contrast_adj

    highlights_adj = -_clamp(exp.highlight_clipping * 400.0, 0.0, 40.0)
    rec["primary.highlights"] = current["primary.highlights"] + highlights_adj

    shadows_adj = _clamp(
        exp.shadow_clipping * 300.0 + style["shadow_lift"] * 35.0, -20.0, 40.0
    )
    rec["primary.shadows"] = current["primary.shadows"] + shadows_adj

    rec["primary.whites"] = current["primary.whites"] + _clamp(
        (0.97 - tc.white_point) * 220.0, -15.0, 20.0
    )
    rec["primary.blacks"] = current["primary.blacks"] + _clamp(
        (0.03 - tc.black_point) * 140.0 - style["shadow_lift"] * 20.0, -22.0, 12.0
    )
    # Move 30% of the way toward a neutral 1.0 transfer.
    rec["primary.gamma"] = tc.gamma + (1.0 - tc.gamma) * 0.3

    if highlights_adj < -8:
        notes.append("Recover highlights")
    if shadows_adj > 8:
        notes.append("Open up shadows")
    if abs(exposure_adj) > 0.12:
        notes.append("Adjust exposure")
    if contrast_adj > 8:
        notes.append("Add contrast")
    elif contrast_adj < -8:
        notes.append("Soften contrast")

    # -- presence -----------------------------------------------------------
    vibrance_adj = _clamp(style["vibrance"] + (0.4 - col.color_richness) * 28.0, -12.0, 30.0)
    rec["presence.vibrance"] = current["presence.vibrance"] + vibrance_adj

    rec["presence.saturation"] = current["presence.saturation"] + _clamp(
        (0.32 - hsv.saturation.mean) * 40.0, 0.0, 10.0  # lift only when muted
    )
    clarity_adj = _clamp(style["clarity"] + (0.05 - con.local_contrast) * 120.0, 0.0, 25.0)
    rec["presence.clarity"] = current["presence.clarity"] + clarity_adj
    rec["presence.texture"] = current["presence.texture"] + _clamp(
        float(style["clarity"]) * 0.6, 0.0, 15.0
    )
    dehaze_adj = _clamp((0.24 - con.global_contrast) * 45.0, 0.0, 12.0)
    # Target is the haze *left over* after applying that much dehaze.
    rec["presence.dehaze"] = max(
        0.0, current["presence.dehaze"] - dehaze_adj * _DEHAZE_SLIDER_TO_HAZE
    )

    if vibrance_adj > 8:
        notes.append("Boost vibrance")
    if clarity_adj > 8:
        notes.append("Add clarity for perceived sharpness")
    if dehaze_adj > 4:
        notes.append("Cut haze for punch")

    # -- tone curve ---------------------------------------------------------
    rec["tone_curve.curve_type"] = _STYLE_CURVE.get(style_name, "gentle_s")
    rec["tone_curve.shadow_lift"] = _clamp(style["shadow_lift"], 0.0, 0.4)
    rec["tone_curve.midtone"] = current["tone_curve.midtone"] + _clamp(
        (0.45 - exp.mean_brightness) * 0.5, -0.2, 0.2
    )
    rec["tone_curve.highlight_rolloff"] = _clamp(0.3 + exp.highlight_clipping * 3.0, 0.0, 1.0)
    rec["tone_curve.contrast_strength"] = _clamp(style["contrast"] + 0.15, 0.0, 1.0)

    # -- colour wheels: establish teal-orange via cool lift / warm gain ------
    separation = float(style["teal_orange"])
    lift_cool = _clamp(separation * 0.55, 0.0, 0.7)
    gain_warm = _clamp(separation * 0.6, 0.0, 0.8)
    gamma_warm = _clamp(separation * 0.2, 0.0, 0.3)  # slight warm midtones (skin)

    # Rule: teal-heavy but orange-light => push warmth into the highlights.
    teal_heavy = cine.teal_dominance > 0.5 and cine.orange_dominance < 0.4
    if teal_heavy:
        gain_warm = _clamp(gain_warm + 0.2, 0.0, 1.0)
        notes.append("Increase orange in highlights")
    if separation > 0.4:
        notes.append("Establish teal-orange separation")

    rec["color_wheels.lift_temp"] = -lift_cool * 100.0
    rec["color_wheels.lift_strength"] = lift_cool
    rec["color_wheels.gamma_temp"] = gamma_warm * 100.0
    rec["color_wheels.gamma_strength"] = gamma_warm
    rec["color_wheels.gain_temp"] = gain_warm * 100.0
    rec["color_wheels.gain_strength"] = gain_warm

    # -- split toning mirrors the wheel tints -------------------------------
    rec["split_toning.shadow_hue"] = 210.0            # teal
    rec["split_toning.shadow_sat"] = lift_cool * 40.0
    rec["split_toning.highlight_hue"] = 40.0          # orange
    rec["split_toning.highlight_sat"] = gain_warm * 40.0

    # -- HSL ----------------------------------------------------------------
    skin_lum_adj = 0.0
    if skin.detected and skin.skin_exposure < 0.5:
        skin_lum_adj = _clamp((0.5 - skin.skin_exposure) * 30.0, 0.0, 15.0)
    if teal_heavy:
        skin_lum_adj = _clamp(skin_lum_adj + 8.0, 0.0, 18.0)

    rec["hsl.orange_sat"] = current["hsl.orange_sat"] + 8.0
    rec["hsl.orange_lum"] = current["hsl.orange_lum"] + skin_lum_adj
    rec["hsl.blue_sat"] = current["hsl.blue_sat"] + (10.0 * separation + 3.0)
    rec["hsl.blue_lum"] = current["hsl.blue_lum"] - 8.0 * separation

    if skin_lum_adj > 4:
        notes.append("Brighten skin (orange luminance)")
    if separation > 0.4:
        notes.append("Deepen background blues for separation")

    # -- subject ------------------------------------------------------------
    warm_bias = _clamp((_WB_NEUTRAL_K - desired_k) / 3000.0, -1.0, 1.0)
    rec["subject.skin_warmth"] = _clamp(
        0.45 + separation * 0.25 + max(0.0, warm_bias) * 0.3, 0.0, 1.0
    )
    rec["subject.skin_luminance"] = (
        _clamp(0.5 + (0.52 - skin.skin_exposure), 0.0, 1.0) if skin.detected else 0.5
    )
    rec["subject.subject_pop"] = _clamp(
        0.4 + rec["presence.clarity"] / 400.0 + separation * 0.2, 0.0, 1.0
    )

    if not notes:
        notes.append("Image is already well balanced; minimal changes.")
    return rec, list(dict.fromkeys(notes))


# ==========================================================================
# The engine
# ==========================================================================
class DecisionEngine:
    """Produces the 45-parameter grade document from an :class:`EngineResult`."""

    def grade(self, result: EngineResult, meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        style_name, style, style_conf = choose_style(result)
        current = measure(result)
        recommended, notes = recommend(result, current, style_name, style)

        base_meta: Dict[str, Any] = {
            "source": result.source,
            "width": result.width,
            "height": result.height,
            "backend": result.backend.get("backend"),
            "elapsed_ms": round(result.elapsed_seconds * 1000.0, 2),
            "deep": result.deep,
        }
        base_meta.update(meta or {})

        palette: List[str] = []
        if result.dominant_colors is not None:
            palette = [c.hex for c in result.dominant_colors.colors[:5]]

        document = schema.assemble(
            current=current,
            recommended=recommended,
            meta=base_meta,
            style={
                "detected": result.summary.mood,
                "target": style_name,
                "confidence": _r(style_conf, 3),
            },
            notes=notes,
            palette=palette,
        )
        return to_serializable(document)

    # Backwards-compatible alias.
    def decide(self, result: EngineResult) -> Dict[str, Any]:
        """Deprecated name for :meth:`grade`."""
        return self.grade(result)


# ==========================================================================
# Renderer bridge
# ==========================================================================
def to_executor_decision(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a grade document into the shape :class:`GradingPlanExecutor` eats.

    The executor speaks *adjustments* (how much to move each slider), which is
    exactly the document's ``delta`` field — with two exceptions:

    * white balance takes an absolute Kelvin target, and the renderer's
      convention is that a value below 6500 K warms the image, so the target is
      ``6500 + delta``;
    * tone curve, colour wheels, split toning and subject describe a target
      *state*, so their ``recommended`` value is passed straight through.
    """
    grade = document.get("grade", {})

    def delta(name: str, default: float = 0.0) -> float:
        entry = grade.get(name) or {}
        value = entry.get("delta")
        return float(value) if value is not None else default

    def rec(name: str, default: float = 0.0) -> Any:
        entry = grade.get(name) or {}
        value = entry.get("recommended")
        return value if value is not None else default

    wheel_strength_gain = float(rec("color_wheels.gain_strength", 0.0))
    return {
        "white_balance": {
            "temperature": _WB_NEUTRAL_K + delta("white_balance.temperature"),
            "tint": delta("white_balance.tint"),
        },
        "primary_corrections": {
            "exposure": delta("primary.exposure"),
            "contrast": delta("primary.contrast"),
            "highlights": delta("primary.highlights"),
            "shadows": delta("primary.shadows"),
            "whites": delta("primary.whites"),
            "blacks": delta("primary.blacks"),
            "gamma": rec("primary.gamma", 1.0),
            "pivot": 0.435,
        },
        "tone_curve": {
            "shadow_lift": rec("tone_curve.shadow_lift", 0.0),
            "midtone": rec("tone_curve.midtone", 0.0),
            "highlight_rolloff": rec("tone_curve.highlight_rolloff", 0.0),
            "contrast_strength": rec("tone_curve.contrast_strength", 0.0),
        },
        "color_wheels": {
            "lift": {"temperature": rec("color_wheels.lift_temp", 0.0),
                     "tint": 0.0, "strength": rec("color_wheels.lift_strength", 0.0)},
            "gamma": {"temperature": rec("color_wheels.gamma_temp", 0.0),
                      "tint": 0.0, "strength": rec("color_wheels.gamma_strength", 0.0)},
            "gain": {"temperature": rec("color_wheels.gain_temp", 0.0),
                     "tint": 0.0, "strength": wheel_strength_gain},
        },
        "split_toning": {
            "shadows": {"hue": rec("split_toning.shadow_hue", 210.0),
                        "sat": rec("split_toning.shadow_sat", 0.0)},
            "highlights": {"hue": rec("split_toning.highlight_hue", 40.0),
                           "sat": rec("split_toning.highlight_sat", 0.0)},
        },
        "hsl_adjustments": {
            "orange": {"hue": 0.0, "saturation": delta("hsl.orange_sat"),
                       "luminance": delta("hsl.orange_lum")},
            "blue": {"hue": 0.0, "saturation": delta("hsl.blue_sat"),
                     "luminance": delta("hsl.blue_lum")},
        },
        "subject_enhancement": {
            "skin_luminance": rec("subject.skin_luminance", 0.5),
        },
        "presence": {
            "vibrance": delta("presence.vibrance"),
            "saturation": delta("presence.saturation"),
            "clarity": delta("presence.clarity"),
            "texture": delta("presence.texture"),
            # Dehaze is subtractive: the document tracks haze *remaining*, so
            # the slider value is how much haze the grade removes.
            "dehaze": max(0.0, -delta("presence.dehaze") / _DEHAZE_SLIDER_TO_HAZE),
        },
    }


def flatten_grade(document: Mapping[str, Any], field: str = "recommended") -> Dict[str, Any]:
    """Flat ``{param: value}`` view of a grade document (see :func:`schema.flatten`)."""
    return flatten(document, field)
