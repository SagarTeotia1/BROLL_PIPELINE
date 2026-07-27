"""Professional grading-plan executor.

Applies a structured, editor-style grading plan — the schema an AI planner emits
for a video-editing platform — to an image.  It supports the operations

    WHITE_BALANCE · PRIMARY_CORRECTION · TONE_CURVE · COLOR_WHEELS · HSL ·
    PRESENCE · VIGNETTE · GRAIN

and honours ``executor_settings.apply_order`` so stages run in the intended
sequence.  Each operation maps to well-known colour-grading maths (documented
inline).  Unknown or missing operations are skipped gracefully, so partial plans
"just work".

The executor runs on the CPU (NumPy + OpenCV/SciPy for the spatial filters used
by clarity/texture/vignette); this is the interactive *application* path, not
the GPU analysis path, so a single clear NumPy implementation is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .grading import _hue_to_rgb
from .utils import hsv_to_rgb, rgb_to_hsv, rgb_to_luminance

# HSL band centres (degrees) for the eight named colour ranges.
_HSL_BANDS: Dict[str, float] = {
    "red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0,
    "aqua": 180.0, "cyan": 180.0, "blue": 240.0, "purple": 275.0, "magenta": 315.0,
}
# Luminance neutral used to interpret an absolute white-balance target.
_WB_NEUTRAL_K = 6500.0


@dataclass
class PlanExecutionResult:
    """Result of executing a grading plan."""

    image: np.ndarray  # graded RGB float [0,1]
    applied_steps: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def is_grading_plan(obj: Any) -> bool:
    """True if ``obj`` is a step-list grading plan (has ``grading_plan``)."""
    return isinstance(obj, dict) and isinstance(obj.get("grading_plan"), list)


def is_grading_decision(obj: Any) -> bool:
    """True if ``obj`` is a *static* grading decision (the compact 10-section JSON).

    Distinguished from a step-list plan by the absence of ``grading_plan`` and
    the presence of the fixed grading sections.
    """
    return (
        isinstance(obj, dict)
        and not isinstance(obj.get("grading_plan"), list)
        and ("color_wheels" in obj or "primary_corrections" in obj)
    )


def build_curve_points(tc: Dict[str, Any]) -> List[List[float]]:
    """Build monotone tone-curve points from parametric curve controls.

    Shared by the decision engine and the executor so a curve is described the
    same way everywhere.  ``tc`` carries ``shadow_lift``/``midtone``/
    ``highlight_rolloff``/``contrast_strength`` (all ~[0,1]).
    """
    shadow_lift = float(tc.get("shadow_lift", 0.0))
    midtone = float(tc.get("midtone", 0.0))
    rolloff = float(tc.get("highlight_rolloff", 0.0))
    c = float(tc.get("contrast_strength", 0.0))
    clamp = lambda v, lo, hi: lo if v < lo else hi if v > hi else v  # noqa: E731
    return [
        [0.0, clamp(shadow_lift * 0.12, 0.0, 0.2)],
        [0.25, clamp(0.25 - c * 0.07 + shadow_lift * 0.1, 0.0, 0.5)],
        [0.5, clamp(0.5 + midtone, 0.2, 0.8)],
        [0.75, clamp(0.75 + c * 0.05 - rolloff * 0.04, 0.5, 1.0)],
        [1.0, clamp(1.0 - rolloff * 0.06, 0.85, 1.0)],
    ]


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep in ``[0,1]`` (0 below edge0, 1 above edge1)."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lum(x: np.ndarray) -> np.ndarray:
    return rgb_to_luminance(np, np.clip(x, 0.0, 1.0))


def _gaussian(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a single-channel image (OpenCV, SciPy fallback)."""
    if sigma <= 0:
        return gray
    if cv2 is not None:
        k = int(max(3, round(sigma * 3) * 2 + 1))
        return cv2.GaussianBlur(gray, (k, k), sigma)
    from scipy.ndimage import gaussian_filter  # pragma: no cover

    return gaussian_filter(gray, sigma)


class GradingPlanExecutor:
    """Executes an editor-style grading plan on an RGB image."""

    _HANDLERS_KEYS = (
        "WHITE_BALANCE", "PRIMARY_CORRECTION", "TONE_CURVE", "COLOR_WHEELS",
        "SPLIT_TONING", "HSL", "PRESENCE", "VIGNETTE", "GRAIN",
    )

    def _handlers(self):
        return {
            "WHITE_BALANCE": self._white_balance,
            "PRIMARY_CORRECTION": self._primary,
            "TONE_CURVE": self._tone_curve,
            "COLOR_WHEELS": self._color_wheels,
            "SPLIT_TONING": self._split_toning,
            "HSL": self._hsl,
            "PRESENCE": self._presence,
            "VIGNETTE": self._vignette,
            "GRAIN": self._grain,
        }

    def apply(self, rgb01: np.ndarray, plan: Dict[str, Any]) -> PlanExecutionResult:
        """Render either a static grading decision or a step-list plan."""
        x = np.clip(np.asarray(rgb01, dtype=np.float32).copy(), 0.0, 1.0)

        if is_grading_decision(plan):
            sequence = self._compile_decision(plan)  # [(op, params), ...]
        else:
            sequence = self._plan_sequence(plan)

        handlers = self._handlers()
        applied: List[str] = []
        skipped: List[str] = []
        for op_name, params in sequence:
            handler = handlers.get(op_name)
            if handler is None or params is None:
                skipped.append(op_name)
                continue
            x = np.clip(handler(x, params), 0.0, 1.0)
            applied.append(op_name)

        notes = plan.get("editor_notes", {})
        warnings = list(notes.get("warnings", [])) if isinstance(notes, dict) else []
        return PlanExecutionResult(
            image=x, applied_steps=applied, skipped_steps=skipped, warnings=warnings
        )

    # -- sequence builders --------------------------------------------------
    def _plan_sequence(self, plan: Dict[str, Any]) -> List[tuple]:
        """Ordered ``[(op, params|None)]`` for a step-list plan (None = skip)."""
        ops: Dict[str, Dict[str, Any]] = {}
        for step in plan.get("grading_plan", []):
            name = str(step.get("operation", "")).upper()
            if name:
                ops[name] = step.get("params", {}) or {}
        stage_to_op = {
            "white_balance": "WHITE_BALANCE", "primary": "PRIMARY_CORRECTION",
            "primary_correction": "PRIMARY_CORRECTION", "tone_curve": "TONE_CURVE",
            "color_wheels": "COLOR_WHEELS", "split_toning": "SPLIT_TONING",
            "hsl": "HSL", "presence": "PRESENCE", "vignette": "VIGNETTE", "grain": "GRAIN",
        }
        order = plan.get("executor_settings", {}).get("apply_order")
        if isinstance(order, list) and order:
            names = [stage_to_op.get(str(s).lower(), str(s).upper()) for s in order]
        else:
            names = list(ops.keys())
        seq: List[tuple] = []
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            seq.append((name, ops.get(name)))  # None => reported as skipped
        return seq

    def _compile_decision(self, d: Dict[str, Any]) -> List[tuple]:
        """Compile a static grading decision into ordered executor operations.

        Fixed canonical order: white balance → primary → tone curve → colour
        wheels → split toning → HSL → presence.  Colour-wheel lift/gamma/gain map
        to shadow/midtone/highlight zone tints; subject-skin luminance folds into
        the orange HSL band.
        """
        seq: List[tuple] = []
        wb = d.get("white_balance")
        if isinstance(wb, dict):
            seq.append(("WHITE_BALANCE", {"temperature": wb.get("temperature", 6500),
                                          "tint": wb.get("tint", 0)}))
        pc = d.get("primary_corrections")
        if isinstance(pc, dict):
            seq.append(("PRIMARY_CORRECTION", {
                "exposure": pc.get("exposure", 0), "contrast": pc.get("contrast", 0),
                "highlights": pc.get("highlights", 0), "shadows": pc.get("shadows", 0),
                "whites": pc.get("whites", 0), "blacks": pc.get("blacks", 0),
                "gamma": pc.get("gamma", 1.0), "pivot": pc.get("pivot", 0.435),
            }))
        tc = d.get("tone_curve")
        if isinstance(tc, dict):
            seq.append(("TONE_CURVE", {"points": build_curve_points(tc)}))
        cw = d.get("color_wheels")
        if isinstance(cw, dict):
            gain_strength = float(cw.get("gain", {}).get("strength", 0.0))
            seq.append(("COLOR_WHEELS", {
                "shadows": _wheel_to_zone(cw.get("lift", {})),
                "midtones": _wheel_to_zone(cw.get("gamma", {})),
                "highlights": _wheel_to_zone(cw.get("gain", {}), luminance=gain_strength * 5.0),
            }))
        st = d.get("split_toning")
        if isinstance(st, dict):
            seq.append(("SPLIT_TONING", st))
        hsl = d.get("hsl_adjustments")
        if isinstance(hsl, dict):
            seq.append(("HSL", self._fold_subject_into_hsl(hsl, d.get("subject_enhancement", {}))))
        pr = d.get("presence")
        if isinstance(pr, dict):
            seq.append(("PRESENCE", {k: pr.get(k, 0) for k in
                                     ("texture", "clarity", "dehaze", "vibrance", "saturation")}))
        return seq

    @staticmethod
    def _fold_subject_into_hsl(hsl: Dict[str, Any], subject: Dict[str, Any]) -> Dict[str, Any]:
        """Add subject skin luminance into the orange band (skin lives in orange)."""
        out = {k: dict(v) if isinstance(v, dict) else v for k, v in hsl.items()}
        skin_lum = float(subject.get("skin_luminance", 0.5)) if isinstance(subject, dict) else 0.5
        orange = dict(out.get("orange", {}))
        orange["luminance"] = float(orange.get("luminance", 0.0)) + max(0.0, skin_lum - 0.5) * 20.0
        out["orange"] = orange
        return out

    # -- operations ---------------------------------------------------------
    def _white_balance(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """Absolute WB target (Kelvin) + tint -> per-channel gains.

        A target below the ~6500K neutral warms the image (boost R, cut B); a
        higher target cools it.  Tint tilts green<->magenta (positive = magenta,
        i.e. reduce green).
        """
        temp = float(p.get("temperature", _WB_NEUTRAL_K))
        tint = float(p.get("tint", 0.0))
        t_norm = np.clip((_WB_NEUTRAL_K - temp) / _WB_NEUTRAL_K, -1.0, 1.0)
        x = x.copy()
        x[..., 0] *= 1.0 + 0.5 * t_norm      # red  warms as temp drops
        x[..., 2] *= 1.0 - 0.5 * t_norm      # blue cools as temp drops
        x[..., 1] *= 1.0 - 0.35 * np.clip(tint / 150.0, -1.0, 1.0)  # tint on green
        return x

    def _primary(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """Exposure (stops) + contrast + tonal-zone sliders (-100..100).

        Highlights/shadows/whites/blacks are luminance-masked luminance shifts
        (added to every channel, preserving hue).  Masks target progressively
        more extreme tones from mid -> extreme.
        """
        x = x * float(2.0 ** float(p.get("exposure", 0.0)))  # exposure
        contrast = float(p.get("contrast", 0.0)) / 100.0
        if contrast != 0.0:
            x = (x - 0.5) * (1.0 + contrast) + 0.5

        lum = _lum(x)[..., None]
        # Region weights (broad -> narrow toward the extremes).
        w_hi = _smoothstep(0.5, 1.0, lum)
        w_sh = 1.0 - _smoothstep(0.0, 0.5, lum)
        w_wh = _smoothstep(0.7, 1.0, lum)
        w_bl = 1.0 - _smoothstep(0.0, 0.3, lum)

        x = x + (float(p.get("highlights", 0.0)) / 100.0) * 0.5 * w_hi
        x = x + (float(p.get("shadows", 0.0)) / 100.0) * 0.5 * w_sh
        x = x + (float(p.get("whites", 0.0)) / 100.0) * 0.5 * w_wh
        x = x + (float(p.get("blacks", 0.0)) / 100.0) * 0.5 * w_bl

        # Optional midtone gamma (>1 brightens mids); pivot kept for API parity.
        gamma = float(p.get("gamma", 1.0))
        if gamma != 1.0:
            x = np.clip(x, 0.0, None) ** (1.0 / gamma)
        return x

    def _split_toning(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """Tint shadows/highlights toward chosen hues (luminance-weighted).

        ``p`` provides ``shadows``/``highlights`` (and optional ``midtones``)
        each with ``{hue, saturation}`` where saturation is ~0..40.  Applied at
        a gentle strength so it complements (not doubles) the colour wheels.
        """
        lum = _lum(x)[..., None]
        shadow_w = np.clip(1.0 - lum * 2.0, 0.0, 1.0)
        highlight_w = np.clip(lum * 2.0 - 1.0, 0.0, 1.0)
        for zone, weight in (("shadows", shadow_w), ("highlights", highlight_w)):
            zp = p.get(zone)
            if not isinstance(zp, dict):
                continue
            sat = float(zp.get("saturation", 0.0)) / 100.0
            if sat > 0.0:
                tint = np.asarray(_hue_to_rgb(float(zp.get("hue", 0.0)))) - 0.5
                x = x + weight * (sat * 0.5) * tint
        return x

    def _tone_curve(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """Optional parametric/point curve.

        Accepts ``{"points": [[in,out], ...]}`` in ``[0,1]`` (applied to
        luminance-preserving RGB via a shared LUT) or parametric
        ``{"highlights","lights","darks","shadows"}`` in -100..100.  Absent =>
        identity.
        """
        pts = p.get("points")
        if isinstance(pts, list) and len(pts) >= 2:
            arr = np.asarray(pts, dtype=np.float64)
            xp_ = np.clip(arr[:, 0], 0.0, 1.0)
            fp_ = np.clip(arr[:, 1], 0.0, 1.0)
            order = np.argsort(xp_)
            return np.interp(x, xp_[order], fp_[order]).astype(np.float32)

        # Parametric four-band curve (Lightroom-style regions).
        params = {k: float(p.get(k, 0.0)) / 100.0 for k in ("shadows", "darks", "lights", "highlights")}
        if not any(params.values()):
            return x
        lum = _lum(x)
        adj = (
            params["shadows"] * (1.0 - _smoothstep(0.0, 0.35, lum))
            + params["darks"] * _bump(lum, 0.25, 0.2)
            + params["lights"] * _bump(lum, 0.7, 0.2)
            + params["highlights"] * _smoothstep(0.65, 1.0, lum)
        )
        return x + 0.4 * adj[..., None]

    def _color_wheels(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """3-way colour grading: tint shadows/midtones/highlights by hue.

        Each zone provides ``{hue, saturation, luminance}`` (saturation &
        luminance in 0..100).  A luminance-masked, hue-directed, luminance-
        preserving offset is added; ``luminance`` lifts/lowers that zone.
        """
        lum = _lum(x)
        weights = {
            "shadows": 1.0 - _smoothstep(0.0, 0.5, lum),
            "midtones": _bump(lum, 0.5, 0.28),
            "highlights": _smoothstep(0.5, 1.0, lum),
        }
        for zone, w in weights.items():
            zp = p.get(zone)
            if not isinstance(zp, dict):
                continue
            sat = float(zp.get("saturation", 0.0)) / 100.0
            if sat != 0.0:
                tint = np.asarray(_hue_to_rgb(float(zp.get("hue", 0.0)))) - 0.5
                x = x + (w[..., None]) * (sat * 0.5) * tint
            lum_adj = float(zp.get("luminance", 0.0)) / 100.0
            if lum_adj != 0.0:
                x = x + (w[..., None]) * (lum_adj * 0.5)
        return x

    def _hsl(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """Per-hue-band selective hue/saturation/luminance in HSV space.

        For each named band a Gaussian membership weight over hue distance
        selects the affected pixels; ``hue`` rotates them (degrees), while
        ``saturation``/``luminance`` (-100..100) scale S and V.
        """
        hsv = rgb_to_hsv(np, x)
        H = hsv[..., 0].copy()
        S = hsv[..., 1].copy()
        V = hsv[..., 2].copy()
        sigma = 30.0
        for band, params in p.items():
            if not isinstance(params, dict):
                continue
            centre = _HSL_BANDS.get(str(band).lower())
            if centre is None:
                continue
            dist = np.abs((H - centre + 180.0) % 360.0 - 180.0)
            w = np.exp(-(dist ** 2) / (2.0 * sigma * sigma))
            H = H + w * float(params.get("hue", 0.0))               # deg shift
            S = S * (1.0 + w * float(params.get("saturation", 0.0)) / 100.0)
            V = V * (1.0 + w * float(params.get("luminance", 0.0)) / 100.0)
        hsv2 = np.stack([H % 360.0, np.clip(S, 0.0, 1.0), np.clip(V, 0.0, 1.0)], axis=-1)
        return hsv_to_rgb(np, hsv2)

    def _presence(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """Texture/clarity (local contrast), dehaze, vibrance, saturation."""
        h, w = x.shape[:2]
        base = min(h, w)
        lum = _lum(x)

        clarity = float(p.get("clarity", 0.0)) / 100.0
        if clarity != 0.0:
            blur = _gaussian(lum, sigma=max(2.0, base * 0.02))
            detail = (lum - blur)[..., None]
            midweight = (1.0 - (2.0 * lum - 1.0) ** 2)[..., None]  # protect extremes
            x = x + clarity * detail * midweight

        texture = float(p.get("texture", 0.0)) / 100.0
        if texture != 0.0:
            blur = _gaussian(lum, sigma=1.5)
            x = x + texture * (lum - blur)[..., None]

        dehaze = float(p.get("dehaze", 0.0)) / 100.0
        if dehaze != 0.0:
            x = (x - 0.5) * (1.0 + 0.4 * dehaze) + 0.5  # add contrast/punch

        # Vibrance + saturation in HSV (vibrance protects already-saturated hues).
        vib = float(p.get("vibrance", 0.0)) / 100.0
        sat = float(p.get("saturation", 0.0)) / 100.0
        if vib != 0.0 or sat != 0.0:
            hsv = rgb_to_hsv(np, np.clip(x, 0.0, 1.0))
            S = hsv[..., 1]
            S = S * (1.0 + sat)
            if vib != 0.0:
                S = S + vib * (1.0 - S) * S * 2.0  # stronger on mid-low sat
            hsv[..., 1] = np.clip(S, 0.0, 1.0)
            x = hsv_to_rgb(np, hsv)
        return x

    def _vignette(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """Radial darkening/brightening. ``amount`` in -100..100 (negative darkens)."""
        amount = float(p.get("amount", 0.0)) / 100.0
        if amount == 0.0:
            return x
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
        feather = float(p.get("feather", 50.0)) / 100.0
        mask = _smoothstep(0.5, 0.5 + max(0.05, feather), r)
        return x * (1.0 + amount * mask[..., None])

    def _grain(self, x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
        """Additive monochrome film grain. ``amount`` in 0..100."""
        amount = float(p.get("amount", 0.0)) / 100.0
        if amount <= 0.0:
            return x
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.03 * amount, size=x.shape[:2])[..., None]
        return x + noise


def _bump(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Gaussian-like midtone bump peaking at ``centre`` (for zone weighting)."""
    return np.exp(-((x - centre) ** 2) / (2.0 * width * width))


def _wheel_to_zone(wheel: Dict[str, Any], luminance: float = 0.0) -> Dict[str, float]:
    """Map a warm/cool colour wheel (temperature/tint/strength) to a zone tint.

    Warm wheels (temperature >= 0) tint toward orange (~40 deg), cool wheels
    toward teal (~210 deg); ``strength`` (0..1) sets the tint saturation.
    """
    temp = float(wheel.get("temperature", 0.0))
    strength = float(wheel.get("strength", 0.0))
    hue = 40.0 if temp >= 0 else 210.0
    return {"hue": hue, "saturation": strength * 100.0, "luminance": luminance}
