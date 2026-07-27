"""Frame preparation and the planes every stage shares.

One :class:`Frame` is built per image and handed to every analysis stage, so
the colour conversions, the luminance histogram and the tonal masks are each
computed once.

Only three representations exist: linear-indexed **RGB** in ``[0,1]``, **HSV**
with hue in degrees, and Rec.709 **luminance**.  There is deliberately no Lab,
XYZ or YCrCb anywhere in this engine — every editor-facing quantity it reports
is definable in RGB/HSV terms, and carrying extra colour spaces was most of what
made the previous pipeline expensive.

Stability
---------
Percentiles are read from a fixed 256-bin luminance histogram rather than by
sorting.  Two frames that differ only by sensor noise land in the same bins, so
they report the same black point — which is the property that matters when the
output drives an automated grade across consecutive frames.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .gpu import Backend, default_backend

#: Rec.709 luminance weights for R, G, B.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: Cross-over edges of the tonal zones. The zones are *soft*: shadow influence
#: fades out between these two luminances while midtone influence fades in, and
#: likewise for midtone/highlight. This mirrors how a lift/gamma/gain panel
#: actually behaves, and it is also what keeps the readings stable — a hard cut
#: at a single luminance means content sitting on that boundary flips zones
#: under sensor noise, and the affected wheel's colour balance jumps with it.
SHADOW_EDGE = (0.15, 0.55)
HIGHLIGHT_EDGE = (0.45, 0.85)

#: Pixels below this saturation are treated as near-neutral and used to
#: estimate white balance.  Chosen so a colourful frame still yields a usable
#: neutral population without admitting genuinely coloured subject matter.
NEUTRAL_SATURATION = 0.25

#: Bin count for the shared luminance histogram.
LUMA_BINS = 256

#: Saturation window over which a pixel starts counting as coloured at all,
#: used by every stage that reads hue. See :attr:`Frame.chroma_gate`.
CHROMA_GATE = (0.04, 0.12)

#: Value window below which a pixel is too dark for its colour to be trusted.
#: HSV saturation is ``(max - min) / max``, so the divisor shrinks with
#: brightness and near-black pixels report wildly unstable saturation — a
#: quarter-stop of sensor noise on a pixel at value 0.05 swings its saturation
#: by tenths. Such pixels also carry no colour an editor would grade.
VALUE_GATE = (0.05, 0.14)


class Frame:
    """A prepared frame plus the planes and masks the stages read.

    Attributes
    ----------
    rgb:
        ``(H, W, 3)`` float32 in ``[0,1]``.
    hsv:
        ``(H, W, 3)`` float32; hue in ``[0,360)``, saturation and value in ``[0,1]``.
    luma:
        ``(H, W)`` float32 Rec.709 luminance in ``[0,1]``.
    """

    def __init__(self, rgb01: np.ndarray, backend: Optional[Backend] = None,
                 source: Optional[str] = None) -> None:
        if rgb01.ndim != 3 or rgb01.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB image, got shape {rgb01.shape}")

        self.backend = backend or default_backend()
        self.source = source

        self.rgb = np.ascontiguousarray(np.clip(rgb01, 0.0, 1.0), dtype=np.float32)
        self.height, self.width = self.rgb.shape[:2]
        self.pixels = self.height * self.width

        self.hsv = self.backend.cvt_color(self.rgb, cv2.COLOR_RGB2HSV)
        self.luma = self.rgb @ _LUMA_WEIGHTS

        # Flattened views; every stage reduces over these.
        self.rgb_flat = self.rgb.reshape(-1, 3)
        self.hue = self.hsv[..., 0].reshape(-1)
        self.sat = self.hsv[..., 1].reshape(-1)
        self.val = self.hsv[..., 2].reshape(-1)
        self.luma_flat = self.luma.reshape(-1)

        self._cdf: Optional[np.ndarray] = None
        self._masks: dict = {}
        self._neutral: Optional[np.ndarray] = None
        self._hue_trig: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._chroma_gate: Optional[np.ndarray] = None
        self._value_gate: Optional[np.ndarray] = None

    # -- construction -------------------------------------------------------
    @classmethod
    def from_path(cls, path: str, backend: Optional[Backend] = None,
                  max_side: Optional[int] = 1024) -> "Frame":
        """Load an image file, downscaled so its longest side is ``max_side``."""
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return cls.from_bgr(bgr, backend=backend, max_side=max_side, source=path)

    @classmethod
    def from_bgr(cls, bgr: np.ndarray, backend: Optional[Backend] = None,
                 max_side: Optional[int] = 1024, source: Optional[str] = None) -> "Frame":
        """Build from an OpenCV BGR frame (uint8 or float)."""
        backend = backend or default_backend()
        bgr = downscale(bgr, max_side, backend)
        rgb = backend.cvt_color(bgr, cv2.COLOR_BGR2RGB)
        return cls(_to_unit_float(rgb), backend=backend, source=source)

    @classmethod
    def from_rgb(cls, rgb: np.ndarray, backend: Optional[Backend] = None,
                 max_side: Optional[int] = 1024, source: Optional[str] = None) -> "Frame":
        """Build from an RGB array (uint8 or float)."""
        backend = backend or default_backend()
        rgb = downscale(rgb, max_side, backend)
        return cls(_to_unit_float(rgb), backend=backend, source=source)

    # -- shared derived data ------------------------------------------------
    def percentiles(self, *fractions: float) -> Tuple[float, ...]:
        """Luminance percentiles (given as fractions of 100) from the shared CDF."""
        if self._cdf is None:
            hist = cv2.calcHist([self.luma], [0], None, [LUMA_BINS], [0.0, 1.0]).ravel()
            total = float(hist.sum())
            self._cdf = np.cumsum(hist) / (total if total > 0 else 1.0)

        targets = np.asarray(fractions, dtype=np.float64) / 100.0
        idx = np.clip(np.searchsorted(self._cdf, targets, side="left"), 0, LUMA_BINS - 1)
        # Bin centre keeps the estimate unbiased rather than always low.
        return tuple(float((i + 0.5) / LUMA_BINS) for i in idx)

    def zone_mask(self, zone: str) -> np.ndarray:
        """Soft membership weights over the flattened frame for a tonal zone.

        ``zone`` is ``"shadows"``, ``"midtones"`` or ``"highlights"``.  Weights
        are in ``[0,1]`` and the three zones sum to 1 at every pixel, so a
        weighted mean over a zone is a proper average and the three
        ``coverage`` figures still add up to the whole frame.

        Weights are multiplied into reductions rather than used to index, so no
        stage ever allocates a copy of the frame.
        """
        cached = self._masks.get(zone)
        if cached is not None:
            return cached

        shadow = 1.0 - _smoothstep(self.luma_flat, *SHADOW_EDGE)
        highlight = _smoothstep(self.luma_flat, *HIGHLIGHT_EDGE)
        weights = {
            "shadows": shadow,
            "highlights": highlight,
            "midtones": np.clip(1.0 - shadow - highlight, 0.0, 1.0),
        }
        if zone not in weights:
            raise ValueError(f"unknown tonal zone: {zone!r}")

        self._masks.update({name: value.astype(np.float32) for name, value in weights.items()})
        return self._masks[zone]

    @property
    def neutral_mask(self) -> np.ndarray:
        """Float32 weights emphasising near-neutral pixels, for white balance.

        Estimating white balance from the whole frame is what makes grey-world
        fail on a sunset or a forest: strongly coloured subject matter drags the
        average and the frame is reported as having a cast it does not have.
        Weighting toward low-saturation pixels measures the light instead of the
        content.

        **Weights, not a hard cut.** The population this selects has to survive
        the frame being white-balanced, and a threshold at a fixed saturation
        does not: warming a near-neutral frame pushes its pixels past the cut,
        they leave the measured set, and the reading barely moves — measured at
        3% of a requested warm shift landing, while the cooling direction worked
        fine. A smooth falloff keeps every pixel in the estimate with a weight
        that changes gradually, so the measurement tracks the correction in both
        directions.

        Clipped pixels are excluded outright: a blown highlight has no colour
        left to measure.
        """
        if self._neutral is None:
            neutrality = 1.0 - _smoothstep(self.sat, NEUTRAL_SATURATION * 0.6,
                                           NEUTRAL_SATURATION * 1.8)
            in_range = ((self.luma_flat > 0.05) & (self.luma_flat < 0.95)).astype(np.float32)
            weights = neutrality * in_range
            if float(weights.sum()) < 0.02 * self.pixels:
                # Nothing neutral enough to measure; fall back to the whole
                # frame and let `confidence` report how little that is worth.
                weights = in_range if float(in_range.sum()) > 0 else np.ones_like(in_range)
            self._neutral = weights.astype(np.float32)
        return self._neutral

    @property
    def neutral_coverage(self) -> float:
        """Fraction of the frame that is near-neutral."""
        return float(self.neutral_mask.sum()) / max(self.pixels, 1)

    def masked_rgb_mean(self, mask: np.ndarray) -> np.ndarray:
        """Mean RGB over a 0/1 mask, as a length-3 float array."""
        count = float(mask.sum())
        if count < 1.0:
            return np.zeros(3, dtype=np.float64)
        return np.array(
            [float(self.rgb_flat[:, c] @ mask) / count for c in range(3)],
            dtype=np.float64,
        )

    def masked_mean(self, values: np.ndarray, mask: np.ndarray) -> float:
        """Mean of ``values`` over a 0/1 mask (0.0 when the mask is empty)."""
        count = float(mask.sum())
        if count < 1.0:
            return 0.0
        return float(values @ mask) / count

    @property
    def hue_cos_sin(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(cos, sin)`` of the hue angle, computed once and shared.

        Ten-odd callers want circular hue statistics — seven HSL bands, three
        split-tone zones, skin — and each recomputing the trigonometry over the
        whole frame dominated the run time.
        """
        if self._hue_trig is None:
            radians = np.deg2rad(self.hue)
            self._hue_trig = (np.cos(radians), np.sin(radians))
        return self._hue_trig

    @property
    def chroma_gate(self) -> np.ndarray:
        """Smooth 0-to-1 weight for "this pixel has colour worth measuring".

        Two conditions, both smooth so a pixel hovering at a cutoff does not
        flip between frames:

        * **Saturated enough.** OpenCV reports hue 0 for any grey pixel, and hue
          0 is the centre of the red band, so without this every neutral in the
          frame is counted as red.
        * **Bright enough.** Saturation divides by the channel maximum, so it is
          unstable to the point of meaninglessness in near-black pixels.

        Shared by every stage that reads hue or saturation.
        """
        if self._chroma_gate is None:
            gate = _smoothstep(self.sat, *CHROMA_GATE) * self.value_gate
            self._chroma_gate = gate.astype(np.float32)
        return self._chroma_gate

    @property
    def value_gate(self) -> np.ndarray:
        """Smooth 0-to-1 weight for "this pixel is bright enough to judge".

        The brightness half of :attr:`chroma_gate`, on its own.

        Kept separate because it does not depend on saturation, and a
        measurement that a *saturation* control has to move must not be gated on
        saturation: raising saturation would push pixels into the measured
        population and lowering it would push them out, so the reading would
        move against the control at one end or the other. Measured going the
        wrong way in both directions before this was split out.
        """
        if self._value_gate is None:
            self._value_gate = _smoothstep(self.val, *VALUE_GATE).astype(np.float32)
        return self._value_gate

    def circular_hue_mean(self, mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Saturation-weighted mean hue in degrees over a mask.

        Hue is an angle, so the arithmetic mean of 350 and 10 degrees is wrong.
        This averages the unit vectors instead.  Weighting by saturation keeps
        near-grey pixels — whose hue is numerically defined but visually
        meaningless — from steering the result.
        """
        w = (self.sat if weights is None else weights) * mask
        total = float(w.sum())
        if total < 1e-6:
            return 0.0
        cos_h, sin_h = self.hue_cos_sin
        x = float(cos_h @ w) / total
        y = float(sin_h @ w) / total
        return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def downscale(image: np.ndarray, max_side: Optional[int],
              backend: Optional[Backend] = None) -> np.ndarray:
    """Downscale so the longest side is at most ``max_side``, preserving aspect.

    Area interpolation averages rather than samples, which is what you want when
    the result is going to be measured: point-sampling a 4K frame down to 1024
    would let a single bright pixel survive and skew the white point.
    """
    if not max_side:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    backend = backend or default_backend()
    return backend.resize(image, size, cv2.INTER_AREA)


def _smoothstep(values: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Hermite smoothstep: 0 below ``edge0``, 1 above ``edge1``, smooth between.

    Chosen over a linear ramp because its derivative is zero at both edges, so a
    pixel drifting across a zone boundary changes its weight gradually at the
    start rather than immediately.
    """
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _to_unit_float(image: np.ndarray) -> np.ndarray:
    """Convert uint8 or float input to float32 in ``[0,1]``."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    out = image.astype(np.float32)
    if float(out.max(initial=0.0)) > 1.5:  # 0-255 floats
        out /= 255.0
    return out
