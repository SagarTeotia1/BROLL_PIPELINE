"""Core utilities: array backend abstraction, image loading, and math helpers.

This module is the foundation of the color-grading analysis engine.  Its central
idea is a *single* array namespace (``xp``) that resolves to CuPy when a CUDA
capable GPU is available and to NumPy otherwise.  Every downstream analyzer is
written against ``xp`` so there is **exactly one** implementation of each
algorithm that transparently runs on the GPU or the CPU.

Design goals
------------
* No duplicated CPU/GPU code paths.  Algorithms use ``ctx.xp`` (NumPy/CuPy).
* Minimal host<->device transfers.  Every colour space is computed **once** and
  cached inside :class:`ImageContext`.
* Fully typed, JSON-serialisable results via :class:`FeatureResult`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
# We attempt to import CuPy and query OpenCV's CUDA device count.  The result is
# a module-level singleton so detection only happens once per process.

try:  # pragma: no cover - depends on the host having CuPy + CUDA
    import cupy as _cupy  # type: ignore

    try:
        _cupy.cuda.runtime.getDeviceCount()
        _HAS_CUPY = _cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _HAS_CUPY = False
except Exception:
    _cupy = None
    _HAS_CUPY = False

try:
    import cv2

    try:
        _HAS_CV2_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        _HAS_CV2_CUDA = False
except Exception:  # pragma: no cover - OpenCV is a hard dependency
    cv2 = None  # type: ignore
    _HAS_CV2_CUDA = False


class Backend:
    """Resolves and exposes the active array backend.

    Attributes
    ----------
    use_gpu:
        ``True`` when CuPy with a usable CUDA device was found.
    xp:
        The array module (``cupy`` or ``numpy``) used for *all* computation.
    """

    def __init__(self, prefer_gpu: bool = True) -> None:
        self.has_cupy: bool = _HAS_CUPY
        self.has_cv2_cuda: bool = _HAS_CV2_CUDA
        self.use_gpu: bool = bool(prefer_gpu and _HAS_CUPY)
        self.xp = _cupy if self.use_gpu else np

    # -- transfer helpers ---------------------------------------------------
    def asarray(self, arr: Any) -> Any:
        """Move ``arr`` onto the active device (host->device if GPU)."""
        return self.xp.asarray(arr)

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Return a host NumPy copy of ``arr`` regardless of its device."""
        if self.use_gpu and isinstance(arr, _cupy.ndarray):  # type: ignore[union-attr]
            return _cupy.asnumpy(arr)  # type: ignore[union-attr]
        return np.asarray(arr)

    def describe(self) -> Dict[str, Any]:
        return {
            "use_gpu": self.use_gpu,
            "backend": "cupy" if self.use_gpu else "numpy",
            "has_cupy": self.has_cupy,
            "has_cv2_cuda": self.has_cv2_cuda,
        }


# A default, lazily shared backend instance.  Callers may create their own.
DEFAULT_BACKEND = Backend()


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------
def to_serializable(obj: Any) -> Any:
    """Recursively convert NumPy/CuPy scalars & arrays into plain Python types.

    Guarantees the output can be passed straight to :func:`json.dumps`.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if _cupy is not None and isinstance(obj, _cupy.ndarray):  # type: ignore[union-attr]
        obj = _cupy.asnumpy(obj)  # type: ignore[union-attr]
    if isinstance(obj, np.ndarray):
        return [to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return to_serializable(dataclasses.asdict(obj))
    return obj


@dataclass
class FeatureResult:
    """Base class for every analyzer result.

    Provides a uniform :meth:`to_dict` that yields JSON-serialisable output and
    a :meth:`flatten` that produces a flat ``{name: float}`` mapping suitable
    for the global feature vector.  Nested dataclasses/lists are handled.
    """

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(dataclasses.asdict(self))

    def flatten(self, prefix: str = "") -> Dict[str, float]:
        """Return a flat mapping of scalar features (arrays/lists are skipped).

        Only finite scalars are emitted so the resulting feature vector never
        contains NaN/Inf.  Nested dataclasses are recursed with a dotted prefix.
        """
        out: Dict[str, float] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            key = f"{prefix}{f.name}"
            _flatten_value(key, value, out)
        return out


def _flatten_value(key: str, value: Any, out: Dict[str, float]) -> None:
    """Helper that appends scalar leaves of ``value`` into ``out``."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            _flatten_value(f"{key}.{f.name}", getattr(value, f.name), out)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_value(f"{key}.{k}", v, out)
        return
    if _cupy is not None and isinstance(value, _cupy.ndarray):  # type: ignore[union-attr]
        value = _cupy.asnumpy(value)  # type: ignore[union-attr]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isfinite(v):
            out[key] = v
        return
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        out[key] = float(value)
        return
    if isinstance(value, (bool, np.bool_)):
        out[key] = float(bool(value))
        return
    # arrays, strings, tuples of colours, etc. are intentionally not flattened.


# ---------------------------------------------------------------------------
# Generic math helpers (backend-agnostic — operate on the passed ``xp``)
# ---------------------------------------------------------------------------
def safe_divide(xp: Any, num: Any, den: Any, eps: float = 1e-8) -> Any:
    """Element-wise division guarded against division by zero."""
    return num / (den + eps)


def normalized_histogram(xp: Any, data: Any, bins: int, value_range: Tuple[float, float]) -> Any:
    """Compute a probability-normalised histogram on the active backend.

    Returns a length-``bins`` vector summing to 1 (or all-zeros for empty
    input).  Works for both NumPy and CuPy since both expose ``histogram``.
    """
    flat = data.reshape(-1)
    hist, _ = xp.histogram(flat, bins=bins, range=value_range)
    hist = hist.astype(xp.float64)
    total = hist.sum()
    if float(total) <= 0.0:
        return hist
    return hist / total


def entropy_from_hist(xp: Any, prob: Any) -> float:
    """Shannon entropy (base-2) of a probability distribution ``prob``.

    ``H = -sum_i p_i * log2(p_i)`` restricted to ``p_i > 0``.  The result is in
    bits and returned as a host ``float``.
    """
    p = prob[prob > 0]
    if int(p.size) == 0:
        return 0.0
    h = -(p * (xp.log(p) / math.log(2.0))).sum()
    return float(h)


def channel_entropy(xp: Any, channel: Any, bins: int, value_range: Tuple[float, float]) -> float:
    """Convenience: entropy of a single channel via its normalised histogram."""
    prob = normalized_histogram(xp, channel, bins, value_range)
    return entropy_from_hist(xp, prob)


def moment_stats(xp: Any, data: Any) -> Dict[str, float]:
    """Return mean/median/variance/std/min/max/p5/p95 for a flat channel.

    All statistics are computed on-device and returned as host floats.  A
    single ``percentile`` call handles both tails to limit kernel launches.
    """
    flat = data.reshape(-1).astype(xp.float64)
    p5, p50, p95 = xp.percentile(flat, xp.asarray([5.0, 50.0, 95.0]))
    mean = flat.mean()
    var = flat.var()
    return {
        "mean": float(mean),
        "median": float(p50),
        "variance": float(var),
        "std": float(xp.sqrt(var)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p5": float(p5),
        "p95": float(p95),
    }


def skewness(xp: Any, data: Any) -> float:
    """Fisher-Pearson skewness ``E[((x-mu)/sigma)^3]`` (0 for symmetric data)."""
    flat = data.reshape(-1).astype(xp.float64)
    mu = flat.mean()
    sigma = flat.std()
    if float(sigma) < 1e-12:
        return 0.0
    return float(((flat - mu) ** 3).mean() / (sigma ** 3))


def kurtosis(xp: Any, data: Any) -> float:
    """Excess kurtosis ``E[((x-mu)/sigma)^4] - 3`` (0 for a normal distribution)."""
    flat = data.reshape(-1).astype(xp.float64)
    mu = flat.mean()
    sigma = flat.std()
    if float(sigma) < 1e-12:
        return 0.0
    return float(((flat - mu) ** 4).mean() / (sigma ** 4) - 3.0)


def clamp01(x: float) -> float:
    """Clamp a scalar to the closed unit interval ``[0, 1]``."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class ChannelStats(FeatureResult):
    """First/second-order statistics of a single scalar channel."""

    mean: float = 0.0
    median: float = 0.0
    variance: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    p5: float = 0.0
    p95: float = 0.0
    entropy: float = 0.0


def channel_stats(
    xp: Any,
    channel: Any,
    bins: int = 256,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> ChannelStats:
    """Build a :class:`ChannelStats` (moments + entropy) for ``channel``."""
    m = moment_stats(xp, channel)
    ent = channel_entropy(xp, channel, bins, value_range)
    return ChannelStats(
        mean=m["mean"],
        median=m["median"],
        variance=m["variance"],
        std=m["std"],
        minimum=m["min"],
        maximum=m["max"],
        p5=m["p5"],
        p95=m["p95"],
        entropy=ent,
    )


# ---------------------------------------------------------------------------
# Colour-space conversions (backend-agnostic, vectorised, no Python loops)
# ---------------------------------------------------------------------------
# All conversions take/return arrays on the *active* backend and are written
# purely with ``xp`` primitives, so they execute identically on NumPy and CuPy.
# Input RGB is assumed to be linearly indexed float in ``[0, 1]`` with shape
# ``(H, W, 3)`` unless otherwise noted.

# sRGB (D65) -> CIE XYZ matrix, IEC 61966-2-1.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
# CIE D65 reference white (normalised so Y = 1.0).
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def rgb_to_hsv(xp: Any, rgb: Any) -> Any:
    """Convert RGB ``[0,1]`` to HSV with H in ``[0,360)``, S,V in ``[0,1]``.

    Standard hexcone model: ``V = max(r,g,b)``, ``S = (V-min)/V``, and hue is a
    piecewise function of which channel is the maximum.  ``xp.where`` keeps the
    computation fully vectorised (no Python branching over pixels).
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    mx = xp.maximum(xp.maximum(r, g), b)
    mn = xp.minimum(xp.minimum(r, g), b)
    delta = mx - mn
    eps = 1e-12

    # Hue per dominant channel; 60 degrees per sextant.
    hue = xp.zeros_like(mx)
    hue = xp.where(mx == r, ((g - b) / (delta + eps)) % 6.0, hue)
    hue = xp.where(mx == g, ((b - r) / (delta + eps)) + 2.0, hue)
    hue = xp.where(mx == b, ((r - g) / (delta + eps)) + 4.0, hue)
    hue = (hue * 60.0) % 360.0
    hue = xp.where(delta < eps, 0.0, hue)  # achromatic -> hue 0

    sat = xp.where(mx < eps, 0.0, delta / (mx + eps))
    val = mx
    return xp.stack([hue, sat, val], axis=-1)


def hsv_to_rgb(xp: Any, hsv: Any) -> Any:
    """Inverse of :func:`rgb_to_hsv`. H in ``[0,360)``, S,V in ``[0,1]`` -> RGB.

    Standard hexcone reconstruction: ``C = V*S`` is the chroma, ``X`` the
    second-largest component, ``m = V - C`` the achromatic offset added back to
    every channel.  Fully vectorised via ``xp.where`` over the six hue sextants.
    """
    h = (hsv[..., 0] % 360.0) / 60.0
    s = xp.clip(hsv[..., 1], 0.0, 1.0)
    v = xp.clip(hsv[..., 2], 0.0, 1.0)
    c = v * s
    x = c * (1.0 - xp.abs(h % 2.0 - 1.0))
    m = v - c
    zero = xp.zeros_like(h)

    r = xp.where(h < 1, c, xp.where(h < 2, x, xp.where(h < 3, zero,
        xp.where(h < 4, zero, xp.where(h < 5, x, c)))))
    g = xp.where(h < 1, x, xp.where(h < 2, c, xp.where(h < 3, c,
        xp.where(h < 4, x, xp.where(h < 5, zero, zero)))))
    b = xp.where(h < 1, zero, xp.where(h < 2, zero, xp.where(h < 3, x,
        xp.where(h < 4, c, xp.where(h < 5, c, x)))))
    return xp.stack([r + m, g + m, b + m], axis=-1)


def _srgb_linearize(xp: Any, c: Any) -> Any:
    """Inverse sRGB companding (gamma expansion) to linear light."""
    return xp.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(xp: Any, rgb: Any) -> Any:
    """Convert sRGB ``[0,1]`` to CIE XYZ (D65), Y in ``[0,1]``.

    Pipeline: gamma-expand to linear RGB, then apply the 3x3 sRGB->XYZ matrix
    via a single tensordot (vectorised matrix product over all pixels).

    The matrix is cast to the image's dtype.  It is declared float64 for
    readability, and NumPy's promotion rules would otherwise silently return a
    float64 result for a float32 image — doubling the memory traffic of Lab,
    which is the array most analyzers read.
    """
    lin = _srgb_linearize(xp, rgb)
    m = xp.asarray(_SRGB_TO_XYZ.T, dtype=rgb.dtype)  # transposed so (H,W,3) @ (3,3) works
    return lin @ m


def xyz_to_lab(xp: Any, xyz: Any) -> Any:
    """Convert CIE XYZ (D65) to CIE L*a*b*.

    Uses the CIE nonlinearity ``f(t)`` with the linear segment near black to
    avoid an infinite slope at the origin::

        f(t) = t^(1/3)                if t > (6/29)^3
        f(t) = t/(3*(6/29)^2) + 4/29  otherwise

    Then ``L*=116 f(Y/Yn)-16``, ``a*=500(f(X/Xn)-f(Y/Yn))``,
    ``b*=200(f(Y/Yn)-f(Z/Zn))``.
    """
    white = xp.asarray(_D65, dtype=xyz.dtype)  # keep float32 in, float32 out
    scaled = xyz / white
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    f = xp.where(
        scaled > delta3,
        xp.cbrt(scaled) if hasattr(xp, "cbrt") else scaled ** (1.0 / 3.0),
        scaled / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    fx = f[..., 0]
    fy = f[..., 1]
    fz = f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return xp.stack([L, a, b], axis=-1)


def rgb_to_lab(xp: Any, rgb: Any) -> Any:
    """Convenience: sRGB ``[0,1]`` -> CIE L*a*b* (via XYZ)."""
    return xyz_to_lab(xp, rgb_to_xyz(xp, rgb))


def rgb_to_ycrcb(xp: Any, rgb: Any) -> Any:
    """Convert RGB ``[0,1]`` to full-range YCrCb (BT.601), all in ``[0,1]``.

    ``Y = 0.299R + 0.587G + 0.114B``; chroma is the scaled colour difference
    offset by 0.5 so neutral grey maps to ``Cr=Cb=0.5`` (matches OpenCV's
    8-bit convention scaled to unity).
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    return xp.stack([y, cr, cb], axis=-1)


def rgb_to_luminance(xp: Any, rgb: Any) -> Any:
    """Rec.709 relative luminance in ``[0,1]`` (``0.2126R+0.7152G+0.0722B``)."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---------------------------------------------------------------------------
# Image loading + shared context
# ---------------------------------------------------------------------------
class ImageContext:
    """Loads an image once and caches every colour space on the active device.

    All analyzers receive the *same* context, so each colour space is computed a
    single time and host<->device transfers are minimised.  Channels are stored
    in analysis-friendly ranges:

    ==========  =====================================================
    Attribute   Meaning / range
    ==========  =====================================================
    ``rgb``     sRGB float, ``[0,1]``, shape ``(H,W,3)``
    ``rgb_u8``  sRGB uint8, ``[0,255]`` (for OpenCV/visualisation)
    ``hsv``     H ``[0,360)``, S ``[0,1]``, V ``[0,1]``
    ``lab``     L ``[0,100]``, a/b roughly ``[-128,127]``
    ``xyz``     CIE XYZ, Y ``[0,1]``
    ``ycrcb``   Y,Cr,Cb ``[0,1]`` (neutral chroma = 0.5)
    ``gray``    Rec.709 luminance, ``[0,1]``
    ==========  =====================================================
    """

    #: Bin count of the cached luminance histogram.  4096 bins over [0,1] gives
    #: a quantisation error of ~0.012% of full scale on any percentile — three
    #: orders of magnitude below the precision any grading decision needs, and
    #: it turns an O(N log N) sort into one O(N) pass.
    LUMA_BINS = 4096

    #: Default window (pixels) for :meth:`local_std` — regional contrast.
    LOCAL_WINDOW = 15
    #: Small window for micro-contrast / detail energy.
    MICRO_WINDOW = 3

    def __init__(
        self,
        rgb01: np.ndarray,
        backend: "Backend | None" = None,
        source_path: str | None = None,
    ) -> None:
        self.backend = backend or DEFAULT_BACKEND
        xp = self.backend.xp
        self.xp = xp
        self.source_path = source_path

        if rgb01.ndim != 3 or rgb01.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) RGB image, got shape {rgb01.shape}")

        self.height: int = int(rgb01.shape[0])
        self.width: int = int(rgb01.shape[1])
        self.n_pixels: int = self.height * self.width

        rgb = xp.asarray(rgb01, dtype=xp.float32)
        rgb = xp.clip(rgb, 0.0, 1.0)
        self.rgb = rgb

        # Eager: needed by essentially every analyzer.
        self.hsv = rgb_to_hsv(xp, rgb)
        self.lab = xyz_to_lab(xp, rgb_to_xyz(xp, rgb))
        self.gray = rgb_to_luminance(xp, rgb)

        # Lazily derived, cached on first use (see the properties below).
        self._xyz: Any = None
        self._ycrcb: Any = None
        self._rgb_u8: Any = None
        self._gray_np: Any = None
        self._luma_cdf: Any = None
        self._local_std: Dict[int, Any] = {}
        self._hue_trig: Any = None

    # -- lazily derived colour spaces ---------------------------------------
    # These are each used by a single consumer, so computing them up front cost
    # a full-resolution allocation (and, on the GPU, a device->host copy) on
    # every frame that never touched them.
    @property
    def xyz(self) -> Any:
        """CIE XYZ (D65). Only the deep ``xyz`` analyzer reads this."""
        if self._xyz is None:
            self._xyz = rgb_to_xyz(self.xp, self.rgb)
        return self._xyz

    @property
    def ycrcb(self) -> Any:
        """YCrCb in ``[0,1]``. Read by the skin and deep ``ycrcb`` analyzers."""
        if self._ycrcb is None:
            self._ycrcb = rgb_to_ycrcb(self.xp, self.rgb)
        return self._ycrcb

    @property
    def rgb_u8(self) -> np.ndarray:
        """8-bit host copy for OpenCV/visualisation."""
        if self._rgb_u8 is None:
            self._rgb_u8 = self.backend.to_numpy(
                (self.rgb * 255.0 + 0.5).astype(self.xp.uint8)
            )
        return self._rgb_u8

    # -- shared derived quantities ------------------------------------------
    @property
    def gray_np(self) -> np.ndarray:
        """Host float32 luminance, computed once and shared.

        Several analyzers hand luminance to OpenCV; without this each one paid
        its own device->host transfer of a full-resolution plane.
        """
        if self._gray_np is None:
            self._gray_np = self.backend.to_numpy(self.gray).astype(np.float32)
        return self._gray_np

    def luma_percentile(self, *percentiles: float) -> Tuple[float, ...]:
        """Luminance percentiles from one cached histogram.

        ``xp.percentile`` sorts the whole frame on every call, and the analyzers
        between them ask for p1/p2/p5/p50/p95/p98/p99. This builds the CDF once
        and reads every percentile off it.
        """
        if self._luma_cdf is None:
            xp = self.xp
            hist, _ = xp.histogram(
                self.gray.reshape(-1), bins=self.LUMA_BINS, range=(0.0, 1.0)
            )
            cdf = xp.cumsum(hist.astype(xp.float64))
            total = float(cdf[-1]) if int(cdf.size) else 0.0
            self._luma_cdf = self.backend.to_numpy(cdf / (total + 1e-12))

        cdf = self._luma_cdf
        # searchsorted on the CDF: the first bin whose cumulative mass reaches q.
        targets = np.asarray(percentiles, dtype=np.float64) / 100.0
        idx = np.searchsorted(cdf, targets, side="left")
        idx = np.clip(idx, 0, self.LUMA_BINS - 1)
        # Bin centre, so the result is unbiased rather than the bin's lower edge.
        return tuple(float((i + 0.5) / self.LUMA_BINS) for i in idx)

    @property
    def hue_cos_sin(self) -> Tuple[Any, Any]:
        """``(cos, sin)`` of the hue angle, flattened and cached.

        Circular hue statistics need these, and the HSV, split-toning and
        local-region analyzers each computed them over the whole frame
        independently.
        """
        if self._hue_trig is None:
            theta = self.hsv[..., 0].reshape(-1) * (math.pi / 180.0)
            self._hue_trig = (self.xp.cos(theta), self.xp.sin(theta))
        return self._hue_trig

    def local_std(self, window: int | None = None) -> np.ndarray:
        """Local luminance standard deviation over a ``window`` box, cached.

        ``Var(x) = E[x^2] - E[x]^2`` with a box filter, so it is O(N) regardless
        of window size.

        The window is a parameter, not a constant, because the two callers mean
        different things by "local": contrast wants *regional* contrast
        (:data:`LOCAL_WINDOW`), while the HDR heuristic wants *micro*-contrast
        (:data:`MICRO_WINDOW`).  Collapsing them onto one window changes what
        the HDR score measures, not merely its scale.
        """
        k = int(window or self.LOCAL_WINDOW)
        cached = self._local_std.get(k)
        if cached is None:
            gray = self.gray_np
            if cv2 is not None:
                mean = cv2.blur(gray, (k, k))
                mean_sq = cv2.blur(gray * gray, (k, k))
            else:  # pragma: no cover - OpenCV is a hard dependency in practice
                mean = _box_filter_np(gray, k)
                mean_sq = _box_filter_np(gray * gray, k)
            var = np.clip(mean_sq - mean * mean, 0.0, None)
            cached = np.sqrt(var)
            self._local_std[k] = cached
        return cached

    # -- flattened (N,3)/(N,) views ----------------------------------------
    @property
    def rgb_flat(self) -> Any:
        return self.rgb.reshape(-1, 3)

    @property
    def hsv_flat(self) -> Any:
        return self.hsv.reshape(-1, 3)

    @property
    def lab_flat(self) -> Any:
        return self.lab.reshape(-1, 3)

    @property
    def gray_flat(self) -> Any:
        return self.gray.reshape(-1)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_path(
        cls,
        path: str,
        backend: "Backend | None" = None,
        max_side: int | None = None,
    ) -> "ImageContext":
        """Load an image from disk (via OpenCV) into an :class:`ImageContext`.

        Parameters
        ----------
        path:
            Image file path.  Decoded with OpenCV, converted BGR->RGB.
        max_side:
            If given, the longest side is downscaled to ``max_side`` (area
            interpolation) to bound analysis cost on very large 4K+ frames.
            Aspect ratio is preserved.
        """
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("OpenCV (cv2) is required to load images")
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        bgr = downscale(bgr, max_side)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return cls(rgb, backend=backend, source_path=path)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        backend: "Backend | None" = None,
        is_bgr: bool = False,
        max_side: int | None = None,
    ) -> "ImageContext":
        """Build a context from an in-memory image array (uint8 or float).

        ``max_side`` is applied **before** the float conversion so a 4K frame is
        never materialised as a full-resolution float array.
        """
        arr = np.asarray(array)
        arr = downscale(arr, max_side)
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            if float(arr.max()) > 1.5:  # heuristics: 0-255 floats
                arr = arr / 255.0
        if is_bgr:
            arr = arr[..., ::-1].copy()
        return cls(arr, backend=backend)


def downscale(image: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale ``image`` so its longest side is at most ``max_side``.

    Returns the input untouched when ``max_side`` is falsy or already satisfied,
    so callers can pass the option straight through.  Area interpolation is used
    because it averages rather than samples — the right choice when the point is
    to measure the frame's statistics, not to look at it.
    """
    if not max_side or cv2 is None:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _box_filter_np(img: np.ndarray, k: int) -> np.ndarray:  # pragma: no cover
    """Separable uniform box filter fallback when OpenCV is unavailable."""
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    cs = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    h, w = img.shape
    out = (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / float(k * k)
    return out[:h, :w]
