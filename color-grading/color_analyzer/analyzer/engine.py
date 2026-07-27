"""Colour-grading analysis engine — the orchestration layer.

:class:`ColorGradingEngine` runs the analyzers against a single shared
:class:`ImageContext` (so each colour space is computed once), assembles the
global feature vector, and derives a compact, human-readable
:class:`GradingSummary` used by the reporting layer.

Core vs deep
------------
The registry is split in two.  :data:`CORE_ANALYZERS` are the ones whose output
actually reaches the 45-parameter grade; :data:`DEEP_ANALYZERS` feed the
histograms, spatial maps and 300+ feature vector used by the reports and the
Streamlit UI.  Deep analyzers are **off by default** because they dominate the
cost — on a 1080p frame they were measured at roughly two thirds of total
analysis time while contributing nothing to the grade.

Pass ``deep=True`` (CLI: ``--deep``) to get everything.  Deep sections are
``None`` when they did not run, so consumers must check before dereferencing.

Adding/removing an analyzer is a one-line change to the two registries — the
feature vector, JSON and report all update automatically because they iterate
the same ordered mapping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .cinematic import CinematicAnalyzer, CinematicFeatures
from .colorfulness import ColorfulnessAnalyzer, ColorfulnessFeatures
from .contrast import ContrastAnalyzer, ContrastFeatures
from .dominant_colors import DominantColorAnalyzer, DominantColorFeatures
from .exposure import ExposureAnalyzer, ExposureFeatures
from .feature_vector import GlobalFeatureVector
from .harmony import HarmonyAnalyzer, HarmonyFeatures
from .histogram import HistogramAnalyzer, HistogramFeatures
from .hsv import HSVAnalyzer, HSVFeatures
from .lab import LabAnalyzer, LabFeatures
from .local_regions import LocalRegionAnalyzer, LocalRegionFeatures
from .perceptual import PerceptualAnalyzer, PerceptualFeatures
from .rgb import RGBAnalyzer, RGBFeatures
from .skin import SkinAnalyzer, SkinFeatures
from .split_toning import SplitToningAnalyzer, SplitToningFeatures
from .tone_curve import ToneCurveAnalyzer, ToneCurveFeatures
from .utils import Backend, FeatureResult, ImageContext, clamp01, to_serializable
from .white_balance import WhiteBalanceAnalyzer, WhiteBalanceFeatures
from .xyz import XYZAnalyzer, XYZFeatures
from .ycrcb import YCrCbAnalyzer, YCrCbFeatures


@dataclass
class GradingSummary(FeatureResult):
    """High-level, human-readable interpretation of the grade."""

    overall_grading: str = ""
    brightness: str = ""
    contrast: str = ""
    temperature: str = ""
    color_temperature_k: float = 6500.0
    dominant_colors: List[str] = field(default_factory=list)  # hex
    split_toning: str = ""
    color_harmony: str = ""
    colorfulness: str = ""
    skin_tone_quality: str = ""
    mood: str = ""
    top_styles: List[Tuple[str, float]] = field(default_factory=list)
    confidence: float = 0.0


# Sections whose output feeds the 45-parameter grade.
CORE_SECTIONS: Tuple[str, ...] = (
    "white_balance", "exposure", "contrast", "tone_curve", "hsv", "lab",
    "colorfulness", "split_toning", "skin", "cinematic", "dominant_colors",
)
# Sections that only feed reports, visualisations and the full feature vector.
DEEP_SECTIONS: Tuple[str, ...] = (
    "rgb", "xyz", "ycrcb", "histogram", "local_regions", "perceptual", "harmony",
)


@dataclass
class EngineResult:
    """Analysis output for a single image.

    Core sections are always populated.  Deep sections are ``None`` unless the
    engine ran with ``deep=True`` — check :attr:`deep` or test for ``None``
    before using them.
    """

    source: Optional[str]
    width: int
    height: int
    backend: Dict[str, Any]
    elapsed_seconds: float

    # -- core ---------------------------------------------------------------
    hsv: HSVFeatures
    lab: LabFeatures
    contrast: ContrastFeatures
    exposure: ExposureFeatures
    tone_curve: ToneCurveFeatures
    white_balance: WhiteBalanceFeatures
    dominant_colors: DominantColorFeatures
    colorfulness: ColorfulnessFeatures
    split_toning: SplitToningFeatures
    skin: SkinFeatures
    cinematic: CinematicFeatures

    feature_vector: GlobalFeatureVector
    summary: GradingSummary

    # -- deep (None unless deep=True) ---------------------------------------
    rgb: Optional[RGBFeatures] = None
    xyz: Optional[XYZFeatures] = None
    ycrcb: Optional[YCrCbFeatures] = None
    histogram: Optional[HistogramFeatures] = None
    local_regions: Optional[LocalRegionFeatures] = None
    perceptual: Optional[PerceptualFeatures] = None
    harmony: Optional[HarmonyFeatures] = None

    deep: bool = False

    # keep the (down-scaled) context for visualisation without re-loading.
    context: ImageContext = field(repr=False, default=None)  # type: ignore[assignment]

    # -- exports ------------------------------------------------------------
    def sections(self) -> Dict[str, FeatureResult]:
        """Ordered mapping of section-name -> analyzer result.

        Sections that did not run are omitted rather than emitted as ``None``,
        so ``to_dict()`` never produces a key with no content.
        """
        ordered = CORE_SECTIONS + DEEP_SECTIONS
        return {
            name: getattr(self, name)
            for name in ordered
            if getattr(self, name) is not None
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full JSON-serialisable dictionary of every result + summary."""
        out: Dict[str, Any] = {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "backend": self.backend,
            "elapsed_seconds": self.elapsed_seconds,
            "deep": self.deep,
            "summary": self.summary.to_dict(),
            "feature_vector": self.feature_vector.to_dict(),
        }
        for name, result in self.sections().items():
            out[name] = result.to_dict()
        return to_serializable(out)


class ColorGradingEngine:
    """Runs the full analysis pipeline and produces an :class:`EngineResult`."""

    def __init__(
        self,
        backend: Optional[Backend] = None,
        prefer_gpu: bool = True,
        deep: bool = False,
    ) -> None:
        self.backend = backend or Backend(prefer_gpu=prefer_gpu)
        self.deep = deep
        self._analyzers = self._build_analyzers(deep)

    @staticmethod
    def _build_analyzers(deep: bool) -> Dict[str, Any]:
        """Ordered registry of ``section -> analyzer instance``."""
        analyzers: Dict[str, Any] = {
            # -- core: everything the 45-parameter grade reads --------------
            "white_balance": WhiteBalanceAnalyzer(),
            "exposure": ExposureAnalyzer(),
            "contrast": ContrastAnalyzer(),
            "tone_curve": ToneCurveAnalyzer(),
            "hsv": HSVAnalyzer(deep=deep),
            "lab": LabAnalyzer(deep=deep),
            "colorfulness": ColorfulnessAnalyzer(),
            "split_toning": SplitToningAnalyzer(),
            "skin": SkinAnalyzer(),
            "cinematic": CinematicAnalyzer(),
            "dominant_colors": DominantColorAnalyzer(),
        }
        if deep:
            analyzers.update({
                "rgb": RGBAnalyzer(),
                "xyz": XYZAnalyzer(),
                "ycrcb": YCrCbAnalyzer(),
                "histogram": HistogramAnalyzer(),
                "local_regions": LocalRegionAnalyzer(),
                "perceptual": PerceptualAnalyzer(),
                "harmony": HarmonyAnalyzer(),
            })
        return analyzers

    # -- public API ---------------------------------------------------------
    def analyze_path(self, path: str, max_side: Optional[int] = None) -> EngineResult:
        """Analyse an image loaded from ``path``."""
        ctx = ImageContext.from_path(path, backend=self.backend, max_side=max_side)
        return self._run(ctx, source=path)

    def analyze_array(
        self,
        array: np.ndarray,
        is_bgr: bool = False,
        max_side: Optional[int] = None,
    ) -> EngineResult:
        """Analyse an in-memory image array (uint8 or float)."""
        ctx = ImageContext.from_array(
            array, backend=self.backend, is_bgr=is_bgr, max_side=max_side
        )
        return self._run(ctx, source=None)

    def analyze_frames(
        self,
        frames: Iterable[np.ndarray],
        is_bgr: bool = True,
        max_side: Optional[int] = 1024,
    ) -> Iterator[EngineResult]:
        """Analyse a stream of decoded video frames, reusing this engine.

        The intended entry point for a video pipeline: backend detection and the
        analyzer instances are created once, and each frame is capped to
        ``max_side`` before analysis.  Defaults to ``is_bgr=True`` because that
        is what OpenCV hands you.
        """
        for frame in frames:
            yield self.analyze_array(frame, is_bgr=is_bgr, max_side=max_side)

    def analyze(self, source: Any, **kwargs: Any) -> EngineResult:
        """Full internal analysis API — returns every extracted metric.

        Dispatch on ``source``: a ``str`` path or an in-memory array.
        """
        if isinstance(source, str):
            return self.analyze_path(source, **kwargs)
        return self.analyze_array(np.asarray(source), **kwargs)

    def decide(self, result: EngineResult) -> Dict[str, Any]:
        """Convert an existing analysis into the 45-parameter grade document."""
        from .decision_engine import DecisionEngine

        return DecisionEngine().grade(result)

    def grade(self, source: Any, **kwargs: Any) -> Dict[str, Any]:
        """Analyse ``source`` and return its 45-parameter grade document.

        Every parameter carries the measured ``current`` value and, when it is
        adjustable, the ``recommended`` target and the ``delta`` between them.
        Feed the document to
        :func:`~color_analyzer.analyzer.decision_engine.to_executor_decision`
        to render it.
        """
        return self.decide(self.analyze(source, **kwargs))

    # Backwards-compatible alias for the pre-45-parameter API.
    def recommend_grade(self, source: Any, **kwargs: Any) -> Dict[str, Any]:
        """Deprecated name for :meth:`grade`."""
        return self.grade(source, **kwargs)

    # -- internals ----------------------------------------------------------
    def _run(self, ctx: ImageContext, source: Optional[str]) -> EngineResult:
        start = time.perf_counter()
        results: Dict[str, FeatureResult] = {}
        for name, analyzer in self._analyzers.items():
            results[name] = analyzer.analyze(ctx)
        elapsed = time.perf_counter() - start

        # Build the ordered global feature vector from the raw results.
        feature_vector = GlobalFeatureVector.from_sections(results)
        summary = self._summarize(results)

        return EngineResult(
            source=source,
            width=ctx.width,
            height=ctx.height,
            backend=self.backend.describe(),
            elapsed_seconds=elapsed,
            feature_vector=feature_vector,
            summary=summary,
            context=ctx,
            deep=self.deep,
            **results,  # type: ignore[arg-type]
        )

    def _summarize(self, r: Dict[str, FeatureResult]) -> GradingSummary:
        """Derive human-readable labels from the numeric results."""
        exposure: ExposureFeatures = r["exposure"]  # type: ignore[assignment]
        contrast: ContrastFeatures = r["contrast"]  # type: ignore[assignment]
        wb: WhiteBalanceFeatures = r["white_balance"]  # type: ignore[assignment]
        dom: DominantColorFeatures = r["dominant_colors"]  # type: ignore[assignment]
        split: SplitToningFeatures = r["split_toning"]  # type: ignore[assignment]
        colourful: ColorfulnessFeatures = r["colorfulness"]  # type: ignore[assignment]
        skin: SkinFeatures = r["skin"]  # type: ignore[assignment]
        cine: CinematicFeatures = r["cinematic"]  # type: ignore[assignment]
        # Harmony is a deep section: absent on the fast path.
        harmony: Optional[HarmonyFeatures] = r.get("harmony")  # type: ignore[assignment]

        brightness = _bucket(
            exposure.mean_brightness,
            [(0.2, "very dark"), (0.4, "dark"), (0.6, "balanced"), (0.8, "bright")],
            "very bright",
        )
        contrast_label = _bucket(
            contrast.global_contrast,
            [(0.2, "flat/low"), (0.4, "moderate"), (0.65, "high")],
            "very high",
        )
        temperature = _temp_label(wb.color_temperature)
        colourfulness = _bucket(
            colourful.color_richness,
            [(0.2, "muted"), (0.45, "moderate"), (0.7, "vivid")],
            "very vivid",
        )

        # Rank the cinematic style scores.
        styles = [
            ("teal-orange", cine.teal_orange_score),
            ("vintage", cine.vintage_score),
            ("commercial", cine.commercial_score),
            ("natural", cine.natural_score),
            ("HDR", cine.hdr_score),
            ("low-key", cine.low_key_score),
            ("high-key", cine.high_key_score),
            ("moody", cine.moody_score),
            ("film", cine.film_look_score),
        ]
        styles.sort(key=lambda kv: kv[1], reverse=True)
        top_styles = [(k, round(v, 3)) for k, v in styles[:3]]
        mood = styles[0][0]
        overall = f"{brightness}, {contrast_label} contrast, {temperature}, {styles[0][0]} look"

        if split.split_tone_confidence > 0.35:
            split_desc = (
                f"shadows ~{int(split.shadows.hue)}deg, highlights ~{int(split.highlights.hue)}deg "
                f"(confidence {split.split_tone_confidence:.2f})"
            )
        else:
            split_desc = "no strong split toning"

        if skin.detected:
            skin_desc = _bucket(
                skin.skin_naturalness,
                [(0.35, "unnatural"), (0.6, "acceptable"), (0.8, "natural")],
                "very natural",
            ) + f" ({skin.skin_percentage * 100:.1f}% of frame)"
        else:
            skin_desc = "no significant skin detected"

        # Overall confidence: decisiveness of the main classifications. Without
        # the deep harmony section its 0.4 share is redistributed across the two
        # remaining terms rather than counted as zero, which would otherwise make
        # every fast-path frame look low-confidence.
        if harmony is not None:
            confidence = clamp01(
                0.4 * harmony.best_confidence
                + 0.3 * styles[0][1]
                + 0.3 * exposure.exposure_quality
            )
            harmony_desc = f"{harmony.best_match} ({harmony.best_confidence:.2f})"
        else:
            confidence = clamp01(0.5 * styles[0][1] + 0.5 * exposure.exposure_quality)
            harmony_desc = "not analysed (deep mode only)"

        return GradingSummary(
            overall_grading=overall,
            brightness=brightness,
            contrast=contrast_label,
            temperature=temperature,
            color_temperature_k=wb.color_temperature,
            dominant_colors=[c.hex for c in dom.colors[:5]],
            split_toning=split_desc,
            color_harmony=harmony_desc,
            colorfulness=colourfulness,
            skin_tone_quality=skin_desc,
            mood=mood,
            top_styles=top_styles,
            confidence=confidence,
        )


def _bucket(value: float, thresholds: List[Tuple[float, str]], default: str) -> str:
    """Map ``value`` to a label using ascending ``(upper_bound, label)`` pairs."""
    for upper, label in thresholds:
        if value < upper:
            return label
    return default


def _temp_label(cct: float) -> str:
    """Human label for a correlated colour temperature in Kelvin."""
    if cct < 3500:
        return "very warm"
    if cct < 5000:
        return "warm"
    if cct < 6500:
        return "neutral-warm"
    if cct < 8000:
        return "neutral-cool"
    return "cool"
