"""Visualisation layer.

Renders the analysis to image files (Matplotlib, non-interactive ``Agg``
backend) and optional interactive 3-D scatter plots (Plotly HTML).  Every
function converts device arrays to host NumPy first, so it works identically on
GPU and CPU.

All heavy scatter plots subsample pixels to keep rendering fast on 4K frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np

from .engine import EngineResult
from .utils import ImageContext


@dataclass
class VisualizationPaths:
    """Mapping of visualisation name -> written file path."""

    files: Dict[str, str] = field(default_factory=dict)

    def add(self, name: str, path: str) -> None:
        self.files[name] = path


class Visualizer:
    """Generates all visual artefacts for an :class:`EngineResult`."""

    def __init__(self, dpi: int = 110, scatter_samples: int = 8000) -> None:
        self.dpi = dpi
        self.scatter_samples = scatter_samples

    def generate_all(self, result: EngineResult, output_dir: str) -> VisualizationPaths:
        """Render every visualisation into ``output_dir``; return their paths.

        The four histogram plots need the deep ``histogram`` section.  Rather
        than fail with an ``AttributeError`` on ``None`` halfway through, the
        requirement is checked up front.
        """
        if result.histogram is None:
            raise ValueError(
                "visualisations require a deep analysis: build the engine with "
                "ColorGradingEngine(deep=True) (CLI: --deep, implied by --visuals)"
            )
        os.makedirs(output_dir, exist_ok=True)
        ctx = result.context
        paths = VisualizationPaths()

        paths.add("rgb_histogram", self._rgb_histogram(result, output_dir))
        paths.add("hsv_histogram", self._hsv_histogram(result, output_dir))
        paths.add("lab_histogram", self._lab_histogram(result, output_dir))
        paths.add("luminance_histogram", self._luminance_histogram(result, output_dir))

        paths.add("brightness_heatmap", self._heatmap(ctx, "brightness", output_dir))
        paths.add("saturation_heatmap", self._heatmap(ctx, "saturation", output_dir))
        paths.add("hue_heatmap", self._heatmap(ctx, "hue", output_dir))

        paths.add("dominant_palette", self._palette(result, output_dir))
        paths.add("shadow_mask", self._tonal_mask(ctx, "shadow", output_dir))
        paths.add("midtone_mask", self._tonal_mask(ctx, "midtone", output_dir))
        paths.add("highlight_mask", self._tonal_mask(ctx, "highlight", output_dir))
        paths.add("tone_curve", self._tone_curve(result, output_dir))

        paths.add("rgb_scatter_3d", self._scatter_3d(ctx, "rgb", output_dir))
        paths.add("lab_scatter_3d", self._scatter_3d(ctx, "lab", output_dir))
        return paths

    # -- histograms ---------------------------------------------------------
    def _rgb_histogram(self, result: EngineResult, out: str) -> str:
        hists = result.histogram.histograms
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.linspace(0, 1, len(hists["r"]))
        for ch, colour in (("r", "red"), ("g", "green"), ("b", "blue")):
            ax.plot(x, hists[ch], color=colour, label=ch.upper(), linewidth=1.2)
        ax.set_title("RGB histogram")
        ax.set_xlabel("intensity")
        ax.set_ylabel("probability")
        ax.legend()
        return self._save(fig, out, "rgb_histogram.png")

    def _hsv_histogram(self, result: EngineResult, out: str) -> str:
        hsv = result.hsv
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
        axes[0].bar(np.linspace(0, 360, len(hsv.hue_histogram)), hsv.hue_histogram,
                    width=360 / max(1, len(hsv.hue_histogram)), color="#8844aa")
        axes[0].set_title("Hue")
        axes[1].bar(np.linspace(0, 1, len(hsv.saturation_histogram)), hsv.saturation_histogram,
                    width=1 / max(1, len(hsv.saturation_histogram)), color="#aa4444")
        axes[1].set_title("Saturation")
        axes[2].bar(np.linspace(0, 1, len(hsv.value_histogram)), hsv.value_histogram,
                    width=1 / max(1, len(hsv.value_histogram)), color="#444444")
        axes[2].set_title("Value")
        fig.tight_layout()
        return self._save(fig, out, "hsv_histogram.png")

    def _lab_histogram(self, result: EngineResult, out: str) -> str:
        hists = result.histogram.histograms
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
        axes[0].plot(np.linspace(0, 100, len(hists["L"])), hists["L"], color="#222222")
        axes[0].set_title("L*")
        axes[1].plot(np.linspace(-128, 128, len(hists["a"])), hists["a"], color="#cc3366")
        axes[1].set_title("a* (green-red)")
        axes[2].plot(np.linspace(-128, 128, len(hists["b_lab"])), hists["b_lab"], color="#3366cc")
        axes[2].set_title("b* (blue-yellow)")
        fig.tight_layout()
        return self._save(fig, out, "lab_histogram.png")

    def _luminance_histogram(self, result: EngineResult, out: str) -> str:
        hists = result.histogram.histograms
        cdf = result.histogram.luminance_cdf
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.linspace(0, 1, len(hists["luma"]))
        ax.fill_between(x, hists["luma"], color="#888888", alpha=0.7, label="pdf")
        ax2 = ax.twinx()
        ax2.plot(np.linspace(0, 1, len(cdf)), cdf, color="#cc5500", label="cdf")
        ax.set_title("Luminance histogram + CDF")
        ax.set_xlabel("luminance")
        return self._save(fig, out, "luminance_histogram.png")

    # -- heatmaps -----------------------------------------------------------
    def _heatmap(self, ctx: ImageContext, kind: str, out: str) -> str:
        if kind == "brightness":
            data = ctx.backend.to_numpy(ctx.gray)
            cmap, title, vmax = "inferno", "Brightness heatmap", 1.0
        elif kind == "saturation":
            data = ctx.backend.to_numpy(ctx.hsv[..., 1])
            cmap, title, vmax = "viridis", "Saturation heatmap", 1.0
        else:  # hue
            data = ctx.backend.to_numpy(ctx.hsv[..., 0])
            cmap, title, vmax = "hsv", "Hue heatmap", 360.0
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return self._save(fig, out, f"{kind}_heatmap.png")

    # -- masks --------------------------------------------------------------
    def _tonal_mask(self, ctx: ImageContext, kind: str, out: str) -> str:
        lum = ctx.backend.to_numpy(ctx.gray)
        if kind == "shadow":
            mask = lum < 0.25
        elif kind == "highlight":
            mask = lum > 0.75
        else:
            mask = (lum >= 0.25) & (lum <= 0.75)
        rgb = ctx.rgb_u8.astype(np.float32) / 255.0
        overlay = rgb.copy()
        overlay[~mask] *= 0.15  # dim non-mask pixels
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f"{kind.capitalize()} mask ({mask.mean() * 100:.1f}%)")
        ax.axis("off")
        return self._save(fig, out, f"{kind}_mask.png")

    # -- palette ------------------------------------------------------------
    def _palette(self, result: EngineResult, out: str) -> str:
        colors = result.dominant_colors.colors
        fig, ax = plt.subplots(figsize=(8, 1.8))
        x = 0.0
        for c in colors:
            w = max(c.percentage, 0.001)
            ax.add_patch(plt.Rectangle((x, 0), w, 1, color=tuple(c.rgb)))
            if w > 0.04:
                ax.text(x + w / 2, 0.5, c.hex, ha="center", va="center",
                        fontsize=7, color=_text_colour(c.rgb))
            x += w
        ax.set_xlim(0, x)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Dominant colour palette (by coverage)")
        return self._save(fig, out, "dominant_palette.png")

    # -- tone curve ---------------------------------------------------------
    def _tone_curve(self, result: EngineResult, out: str) -> str:
        tc = result.tone_curve
        q = np.asarray(tc.curve_samples)
        p = np.linspace(0, 1, len(q))
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        ax.plot(p, q, color="#cc5500", linewidth=2, label="tone curve Q(p)")
        ax.plot([0, 1], [0, 1], "--", color="#888888", label="identity")
        ax.scatter([0, 1], [tc.black_point, tc.white_point], color="black", zorder=5)
        ax.set_title(f"Tone curve (gamma≈{tc.gamma:.2f}, S={tc.s_curve_strength:+.2f})")
        ax.set_xlabel("input rank")
        ax.set_ylabel("output luminance")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return self._save(fig, out, "tone_curve.png")

    # -- 3-D scatter (Plotly HTML, subsampled) ------------------------------
    def _scatter_3d(self, ctx: ImageContext, space: str, out: str) -> str:
        try:
            import plotly.graph_objects as go
        except Exception:  # pragma: no cover
            return ""
        rgb = ctx.rgb_flat
        n = int(rgb.shape[0])
        step = max(1, n // self.scatter_samples)
        rgb_np = ctx.backend.to_numpy(rgb[::step])
        colours = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in rgb_np]
        if space == "rgb":
            pts = rgb_np
            axis = ("R", "G", "B")
            title = "RGB colour cloud"
        else:
            lab_np = ctx.backend.to_numpy(ctx.lab_flat[::step])
            pts = lab_np
            axis = ("L*", "a*", "b*")
            title = "L*a*b* colour cloud"
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                    marker=dict(size=2, color=colours, opacity=0.7),
                )
            ]
        )
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title=axis[0], yaxis_title=axis[1], zaxis_title=axis[2]),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        path = os.path.join(out, f"{space}_scatter_3d.html")
        fig.write_html(path, include_plotlyjs="cdn")
        return path

    # -- helpers ------------------------------------------------------------
    def _save(self, fig, out: str, name: str) -> str:
        path = os.path.join(out, name)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path


def _text_colour(rgb) -> str:
    """Pick black/white label text for legibility on a coloured swatch."""
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "white" if luminance < 0.5 else "black"
