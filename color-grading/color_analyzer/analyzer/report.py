"""Reporting layer — plain-text summary and the optional HTML dashboard.

The image's measured grade is no longer written here.  It lives in the
``current`` field of every parameter in the unified ``grade.json`` document
(see :mod:`color_analyzer.analyzer.schema`), so a separate ``report.json``
would be the same numbers under different key names — exactly the drift the
single-source-of-truth schema exists to prevent.

The HTML dashboard (with the visualisations) is only written when visuals are
supplied, which requires a deep-mode analysis.

A :class:`rich`-based console printer is also provided for terminal output.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from typing import Dict, Optional

from .engine import EngineResult
from .visualization import VisualizationPaths


@dataclass
class ReportPaths:
    """Paths of the written report artefacts (``html_path`` empty if skipped)."""

    summary_path: str
    html_path: str = ""


class ReportGenerator:
    """Writes the plain-text summary and the optional HTML dashboard."""

    def generate(
        self,
        result: EngineResult,
        output_dir: str,
        visuals: Optional[VisualizationPaths] = None,
    ) -> ReportPaths:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "summary.txt")

        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(self.summary_text(result))

        html_path = ""
        if visuals is not None:  # HTML dashboard only when visuals are generated.
            html_path = os.path.join(output_dir, "report.html")
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(self.html(result, output_dir, visuals))

        return ReportPaths(summary_path=summary_path, html_path=html_path)

    # -- plain text ---------------------------------------------------------
    def summary_text(self, result: EngineResult) -> str:
        s = result.summary
        lines = [
            "=" * 60,
            "COLOUR GRADING ANALYSIS SUMMARY",
            "=" * 60,
            f"Source            : {result.source or '<array>'}",
            f"Resolution        : {result.width} x {result.height}",
            f"Backend           : {result.backend['backend']} "
            f"(gpu={result.backend['use_gpu']})",
            f"Analysis time     : {result.elapsed_seconds:.3f} s",
            "-" * 60,
            f"Overall grading   : {s.overall_grading}",
            f"Brightness        : {s.brightness}",
            f"Contrast          : {s.contrast}",
            f"Temperature       : {s.temperature} ({s.color_temperature_k:.0f} K)",
            f"Colourfulness     : {s.colorfulness}",
            f"Colour harmony    : {s.color_harmony}",
            f"Split toning      : {s.split_toning}",
            f"Skin tone quality : {s.skin_tone_quality}",
            f"Mood              : {s.mood}",
            f"Dominant colours  : {', '.join(s.dominant_colors)}",
            "Top styles        : "
            + ", ".join(f"{k} {v:.2f}" for k, v in s.top_styles),
            f"Confidence        : {s.confidence:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines) + "\n"

    # -- HTML ---------------------------------------------------------------
    def html(
        self,
        result: EngineResult,
        output_dir: str,
        visuals: Optional[VisualizationPaths],
    ) -> str:
        s = result.summary

        def rel(path: str) -> str:
            return html.escape(os.path.relpath(path, output_dir).replace(os.sep, "/"))

        swatches = "".join(
            f'<span class="swatch" style="background:{html.escape(hexc)}" title="{html.escape(hexc)}"></span>'
            for hexc in s.dominant_colors
        )

        rows = [
            ("Overall grading", s.overall_grading),
            ("Brightness", s.brightness),
            ("Contrast", s.contrast),
            ("Temperature", f"{s.temperature} ({s.color_temperature_k:.0f} K)"),
            ("Colourfulness", s.colorfulness),
            ("Colour harmony", s.color_harmony),
            ("Split toning", s.split_toning),
            ("Skin tone quality", s.skin_tone_quality),
            ("Mood", s.mood),
            ("Confidence", f"{s.confidence:.2f}"),
        ]
        table_rows = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows
        )
        styles_html = "".join(
            f'<div class="bar"><span>{html.escape(k)}</span>'
            f'<div class="track"><div class="fill" style="width:{v*100:.0f}%"></div></div>'
            f'<b>{v:.2f}</b></div>'
            for k, v in s.top_styles
        )

        image_order = [
            ("RGB histogram", "rgb_histogram"),
            ("HSV histogram", "hsv_histogram"),
            ("LAB histogram", "lab_histogram"),
            ("Luminance histogram", "luminance_histogram"),
            ("Dominant palette", "dominant_palette"),
            ("Tone curve", "tone_curve"),
            ("Brightness heatmap", "brightness_heatmap"),
            ("Saturation heatmap", "saturation_heatmap"),
            ("Hue heatmap", "hue_heatmap"),
            ("Shadow mask", "shadow_mask"),
            ("Midtone mask", "midtone_mask"),
            ("Highlight mask", "highlight_mask"),
        ]
        gallery = ""
        scatter_links = ""
        if visuals is not None:
            cards = []
            for title, key in image_order:
                path = visuals.files.get(key)
                if path and path.lower().endswith(".png") and os.path.exists(path):
                    cards.append(
                        f'<figure><img src="{rel(path)}" alt="{html.escape(title)}"/>'
                        f"<figcaption>{html.escape(title)}</figcaption></figure>"
                    )
            gallery = "".join(cards)
            for title, key in (("RGB 3-D cloud", "rgb_scatter_3d"), ("Lab 3-D cloud", "lab_scatter_3d")):
                path = visuals.files.get(key)
                if path and os.path.exists(path):
                    scatter_links += f'<a href="{rel(path)}" target="_blank">{html.escape(title)}</a> '

        return _HTML_TEMPLATE.format(
            source=html.escape(str(result.source or "<array>")),
            width=result.width,
            height=result.height,
            backend=html.escape(result.backend["backend"]),
            elapsed=f"{result.elapsed_seconds:.3f}",
            n_features=len(result.feature_vector),
            swatches=swatches,
            table_rows=table_rows,
            styles=styles_html,
            gallery=gallery,
            scatter_links=scatter_links,
        )

    # -- console ------------------------------------------------------------
    def print_console(self, result: EngineResult) -> None:
        """Pretty-print the summary to the terminal using ``rich`` if present."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
        except Exception:  # pragma: no cover
            print(self.summary_text(result))
            return

        # ``safe_box`` + explicit stdout keeps rendering robust on legacy
        # Windows consoles that use a non-UTF codepage.
        console = Console(safe_box=True)
        s = result.summary
        table = Table(show_header=False, box=None)
        table.add_column(style="bold cyan")
        table.add_column()
        for k, v in [
            ("Overall grading", s.overall_grading),
            ("Brightness", s.brightness),
            ("Contrast", s.contrast),
            ("Temperature", f"{s.temperature} ({s.color_temperature_k:.0f} K)"),
            ("Colourfulness", s.colorfulness),
            ("Colour harmony", s.color_harmony),
            ("Split toning", s.split_toning),
            ("Skin tone quality", s.skin_tone_quality),
            ("Mood", s.mood),
            ("Dominant colours", ", ".join(s.dominant_colors)),
            ("Top styles", ", ".join(f"{k2} {v2:.2f}" for k2, v2 in s.top_styles)),
            ("Confidence", f"{s.confidence:.2f}"),
        ]:
            table.add_row(k, str(v))
        try:
            console.print(
                Panel(
                    table,
                    title="[bold]Colour Grading Analysis[/bold]",
                    subtitle=f"{result.backend['backend']} - {result.elapsed_seconds:.2f}s - "
                    f"{len(result.feature_vector)} features",
                )
            )
        except UnicodeEncodeError:  # pragma: no cover - legacy console fallback
            print(self.summary_text(result))


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Colour Grading Report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
          background: #12141a; color: #e6e8ee; }}
  h1 {{ margin: 0 0 .25rem; font-size: 1.6rem; }}
  .meta {{ color: #9aa0ad; font-size: .85rem; margin-bottom: 1.5rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background: #1b1e27; border: 1px solid #2a2e3a; border-radius: 12px; padding: 1.2rem; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; color: #9aa0ad; font-weight: 600; padding: .35rem .5rem;
        width: 42%; vertical-align: top; }}
  td {{ padding: .35rem .5rem; }}
  .swatch {{ display: inline-block; width: 40px; height: 40px; border-radius: 8px;
             margin-right: 6px; border: 1px solid #3a3f4c; }}
  .bar {{ display: grid; grid-template-columns: 90px 1fr 42px; align-items: center;
          gap: .6rem; margin: .4rem 0; font-size: .9rem; }}
  .track {{ background: #2a2e3a; border-radius: 6px; height: 10px; overflow: hidden; }}
  .fill {{ background: linear-gradient(90deg,#f0a020,#e05a2b); height: 100%; }}
  .gallery {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(260px,1fr));
              gap: 1rem; margin-top: 1.5rem; }}
  figure {{ margin: 0; background: #1b1e27; border: 1px solid #2a2e3a; border-radius: 10px;
            padding: .5rem; }}
  figure img {{ width: 100%; border-radius: 6px; display: block; }}
  figcaption {{ text-align: center; color: #9aa0ad; font-size: .8rem; padding-top: .4rem; }}
  a {{ color: #f0a020; }}
</style></head><body>
  <h1>Colour Grading Analysis</h1>
  <div class="meta">{source} &middot; {width}&times;{height} &middot; backend: {backend}
      &middot; {elapsed}s &middot; {n_features} features</div>
  <div class="grid">
    <div class="card">
      <h3>Summary</h3>
      <table>{table_rows}</table>
      <h3 style="margin-top:1rem">Dominant colours</h3>
      <div>{swatches}</div>
    </div>
    <div class="card">
      <h3>Top styles</h3>
      {styles}
      <h3 style="margin-top:1rem">Interactive</h3>
      <div>{scatter_links}</div>
    </div>
  </div>
  <div class="gallery">{gallery}</div>
</body></html>
"""
