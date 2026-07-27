"""Streamlit UI for the colour-grading analysis engine.

Upload an image and get the full grading analysis interactively: summary,
dominant palette, style scores, per-space statistics, tonal masks/heatmaps,
tone curve, interactive 3-D colour clouds, and downloadable JSON + feature
vector.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any, Dict

import numpy as np
import streamlit as st
from PIL import Image

from color_analyzer.analyzer.decision_engine import DecisionEngine, to_executor_decision
from color_analyzer.analyzer.engine import ColorGradingEngine, EngineResult
from color_analyzer.analyzer.grading import ColorGrader, GradingParams
from color_analyzer.analyzer.grading_plan import (
    GradingPlanExecutor,
    is_grading_decision,
    is_grading_plan,
)
from color_analyzer.analyzer.utils import Backend


def _require_streamlit_runtime() -> None:
    """Exit with a usable message when launched as `python app.py`.

    Streamlit only provides a script-run context under its own CLI. Run
    directly, every `st.*` call below would instead emit a "missing
    ScriptRunContext" warning — dozens of them, with no app and no hint that
    the command was simply wrong.
    """
    from streamlit.runtime import exists

    if not exists():
        import sys

        print(
            "This is a Streamlit app; run it with the Streamlit CLI:\n\n"
            "    streamlit run app.py\n\n"
            "For the command-line interface instead:\n\n"
            "    python main.py frame.jpg -o outputs\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


_require_streamlit_runtime()

st.set_page_config(page_title="Colour Grading Analyzer", page_icon="🎨", layout="wide")


# ---------------------------------------------------------------------------
# Cached engine (built once per backend choice)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine(force_cpu: bool, deep: bool = False) -> ColorGradingEngine:
    """Engine for the UI.

    The Analyze tab needs ``deep=True`` — it plots histograms and the full
    feature vector, which only the deep analyzers produce.  The Grade tab only
    needs the 45 parameters, so it uses the fast path.
    """
    return ColorGradingEngine(backend=Backend(prefer_gpu=not force_cpu), deep=deep)


@st.cache_data(show_spinner=False)
def analyze_bytes(data: bytes, force_cpu: bool, max_side: int) -> Dict[str, Any]:
    """Analyse raw image bytes and return a plain-dict payload for the UI.

    Cached on the (bytes, backend, max_side) key so re-renders are instant.
    Only JSON-friendly data + a few numpy arrays for plotting are returned so
    Streamlit can hash/serialise the result.
    """
    image = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    # Deep: this payload feeds the histogram plots and the feature-vector table.
    engine = get_engine(force_cpu, deep=True)
    result = engine.analyze_array(rgb, max_side=max_side if max_side > 0 else None)
    return _payload(result, result.context)


def _payload(result: EngineResult, ctx: Any) -> Dict[str, Any]:
    """Extract everything the UI needs (dicts + small arrays) from a result."""
    xp = ctx.backend
    # Sub-sample colour cloud for the 3-D scatter (keeps the page light).
    flat = ctx.rgb_flat
    n = int(flat.shape[0])
    step = max(1, n // 6000)
    rgb_pts = xp.to_numpy(flat[::step])
    lab_pts = xp.to_numpy(ctx.lab_flat[::step])
    return {
        "grade": DecisionEngine().grade(result),
        "full": result.to_dict(),
        "summary": result.summary.to_dict(),
        "dominant": [c.to_dict() for c in result.dominant_colors.colors],
        "harmony": result.harmony.to_dict(),
        "cinematic": result.cinematic.to_dict(),
        "histograms": result.histogram.histograms,
        "luminance_cdf": result.histogram.luminance_cdf,
        "tone_curve": result.tone_curve.to_dict(),
        "feature_names": list(result.feature_vector.names),
        "feature_values": result.feature_vector.to_array().astype(float),
        "gray": xp.to_numpy(ctx.gray),
        "hue": xp.to_numpy(ctx.hsv[..., 0]),
        "sat": xp.to_numpy(ctx.hsv[..., 1]),
        "rgb_pts": rgb_pts,
        "lab_pts": lab_pts,
        "backend": result.backend,
        "elapsed": result.elapsed_seconds,
        "width": result.width,
        "height": result.height,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_grade_table(doc: Dict[str, Any]) -> None:
    """Render the 45-parameter grade grouped by section.

    Read-only parameters (style scores, quality flags) have no recommendation,
    so their rows show a measurement and nothing else.
    """
    from color_analyzer.analyzer.schema import GROUPS, PARAMS

    grade = doc.get("grade", {})
    for group in GROUPS:
        rows = []
        for param in (p for p in PARAMS if p.group == group):
            entry = grade.get(param.name, {})
            rows.append({
                "parameter": param.field,
                "current": entry.get("current"),
                "recommended": entry.get("recommended", "—"),
                "delta": entry.get("delta", "—"),
                "unit": param.unit or "",
            })
        with st.expander(group.replace("_", " ").title(), expanded=group in ("white_balance", "primary")):
            st.dataframe(rows, hide_index=True, use_container_width=True)


def render_summary(p: Dict[str, Any]) -> None:
    s = p["summary"]
    st.subheader("Overall grading")
    st.markdown(f"### {s['overall_grading']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brightness", s["brightness"])
    c2.metric("Contrast", s["contrast"])
    c3.metric("Temperature", f"{s['color_temperature_k']:.0f} K", s["temperature"])
    c4.metric("Confidence", f"{s['confidence']*100:.0f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Colourfulness", s["colorfulness"])
    c2.metric("Colour harmony", s["color_harmony"])
    c3.metric("Mood", s["mood"])

    st.caption(f"Split toning — {s['split_toning']}")
    st.caption(f"Skin tone — {s['skin_tone_quality']}")


def render_palette(p: Dict[str, Any]) -> None:
    st.subheader("Dominant colour palette")
    colors = p["dominant"]
    cols = st.columns(len(colors))
    for col, c in zip(cols, colors):
        hexc = c["hex"]
        pct = c["percentage"] * 100.0
        col.markdown(
            f"<div style='background:{hexc};height:64px;border-radius:8px;"
            f"border:1px solid #3a3f4c'></div>"
            f"<div style='text-align:center;font-size:0.8rem;padding-top:4px'>"
            f"{hexc}<br/>{pct:.1f}%</div>",
            unsafe_allow_html=True,
        )


def render_styles(p: Dict[str, Any]) -> None:
    st.subheader("Cinematic style scores")
    cine = p["cinematic"]
    order = [
        "teal_orange_score", "film_look_score", "commercial_score", "natural_score",
        "vintage_score", "moody_score", "hdr_score", "low_key_score", "high_key_score",
        "teal_dominance", "orange_dominance",
    ]
    for key in order:
        label = key.replace("_score", "").replace("_", " ")
        st.progress(min(1.0, float(cine[key])), text=f"{label} — {cine[key]:.2f}")


def render_histograms(p: Dict[str, Any]) -> None:
    import plotly.graph_objects as go

    h = p["histograms"]
    st.subheader("Histograms")
    x = np.linspace(0, 1, len(h["r"]))
    fig = go.Figure()
    for ch, colour in (("r", "#e04b4b"), ("g", "#4bbf4b"), ("b", "#4b7be0")):
        fig.add_trace(go.Scatter(x=x, y=h[ch], line=dict(color=colour), name=ch.upper()))
    fig.add_trace(
        go.Scatter(x=x, y=h["luma"], line=dict(color="#cccccc", dash="dot"), name="luma")
    )
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def render_tone_curve(p: Dict[str, Any]) -> None:
    import plotly.graph_objects as go

    tc = p["tone_curve"]
    q = np.asarray(tc["curve_samples"])
    x = np.linspace(0, 1, len(q))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=q, line=dict(color="#e05a2b", width=3), name="tone curve"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#888", dash="dash"),
                             name="identity"))
    fig.update_layout(
        title=f"gamma≈{tc['gamma']:.2f} · S={tc['s_curve_strength']:+.2f} · "
        f"black={tc['black_point']:.2f} white={tc['white_point']:.2f}",
        height=360, margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="input rank", yaxis_title="output luminance",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_maps(p: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.subheader("Heatmaps & tonal masks")
    gray = p["gray"]
    tabs = st.tabs(["Brightness", "Saturation", "Hue", "Shadows", "Midtones", "Highlights"])

    def _heat(data, cmap, vmax):
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    def _mask(mask, title):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(np.where(mask, gray, gray * 0.15), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title} ({mask.mean()*100:.1f}%)")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tabs[0]:
        _heat(gray, "inferno", 1.0)
    with tabs[1]:
        _heat(p["sat"], "viridis", 1.0)
    with tabs[2]:
        _heat(p["hue"], "hsv", 360.0)
    with tabs[3]:
        _mask(gray < 0.25, "Shadows")
    with tabs[4]:
        _mask((gray >= 0.25) & (gray <= 0.75), "Midtones")
    with tabs[5]:
        _mask(gray > 0.75, "Highlights")


def render_scatter(p: Dict[str, Any]) -> None:
    import plotly.graph_objects as go

    st.subheader("3-D colour clouds")
    space = st.radio("Space", ["RGB", "Lab"], horizontal=True, key="scatter_space")
    pts = p["rgb_pts"] if space == "RGB" else p["lab_pts"]
    colours = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in p["rgb_pts"]]
    axis = ("R", "G", "B") if space == "RGB" else ("L*", "a*", "b*")
    fig = go.Figure(
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(size=2, color=colours, opacity=0.7),
        )
    )
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis_title=axis[0], yaxis_title=axis[1], zaxis_title=axis[2]),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_downloads(p: Dict[str, Any]) -> None:
    st.subheader("Downloads")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "report.json",
        data=json.dumps(p["full"], indent=2),
        file_name="report.json",
        mime="application/json",
    )
    fv = {n: float(v) for n, v in zip(p["feature_names"], p["feature_values"])}
    c2.download_button(
        "feature_vector.json",
        data=json.dumps(fv, indent=2),
        file_name="feature_vector.json",
        mime="application/json",
    )
    buf = io.BytesIO()
    np.save(buf, p["feature_values"])
    c3.download_button(
        "feature_vector.npy", data=buf.getvalue(), file_name="feature_vector.npy"
    )


# ---------------------------------------------------------------------------
# Grade mode helpers
# ---------------------------------------------------------------------------
# Which grading controls appear under which UI group.
PARAM_GROUPS: Dict[str, list] = {
    "Tone": ["exposure", "contrast", "gamma", "black_point", "white_point",
             "highlights", "shadows"],
    "Colour": ["temperature", "tint", "saturation", "vibrance"],
    "Split toning": ["split_shadow_hue", "split_shadow_strength",
                     "split_highlight_hue", "split_highlight_strength"],
}


def _load_rgb(data: bytes, max_side: int) -> np.ndarray:
    """Decode image bytes to float RGB ``[0,1]``, optionally downscaled."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if max_side > 0:
        w, h = image.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.asarray(image, dtype=np.float32) / 255.0


def _to_png_bytes(rgb01: np.ndarray) -> bytes:
    arr = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _ingest_grade_json(upload) -> None:
    """If a *new* grading JSON was uploaded, push its values into the controls.

    Accepts a grading-params file (direct) or an analysis ``report.json`` (from
    which a look preset is derived).  Runs BEFORE the sliders are created so the
    updated session-state values populate them this same run.
    """
    if upload is None:
        return
    raw = upload.getvalue()
    digest = hashlib.md5(raw).hexdigest()
    if st.session_state.get("_grade_json_hash") == digest:
        return  # already applied
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        st.session_state["_grade_note"] = f"⚠️ Could not parse JSON: {exc}"
        st.session_state["_grade_json_hash"] = digest
        return

    st.session_state["_grade_json_hash"] = digest

    # A structured grading *plan* or a static grading *decision* takes the
    # executor path (not the sliders).
    if is_grading_plan(obj) or is_grading_decision(obj):
        st.session_state["_grade_plan"] = obj
        st.session_state["_grade_ai"] = is_grading_decision(obj)
        goal = obj.get("goal", {})
        style = goal.get("style") or obj.get("style_profile", {}).get("primary_style", "grade")
        st.session_state["_grade_note"] = f"✅ Loaded {'grading decision' if is_grading_decision(obj) else 'grading plan'} — {style}."
        return

    # Otherwise it's slider values; leaving plan mode if a plan was active.
    st.session_state.pop("_grade_plan", None)
    if GradingParams.is_grading_dict(obj):
        params = GradingParams.from_dict(obj)
        note = "✅ Loaded grading controls from JSON."
    elif isinstance(obj, dict) and ("feature_vector" in obj or "summary" in obj):
        params = GradingParams.from_analysis(obj)
        note = "✅ Derived a look preset from an analysis report.json."
    else:
        params = GradingParams.from_dict(obj)
        note = "✅ Loaded grading values from JSON."

    for key, value in params.to_dict().items():
        st.session_state[f"g_{key}"] = value
    st.session_state["_grade_note"] = note


def _current_params() -> GradingParams:
    """Read the live slider values from session-state into a GradingParams."""
    defaults = GradingParams().to_dict()
    current = {k: float(st.session_state.get(f"g_{k}", v)) for k, v in defaults.items()}
    return GradingParams.from_dict(current)


def render_decision_details(decision: Dict[str, Any]) -> None:
    """Render the static grading-decision sections (creative style, quality, params)."""
    cs = decision.get("creative_style", {})
    if cs:
        st.markdown("**Target style profile**")
        cols = st.columns(2)
        for i, (key, value) in enumerate(cs.items()):
            if isinstance(value, (int, float)):
                cols[i % 2].progress(min(1.0, float(value)), text=f"{key} — {value:.2f}")

    qc = decision.get("quality_checks", {})
    if qc:
        chips = "  ".join(f"{'✅' if v else '⛔'} {k.replace('_', ' ')}" for k, v in qc.items())
        st.caption(chips)

    labels = {
        "white_balance": "White balance", "primary_corrections": "Primary corrections",
        "presence": "Presence", "color_wheels": "Colour wheels", "tone_curve": "Tone curve",
        "hsl_adjustments": "HSL", "split_toning": "Split toning",
        "subject_enhancement": "Subject enhancement",
    }
    with st.expander("Grading parameters (static schema)", expanded=True):
        for key, label in labels.items():
            if key in decision:
                st.markdown(f"**{label}**")
                st.json(decision[key], expanded=False)


def render_plan_details(plan: Dict[str, Any], result) -> None:
    """Show the grading plan's steps, style profile, warnings and predictions."""
    goal = plan.get("goal", {})
    style = plan.get("style_profile", {})
    if goal or style:
        bits = []
        if goal.get("style"):
            bits.append(f"**Style:** {goal['style']}")
        if style.get("mood"):
            bits.append(f"**Mood:** {style['mood']}")
        if goal.get("output_medium"):
            bits.append(f"**Output:** {goal['output_medium']}")
        if bits:
            st.markdown(" · ".join(bits))
        if goal.get("user_intent"):
            st.caption(goal["user_intent"])

    for warn in result.warnings:
        st.warning(warn)

    # Editor notes (decision-engine format: current look + recommendations).
    notes = plan.get("editor_notes", {})
    if isinstance(notes, dict) and (notes.get("current_look") or notes.get("recommendations")):
        n1, n2 = st.columns(2)
        with n1:
            if notes.get("current_look"):
                st.markdown("**Current look**")
                for item in notes["current_look"]:
                    st.markdown(f"- {item}")
        with n2:
            if notes.get("recommendations"):
                st.markdown("**Recommendations**")
                for item in notes["recommendations"]:
                    st.markdown(f"- {item}")
        if notes.get("summary"):
            st.caption(notes["summary"])

    # Target creative-style scores (decision-engine format).
    cs = plan.get("creative_style")
    if isinstance(cs, dict) and cs:
        st.markdown("**Target style profile**")
        for key, value in cs.items():
            if isinstance(value, (int, float)):
                st.progress(min(1.0, float(value)), text=f"{key} — {value:.2f}")

    with st.expander("Grading plan steps", expanded=True):
        for step in plan.get("grading_plan", []):
            op = step.get("operation", "?")
            reason = step.get("reason", "")
            conf = step.get("confidence")
            badge = "✅" if op in result.applied_steps else "⏭️"
            conf_txt = f" · conf {conf:.2f}" if isinstance(conf, (int, float)) else ""
            st.markdown(f"{badge} **{op}**{conf_txt}  \n  <small>{reason}</small>",
                        unsafe_allow_html=True)
            st.json(step.get("params", {}), expanded=False)
        if result.skipped_steps:
            st.caption("Skipped (not in plan / unsupported): " + ", ".join(result.skipped_steps))

    pred = plan.get("predicted_result")
    if isinstance(pred, dict):
        with st.expander("Predicted result"):
            st.json(pred, expanded=True)


def grade_mode(data: bytes, force_cpu: bool, max_side: int) -> None:
    st.subheader("🎨 Grade — apply colour grading to the image")
    st.caption(
        "Upload a **grading plan JSON** (editor-style operations) to apply it, or a "
        "flat **grading JSON** / analysis `report.json` to drive the manual sliders."
    )

    # Ensure every slider control has a session-state value before any widget.
    for key, value in GradingParams().to_dict().items():
        st.session_state.setdefault(f"g_{key}", value)

    src_col, ai_col = st.columns([3, 2])
    with src_col:
        grade_json = st.file_uploader(
            "Upload grading JSON / plan", type=["json"], key="grade_json_up"
        )
    with ai_col:
        st.write("")
        st.write("")
        if st.button("🤖 Auto-grade this image (AI decision)"):
            rgb_full = _load_rgb(data, max_side)
            with st.spinner("Analysing & deciding a grade…"):
                # Fast path: the grade needs only the core analyzers.
                doc = get_engine(force_cpu, deep=False).grade(rgb_full)
            # The executor speaks the section-shaped decision, not the document.
            st.session_state["_grade_plan"] = to_executor_decision(doc)
            st.session_state["_grade_doc"] = doc
            st.session_state["_grade_ai"] = True
            st.session_state["_grade_note"] = (
                f"🤖 AI-generated grade — {doc.get('style', {}).get('target', 'plan')}."
            )
            st.session_state["_grade_json_hash"] = None
            st.rerun()

    _ingest_grade_json(grade_json)
    if st.session_state.get("_grade_note"):
        st.info(st.session_state["_grade_note"])

    rgb = _load_rgb(data, max_side)
    plan = st.session_state.get("_grade_plan")

    # ---- Plan-executor path ----------------------------------------------
    if plan is not None:
        with st.sidebar:
            st.header("Grading plan")
            if st.button("↺ Clear plan (use manual sliders)"):
                for k in ("_grade_plan", "_grade_note", "_grade_json_hash", "_grade_ai"):
                    st.session_state.pop(k, None)
                st.rerun()
        result = GradingPlanExecutor().apply(rgb, plan)
        graded = result.image
        plan_name = "grade.json" if st.session_state.get("_grade_ai") else "plan.json"

        before, after = st.columns(2)
        with before:
            st.markdown("**Original**")
            st.image(_to_png_bytes(rgb), use_container_width=True)
        with after:
            st.markdown("**Graded (plan applied)**")
            st.image(_to_png_bytes(graded), use_container_width=True)

        st.caption("Applied: " + " → ".join(result.applied_steps))
        _grade_downloads_and_analyze(graded, force_cpu, json.dumps(plan, indent=2), plan_name)
        st.markdown("---")
        if is_grading_decision(plan):
            render_decision_details(plan)
        else:
            render_plan_details(plan, result)
        return

    # ---- Manual slider path ----------------------------------------------
    ranges = GradingParams.ranges()
    with st.sidebar:
        st.header("Grading controls")
        if st.button("↺ Reset to neutral"):
            for key, value in GradingParams().to_dict().items():
                st.session_state[f"g_{key}"] = value
            st.session_state.pop("_grade_note", None)
            st.session_state.pop("_grade_json_hash", None)
            st.rerun()
        for group, keys in PARAM_GROUPS.items():
            with st.expander(group, expanded=(group == "Colour")):
                for key in keys:
                    lo, hi = ranges[key]
                    st.slider(
                        key.replace("_", " "),
                        float(lo), float(hi),
                        key=f"g_{key}",
                        step=(hi - lo) / 200.0,
                    )

    params = _current_params()
    grader = ColorGrader(Backend(prefer_gpu=not force_cpu))
    graded = grader.apply(rgb, params)

    before, after = st.columns(2)
    with before:
        st.markdown("**Original**")
        st.image(_to_png_bytes(rgb), use_container_width=True)
    with after:
        st.markdown("**Graded**")
        st.image(_to_png_bytes(graded), use_container_width=True)

    st.markdown("---")
    _grade_downloads_and_analyze(
        graded, force_cpu, json.dumps({"grading": params.to_dict()}, indent=2), "grading.json"
    )
    st.caption(
        "Preview/download resolution follows the sidebar 'Downscale' setting "
        "(set it to 0 for full resolution)."
    )


def _grade_downloads_and_analyze(
    graded: np.ndarray, force_cpu: bool, json_payload: str, json_name: str
) -> None:
    """Shared download buttons + optional re-analysis of a graded image."""
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "⬇️ graded image (PNG)", data=_to_png_bytes(graded),
        file_name="graded.png", mime="image/png",
    )
    d2.download_button(
        f"⬇️ {json_name}", data=json_payload, file_name=json_name, mime="application/json",
    )
    if d3.button("🔍 Analyze graded result"):
        with st.spinner("Analysing graded image…"):
            # Deep, because _payload reads the histogram section.
            res = get_engine(force_cpu, deep=True).analyze_array(graded)
            pg = _payload(res, res.context)
        st.markdown("### Analysis of the graded result")
        render_summary(pg)
        render_palette(pg)
        render_styles(pg)


# ---------------------------------------------------------------------------
# Analyze mode
# ---------------------------------------------------------------------------
def analyze_mode(data: bytes, force_cpu: bool, max_side: int, name: str) -> None:
    left, right = st.columns([1, 1])
    with left:
        st.image(data, caption=name, use_container_width=True)

    with st.spinner("Analysing…"):
        p = analyze_bytes(data, force_cpu, max_side)

    with right:
        render_summary(p)
    st.caption(
        f"Analysed {p['width']}×{p['height']} on **{p['backend']['backend']}** "
        f"in {p['elapsed']:.2f}s · {len(p['feature_names'])} features"
    )

    st.markdown("---")
    st.subheader("Grade — 45 parameters (current → recommended)")
    doc = p["grade"]
    st.caption(
        f"Style target: **{doc['style'].get('target')}** "
        f"(confidence {doc['style'].get('confidence')}) · "
        + " · ".join(doc.get("notes", []))
    )
    st.download_button(
        "⬇️ grade.json", data=json.dumps(doc, indent=2),
        file_name="grade.json", mime="application/json",
    )
    render_grade_table(doc)

    st.markdown("---")
    render_palette(p)
    render_styles(p)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        render_histograms(p)
    with c2:
        render_tone_curve(p)

    st.markdown("---")
    render_maps(p)

    st.markdown("---")
    render_scatter(p)

    st.markdown("---")
    render_downloads(p)

    with st.expander(f"Raw feature vector ({len(p['feature_names'])} values)"):
        st.dataframe(
            {"feature": p["feature_names"], "value": [float(v) for v in p["feature_values"]]},
            use_container_width=True,
            height=400,
        )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🎨 Colour Grading Studio")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["🤖 AI Grade", "🔍 Analyze", "🎨 Grade"], index=0)
        backend = Backend()
        st.info(
            f"Backend: **{backend.describe()['backend']}** "
            f"({'GPU' if backend.use_gpu else 'CPU'})"
        )
        force_cpu = st.checkbox("Force CPU", value=not backend.use_gpu)
        max_side = st.slider(
            "Downscale long side (px, 0 = full res)", 0, 3840, 1600, step=160,
            help="Bounds cost on 4K frames. 0 keeps full resolution.",
        )
        st.markdown("---")
        st.caption("CLI: `python main.py frame.jpg -o outputs`")

    if mode == "🤖 AI Grade":
        st.caption(
            "Measure the frame's editable colour state, hand it to a grading "
            "model, and render what it sends back."
        )
    elif mode == "🔍 Analyze":
        st.caption(
            "Extract an image's full colour-grading fingerprint "
            "(tone curve, white balance, split toning, harmony, cinematic look)."
        )
    uploaded = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
    )
    if uploaded is None:
        st.info("⬆️ Upload an image to begin.")
        return

    data = uploaded.getvalue()
    if mode == "🤖 AI Grade":
        # Imported here, not at module scope: the editorial engine is
        # self-contained and this keeps the older pipeline's import graph clear
        # of it.
        from color_analyzer.editorial.ui import render as render_ai_grade

        render_ai_grade(data, force_cpu, max_side)
    elif mode == "🔍 Analyze":
        analyze_mode(data, force_cpu, max_side, uploaded.name)
    else:
        grade_mode(data, force_cpu, max_side)


if __name__ == "__main__":
    main()
