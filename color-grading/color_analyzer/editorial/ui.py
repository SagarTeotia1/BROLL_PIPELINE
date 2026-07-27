"""Streamlit view for the grading loop.

Kept out of ``app.py`` so the editorial engine stays independent of the older
pipeline: this module imports nothing from ``color_analyzer.analyzer``, and
``app.py`` reaches it through one function.

The loop has two halves and the UI mirrors them — build a payload for the model,
then render what it sends back — with the frame's measured controls visible
between the two so a person can see what the model was given.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from . import controls as controls_module
from .analyzer import EditorialAnalyzer
from .apply import apply_controls
from .prompt import build_payload

@st.cache_resource
def _analyzer(force_cpu: bool, max_side: int) -> EditorialAnalyzer:
    from .gpu import Backend

    return EditorialAnalyzer(backend=Backend(prefer_gpu=not force_cpu),
                             max_side=max_side or None)


@st.cache_data(show_spinner=False)
def _analyse(data: bytes, force_cpu: bool, max_side: int) -> Dict[str, Any]:
    """Measure an uploaded image. Cached on the bytes, so re-renders are free."""
    rgb = _decode(data)
    return _analyzer(force_cpu, max_side).analyze_rgb(rgb)


def _decode(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def render(data: bytes, force_cpu: bool, max_side: int) -> None:
    """Draw the whole loop for one uploaded image."""
    rgb = _decode(data)
    with st.spinner("Measuring the frame…"):
        state = _analyse(data, force_cpu, max_side)

    st.caption(
        f"{state['meta']['width']}×{state['meta']['height']} on "
        f"**{state['meta']['device']}** · {state['meta']['elapsed_ms']:.0f} ms · "
        f"{state['look']['overall_look']} · mood **{state['look']['mood']}**"
    )

    step_one, step_two = st.tabs(["1 · Build the payload", "2 · Apply the reply"])
    with step_one:
        _render_payload(state)
    with step_two:
        _render_apply(rgb, state, force_cpu, max_side)

    st.markdown("---")
    _render_controls_table(state)


# ---------------------------------------------------------------------------
# step 1 — payload out
# ---------------------------------------------------------------------------
def _render_payload(state: Dict[str, Any]) -> None:
    st.subheader("Send this to your grading model")

    instruction = st.text_input(
        "What should the grade do?",
        value="dark cinematic grading",
        help="Plain language. This is the only instruction the model gets about "
             "the look you want.",
    )
    payload = build_payload(state, instruction)
    text = json.dumps(payload, indent=2)

    left, right = st.columns([1, 1])
    left.download_button(
        "⬇️ payload.json", data=text, file_name="payload.json",
        mime="application/json", width='stretch',
    )
    right.metric("Payload size", f"{len(text) / 1024:.1f} KB")

    notes = payload["context"].get("notes", [])
    if notes:
        with st.expander(f"What the model is told about this frame ({len(notes)})",
                         expanded=True):
            for note in notes:
                st.markdown(f"- {note}")

    with st.expander("Full payload"):
        st.code(text, language="json")

    st.info(
        "The model returns the **`controls`** object with values changed. Those "
        "values are a *target state*, not adjustments: `temperature: 5100` means "
        "the frame is at 5100 K now — return 4000 to make it warmer.",
        icon="💡",
    )


# ---------------------------------------------------------------------------
# step 2 — reply in, graded frame out
# ---------------------------------------------------------------------------
def _render_apply(rgb: np.ndarray, state: Dict[str, Any],
                  force_cpu: bool, max_side: int) -> None:
    st.subheader("Paste what the model sent back")

    pasted = st.text_area(
        "Reply JSON", height=200, key="editorial_reply_text",
        placeholder='{"white_balance": {"temperature": 7400, ...}, ...}',
        help="A ```json fence around it is fine, and so is the whole payload "
             "rather than just the controls.",
    )
    uploaded = st.file_uploader("…or upload reply.json", type=["json"],
                                key="editorial_reply_file")

    raw = uploaded.getvalue().decode("utf-8") if uploaded is not None else pasted
    protect_skin = st.checkbox(
        "Protect skin tones", value=True,
        help="Holds global moves back on skin so a heavy grade does not turn "
             "faces grey. Turn off to apply exactly what the model asked for.",
    )

    if not raw.strip():
        st.info("⬆️ Paste or upload the model's reply to render it.")
        return

    reply, error = _parse(raw)
    if error:
        st.error(error)
        return

    with st.spinner("Rendering the grade…"):
        result = apply_controls(rgb, reply, source=state, protect_skin=protect_skin,
                                max_side=max_side or None)

    if not result.applied:
        st.warning(
            "Nothing was applied — the reply matches the frame's current state. "
            "Did the model return the payload unchanged?"
        )
    else:
        st.success(f"Applied: {', '.join(result.applied)}")

    for note in result.warnings():
        st.warning(note, icon="⚠️")
    if result.clamped:
        with st.expander(f"Clamped into range ({len(result.clamped)})"):
            for note in result.clamped:
                st.markdown(f"- {note}")
    if result.ignored:
        with st.expander(f"Ignored ({len(result.ignored)})"):
            for note in result.ignored:
                st.markdown(f"- {note}")

    before, after = st.columns(2)
    before.image(np.clip(rgb, 0, 1), caption="Before", width='stretch')
    after.image(result.image, caption="After", width='stretch')

    st.download_button(
        "⬇️ graded.png", data=_to_png(result.image), file_name="graded.png",
        mime="image/png", width='stretch',
    )

    with st.expander("What the graded frame measures now"):
        graded_state = _analyzer(force_cpu, max_side).analyze_rgb(result.image)
        _render_comparison(state, graded_state, reply)


def _parse(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Read a model's reply, tolerating a markdown fence and a wrapped payload."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return None, f"That is not valid JSON: {error}"
    if isinstance(parsed, dict) and "controls" in parsed:
        parsed = parsed["controls"]   # the whole payload came back
    if not isinstance(parsed, dict):
        return None, f"Expected a JSON object, got {type(parsed).__name__}."
    return parsed, None


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _render_controls_table(state: Dict[str, Any]) -> None:
    """The frame's measured controls, grouped as the payload presents them."""
    st.subheader("Measured controls")
    flat = controls_module.flatten(state)

    groups: Dict[str, list] = {}
    for control in controls_module.CONTROLS:
        if control.path not in flat:
            continue
        group = control.path.split(".", 1)[0]
        groups.setdefault(group, []).append({
            "control": control.path.split(".", 1)[1],
            "value": flat[control.path],
            "range": f"{control.lo:g} … {control.hi:g}",
            "unit": control.unit or "",
        })

    columns = st.columns(2)
    for index, (group, rows) in enumerate(groups.items()):
        with columns[index % 2]:
            with st.expander(f"{group.replace('_', ' ').title()} ({len(rows)})",
                             expanded=group in ("white_balance", "tone")):
                st.dataframe(rows, hide_index=True, width='stretch')


def _render_comparison(before: Dict[str, Any], after: Dict[str, Any],
                       asked: Dict[str, Any]) -> None:
    """Before / asked-for / after, for the controls the reply actually changed."""
    before_flat = controls_module.flatten(before)
    after_flat = controls_module.flatten(after)
    asked_flat = controls_module.flatten(asked)

    rows = [
        {
            "control": path,
            "before": before_flat.get(path),
            "asked": asked_flat[path],
            "after": after_flat.get(path),
        }
        for path in asked_flat
        if path in before_flat and asked_flat[path] != before_flat[path]
    ]
    if not rows:
        st.caption("The reply did not change any control.")
        return

    st.dataframe(rows, hide_index=True, width='stretch')
    st.caption(
        "A single control lands close to what was asked. A large combined grade "
        "will not: every operation is calibrated against the original "
        "measurement, so several big moves at once interact. Grade in smaller "
        "steps if a value matters exactly."
    )


def _to_png(rgb01: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()
