"""Colour stage: saturation, vibrance and colourfulness.

The three are distinct controls in every editor and they are measured as three
distinct quantities here:

* **saturation** — the overall level, the whole frame's mean HSV saturation.
* **vibrance** — how much saturation headroom the *muted* pixels still have.
  Vibrance sliders act mainly on unsaturated colours, so the reading that
  predicts what one will do is the state of those pixels, not the average.
* **colourfulness** — perceived chromatic richness, which depends on both the
  mean and the spread of chroma. A frame with one screaming red object and a
  grey background has low mean saturation but reads as colourful.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .frame import Frame
from .scales import centred, clamp, label, ratio, slider

#: Mean HSV saturation of a normally graded frame; the saturation slider's zero.
NEUTRAL_SATURATION = 0.35

#: That percentile on a normally graded frame; the vibrance slider's zero, and
#: the deflection that reads as full scale.
#:
#: Calibrated against the range the measurement actually takes: across muted
#: footage, mixed scenes, saturated graphics and a dark warm interior it spans
#: roughly 0.15 to 0.70. Centring on 0.20 with a 0.20 span — the values that
#: suited the earlier windowed-mean definition — railed two thirds of those
#: frames at +100 and reported nothing useful.
NEUTRAL_MUTED_SATURATION = 0.35
MUTED_SPAN = 0.35

#: Hasler-Susstrunk index of a vividly colourful image, used to normalise to 0-100.
VIVID_COLORFULNESS = 110.0


def analyze_color(frame: Frame) -> Dict[str, Any]:
    """Measure saturation, vibrance and colourfulness."""
    saturation = float(frame.sat.mean())

    return {
        "saturation": centred(saturation, neutral=NEUTRAL_SATURATION, span=0.35),
        "vibrance": centred(
            _muted_saturation(frame), neutral=NEUTRAL_MUTED_SATURATION, span=MUTED_SPAN
        ),
        "colorfulness": slider(_colorfulness(frame), 0.0, 100.0),
        "mean_saturation": ratio(saturation),
    }


def _muted_saturation(frame: Frame) -> float:
    """Saturation of the frame's quiet colours — the vibrance headroom.

    A saturation mean weighted by ``value_gate * (1 - saturation)``: the gate
    drops near-blacks, whose saturation is numerically unstable, and the
    ``(1 - saturation)`` term makes already-vivid colours count for little.

    Note the **value** gate, not the chroma gate. Gating on saturation would
    make the measured population depend on the very quantity the control moves:
    raising vibrance pushes pixels into it, lowering vibrance pushes them out,
    and either way the population shift outweighs the value shift. Measured
    moving the wrong way in *both* directions before the two gates were split.

    A frame whose quiet colours are already saturated has little vibrance
    headroom left; one with plenty of muted colour has a lot.

    **The weighting is the same function the vibrance control applies**, and
    that is the point of it. Two earlier definitions were rejected for coupling
    with the operation rather than tracking it:

    * A *windowed mean* over a fixed saturation band, or a median split, needed
      a rule for frames with no muted band at all, and every version of that
      rule was a threshold something could cross — tens of slider units of drift
      between otherwise identical frames.
    * A *percentile of the chroma-gated population* was stable but barely
      controllable, for the same reason: raising vibrance lifts near-neutral
      pixels over the gate and into the measured population, where — still the
      least saturated things in frame — they pull the percentile down.

    Weighting by exactly what the operation moves leaves the measurement
    monotonic in the control, which is what makes the analyse-edit-render loop
    converge.
    """
    sat = frame.sat
    if sat.size == 0:
        return 0.0

    weights = frame.value_gate * (1.0 - sat)
    total = float(weights.sum())
    if total <= 0.0:
        return float(sat.mean())  # no chromatic content at all
    return float(sat @ weights) / total


def _colorfulness(frame: Frame) -> float:
    """Hasler-Susstrunk colourfulness, rescaled to 0-100.

    Opponent channels ``rg = R - G`` and ``yb = 0.5(R + G) - B`` are formed in
    plain RGB — no perceptual colour space is needed — and the metric combines
    the spread and the magnitude of the resulting chroma cloud::

        M = sqrt(sigma_rg^2 + sigma_yb^2) + 0.3 * sqrt(mu_rg^2 + mu_yb^2)

    The published index runs roughly 0-110 over natural images, so it is divided
    by :data:`VIVID_COLORFULNESS` to land on a 0-100 slider.
    """
    rgb = frame.rgb_flat.astype(np.float32) * 255.0
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]

    std = float(np.sqrt(rg.var() + yb.var()))
    mean = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    index = std + 0.3 * mean
    return clamp(index / VIVID_COLORFULNESS * 100.0, 0.0, 100.0)


def describe_colorfulness(color: Dict[str, Any]) -> str:
    """Word for the frame's chromatic richness."""
    return label(
        color["colorfulness"],
        [(18, "monochrome"), (32, "muted"), (52, "natural"), (72, "rich")],
        "vivid",
    )
