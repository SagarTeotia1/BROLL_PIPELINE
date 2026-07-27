# editorial — editor-friendly colour analysis

Describes the **editable colour state** of a frame: the controls a colourist
touches in Resolve, Lumetri or Lightroom, and nothing else.

No histograms, entropy, variance, kurtosis, skewness, statistical moments,
feature vectors, or Lab/XYZ/YCrCb statistics — and that is enforced, not just
documented. `schema.validate()` walks every key at every depth and rejects any
that names a statistical quantity, plus any array longer than 16 entries. It
runs on every document the analyzer produces.

OpenCV and NumPy only. CUDA offload through `cv2.cuda` where the installed
OpenCV provides it.

## The grading loop

```
frame ──► analyse ──► controls JSON ──► your model ──► edited JSON ──► render ──► graded frame
```

```bash
# 1. build a payload for the model: 47 controls, read-only context, instructions
python -m color_analyzer.editorial.cli frame.jpg \
    --prompt "dark cinematic grading" -o payload.json

# 2. send it to the model, save the reply (a markdown fence around it is fine)

# 3. render the reply
python -m color_analyzer.editorial.cli frame.jpg --apply reply.json -o graded.png
```

The values are a **state, not adjustments**. `temperature: 5200` means the frame
currently sits at 5200 K; the model returns 4200 to make it warmer and the
renderer works out the difference. That is why the render needs both — it
re-measures the frame, or takes `--source state.json` if you already have it.

```python
from color_analyzer.editorial import EditorialAnalyzer, apply_controls, build_payload
from color_analyzer.editorial import controls

analyzer = EditorialAnalyzer()
state = analyzer.analyze_rgb(rgb)

payload = build_payload(state, "dark cinematic grading")   # -> your model
reply   = ...                                              # <- its JSON

result = apply_controls(rgb, reply, source=state)
result.image      # graded RGB float32
result.applied    # ['white_balance', 'levels', 'exposure', ...]
result.ignored    # what the model sent that could not be used
result.clamped    # what had to be pulled into range
```

`apply_controls` never raises on a bad payload. Numbers as strings, invented
keys, read-only fields "updated", values an order of magnitude out of range,
markdown fences — all reported, none fatal.

Skin is held back by default so a heavy grade does not turn faces grey; pass
`--no-protect-skin` (or `protect_skin=False`) to apply at full strength.

## Analysis only

```bash
python -m color_analyzer.editorial.cli frame.jpg
python -m color_analyzer.editorial.cli frames/ -o out/
python -m color_analyzer.editorial.cli clip.mp4 --every 24 -o out/
python -m color_analyzer.editorial.cli frame.jpg --max-side 0   # native resolution
```

```python
from color_analyzer.editorial import EditorialAnalyzer

analyzer = EditorialAnalyzer()
state = analyzer.analyze_path("frame.jpg")

state["white_balance"]["temperature"]   # 5200
state["wheels"]["lift"]["red"]          # 63
state["hsl"]["orange"]["saturation"]    # 14
state["look"]["mood"]                   # "warm"
```

Video, reusing one analyzer:

```python
for state in analyzer.analyze_frames(decoded_bgr_frames):
    ...
```

## Output

```json
{
  "meta":  { "width": 1024, "height": 576, "elapsed_ms": 154, "device": "cpu" },
  "look":  { "overall_look": "neutral-warm, punchy contrast, natural",
             "mood": "warm", "brightness": "dark", "contrast": "punchy",
             "colorfulness": "natural", "temperature": "neutral-warm",
             "confidence": 0.59 },
  "white_balance": { "temperature": 5200, "tint": 4,
                     "neutral_coverage": 0.59, "confidence": 0.71 },
  "tone":  { "exposure": -0.48, "brightness": -26, "contrast": 39, "gamma": 1.43,
             "black_point": 0.002, "white_point": 0.822,
             "clipped_shadows": 0.006, "clipped_highlights": 0.0 },
  "color": { "saturation": -18, "vibrance": -55, "colorfulness": 41,
             "mean_saturation": 0.287 },
  "wheels": { "lift":  { "red": 63, "green": -21, "blue": -42, "luma": -13, "coverage": 0.511 },
              "gamma": { "red": 35, "green":  -7, "blue": -29, "luma":  -3, "coverage": 0.307 },
              "gain":  { "red": 13, "green":  -2, "blue": -11, "luma": -82, "coverage": 0.182 } },
  "split_toning": { "shadows":    { "hue": 354, "saturation": 89 },
                    "highlights": { "hue":  24, "saturation": 28 },
                    "separation": 30, "balance": -34, "strength": 7 },
  "palette": [ { "hex": "#1d1719", "rgb": [29,23,25], "coverage": 0.312,
                 "saturation": 19, "role": "dominant" } ],
  "skin_tone": { "detected": true, "coverage": 0.307, "hue": 20,
                 "saturation": 37, "luminance": 43,
                 "tone": "medium-deep warm", "warmth": 24 },
  "hsl": { "red": { "presence": 0.13, "hue": 33, "saturation": 18, "luminance": -15 },
           "orange": {}, "yellow": {}, "green": {}, "cyan": {}, "blue": {}, "purple": {} }
}
```

Scales are the ones an editor already knows: -100..100 for sliders, Kelvin for
temperature, degrees for hue, stops for exposure. A model can move a value
without first learning what the number means.

## Modules

| Module | Stage |
|---|---|
| `gpu.py` | CUDA/CPU dispatch for resize, colour conversion, blur |
| `frame.py` | Loading, downscale, shared RGB/HSV/luma planes, tonal zones, gates |
| `scales.py` | Slider mapping, quantisation, labelling |
| `tone.py` | exposure, brightness, contrast, gamma, black/white point, clipping |
| `white_balance.py` | temperature, tint |
| `color.py` | saturation, vibrance, colourfulness |
| `wheels.py` | lift / gamma / gain |
| `split_toning.py` | shadow and highlight tints, separation, balance |
| `palette.py` | dominant swatches |
| `skin.py` | skin coverage, hue, saturation, luminance, tone |
| `hsl.py` | seven hue bands |
| `look.py` | overall look and mood |
| `schema.py` | output contract and the forbidden-content guard |
| `analyzer.py` | orchestration |
| `controls.py` | the 47 editable controls; extracting and sanitising them |
| `apply.py` | rendering a target state onto a frame |
| `prompt.py` | building the payload a grading model is asked to edit |

Each stage is a plain function taking a `Frame` and returning a dict, so they
can be run, tested and replaced individually.

## Stability

The engine promises two different things:

* **Determinism** — the same frame always gives byte-identical output. Nothing
  is seeded randomly; the palette's clustering is seeded from a fixed coarse RGB
  grid rather than k-means++.
* **Stability** — *similar* frames give near-identical output. This is the one
  that matters: if consecutive frames of a shot wobble, an automated grade
  driven by them flickers, and determinism does not help at all.

Measured drift on real 1080p footage, worst field, four noise seeds:

| Sensor noise | Worst drift |
|---|---|
| ±1/255 | ~2 slider units |
| ±2.5/255 | ~9 |
| ±5/255 | ~16 |

Resolution changes and small reframes move nothing. The tests assert this by
perturbing frames the way a camera does.

### The one design rule that produced most of it

**Every hard threshold on a continuous quantity is a stability bug.** It came up
five times during development, in five disguises, each worth tens of slider
units of drift between otherwise identical frames:

| Where | Symptom | Fix |
|---|---|---|
| HSL band edges | a hue at a boundary flipped bands | raised-cosine membership |
| Tonal zone cuts at 0.33 / 0.66 | content on the boundary flipped zones | smoothstep zones, partition of unity |
| Vibrance's median split | the split moved with the frame's biggest cluster | percentile of the chromatic distribution |
| Chroma/value gates | pixels flickered in and out | smoothstep gates |
| A population-size `if` | flipped between two whole formulas | blend, don't branch |

If you add a stage, do not write `if x > threshold` over pixel data.

## What the renderer can and cannot deliver

Every control was driven by a known amount on real footage and the resulting
reading measured. Per-control, one at a time, the delivered fraction sits in
**0.85–1.14** — the response factors in `apply.RESPONSE` are set from those
measurements, not guessed.

Two things that follow, and that a model driving these controls needs to know:

**A large combined grade does not compose.** Each operation is calibrated
against the *original* measurement, so asking for less exposure, more contrast
and a new black point at once fires all three at full strength against a frame
the other two are also changing. Measured: a requested white point of 0.88
landed at 0.53. The individual controls are accurate; the sum of several large
ones is not.

Iterating — measure the result, apply what is left, repeat — was implemented and
then removed. The readings are coupled: moving exposure changes the contrast
reading, moving contrast changes the exposure reading. Per-control feedback on a
coupled system oscillates rather than converging (a requested contrast of 94
landed at −16, then 33, then −58 across three passes; damping to 45% only slowed
the swing). Converging would need a solver that models the coupling, which this
does not attempt.

**Darkening lowers measured contrast.** Contrast is reported as the absolute
p95−p5 luminance spread, so halving the signal roughly halves it. Dropping
exposure 0.8 stops took one frame's contrast reading from −20 to −68 on its own,
while a +30 contrast request delivered +29 on its own. "Darker *and* punchier"
needs a much larger contrast request than the arithmetic suggests. This is
stated in the generated prompt so the model compensates.

## Known limitations

**Palette on smooth gradients.** A continuous gradient has no clusters, so any
partition of it is arbitrary and its swatch boundaries drift under noise —
measured at ~38 levels on a synthetic ramp against 2-3 on real footage. Not
fixable by better seeding; it is a property of the problem.

**Gamma is compressed toward 1.0.** Normalising by the tonal range is what makes
the reading independent of exposure, and it also pulls the estimate inward: a
ramp raised to 2.2 measures ~2.1, one raised to 0.45 measures ~0.7. The reading
is monotonic, so it ranks and steers correctly, but it is not an exact exponent.
Not normalising would buy accuracy at the cost of confounding gamma with
exposure, which is the worse failure for a control a model will adjust.

**Saturation is noise-sensitive, inherently.** HSV saturation is
`(max - min) / max` over three channels, so independent per-channel noise pushes
`max` up and `min` down and biases it upward — more so on dark frames, where the
divisor is small. A pre-blur would hide it at the cost of an undocumented bias
in the readings themselves. The sensitivity is measured and documented instead.

**White balance measures the light, not the content.** The estimate uses only
near-neutral pixels, so a brown-toned scene under neutral light reads ~6500 K
rather than the ~3800 K grey-world would report. That is the intended behaviour
and it is usually right, but a scene with a genuine strong cast and no neutral
surfaces will have its cast understated. `confidence` reports both how much of
the frame was neutral and how neutral it actually was; check it before acting.

**The CUDA path is untested.** The stock `opencv-python` wheel exposes the
`cv2.cuda` namespace with no kernels and zero devices, which is what this was
developed against. The dispatch, capability probing and fallbacks are written
and exercised on the CPU path; the GPU branches themselves have never run.
`Backend.describe()` reports which operations are actually being offloaded.

## Relationship to the rest of the package

`color_analyzer.editorial` is self-contained and shares no code with the older
`color_analyzer.analyzer` pipeline, which still exists and still produces the
45-parameter grading document with its `current`/`recommended` pairs. The two
answer different questions — "what colour state is this frame in" versus "what
grade should be applied to it" — and the older one is where the recommendation
heuristics live.

## Tests

```bash
python -m pytest color_analyzer/editorial/tests/ -q      # 117 tests
```

* `test_contract.py` — sections, required controls, and the forbidden-content
  guard (including that short bans like `lab` match on token boundaries, so a
  key called `label` is not rejected).
* `test_stages.py` — does each reading point the right way: a warm frame reads
  below neutral Kelvin, a coloured subject does not shift white balance, a
  traffic cone is not mistaken for skin, a uniform cast is not a split tone.
* `test_stability.py` — determinism, and drift under sensor noise, reframing and
  resolution change.
* `test_apply.py` — the render loop: identity (an unedited payload must return
  the frame untouched, which is what guards the delta arithmetic), per-control
  round trips, monotonicity, robustness against malformed model output, and that
  skin protection measurably moderates the grade on skin.
