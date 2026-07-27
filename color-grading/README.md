# color_analyzer — Colour-Grading Analysis Engine

Points at a frame and returns **45 colour-grading parameters**, each with the
value the frame currently has and the value it should be graded to. Built for an
intermediate colourist's panel and for running over video frames at speed.

> This is **analysis and recommendation, not enhancement**. Nothing about the
> image is modified unless you explicitly render the grade.

## Key properties

- **A 45-parameter contract.** One schema (`analyzer/schema.py`) is the single
  source of truth for what comes out; the decision engine and the report layer
  both iterate it, so the output cannot drift between them.
- **Fast by default.** A 1080p frame is graded in ~300 ms on CPU; a full-length
  analysis with histograms and the 250-feature vector is opt-in via `--deep`.
- **Single algorithm, GPU *or* CPU.** Every computation is written against one
  array namespace (`xp`) that resolves to **CuPy** when a CUDA GPU is present and
  **NumPy** otherwise. There is no duplicated CPU/GPU code.
- **Minimal recomputation.** Each colour space, the luminance histogram, the
  local-contrast maps and the hue trigonometry are computed **once** in a shared
  `ImageContext` and reused by every analyzer.
- **Fully typed & documented.** Every analyzer returns a frozen dataclass; every
  metric is implemented mathematically with the equation in the comments.

## Install

```bash
pip install -r requirements.txt
# optional GPU: pip install cupy-cuda12x   (matching your CUDA toolkit)
```

## CLI

```bash
# grade one frame -> outputs/grade.json (+ summary.txt)
python main.py frame.jpg -o outputs

# also write a flat {parameter: recommended} mapping, and render the grade
python main.py frame.jpg -o outputs --flat --render

# full analysis: histograms, spatial maps, feature vector, plots, report.html
python main.py frame.jpg -o outputs --deep --visuals

# analyse at native resolution instead of the 1024px default
python main.py frame.jpg -o outputs --max-side 0

# a folder, forced onto the CPU
python main.py frames/ -o outputs --cpu
```

## The output

`grade.json` is one document. Every parameter carries what was **measured** and,
when it is adjustable, the **target** and the **delta** between them — and the
delta is exactly what a renderer applies.

```json
{
  "meta":  { "width": 1024, "height": 576, "elapsed_ms": 122.4, "backend": "numpy", "deep": false },
  "style": { "detected": "moody", "target": "cinematic", "confidence": 0.78 },
  "grade": {
    "white_balance.temperature": { "current": 7250, "recommended": 6200, "delta": -1050 },
    "primary.contrast":          { "current": 18.0, "recommended": 26.0, "delta": 8.0 },
    "creative_style.cinematic":  { "current": 0.74 },
    "quality.noise_risk":        { "current": "low" }
  },
  "notes":   ["Increase warmth", "Recover highlights", "Brighten skin"],
  "palette": ["#2b3a4a", "#c98b5e", "#8e7a63", "#43332f", "#b37845"]
}
```

### The 45 parameters

| Group | n | Parameters |
|---|--:|---|
| `white_balance` | 2 | temperature (K), tint |
| `primary` | 7 | exposure, contrast, highlights, shadows, whites, blacks, gamma |
| `presence` | 5 | vibrance, saturation, clarity, texture, dehaze |
| `tone_curve` | 5 | curve_type, shadow_lift, midtone, highlight_rolloff, contrast_strength |
| `color_wheels` | 6 | lift/gamma/gain × (temp, strength) |
| `split_toning` | 4 | shadow_hue, shadow_sat, highlight_hue, highlight_sat |
| `hsl` | 4 | orange_sat, orange_lum, blue_sat, blue_lum |
| `subject` | 4 | skin_present\*, skin_warmth, skin_luminance, subject_pop |
| `creative_style` | 4 | cinematic\*, commercial\*, natural\*, moody\* |
| `quality` | 4 | highlight_clipping\*, shadow_crush\*, dynamic_range\*, noise_risk\* |

`*` = **read-only**: measured, with no `recommended` value. Asking for a
recommended `shadow_crush` is meaningless.

Only the two HSL bands that are actually derived from the image are exposed —
orange (where skin lives) and blue (background separation). Earlier versions
emitted six bands, four of which were hard-coded constants.

## Library API

```python
from color_analyzer import ColorGradingEngine

engine = ColorGradingEngine()            # fast path, auto GPU/CPU
doc = engine.grade("frame.jpg")          # the 45-parameter document

doc["grade"]["white_balance.temperature"]["delta"]   # -1050
doc["style"]["target"]                               # "cinematic"
doc["notes"]                                         # editor-facing sentences
```

Render it:

```python
from color_analyzer import to_executor_decision, GradingPlanExecutor

plan   = to_executor_decision(doc)
graded = GradingPlanExecutor().apply(rgb01, plan).image
```

Flat view for a renderer that wants plain sliders:

```python
from color_analyzer.analyzer.schema import flatten
flatten(doc, "recommended")   # {"white_balance.temperature": 6200, ...}
flatten(doc, "current")       # the measurement, same keys
```

### Video

`analyze_frames` reuses one engine (backend detection and analyzer construction
happen once) and caps each frame to `max_side` before analysis:

```python
import cv2
from color_analyzer import ColorGradingEngine

engine = ColorGradingEngine()
cap = cv2.VideoCapture("clip.mp4")

def frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame

for result in engine.analyze_frames(frames(), max_side=1024):
    doc = engine.decide(result)
```

## Fast vs deep

The engine has two registries. **Core** analyzers are the ones the 45 parameters
are built from; **deep** analyzers feed the histograms, spatial maps and the
250-feature vector used by the reports and the Streamlit UI.

| | Core (default) | Deep (`--deep`) |
|---|---|---|
| Analyzers | 11 | 18 |
| Feature vector | ~115 | ~250 |
| Produces | `grade.json`, `summary.txt` | + `report.html`, plots, `feature_vector.npy` |

Deep sections are `None` when they did not run — check before dereferencing.
`--visuals` implies `--deep`, and asking for visualisations from a fast analysis
raises rather than failing deep inside a plotting call.

## Performance

Measured on an Intel i5, CPU/NumPy backend (no CUDA on this machine), median of
10+ runs — reproduce with `benchmarks/bench.py`:

| Case | Before | After |
|---|--:|--:|
| 1080p, native resolution | 4177 ms | **1018 ms** |
| 1080p, default 1024px cap | — | **307 ms** |
| 4K, default 1024px cap | 26 456 ms | **366 ms** |
| 1080p @1024, `--deep` | — | **838 ms** |
| 100 decoded 1080p video frames | ~417 s | **24.4 s** (244 ms/frame) |

Where it came from, largest first:

1. **Skipping the deep analyzers.** They were ~69% of analysis time and fed
   nothing into the grade. `histogram` alone was 38% of the total: for ten
   channels it sorted the frame and built `(x-µ)³` and `(x-µ)⁴` temporaries.
2. **The 1024px default.** Colour statistics converge well below delivery
   resolution; a 4K frame costs 8× the work for no measurable change in the
   grade. `--max-side 0` restores native analysis.
3. **A float64 leak.** `rgb_to_xyz` multiplied a float32 image by a float64
   matrix, so Lab — the array most analyzers read — came back float64 and every
   downstream pass moved twice the memory it needed to.
4. **Shared derived quantities.** One luminance histogram now serves every
   percentile (they were separate full sorts), one local-std map per window
   serves contrast and the HDR score, and hue trig is computed once.
5. **Reduction rewrites.** Split toning reduces its three tonal zones with dot
   products against 0/1 masks instead of fancy-indexed copies; K-means uses a
   segment sum rather than a Python loop over `k` masked passes per iteration.

### Was the maths preserved?

Across 111 analyzer features on three real frames, comparing against the
pre-refactor commit: **81 bit-identical, and the largest change anywhere is
0.31%** — float32 precision and 4096-bin histogram quantisation. Reproduce the
comparison with a git worktree of the previous commit.

One change was *not* cosmetic and was reverted: folding the HDR heuristic's
3-pixel micro-contrast window into the contrast analyzer's 15-pixel regional
window moved `hdr_score` by 53%, because those windows measure different things.
`ImageContext.local_std(window)` now caches per window.

## Analyzers

**Core** (`analyzer/`):

| Module | What it extracts |
|---|---|
| `white_balance` | CCT (Kelvin), tint, grey-world gains, directional casts, neutrality |
| `exposure` | tonal-zone %, clipping, gamma estimate, exposure-quality score |
| `contrast` | RMS, Michelson, global, local, dynamic range (stops), Laplacian variance |
| `tone_curve` | black/white point, gamma, S-curve, lifted/crushed, highlight rolloff |
| `hsv`, `lab` | circular hue statistics, saturation stats, warmth, chroma |
| `colorfulness` | Hasler-Süsstrunk, opponent variance, average chroma, richness |
| `split_toning` | shadow/mid/highlight tint hue, saturation, temperature + confidence |
| `skin` | skin %, hue, saturation, exposure, consistency, naturalness |
| `cinematic` | teal-orange, vintage, commercial, natural, HDR, low/high-key, moody, film, per-band HSL deviations |
| `dominant_colors` | GPU/CPU K-means palette, coverage %, diversity, palette distance |

**Deep only**: `rgb`, `xyz`, `ycrcb` (per-space statistics), `histogram`
(entropy, skew, kurtosis, peaks, CDF), `local_regions` (4×4 spatial grid),
`perceptual` (CIE76 ΔE, colour complexity), `harmony` (complementary /
analogous / triadic / split-comp / tetradic / mono).

## Architecture

```
ImageContext  (colour spaces, luminance CDF, local-std maps, hue trig — each once)
      │
      ▼
ColorGradingEngine ── core analyzers ──┐
                   └─ deep analyzers ──┤ (opt-in)
                                       ▼
                            {section: FeatureResult}
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
            DecisionEngine     GlobalFeatureVector    GradingSummary
             measure() +                                    │
             recommend()                                    │
                    │                                       │
                    ▼                                       ▼
            schema.assemble()                    Visualizer / ReportGenerator
                    │                                  (deep only)
                    ▼
             grade.json (45 params)
                    │
                    ▼
          to_executor_decision() ──► GradingPlanExecutor ──► graded image
```

Adding a parameter means adding it to `schema.PARAMS` and populating it in
`decision_engine.measure()` / `recommend()`; `schema.assemble` refuses to emit
anything not declared, and `schema.validate` is what the tests assert against.

## Streamlit app

```bash
streamlit run app.py
```

- **🔍 Analyze** — full deep analysis: the 45-parameter table grouped by section,
  dominant palette, style bars, histograms, heatmaps, tonal masks, interactive
  3-D colour clouds.
- **🎨 Grade** — apply a grade like a video-editor colour panel, with live
  before/after. *Auto-grade this image* runs the fast path and loads the result.

## Tests

```bash
python -m pytest color_analyzer/tests/ -q      # 80 tests
```

- `test_schema.py` — the contract: exactly 45 parameters, read-only parameters
  carry no recommendation, every value inside its declared range, `delta` equals
  `recommended - current`, `assemble` rejects undeclared keys.
- `test_decision_engine.py` — behaviour: a cool frame is warmed, a warm frame is
  cooled, clipped highlights are recovered, a dark frame gets positive exposure,
  and the grade renders through the executor.
- `test_engine.py` — fast/deep parity (the core sections must be *identical*
  between modes, which is what guards the shared caches), resolution capping,
  `analyze_frames`, reports and visuals.
- Plus colour-space correctness and per-analyzer behaviour.

## Known limitation: no ground truth

The recommendation rules are **hand-tuned heuristics**, not learned from data.
`STYLE_TARGETS` in `decision_engine.py` says a cinematic frame wants 6200 K and
0.80 teal-orange separation; those numbers are a defensible opinion, not a
measurement, and no professional colourist has validated them.

The tests check that the engine is *self-consistent* — a cool frame gets warmed —
not that the resulting grade is *good*. Establishing that needs a corpus of
professionally graded before/after pairs to measure the recommendations against.
Treat the `current` half of the output (which is measured) with more confidence
than the `recommended` half (which is inferred).
