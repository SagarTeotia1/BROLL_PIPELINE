# Face Recognition + Emotion Timeline

Production pipeline for an AI video editor. Register your cast (name + photos), point it
at a video, and it returns an **event-based** emotion timeline for *those people only* —
one record per emotion span, never per frame.

**Cast-only by design.** Faces that match no registered actor are tracked (the tracker
needs them to keep boxes apart) but never classified, never shown and never written to
the JSON. Extras and background crowd cost nothing.

```json
{
  "video": "input video.mp4",
  "video_duration": 145.12,
  "fps": 25.0,
  "processed_fps": 8.689,
  "analysis_time_seconds": 66.215,
  "analysis_time": "1 m 06 s",
  "expression_changes": 30,
  "actors": [
    {
      "id": 3, "name": "samay",
      "first_seen": 3.08, "last_seen": 126.4, "screen_time": 77.68,
      "events": 24, "expression_changes": 20, "mean_similarity": 0.6084,
      "emotion_totals": { "Fear": 34.76, "Surprise": 22.52, "Happy": 20.4 }
    }
  ],
  "events": [
    { "actor": "samay",  "emotion": "Happy",    "start": 3.08,  "end": 5.0,   "duration": 1.92,  "confidence": 0.496 },
    { "actor": "samay",  "emotion": "Fear",     "start": 5.0,   "end": 14.08, "duration": 9.08,  "confidence": 0.603 },
    { "actor": "balraj", "emotion": "Angry",    "start": 15.08, "end": 36.0,  "duration": 20.92, "confidence": 0.686 },
    { "actor": "balraj", "emotion": "Surprise", "start": 38.24, "end": 47.28, "duration": 9.04,  "confidence": 0.685 }
  ]
}
```

That is the whole default file: who, what expression, from when to when. Pass `--full`
to append the per-stage profiler dump and per-event debug fields (track ids, frame
numbers, sample counts) — diagnostics, not results.

`expression_changes` counts real transitions to a *different* emotion — an actor's first
event is an appearance, and leaving frame then returning still Neutral is not a change.
The top-level total and the per-actor counts come from the same event list, so they can
never disagree.

**Empty `events`?** The cast in that clip probably isn't the cast you registered:

```bash
python cli.py probe clip.mp4
```

It reports the best similarity each registered actor reaches, and tells you whether to
lower `recognition.similarity_threshold` or accept that they are simply not there.

**Stack** — PyTorch · ONNX Runtime (CUDA/TensorRT) · SCRFD · ArcFace · HSEmotion ·
ByteTrack · PyAV/NVDEC · OpenCV · PySide6.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

---

## 1. Install

```bash
# 1. PyTorch first - it also supplies the CUDA/cuDNN DLLs ONNX Runtime loads on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. everything else
pip install -r requirements.txt
```

> **CUDA major version must match.** `onnxruntime-gpu` 1.19–1.22 are CUDA 12 builds;
> 1.23+ are CUDA 13. Pair them with a torch build of the same major version. A mismatch
> makes ONNX Runtime fall back to the CPU provider — the engine detects this and logs an
> explicit error rather than pretending to be fast.

Model weights (~180 MB) download automatically on first use into `models/weights/`:

| Model | File | Role |
|---|---|---|
| SCRFD-10G-BNKPS | `det_10g.onnx` | face detection + 5 landmarks |
| ArcFace R50 (WebFace600K) | `w600k_r50.onnx` | 512-d identity embedding |
| HSEmotion EfficientNet-B0 | `enet_b0_8_best_vgaf.onnx` | 8-class expression |

```bash
python cli.py models        # show the zoo and what is cached
```

---

## 2. Use

### GUI

```bash
python app.py
python app.py --video clip.mp4
```

The window is analysis-first: a small player column on the left, and the **expression
change table** filling the rest — every row is one emotion span with start, end,
duration, confidence and match similarity. Double-click a row to jump the player there.
The bottom bar shows **analysis time**, remaining ETA, realtime factor, live change
count, GPU load and VRAM; when the run finishes it freezes on
`analysed in 01:05 (2.22x realtime)`.

1. **Cast** — enter a name, add photos, hit register.
2. **Open video**, then **Analyze**.
3. Rows appear as the pipeline closes each event; the timeline lane fills in underneath.
4. **Export** writes JSON + CSV + two PNGs.

### Headless

```bash
python cli.py register --name "John" --images ./cast/john ./extra/john_02.jpg
python cli.py analyze clip.mp4
python cli.py analyze clip.mp4 --set sampling.frame_stride=2 --set recognition.similarity_threshold=0.45
python cli.py cast --list
```

### Benchmarks

```bash
python benchmarks/benchmark.py micro                     # per-model latency vs batch size
python benchmarks/benchmark.py pipeline clip.mp4         # end-to-end + stage profile
python benchmarks/benchmark.py sweep clip.mp4 --strides 2 4 8
```

### Tests

```bash
python -m unittest discover -s tests -v      # 34 CPU-only tests, no weights needed
```

---

## 3. How it stays fast

| Lever | Effect |
|---|---|
| **Cast-only analysis** | HSEmotion runs only on recognised actors — 911 → **38 calls** on the test clip |
| 1-in-4 frame sampling | 30 fps → 7.5 analysed fps |
| Decoder-side downscale | swscale emits analysis resolution during YUV→BGR, not after |
| NVDEC via PyAV | GPU video decode, software fallback is automatic |
| ByteTrack identity caching | ArcFace runs on track birth / ambiguity / every `reid_interval`, not per frame |
| Cross-frame batching | ArcFace + HSEmotion batch faces from several frames at once |
| Quality gating | blurry, tiny and extreme-profile faces never reach a network |
| Scene-cut boost | stride drops to 1 for a moment after a cut, then decays back |
| FP16 (validated) | half precision per model, only where it provably matches FP32 |
| Pinned memory + IOBinding | uint8 staging, `non_blocking` upload, ORT bound to CUDA tensor pointers |
| Per-stage CUDA streams | detection / recognition / emotion overlap |

Measured on the target machine (RTX 3050 Laptop 6 GB, i5, Windows 11), 1280×720 H.264
@ 30 fps — full numbers in [benchmarks/RESULTS.md](benchmarks/RESULTS.md):

| | |
|---|---|
| End-to-end | **2.1–2.2× realtime**, 8.7 analysed fps, 1.3 GB VRAM |
| ArcFace batch 1 → 16 | 146 → **577 faces/s** (4.0× from batching) |
| HSEmotion batch 1 → 16 | 241 → **566 faces/s** (2.4×) |
| SCRFD @640² | 83.8 frames/s (335 source fps at stride 4) |
| Identity caching | **57 ArcFace calls for 911 faces** — ~94 % of the work removed |
| GPU utilisation | ~19 % — the run is *decode-bound*, not compute-bound |

Because the bottleneck is video decoding rather than the networks, a smaller
`frame_stride`, a bigger emotion model or a much larger cast are all close to free.

### Two model-level gotchas this codebase handles explicitly

Both were found by measurement, not assumption, and both are re-validated per machine and
cached (`models/weights/batch/`, `models/weights/fp16/`):

* **SCRFD is batch-1 only.** Rewriting its ONNX batch dimension to a symbolic one
  "works" and returns *silently wrong* boxes — feeding the same image twice yields two
  different results. `models/batch_patch.py` validates a patched graph before trusting
  it, and pins the detector to batch 1 when it fails. The per-face models (ArcFace,
  HSEmotion) are genuinely dynamic and do batch.
* **HSEmotion breaks in FP16 on the CUDA provider** (NaN logits), while being fine in
  FP32 and fine in FP16 on CPU. `models/precision.py` compares FP16 against FP32 on the
  target provider and keeps FP32 for that model automatically.

---

## 4. Configuration

Everything lives in [configs/default.yaml](configs/default.yaml), typed by
[configs/config.py](configs/config.py). The knobs you will actually touch:

```yaml
sampling:
  frame_stride: 4               # 2 = denser analysis, 8 = faster
recognition:
  cast_only: true               # false = also analyse people who are not registered
  similarity_threshold: 0.38    # raise for a stricter cast match
  reid_interval: 45             # sampled frames between identity refreshes
emotion:
  smoothing: hybrid             # majority | ema | hybrid
  window: 7                     # larger = steadier, slower to react
timeline:
  min_event_duration: 0.3       # shorter spans are absorbed
  include_unknown: false        # also time-line unrecognised faces
gpu:
  use_tensorrt: false           # first run builds engines (slow), then 20-40% faster
```

Override without editing the file:

```bash
python cli.py analyze clip.mp4 --set emotion.window=11 --set gpu.use_tensorrt=true
```

---

## 5. Output

`outputs/<video>/` receives:

| File | Contents |
|---|---|
| `<video>_timeline.json` | video info, actors, events, full run statistics |
| `<video>_timeline.csv` | one row per event |
| `<video>_timeline.png` | Gantt-style lanes, one per actor, coloured by emotion |
| `<video>_emotions.png` | screen time per emotion per actor |

Emotions: Angry · Contempt · Disgust · Fear · Happy · Neutral · Sad · Surprise.

---

## 6. Layout

```
app.py          GUI entry            cli.py          headless entry
configs/        typed config + YAML  utils/          logging, GPU, profiling, image ops
models/         ONNX engine, zoo,    detector/       SCRFD + quality gate
                batch & precision
                probes
recognition/    ArcFace, SQLite cast tracking/       Kalman + ByteTrack + identity state
                DB, gallery matcher
emotion/        HSEmotion + smoothing timeline/      events, engine, exporters
pipeline/       decode, sample, scene gui/           PySide6 widgets
                detect, batching,
                orchestration
benchmarks/     throughput harness   tests/          unit tests
```

---

## 7. Troubleshooting

**"onnxruntime could not activate a GPU provider"** — CUDA major-version mismatch; see
the install note above.

**Everything is "Unknown"** — no cast registered, or `similarity_threshold` is too high.
`python cli.py cast --list` shows what is enrolled; the registration report explains
which reference photos were rejected and why.

**Model download fails** — mirrors may be blocked. Drop `det_10g.onnx`,
`w600k_r50.onnx` and `enet_b0_8_best_vgaf.onnx` into `models/weights/` by hand; the
loader uses local files when present.

**Playback stutters** — it should not: playback owns a separate decoder thread. If it
does, lower `gui.preview_width` or `gui.playback_fps_cap`.
#   f a c e _ a n a l y s i s - m o d e l  
 