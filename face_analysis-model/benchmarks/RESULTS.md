# Benchmark results

Measured on the target hardware, not extrapolated.

**Machine** — NVIDIA GeForce RTX 3050 6 GB Laptop GPU (sm_86) · Intel i5 · 24 GB RAM ·
Windows 11 · CUDA 12.1 · onnxruntime-gpu 1.22 · torch 2.5.1+cu121 · Python 3.11.

Reproduce with:

```bash
python benchmarks/benchmark.py micro --batches 1 4 8 16 32
python benchmarks/benchmark.py pipeline <clip.mp4>
```

---

## 1. Per-model throughput vs batch size

`python benchmarks/benchmark.py micro --batches 1 4 8 16 32 --iterations 15`

| Model | Precision | Batch | Mean latency | p95 | Throughput |
|---|---|---:|---:|---:|---:|
| SCRFD-10G @640² | FP16 | 1 | 11.94 ms | 12.45 ms | **83.8 frames/s** |
| ArcFace R50 @112² | FP16 | 1 | 6.87 ms | 8.99 ms | 145.6 faces/s |
| ArcFace R50 | FP16 | 4 | 10.60 ms | 11.35 ms | 377.2 faces/s |
| ArcFace R50 | FP16 | 8 | 15.57 ms | 16.19 ms | 513.8 faces/s |
| ArcFace R50 | FP16 | **16** | 27.73 ms | 28.46 ms | **576.9 faces/s** |
| ArcFace R50 | FP16 | 32 | 56.46 ms | 57.98 ms | 566.8 faces/s |
| HSEmotion B0 @224² | FP32 | 1 | 4.15 ms | 5.75 ms | 240.7 faces/s |
| HSEmotion B0 | FP32 | 4 | 9.05 ms | 9.67 ms | 442.0 faces/s |
| HSEmotion B0 | FP32 | 8 | 15.31 ms | 15.80 ms | 522.6 faces/s |
| HSEmotion B0 | FP32 | **16** | 28.25 ms | 29.58 ms | **566.3 faces/s** |
| HSEmotion B0 | FP32 | 32 | 55.37 ms | 57.63 ms | 577.9 faces/s |

**Why batching is mandatory.** Going from batch 1 to batch 16 is a **4.0×** throughput
gain for ArcFace and **2.4×** for HSEmotion — at batch 1 the GPU spends most of its time
on launch overhead rather than convolutions. Batch 32 buys nothing on a 6 GB card, which
is why the default is 16 (`recognition.batch_size`, `emotion.batch_size`).

SCRFD stays at batch 1 because its published ONNX export is not batch-safe — see
`models/batch_patch.py`. At 83.8 frames/s it still supports 335 source fps at the default
1-in-4 stride, so it is nowhere near the bottleneck.

HSEmotion runs FP32 because `models/precision.py` measured NaN outputs from the FP16
conversion on the CUDA provider. The validator makes that decision automatically and
caches it.

---

## 2. End-to-end pipeline

`python benchmarks/benchmark.py pipeline "<clip>"` — H.264, NVDEC via PyAV.

| Clip | Resolution | Duration | Source fps | Frames | Analysed | Wall time | Realtime | Analysed fps | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `input video.mp4` | 1920×1080 | 145.1 s | 25 | 3628 | 1261 (34.8 %) | 66.2 s | **2.19×** | 8.69 | 1.3 GB |
| `input_video2.mp4` | 1280×720 | 83.6 s | 30 | 2508 | 737 (29.4 %) | 37.8 s | **2.21×** | 8.82 | 1.3 GB |

**Why 34.8 % and not 25 %.** The base stride of 4 would analyse 907 of 3628 frames.
Scene-cut detection found 25 shot boundaries and dropped the stride to 1 for a dozen
frames after each, adding ~350 analysed frames. That is the feature working as designed —
new shots get characterised immediately — but it is worth knowing when reading the
throughput numbers. `sampling.scene_cut_enabled: false` gives a flat 25 %.

¹ with `timeline.include_unknown=true`; the cast registered for that run does not appear
in the clip.

### Stage profile (`input video.mp4`)

| Stage | Calls | fps | Avg ms | p95 ms |
|---|---:|---:|---:|---:|
| decode | 3628 | 59.7 | — | — |
| sample (+ scene cut) | 3628 | — | 2.30 | 5.02 |
| detect | 1261 | 18.8 | 20.61 | 46.84 |
| track | 1261 | 18.1 | 0.08 | 0.15 |
| recognition | **57** | 0.90 | 72.56 | 302.94 |
| emotion | 911 | 18.4 | 114.55 | 183.70 |

### Cast-only mode

With `recognition.cast_only: true` (the default), HSEmotion runs only on faces that
matched a registered actor. On the same clip:

| | Analyse everyone | Cast-only |
|---|---:|---:|
| HSEmotion calls | 911 | **38** |
| Emotion stage share of GPU time | 104 s | 4 s |
| Events in the JSON | 26 (mostly `Unknown`) | 4 (named actors) |

Extras and background crowd are still tracked — the tracker needs them to keep boxes
apart — but they never reach the classifier, the UI or the output.

### The two numbers that matter

**57 ArcFace calls for 911 faces across 1261 analysed frames.** Track-scoped identity
caching (embed on track birth / ambiguity / every `reid_interval`) removes ~94 % of the
recognition work. A naive "embed every face every frame" pipeline would have embedded all
911 faces — at 577 faces/s that is ~1.6 s of extra GPU time on this 2.4-minute clip, and
the gap widens with crowd density and longer takes.

**GPU utilisation peaks around 19 %.** The run is *decode-bound*, not compute-bound: at
~60 fps decode the pipeline is already 2× realtime, and the accelerators are idle most of
the time. Practical consequences:

* denser sampling is nearly free — `sampling.frame_stride=2` roughly doubles analysed fps
  without doubling wall time;
* the heavier `hsemotion_enet_b2_8` model fits comfortably in the budget;
* more actors in the gallery cost essentially nothing (the match is one small matmul).

Decode was already optimised: frames are downscaled to the analysis resolution *inside*
the swscale colour conversion (`VideoSource(target_long_side=...)`) instead of being
converted at full size and resized afterwards, and the scene-cut histogram runs on a
strided subsample. Further gains would need a decoder that keeps frames on the GPU
end-to-end (CUDA-mapped NVDEC surfaces), which is the natural next step.

---

## 3. Accuracy spot-check

Gallery of 2 actors (3 reference photos each), matched against faces sampled from three
clips, `recognition.similarity_threshold = 0.38`:

| Clip | Best similarity, correct actor | Best similarity, other actors |
|---|---:|---:|
| `input video.mp4` (contains BodyGuard) | **0.980** | 0.167 |
| `Timeline 1.mp4` | — (neither actor present) | 0.183 |
| `input_video2.mp4` | — (neither actor present) | 0.201 |

True matches sit an order of magnitude above the impostor scores, and the default
threshold of 0.38 falls in the empty band between them — no false positives were produced
on the two clips where neither actor appears.
