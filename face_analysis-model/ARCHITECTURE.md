# Face Recognition + Emotion Timeline — Architecture

Production pipeline for an AI video editor. Ingests a video + a registered cast, emits an
**event-based** emotion timeline (one event per emotion *change*, never per frame).

Target hardware: RTX 3050 Laptop (6 GB VRAM), i5, 24 GB RAM, Windows 11, CUDA 12.x.

---

## 1. Design principles

| Principle | Consequence in this codebase |
|---|---|
| The GPU must never wait on Python | Every stage is a thread with bounded queues; decode / detect / recognize / emote overlap. |
| Never infer one face at a time | `BatchCollector` aggregates faces across *frames* until `batch_size` or `max_latency_ms`. |
| Don't recompute identity | ByteTrack gives persistent IDs; ArcFace runs only on track birth / low identity confidence / periodic re-check. |
| Zero-copy where it matters | ONNX Runtime `IOBinding` is bound directly to **CUDA torch tensors** (`data_ptr()`), so no H2D round-trip between stages. |
| Bounded memory | Ring of pre-allocated pinned host buffers + persistent device buffers; queues are size-capped and drop-oldest for preview traffic only. |
| Playback is sacred | The GUI renders from its own decoder at wall-clock pace. Inference results arrive by signal and are *overlaid*; a slow model can never stall the video. |

---

## 2. Process / thread topology

```
                    ┌──────────────────────── GUI process (PySide6) ────────────────────────┐
                    │                                                                       │
 ┌───────────┐      │  ┌──────────────┐   frames    ┌─────────────┐                          │
 │  video    │──────┼─▶│ PlaybackThr. │────────────▶│ VideoWidget │  (wall-clock paced)      │
 │  file     │      │  └──────────────┘             └─────────────┘                          │
 └───────────┘      │                                      ▲ overlay boxes                   │
        │           │  ┌────────────────────────────────────┴─────────────────┐              │
        │           │  │ PipelineBridge (Qt signals, queued connections)       │              │
        │           │  └────────────────────────────────────▲─────────────────┘              │
        │           └───────────────────────────────────────┼────────────────────────────────┘
        │                                                   │ FrameResult / TimelineEvent
        ▼                                                   │
┌──────────────────────────────── AnalysisPipeline (worker threads) ──────────────────────────┐
│                                                                                             │
│  DecodeThread          SampleThread         DetectThread        TrackThread                 │
│  PyAV/NVDEC ──q0──▶ adaptive sampler ──q1──▶ SCRFD (batched) ──q2──▶ ByteTrack + quality     │
│  (or cv2)           + scene-cut boost                                     │                 │
│                                                                           ▼                 │
│                                              ┌────────── BatchCollector (cross-frame) ─────┐ │
│                                              │  ArcFace (only "needs id" tracks)          │ │
│                                              │  HSEmotion (all live tracks)               │ │
│                                              └────────────────────┬───────────────────────┘ │
│                                                                   ▼                         │
│                                            TimelineThread: smoothing → event open/close     │
│                                                                   │                         │
└───────────────────────────────────────────────────────────────────┼─────────────────────────┘
                                                                    ▼
                                                 outputs/*.json  *.csv  *_timeline.png
```

All inter-stage links are `queue.Queue(maxsize=N)` (producer/consumer). Back-pressure is
explicit and measured: `utils/profiling.py` reports per-stage FPS and queue depth so a
bottleneck is visible instead of mysterious.

---

## 3. Stage contracts

| Stage | Module | In | Out |
|---|---|---|---|
| Decode | `pipeline/frame_source.py` | path | `RawFrame(idx, pts, bgr)` |
| Sample | `pipeline/sampler.py` | `RawFrame` | `RawFrame` (1-of-N, boosted after cuts) |
| Detect | `detector/scrfd.py` | frame batch | `Detection(bbox, score, kps5)` |
| Quality | `detector/face_quality.py` | `Detection` + frame | blur / size / yaw gates |
| Track | `tracking/byte_tracker.py` | detections | `FaceTrack(track_id, …)` persistent |
| Recognize | `recognition/arcface.py` + `matcher.py` | aligned 112² batch | 512-d embedding → `(actor, similarity)` |
| Emotion | `emotion/hsemotion.py` | aligned 224² batch | `(emotion, confidence)` |
| Smooth | `emotion/smoothing.py` | per-track stream | stable emotion |
| Timeline | `timeline/timeline_engine.py` | stable emotion | `TimelineEvent(actor, emotion, start, end)` |

---

## 4. GPU execution model

* **Providers**: `CUDAExecutionProvider` (cuDNN-9 heuristics, `arena_extend_strategy=kSameAsRequested`)
  with CPU fallback. `TensorrtExecutionProvider` is used automatically when present.
* **Precision**: models are converted once to FP16 (cached in `models/weights/fp16/`); inputs are
  fed as `torch.float16` CUDA tensors. FP32 path stays available via config.
* **Streams**: preprocessing runs on a dedicated `torch.cuda.Stream`; ORT is handed the same
  stream handle through `run_with_iobinding(run_options)` after a stream sync, so H2D copies of
  batch *n+1* overlap compute of batch *n*.
* **Pinned memory**: `utils/gpu.PinnedRing` hands out page-locked staging buffers, reused forever,
  copied with `non_blocking=True`.
* **channels_last**: applied to the torch-side preprocessing tensors (NHWC is also the natural
  layout of decoded frames, so this removes a transpose).
* **torch.compile**: applied to the fused normalize/resize kernel only (`utils/image_ops.py`);
  guarded by config + try/except because the win is ~0.4 ms/batch and compile cost is 20 s.

Why ONNX Runtime and not pure PyTorch for the three networks: SCRFD/ArcFace/HSEmotion are
static-graph CNNs; ORT's CUDA EP fuses conv+bn+act and avoids Python dispatch entirely. Torch
remains the memory/stream owner, which is why IOBinding-to-`data_ptr` matters.

---

## 5. Compute-saving policy (the reason this beats real time)

1. **1-in-4 frame sampling** (configurable) → 7.5 analysed FPS on 30 FPS source.
2. **Scene-cut boost**: histogram-correlation cut detector temporarily drops the stride to 1
   for `boost_frames`, then decays back — new shots get dense sampling, static shots don't.
3. **Adaptive sampling**: if no track changed identity/emotion for `calm_window` sampled frames,
   stride relaxes up to `max_stride`.
4. **Identity caching**: ArcFace runs on a track only when (a) the track is new, (b) its
   similarity margin is below threshold, or (c) `reid_interval` sampled frames elapsed.
   Steady state: ~1 ArcFace call per actor per 2 seconds instead of 7.5/s.
5. **Quality gating**: blurry (variance-of-Laplacian), tiny (<`min_face_px`) and extreme-profile
   faces (5-point symmetry ratio) are dropped before any embedding work.

---

## 6. Data stores

* `cache/cast_database.db` — SQLite: `actors(id, name, created, updated)`,
  `embeddings(actor_id, vector BLOB float32[512], source_image, quality)`.
  Averaged + L2-normalised centroid per actor is materialised in `actor_centroids`.
  A `.pkl` mirror is written for portability (`cache/cast_database.pkl`).
* `models/weights/` — ONNX zoo, auto-downloaded and SHA-checked.
* `outputs/<video-stem>/` — `timeline.json`, `timeline.csv`, `timeline.png`, `run.log`.

---

## 7. Module map

```
app.py            GUI entry point            cli.py           headless batch entry
configs/          config dataclasses + YAML  utils/           logging, gpu, profiling, imageops
models/           ORT engine + model zoo     detector/        SCRFD + quality gates
recognition/      ArcFace, DB, matcher       tracking/        Kalman + ByteTrack
emotion/          HSEmotion + smoothing      timeline/        events, engine, exporters
pipeline/         decode/sample/orchestrate  gui/             PySide6 widgets
benchmarks/       throughput harness         tests/           unit sanity tests
```
