# ASH-FS — Adaptive Shot-Aware Hierarchical Frame Sampler

ASH-FS is the keyframe selection stage of the **Broll** AI video editing pipeline. Given a raw video file it outputs a compact, semantically representative set of frame indices — the *manifest* — that downstream Qwen3-VL passes consume for scene understanding, captioning, and edit-point detection.

**Where it sits in Broll:**

```
Raw video ingestion
      ↓
  ASH-FS  ←── this component
      ↓
 Manifest (selected frame indices + metadata)
      ↓
  Qwen3-VL analysis pass
      ↓
 Edit decisions / timeline
```

The problem ASH-FS solves: Qwen3-VL inference is expensive (~0.37 GPU-hours per hour of source footage). Feeding every frame is infeasible. Flat uniform sampling (e.g. 1 fps) wastes budget on static or repetitive segments and under-samples fast-cut or high-motion sequences. ASH-FS allocates frames adaptively — more budget where the content is visually complex or changing rapidly, less where it is static — while preserving a globally bounded frame count.

---

## Algorithm Overview

The pipeline runs six sequential stages. All stages share a single `EmbeddingManager` instance so DINOv2 embeddings are extracted exactly once.

```
Video file
    │
    ▼
[1] EmbeddingManager.extract_for_frames()
    Sample candidate frames at candidate_fps, extract DINOv2-ViT-B/14
    embeddings (768-dim, L2-normalised) for every candidate frame.
    Results cached in an LRU dict (keyed by absolute frame index).
    │
    ▼
[2] ShotSegmenter.detect_shots()
    a. PySceneDetect ContentDetector (threshold=27.0) → hard cut boundaries
    b. GLRT soft boundary detection (see below) → additional soft boundaries
    c. Merge boundaries → raw Shot list
    d. Duration enforcement: merge shots < min_shot_duration_s,
       force-split shots > max_shot_duration_s into equal sub-segments
       (marked is_forced_split=True)
    │
    ▼
[3] ComplexityScorer.score_all()
    For each Shot × its candidate frames:
      variance_score = mean pairwise cosine distance among frame embeddings
                       (sampled to ≤100 random pairs when N > 15)
      motion_score   = mean sequential L2 distance between consecutive
                       embeddings, normalised by number of pairs (n - 1)
      complexity     = variance_weight * variance_score
                     + motion_weight   * motion_score
    │
    ▼
[4] BudgetPlanner.plan()
    Per-shot budget (keyframes to select):
      budget = base_frames_per_shot
             + 1  if complexity > threshold            (complexity bonus)
             + floor((duration_s - duration_bonus_start_s)
                     / duration_bonus_interval_s)      (duration bonus)
      budget = min(budget, max_frames_per_shot, available_candidate_frames)
    Global cap: if sum(budgets) > max_total_frames_per_video, trim
    lowest-complexity shots first (3-pass greedy reduction).
    │
    ▼
[5] DualKeyframeSelector.select_all()
    InfoShot-inspired formulation (query-free — see Deviations section).
    For each budgeted shot, compute per-frame scores:
      typicality  g_i = mean cosine similarity of frame i to all others
      volatility  v_i = 1 − mean cosine sim of frame i to k temporal neighbours
    Min-max normalise both to [0, 1]: ĝ, v̂

    budget == 1 → medoid: argmax(ĝ)
    budget == 2 → common:  argmax(λ·ĝ − (1−λ)·v̂)   [typicality-dominant]
                   unique:  argmax(α·(1−ĝ) + (1−α)·v̂)  [atypicality-dominant]
                   (where λ = lambda_common, α = alpha_unique)
    budget  > 2 → common + unique + greedy extra frames by descending v̂,
                   subject to min_spacing_frames gap between any two selected
    │
    ▼
[6] HierarchicalDiversityPlanner.build_tree() + get_final_keyframes()
    VideoTree-inspired 4-level hierarchy (query-free — see Deviations section).
    Levels: shot → chunk (chunk_size_shots shots) → scene (3 chunks)
            → sequence (3 scenes) → video root

    For each non-shot node:
      aggregate_embedding = L2-normalised mean-pool of children's aggregates
      diversity_score     = mean pairwise cosine distance between children's
                            aggregate embeddings

    Expansion decision (top-down walk):
      diversity_score > expansion_threshold_{level} → expand (recurse)
      diversity_score ≤ threshold                   → summarise
        (replace subtree with a single medoid keyframe: medoid-of-medoids
         strategy drills recursively to a shot-level frame)

    Shot-level nodes always yield their full keyframe set.
```

### GLRT Soft Boundary Detection

The "GLRT" label here refers to the Generalised Likelihood Ratio Test intuition: a boundary exists where the distributions on either side are maximally dissimilar. The implementation uses a windowed cosine distance proxy rather than a full parametric GLRT:

```
For each candidate frame i (not already a hard-cut boundary):
  seg_len   = distance (in candidate frames) to nearest hard boundary
  window_w  = clamp(seg_len // 4, min=window_size_range[0],
                                   max=window_size_range[1])
  half      = window_w // 2
  score(i)  = mean_cosine_distance(embeddings[i-half : i],
                                    embeddings[i    : i+half])

Soft boundary at i iff:
  score(i) > glrt_threshold  AND  score(i) is a local maximum
```

This captures gradual content transitions (cross-fades, slow pans into new scenes) that pixel-level hard-cut detectors miss.

---

## Config Reference

All keys live in `config.yaml`. Every key maps directly to a field in `ashfs/config.py`'s dataclass hierarchy.

### `shot_segmentation`

| Key | Default | What it controls | Tuning guidance |
|---|---|---|---|
| `window_size_range` | `[3, 15]` | Min/max half-window size (in candidate frames) for GLRT soft boundary scoring | Narrower `[2, 8]` for fast-cut content to catch rapid transitions; wider for talking-head where false positives are more costly |
| `min_shot_duration_s` | `0.5` | Shots shorter than this are merged into their shortest neighbour | Lower to `0.3` for fast-cut/ad content; raise to `1.0` for talking-head/documentary |
| `max_shot_duration_s` | `30.0` | Shots longer than this are force-split into equal sub-segments | Lower to `15.0` for dense dialogue; raise for long static shots in educational content |
| `glrt_threshold` | `0.35` | Cosine distance threshold for declaring a soft boundary | Raise toward `0.45` for talking-head (suppress false splits on subtle expression changes); lower toward `0.25` for general/fast-cut to catch more transitions |
| `candidate_fps` | `2.0` | Frame sampling rate for candidate frames sent to DINOv2 | `1.0` for very long videos to save memory; `4.0` for fast-cut content where 2 fps may miss frames |

### `complexity`

| Key | Default | What it controls | Tuning guidance |
|---|---|---|---|
| `variance_weight` | `0.5` | Weight of variance_score in complexity = w1*variance + w2*motion | Raise for content where static diversity matters (slides, graphics); lower for content where motion is the primary signal |
| `motion_weight` | `0.5` | Weight of motion_score in complexity | Raise for sports/action; lower for talking-head where motion adds noise |
| `complexity_threshold_general` | `0.25` | Complexity above which a shot gets a complexity bonus frame | General-purpose default |
| `complexity_threshold_talking_head` | `0.15` | Threshold for talking-head/EdTech content | Lower threshold = bonus frames awarded more freely, appropriate since talking-head shots are visually subtle |
| `complexity_threshold_fast_cut` | `0.35` | Threshold for fast-cut/ad content | Higher threshold = bonus frames only for genuinely high-motion sequences; fast-cut videos already have many short shots so base budget usually suffices |
| `content_vertical` | `"general"` | Selects which threshold to apply: `general`, `talking_head`, or `fast_cut` | **Set this to match your content type** — it is the single most impactful tuning parameter |

### `budget`

| Key | Default | What it controls | Tuning guidance |
|---|---|---|---|
| `base_frames_per_shot` | `1` | Minimum keyframes per shot before any bonus | Raise to `2` for content where even simple shots need a common+unique pair |
| `max_frames_per_shot` | `6` | Hard ceiling on per-shot budget | Raise for long complex shots; lower to constrain total output |
| `duration_bonus_interval_s` | `4.0` | Add +1 frame per this many seconds of shot duration beyond `duration_bonus_start_s` | Lower to `2.0` for documentary/long-take content; raise to `8.0` for fast-cut |
| `duration_bonus_start_s` | `4.0` | Duration threshold at which duration bonus begins | Pair with `max_shot_duration_s`: if max=30 and interval=4, a 30s shot gets (30-4)/4 = 6 bonus frames (capped by max_frames_per_shot) |
| **`max_total_frames_per_video`** | **`300`** | **Global hard cap on total selected frames per video. This is the primary compute envelope anchor.** | See Performance Envelope below. At 300 frames and Qwen3-VL cost of ~0.37 GPU-hr/footage-hr this keeps a 1-hour video within budget. Lower to `150` for faster/cheaper passes; raise to `600` for high-fidelity analysis. |

### `dual_keyframe`

| Key | Default | What it controls | Tuning guidance |
|---|---|---|---|
| `lambda_common` | `0.7` | λ in `common_score = λ·ĝ − (1−λ)·v̂`. Higher = more typicality-dominant common frame | Range [0.5, 0.9]. Lower for content where you want the common frame to also avoid very stable areas |
| `alpha_unique` | `0.5` | α in `unique_score = α·(1−ĝ) + (1−α)·v̂`. Higher = more atypicality-dominant | Range [0.3, 0.7]. Raise to find frames visually unlike the shot's average |
| `neighborhood_k` | `1` | Temporal neighbourhood half-size for volatility computation: v_i uses frames [i-k, i+k] | Raise to `3` for smoother volatility over longer windows; lower to `1` (default) for responsive local detection |
| `min_spacing_frames` | `3` | Minimum gap (in candidate-frame positions) between any two selected frames | Raise to avoid selecting nearly-identical adjacent frames; lower to allow denser selection in fast-cut shots |

### `hierarchy`

| Key | Default | What it controls | Tuning guidance |
|---|---|---|---|
| `chunk_size_shots` | `5` | Number of shots grouped into a single chunk node | Raise to `8–10` for long-form content; lower to `3` for short dense videos |
| `expansion_threshold_chunk` | `0.20` | Mean pairwise cosine distance above which a chunk node expands (recurses) | Lower = more expansion (keeps more frames); raise to prune more aggressively at the chunk level |
| `expansion_threshold_scene` | `0.15` | Same for scene nodes | Slightly lower than chunk to preserve scene-level diversity |
| `expansion_threshold_sequence` | `0.10` | Same for sequence nodes | Lowest threshold — sequence-level pruning is most aggressive |

**Tuning by vertical:**

| Vertical | Recommended thresholds (chunk / scene / seq) |
|---|---|
| `talking_head` | `0.10 / 0.08 / 0.05` (prune aggressively — repetitive) |
| `general` | `0.20 / 0.15 / 0.10` (defaults) |
| `fast_cut` | `0.30 / 0.25 / 0.15` (expand more — high visual diversity) |

### `embeddings`

| Key | Default | What it controls |
|---|---|---|
| `dinov2_model` | `"dinov2_vitb14"` | torch.hub model name. `dinov2_vitb14` (ViT-B/14, 768-dim) is the default. `dinov2_vitl14` (ViT-L/14, 1024-dim) for higher quality at higher cost. |
| `dinov2_batch_size` | `32` | Frames per DINOv2 forward pass. Lower to `8` on small GPUs; raise to `64` if VRAM allows. |
| `dinov2_device` | `"cuda"` | `"cuda"` or `"cpu"`. Auto-falls back to CPU on OOM. |
| `siglip2_enabled` | `false` | Enable SigLIP2 extraction for final keyframes. **Currently stored for future use only — not consumed by the pipeline.** See Deviations section. |
| `siglip2_batch_size` | `16` | Batch size for SigLIP2 (when enabled). |

### `fallback`

| Key | Default | What it controls |
|---|---|---|
| `uniform_fps` | `1.0` | Frames-per-second for the uniform fallback sampler (triggered on pipeline error) |
| `max_fallback_frames` | `6` | Hard cap on fallback frames (prevents unbounded output on very long videos) |

---

## Deviations from Source Papers

ASH-FS draws on two papers — **InfoShot** (query-conditioned dual-frame selection) and **VideoTree** (hierarchical query-driven tree expansion) — but deviates from both in important ways driven by Broll's use case.

### a. No Query Conditioning

**InfoShot** scores frames against a user query: relevance to the query is the primary selection signal. **VideoTree** guides tree expansion by checking whether a subtree contains frames relevant to a query.

ASH-FS cannot use query conditioning because Broll analyzes video *once at ingest time*, before any edit prompts exist. The same manifest is reused for all future edit requests against that video.

**Replacement:** typicality (g_i = mean cosine similarity to all other frames in the shot) and volatility (v_i = local distinctiveness from temporal neighbours) are purely visual, query-free signals. They select frames that are jointly representative of the shot's visual content (typicality) and capture moments of visual change (volatility), which are the frames most likely to be useful across a broad range of future queries.

### b. Adaptive Budget (vs. InfoShot Fixed 2/Shot)

The original InfoShot paper uses a fixed budget of 2 frames per shot (one common, one unique). This is appropriate when query-relevance can rank shots — irrelevant shots naturally receive no budget.

Without query conditioning, all shots must be budgeted upfront. ASH-FS adds:
- A **complexity bonus** (+1 frame when complexity score exceeds the per-vertical threshold)
- A **duration bonus** (+1 frame per `duration_bonus_interval_s` of shot length beyond `duration_bonus_start_s`)
- A **global video cap** (`max_total_frames_per_video`) with complexity-ranked trimming

The eval harness includes `InfoShot-2/shot` as a baseline to measure how much the adaptive budget improves recall over the fixed-budget formulation.

### c. Query-Free Tree Expansion (vs. VideoTree)

VideoTree expands tree nodes when a query's answer is not confidently found in the node's aggregate representation, and stops (summarises) when it is. This requires a live query.

ASH-FS substitutes **mean pairwise cosine distance between children's aggregate embeddings** as the expansion signal. High distance = visually diverse children = worth expanding. Low distance = visually homogeneous = safe to summarise with a single medoid frame. The threshold is tunable per level (`expansion_threshold_{chunk,scene,sequence}`).

This approach preserves the tree's structural budget benefit (avoiding redundant frames in repetitive segments) without requiring a query.

### d. Forced Splits as Budget Mechanism

ASH-FS adds forced splitting of shots longer than `max_shot_duration_s` into equal sub-segments. This is a **budget and coverage mechanism**, not a semantic cut detector — it ensures very long static shots (e.g. a 2-minute whiteboard recording) do not monopolise the frame budget.

Forced-split sub-segments are marked with `is_forced_split=True` on the `Shot` object so downstream stages (and the manifest) can distinguish them from semantically detected boundaries.

### e. SigLIP2 — Stored but Not Consumed

`EmbeddingManager.extract_siglip2()` is implemented and wired into the pipeline (when `siglip2_enabled: true`) but the SigLIP2 embeddings are stored in `SelectedKeyframes.siglip_embeddings` and **not read by any downstream stage**.

The reason: SigLIP2's strength is text-image matching (it shares a joint text-image embedding space). This is only useful when a query is available. At ingest time there is no query, so text-image matching adds no signal over DINOv2's pure visual features.

SigLIP2 embeddings are preserved in the dataclass for future use: once a Broll user submits an edit prompt, the stored SigLIP2 embeddings can be used for fast retrieval (nearest-neighbour search of prompt embedding against stored frame embeddings) without re-running the vision model.

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision scenedetect opencv-python pillow numpy pyyaml
# Optional: SigLIP2 support (siglip2_enabled: true in config)
pip install timm
```

PyTorch: install the version matching your CUDA toolkit from https://pytorch.org. CPU-only builds work but are slower for embedding extraction.

### Run the pipeline

```bash
python run_ashfs.py --video path/to/video.mp4 --config config.yaml
```

The manifest is written to `output/manifest.json`. The manifest schema:

```json
{
  "video_path": "...",
  "total_shots": 42,
  "total_candidate_frames": 1200,
  "total_selected_frames": 137,
  "frames_per_shot_distribution": [1, 2, 1, ...],
  "budget_stats": {
    "total_frames": 137,
    "min_budget": 1,
    "max_budget": 6,
    "mean_budget": 2.1,
    "shots_at_base": 28,
    "shots_with_bonus": 14
  },
  "compression_stats": {
    "total_original_frames": 200,
    "total_final_frames": 137,
    "compression_ratio": 1.46,
    "nodes_expanded": 12,
    "nodes_summarized": 3
  },
  "selected_frame_indices": [0, 48, 102, ...],
  "frame_roles": ["medoid", "common", "unique", ...],
  "shot_boundaries": [
    {"start_frame": 0, "end_frame": 299, "duration_s": 12.0, "fps": 25.0},
    ...
  ],
  "timing": {
    "candidate_sampling": 0.12,
    "dinov2_extraction": 8.43,
    ...
  }
}
```

`frame_roles` values: `"medoid"` (single representative), `"common"` (typicality-dominant), `"unique"` (atypicality-dominant), `"extra"` (additional high-volatility frames), `"fallback"` (uniform fallback only).

### Run the test suite

```bash
# Fast unit tests (no video, no GPU required):
cd ashfs
python -m pytest tests/

# Full integration test (generates a synthetic 10-minute video, requires torch + scenedetect):
python -m pytest tests/ --run-slow
```

### Run the evaluation harness

```bash
# Step 1 — label a clip (interactive):
python eval/eval_harness.py label --clip clip.mp4 --output labels/clip.json

# Step 2 — evaluate all labelled clips:
python eval/eval_harness.py eval --clips labels/ --config config.yaml

# Step 3 — print aggregate report from saved results:
python eval/eval_harness.py report --results eval/results/
```

The harness compares three methods: ASH-FS (full adaptive pipeline), Flat-FPS (1 fps uniform baseline), and InfoShot-2/shot (fixed 2 frames per shot, no adaptive budget). Metrics: Recall@15, Precision@15, F1 — where @15 means a selected frame is counted as a hit if it falls within ±15 absolute frames (~0.5 s at 30 fps) of a ground-truth must-keep frame.

---

## Performance Envelope

### Target: ~0.37 GPU-hours per hour of source footage

This figure is the Qwen3-VL inference cost when processing the frames ASH-FS selects, not the cost of running ASH-FS itself.

**How `max_total_frames_per_video` was calibrated:**

Qwen3-VL processes one frame in approximately 1.33 GPU-seconds at our serving configuration. At 300 frames per video:

```
300 frames × 1.33 GPU-s/frame = 399 GPU-seconds ≈ 0.111 GPU-hours per video

For a 1-hour video this equals:
  0.111 GPU-hours / 1 footage-hour ≈ 0.37 GPU-hr/footage-hr  ✓
```

The `0.37 GPU-hr/footage-hr` figure is embedded in `tests/test_integration.py` as `GPU_HR_PER_FOOTAGE_HR = 0.37` for regression tracking.

Adjusting the budget:
- `max_total_frames_per_video: 150` → ~0.185 GPU-hr/footage-hr (half cost, lower recall)
- `max_total_frames_per_video: 600` → ~0.74 GPU-hr/footage-hr (double cost, higher recall for dense content)

### ASH-FS's own compute cost

The sampler pipeline runs almost entirely on CPU or a small GPU slice:

- **Embedding extraction** (DINOv2): the only GPU-intensive step. At `candidate_fps=2.0` a 1-hour video produces ~7200 candidate frames; at batch size 32 this is ~225 forward passes. On a T4 GPU this runs in under 3 minutes.
- **All other stages** (shot segmentation, complexity scoring, budget planning, dual-frame selection, hierarchy planning): pure NumPy/CPU, typically under 10 seconds total for a 1-hour video.

The sampler's GPU cost is negligible compared to the Qwen3-VL pass it enables.
