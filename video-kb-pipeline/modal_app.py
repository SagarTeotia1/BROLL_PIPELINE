"""Modal.com deployment layer for the video-kb-pipeline.

SETUP REQUIRED — create one Modal secret named "video-kb-secrets" in the
Modal dashboard (https://modal.com/secrets) containing ALL environment
variables that the pipeline needs:

    DATABASE_URL            postgresql+asyncpg://user:pass@host:5432/dbname
    PINECONE_API_KEY        ...
    PINECONE_INDEX_NAME     video-kb
    PINECONE_ENVIRONMENT    us-east-1
    R2_ENDPOINT_URL         https://<account>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID        ...
    R2_SECRET_ACCESS_KEY    ...
    R2_BUCKET_NAME          ...
    OPENAI_API_KEY          sk-...   (optional — NOT used for embeddings; local model used instead)
    HF_TOKEN                hf_...   (optional — required if model repo is gated)

Deploy with:
    modal deploy modal_app.py

Run ad-hoc:
    modal run modal_app.py::run_full_pipeline --video-path videos/my_video.mp4

Architecture
------------
  run_full_pipeline  (orchestrator, image=base_image, no GPU)
      │  downloads video ONCE → /vol/videos/{video_id}.mp4 (shared volume)
      ├── run_l1_modal         (L1, gpu=L40S, l1_image)
      │     ASHFS + Whisper in parallel threads, reads from shared volume
      ├── run_level2_modal     (L2, gpu=L40S, face_color_image)
      │     face analysis + color grading, reads from shared volume
      └── QwenAnalyserModal    (L3, gpu=L40S, qwen_image, @app.cls)
            Qwen VL inference + KB write
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

app = modal.App("video-kb-pipeline")

# ---------------------------------------------------------------------------
# Persistent model caches — survives container restarts, avoids re-downloads
# ---------------------------------------------------------------------------
model_cache = modal.Volume.from_name("video-kb-model-cache", create_if_missing=True)
# /vol/models is never created by pip/apt during image build, so Modal can mount cleanly.
_MODEL_CACHE_VOLUMES = {"/vol/models": model_cache}

# Ephemeral video staging volume — video downloaded once by the orchestrator,
# shared across all L1/L2/L3 containers without re-downloading.
# Files written here are deleted by the orchestrator after the pipeline finishes.
video_staging = modal.Volume.from_name("video-kb-video-staging", create_if_missing=True)
_VIDEO_STAGING_VOLUMES = {"/vol/videos": video_staging}
_ALL_VOLUMES = {"/vol/models": model_cache, "/vol/videos": video_staging}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Absolute path to THIS file's directory — the video-kb-pipeline root.
_THIS_DIR = Path(__file__).parent.resolve()

# Parent directory that contains the sibling modules (ashfs, color-grading,
# face_analysis-model).
_SIBLING_ROOT = _THIS_DIR.parent.resolve()


# ---------------------------------------------------------------------------
# Images — Modal 1.x
# Rule: add_local_dir() MUST come last — no pip/apt after it.
# Strategy: build pip-only "_pkgs" images first, then attach local dirs at end.
# ---------------------------------------------------------------------------

_BASE_PKGS = [
    "asyncpg>=0.29",
    "pgvector>=0.3",
    "pinecone>=3.0",
    "boto3>=1.35",
    "pydantic>=2.7",
    "pydantic-settings>=2.4",
    "tenacity>=8.0",
    "nanoid>=2.0",
    "numpy>=1.26",
    "opencv-python-headless>=4.10",
    "ffmpeg-python>=0.2",
    "neo4j>=5.0",
]

# pip-only base (no local files) — used as parent for derived images.
# libgl1 + libglib2.0-0 needed for OpenCV headless. ffmpeg binary is NOT
# installed here — it pulls 80+ packages (Cairo/Pango/fontconfig/libav*)
# and causes build OOM. Only whisper_image installs the ffmpeg binary.
_base_pkgs = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libgl1", "libglib2.0-0"])
    .pip_install(_BASE_PKGS)
    .env({
        "HF_HOME": "/vol/models/huggingface",
        "TORCH_HOME": "/vol/models/torch",
        # XDG_CACHE_HOME intentionally NOT set — pip uses it during image build
        # steps and would fill /vol/models before the volume can be mounted.
    })
)

# base_image = _base_pkgs + pipeline source (add_local_dir last)
base_image = _base_pkgs.add_local_dir(str(_THIS_DIR), remote_path="/app")

# Level 1 (ASHFS + Whisper): single CUDA runtime image runs both in parallel threads.
# CUDA runtime base needed for ctranslate2 (faster-whisper GPU inference).
# torch cu124 wheel for DINOv2 / PyTorch CUDA. ffmpeg binary for audio extraction.
# One container = one video download, one cold start, same GPU, lower wall-clock.
l1_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install(["ffmpeg", "libgl1", "libglib2.0-0"])
    .pip_install(_BASE_PKGS)
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install([
        "faster-whisper>=1.0",
        "scenedetect>=0.6",
        "timm>=1.0",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    ])
    .env({
        "HF_HOME": "/vol/models/huggingface",
        "TORCH_HOME": "/vol/models/torch",
    })
    .add_local_dir(str(_THIS_DIR), remote_path="/app")
    .add_local_dir(str(_SIBLING_ROOT / "ashfs"), remote_path="/app_modules/ashfs", ignore=[".git"])
)

# Face + color: CUDA 12.4 + cuDNN 9 runtime base so onnxruntime-gpu can use GPU.
# debian_slim has no cuDNN → ONNX falls back to CPU. This base provides libcudnn.so.9.
face_color_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install(["libgl1", "libglib2.0-0", "ffmpeg"])
    .pip_install(_BASE_PKGS)
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install([
        "onnxruntime-gpu==1.20.0",
        "onnx>=1.16",
        "onnxconverter-common>=1.14",   # enables FP16 ONNX model conversion
        "av>=12.0",
        "requests>=2.32",
        "scipy>=1.13",
        "scikit-learn>=1.5",
        "scikit-image>=0.24",
        "matplotlib>=3.9",
        "numba>=0.60",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "pyannote.audio>=3.1",   # Stage 0: speaker diarization
    ])
    # pyannote.audio pulls torchaudio -> nvidia-cudnn-cu12 (pip wheel), which
    # shadows the base image's apt-installed system cuDNN that onnxruntime-gpu
    # was built against. Loading the wrong cuDNN aborts with "undefined symbol
    # cudnnGetLibConfig" (SIGABRT). Uninstall the pip wheel so onnxruntime-gpu
    # keeps using the working system cuDNN — same fix pattern as the
    # torchcodec/qwen_image conflict below.
    .run_commands("pip uninstall -y nvidia-cudnn-cu12 2>/dev/null || true")
    # pyannote.audio's dependency resolution (speechbrain / pytorch-lightning
    # transitively constrain torchvision) silently upgrades torchvision past
    # the pin above, breaking its compiled-op ABI match with torch==2.5.1
    # ("operator torchvision::nms does not exist"). Force-reinstall the exact
    # matched torch/torchvision/torchaudio trio as the final step so nothing
    # installed after this can drift them apart again.
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # pyannote.audio pulls in torchcodec, which needs CUDA 13's libnvrtc.so.13
    # — not present in this CUDA 12.4 image. Same conflict already solved for
    # qwen_image below; pyannote's Audio class falls back to
    # torchaudio/soundfile decoding when torchcodec is absent.
    .run_commands("pip uninstall -y torchcodec 2>/dev/null || true")
    .env({
        "HF_HOME": "/vol/models/huggingface",
        "TORCH_HOME": "/vol/models/torch",
        "INSIGHTFACE_HOME": "/vol/models/insightface",   # SCRFD + ArcFace ONNX models
        "ORT_CACHE_DIR": "/vol/models/ort",              # onnxruntime GPU session cache
    })
    .add_local_dir(str(_THIS_DIR), remote_path="/app")
    .add_local_dir(str(_SIBLING_ROOT / "face_analysis-model"), remote_path="/app_modules/face_analysis", ignore=[".git"])
    .add_local_dir(str(_SIBLING_ROOT / "color-grading"), remote_path="/app_modules/color_grading", ignore=[".git"])
)

# Qwen / vLLM + local embedder
# Start from CUDA 12.4 base — avoids pip pulling nvidia-cublas-cu13 (CUDA 13)
# which breaks on Modal's CUDA 12.x hardware. vllm + torch install cleanly
# against system CUDA when the base image already provides it.
qwen_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(["ffmpeg", "libgl1", "libglib2.0-0", "git"])
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install([
        "vllm>=0.6",
        "transformers>=4.45",
        "accelerate>=0.34",
        "pillow>=10.4",
        "sentence-transformers>=3.0,<4.0",  # 4.0+ pulls torchcodec which needs libnvrtc.so.13 (CUDA 13)
    ] + _BASE_PKGS)
    .run_commands("pip uninstall -y torchcodec 2>/dev/null || true")  # nuke torchcodec — needs libnvrtc.so.13 (CUDA 13), container has CUDA 12.4
    .env({
        "HF_HOME": "/vol/models/huggingface",
        "TORCH_HOME": "/vol/models/torch",
        "VLLM_CACHE_ROOT": "/vol/models/vllm",           # torch.compile cache — persists across cold starts (~40s saved)
        "TRANSFORMERS_CACHE": "/vol/models/huggingface",  # redundant with HF_HOME but some libs check this
        "SENTENCE_TRANSFORMERS_HOME": "/vol/models/sentence_transformers",
    })
    .add_local_dir(str(_THIS_DIR), remote_path="/app")
)

# ---------------------------------------------------------------------------
# Model pre-warm functions — run once via: modal run modal_app.py::warmup_models
# Downloads every model to the persistent volume so all pipeline runs are instant.
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100",
    image=l1_image,
    timeout=3600,
    volumes=_ALL_VOLUMES,
)
def _warmup_l1_models() -> None:
    """Pre-download DINOv2 (torch.hub) and Whisper large-v3 (HuggingFace) to volume."""
    import sys
    sys.path.insert(0, "/app")

    # DINOv2 ViT-B/14 — ASHFS uses this for all keyframe embeddings
    import torch
    logger.info("Downloading DINOv2 ViT-B/14 to TORCH_HOME=%s ...", "/vol/models/torch")
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    logger.info("DINOv2 cached.")

    # Whisper large-v3 — faster-whisper downloads via huggingface_hub → HF_HOME
    from faster_whisper import WhisperModel  # type: ignore[import]
    logger.info("Downloading Whisper large-v3 to HF_HOME=%s ...", "/vol/models/huggingface")
    WhisperModel("large-v3", device="cuda", compute_type="float16")
    logger.info("Whisper large-v3 cached.")


@app.function(
    gpu="A100",
    image=qwen_image,
    timeout=7200,
    volumes=_ALL_VOLUMES,
)
def _warmup_l3_models() -> None:
    """Pre-download Qwen3-VL-8B and BAAI/bge-large-en-v1.5 to volume."""
    import sys
    sys.path.insert(0, "/app")

    # BAAI/bge-large-en-v1.5 — used by LocalEmbedder for fact/scene embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    logger.info("Downloading BAAI/bge-large-en-v1.5 ...")
    SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    logger.info("bge-large-en-v1.5 cached.")

    # Qwen3-VL-8B-Instruct — vLLM downloads tokenizer + weights via HF on first load
    from vllm import LLM, SamplingParams  # type: ignore[import]
    logger.info("Downloading Qwen/Qwen3-VL-8B-Instruct via vLLM ...")
    LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=16384,
        max_num_seqs=48,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "max_pixels": 602112,
            "min_pixels": 200704,
        },
        trust_remote_code=True,
    )
    logger.info("Qwen3-VL-8B-Instruct cached.")


@app.local_entrypoint()
def warmup_models() -> None:
    """Download all pipeline models to the persistent volume cache.

    Run once before first pipeline execution:
        modal run modal_app.py::warmup_models

    Both L1 and L3 model downloads run in parallel on separate containers.
    After this completes, every subsequent pipeline run loads from cache —
    no network download on hot starts.
    """
    import concurrent.futures

    logger.info("Warming up model caches in parallel (L1 + L3) ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        l1_future = pool.submit(_warmup_l1_models.remote)
        l3_future = pool.submit(_warmup_l3_models.remote)
        l1_future.result()
        logger.info("L1 models (DINOv2 + Whisper) cached.")
        l3_future.result()
        logger.info("L3 models (Qwen + bge-large) cached.")
    logger.info("All models cached. Pipeline cold starts will be instant from now on.")


# ---------------------------------------------------------------------------
# Shared helper: download video inside a Modal container
# ---------------------------------------------------------------------------

def _download_video_in_container(video_url: str) -> str:
    """Download video from URL into container /tmp. Returns local path.

    Each sub-function container has its own isolated /tmp — the orchestrator's
    downloaded copy is NOT accessible here.  This helper re-downloads the video
    so the container can open it with cv2/ffmpeg.
    """
    import tempfile
    import urllib.request as _req

    suffix = ".mp4" if ".mp4" in video_url.lower() else ".video"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    req = _req.Request(
        video_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; video-kb-pipeline/1.0)"},
    )
    with _req.urlopen(req) as resp, open(tmp.name, "wb") as f:
        f.write(resp.read())
    return tmp.name


# ---------------------------------------------------------------------------
# Level 1: ASHFS + Whisper in parallel inside one container
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100",
    image=l1_image,
    timeout=3600,
    secrets=[modal.Secret.from_name("video-kb-secrets")],
    volumes=_ALL_VOLUMES,
)
def run_l1_modal(video_path: str, video_id: str) -> dict:
    """Run ASHFS keyframe sampling + Whisper transcription in parallel.

    One container download, one cold start, both workloads share the A100 GPU.
    ASHFS uses PyTorch CUDA (DINOv2 ~330MB VRAM).
    Whisper uses ctranslate2 CUDA (large-v3 ~3GB VRAM).
    Both fit comfortably on 80GB A100.

    Returns:
        Dict with keys:
        - "manifest":  raw ASHFS manifest dict
        - "segments":  list of serialised TranscriptSegment dicts
    """
    import asyncio
    import dataclasses
    import sys

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app_modules/ashfs")

    from pipeline.level1.ashfs_runner import run_ashfs  # type: ignore[import]
    from pipeline.level1.whisper_runner import run_whisper  # type: ignore[import]

    # Video is pre-staged by the orchestrator at /vol/videos/{video_id}.mp4.
    # Reload the volume so this container sees the committed file.
    video_staging.reload()
    local_path = video_path  # already an absolute /vol/videos/... path
    logger.info("run_l1_modal: video_id=%s reading from volume path=%s — launching ASHFS + Whisper in parallel", video_id, local_path)

    import time as _t
    from shared.utils import StepTimer  # type: ignore[import]
    l1_inner_timer = StepTimer()

    def _run_ashfs() -> tuple[dict, float]:
        t0 = _t.perf_counter()
        manifest = run_ashfs(local_path, video_id, "/tmp")
        elapsed = _t.perf_counter() - t0
        logger.info(
            "ASHFS complete in %.0fs: %d shots, %d keyframes",
            elapsed, manifest.get("total_shots", 0), manifest.get("total_selected_frames", 0),
        )
        return manifest, elapsed

    def _run_whisper() -> tuple[list[dict], float]:
        t0 = _t.perf_counter()
        segs = run_whisper(local_path, video_id, "/tmp")
        elapsed = _t.perf_counter() - t0
        logger.info("Whisper complete in %.0fs: %d segments", elapsed, len(segs))
        return [dataclasses.asdict(s) for s in segs], elapsed

    async def _run_parallel() -> tuple[dict, list[dict]]:
        (manifest, ashfs_s), (segments, whisper_s) = await asyncio.gather(
            asyncio.to_thread(_run_ashfs),
            asyncio.to_thread(_run_whisper),
        )
        l1_inner_timer.record("ashfs", ashfs_s)
        l1_inner_timer.record("whisper", whisper_s)
        return manifest, segments

    manifest, segments = asyncio.run(_run_parallel())
    l1_inner_timer.log(logger, "L1 container — ASHFS vs Whisper")
    return {"manifest": manifest, "segments": segments, "timings": l1_inner_timer.summary()["steps"]}


# ---------------------------------------------------------------------------
# Level 2: Face analysis + Color grading
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100",
    image=face_color_image,
    timeout=7200,
    secrets=[modal.Secret.from_name("video-kb-secrets")],
    volumes=_ALL_VOLUMES,
)
def run_level2_modal(
    video_path: str,
    video_id: str,
    shots_data: list[dict],
    cast_db_r2_key: str | None = None,
    cast_list: list[dict] | None = None,
    keyframe_timestamps: list[float] | None = None,
) -> dict:
    """Run Level-2 face analysis and color grading.

    Face analysis uses ArcFace (identity) + HSEmotion (expression) from the
    face_analysis-model sibling module.  Color grading uses the ColorGradingEngine
    from the color-grading sibling module.

    Cast identity resolution priority:
      1. cast_list (inline JSON) — images downloaded + embedded on this container.
      2. cast_db_r2_key — pre-built SQLite DB downloaded from R2.
      3. Neither — all persons labelled Unknown (PU1, PU2, ...).

    Args:
        video_path:           Local path to the video file.
        video_id:             Stable identifier for this video.
        shots_data:           ShotRecord dicts from Level 1 (loaded from DB by orchestrator).
        cast_db_r2_key:       R2 key for a pre-built cast SQLite database, or None.
        cast_list:            Inline cast — [{"name": "Alice", "images": ["https://..."]}].
        keyframe_timestamps:  Timestamps (s) of ASHFS keyframes — face appearances are
                              sampled at these points so L3 cross-reference is exact.

    Returns:
        Dict with keys "persons", "appearances", "timeline_events", "color_grades",
        each containing a list of serialised dataclass dicts.
    """
    import dataclasses
    import sys
    import tempfile
    import os

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app_modules/color_grading")
    # NOTE: /app_modules/face_analysis is NOT added here — face_runner.py
    # manages its own sys.path isolation to avoid shadowing our pipeline package.

    from pipeline.level2.face_runner import run_face_analysis  # type: ignore[import]
    from pipeline.level2.color_runner import run_color_grading  # type: ignore[import]
    from pipeline.level2.diarization_runner import run_diarization  # type: ignore[import]
    from pipeline.level2.fusion import fuse_diarization_with_faces  # type: ignore[import]
    from shared.types import ShotRecord  # type: ignore[import]

    shots = [ShotRecord(**s) for s in shots_data]
    # Cast count known upfront (user-provided) is a much stronger diarization
    # hint than letting pyannote guess num_speakers — pass it through when
    # available. Falls back to auto-detection when cast size is unknown.
    num_speakers_hint = len(cast_list) if cast_list else None

    # Build / download cast DB — cast_list takes priority over cast_db_r2_key
    cast_db_path: str | None = None
    cast_db_tmp = None

    if cast_list:
        # Build cast DB on-the-fly from inline JSON (images downloaded here on GPU container)
        from pipeline.level2.face_runner import build_cast_db_from_cast_list  # type: ignore[import]

        cast_db_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        cast_db_tmp.close()
        cast_db_path = cast_db_tmp.name
        logger.info(
            "Building cast DB from %d inline cast members → %s", len(cast_list), cast_db_path
        )
        n_registered = build_cast_db_from_cast_list(cast_list, cast_db_path)
        logger.info("Cast DB built: %d/%d members registered", n_registered, len(cast_list))

    elif cast_db_r2_key:
        from knowledge_base.r2.uploader import download_bytes  # type: ignore[import]

        logger.info("Downloading cast DB from R2: %s", cast_db_r2_key)
        cast_db_bytes = download_bytes(cast_db_r2_key)
        cast_db_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        cast_db_tmp.write(cast_db_bytes)
        cast_db_tmp.close()
        cast_db_path = cast_db_tmp.name
        logger.info("Cast DB downloaded → %s (%d bytes)", cast_db_path, len(cast_db_bytes))

        # Also download the .pkl sidecar — the face pipeline needs it for fast
        # gallery search.  Derive the R2 key by replacing the .db suffix.
        cast_pkl_r2_key = (
            cast_db_r2_key[:-3] + ".pkl" if cast_db_r2_key.endswith(".db")
            else cast_db_r2_key + ".pkl"
        )
        cast_pkl_local = cast_db_path[:-3] + ".pkl"
        try:
            pkl_bytes = download_bytes(cast_pkl_r2_key)
            with open(cast_pkl_local, "wb") as _f:
                _f.write(pkl_bytes)
            logger.info(
                "Cast DB .pkl downloaded → %s (%d bytes)", cast_pkl_local, len(pkl_bytes)
            )
        except Exception as _pkl_exc:
            # .pkl is optional — the face pipeline will regenerate it from the .db.
            logger.warning(
                "Could not download cast .pkl from R2 (%s): %s — pipeline will regenerate it",
                cast_pkl_r2_key, _pkl_exc,
            )

    # Video pre-staged by orchestrator on shared volume — reload to see it.
    video_staging.reload()
    logger.info("run_level2_modal: using volume video path=%s", video_path)

    import time as _t
    from shared.utils import StepTimer  # type: ignore[import]
    l2_inner_timer = StepTimer()

    # Pre-import torch/torchvision here, in the main thread, BEFORE the face
    # + color + diarization threads below start. torchvision's internal
    # submodule init (torchvision.extension) is not safe against two threads
    # triggering its first import at the same time — face_analysis (via
    # torch) and diarization (via torchaudio) racing on this causes
    # "partially initialized module 'torchvision' ... circular import".
    # Importing once up front caches it in sys.modules so the threads below
    # never race on it.
    import torch  # type: ignore[import]  # noqa: F401
    try:
        import torchvision  # type: ignore[import]  # noqa: F401
    except ImportError:
        pass

    # Face (GPU bursts), color grading (pure CPU), and diarization (audio-only,
    # short GPU burst) have zero resource conflict. Run all three in parallel
    # threads → wall clock = max(face, color, diarization) not sum.
    import concurrent.futures as _cf
    face_exc: BaseException | None = None
    color_exc: BaseException | None = None
    diar_exc: BaseException | None = None
    face_result: dict = {}
    color_result: list = []
    diar_turns: list[dict] = []

    def _run_face():
        nonlocal face_exc
        try:
            t0 = _t.perf_counter()
            result = run_face_analysis(
                video_path, video_id, cast_db_path,
                keyframe_timestamps=keyframe_timestamps,
                shots=[dataclasses.asdict(s) for s in shots],
            )
            l2_inner_timer.record("face_analysis", _t.perf_counter() - t0)
            return result
        except Exception as exc:
            face_exc = exc
            raise

    def _run_color():
        nonlocal color_exc
        try:
            t0 = _t.perf_counter()
            result = run_color_grading(video_path, video_id, shots)
            l2_inner_timer.record("color_grading", _t.perf_counter() - t0)
            return result
        except Exception as exc:
            color_exc = exc
            raise

    def _run_diarization():
        nonlocal diar_exc
        try:
            t0 = _t.perf_counter()
            result = run_diarization(
                video_path, video_id, "/tmp", num_speakers=num_speakers_hint,
            )
            l2_inner_timer.record("diarization", _t.perf_counter() - t0)
            return result
        except Exception as exc:
            # Non-fatal — Stage 0 is a fusion enhancement, not required for L2 to succeed.
            diar_exc = exc
            logger.warning("Diarization failed for video_id=%s (non-fatal): %s", video_id, exc)
            return []

    logger.info("run_level2_modal: face + color + diarization in parallel for video_id=%s", video_id)
    try:
        with _cf.ThreadPoolExecutor(max_workers=3) as _pool:
            _face_fut = _pool.submit(_run_face)
            _color_fut = _pool.submit(_run_color)
            _diar_fut = _pool.submit(_run_diarization)
            face_result = _face_fut.result()
            color_result = _color_fut.result()
            diar_turns = _diar_fut.result()
    finally:
        if cast_db_tmp:
            # Remove both .db and .pkl sidecar — build_cast_db_from_cast_list
            # writes a .pkl via db.export_pickle() alongside the .db.
            for _ext in ("", ".pkl"):
                _path = cast_db_tmp.name.removesuffix(".db") + (".db" if not _ext else _ext)
                try:
                    os.unlink(_path)
                except OSError:
                    pass

    speaker_turns = fuse_diarization_with_faces(
        video_id, diar_turns, face_result["appearances"]
    )

    _l2_inner = l2_inner_timer.summary()
    logger.info(
        "run_level2_modal complete in %.0fs: persons=%d appearances=%d timeline=%d color=%d speaker_turns=%d | timings=%s",
        _l2_inner["total_s"],
        len(face_result["persons"]),
        len(face_result["appearances"]),
        len(face_result["timeline_events"]),
        len(color_result),
        len(speaker_turns),
        _l2_inner["steps"],
    )

    return {
        "persons": [dataclasses.asdict(p) for p in face_result["persons"]],
        "appearances": [dataclasses.asdict(a) for a in face_result["appearances"]],
        "timeline_events": [dataclasses.asdict(e) for e in face_result["timeline_events"]],
        "color_grades": [dataclasses.asdict(c) for c in color_result],
        "speaker_turns": [dataclasses.asdict(s) for s in speaker_turns],
        "timings": _l2_inner["steps"],
    }


# ---------------------------------------------------------------------------
# Level 3: Qwen VL analysis (Modal class — model loaded once per container)
# ---------------------------------------------------------------------------


@app.cls(
    gpu="A100",
    image=qwen_image,
    timeout=14400,
    secrets=[modal.Secret.from_name("video-kb-secrets")],
    volumes=_ALL_VOLUMES,
)
class QwenAnalyserModal:
    """Modal class that loads the Qwen VL model once and analyses any number of videos.

    The model is loaded in the @modal.enter() lifecycle hook so it is
    initialised exactly once per container, not on every method call.
    """

    @modal.enter()
    def load(self) -> None:
        """Load the Qwen VL vLLM engine.  Called once at container startup."""
        import sys

        sys.path.insert(0, "/app")

        from models.qwen_vllm import QwenVLLM  # type: ignore[import]

        logger.info("QwenAnalyserModal: loading vLLM engine")
        self.qwen = QwenVLLM.get()
        self.qwen.load()
        logger.info("QwenAnalyserModal: engine ready")

    @modal.method()
    async def analyse_video(
        self,
        video_id: str,
        keyframe_ids: list[str] | None = None,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> dict:
        """Run Qwen VL inference on a subset of keyframes and write L3 KB records.

        Args:
            video_id: Stable identifier for the video.
            keyframe_ids: Optional explicit list of keyframe IDs to process.
                When None, all keyframes for the video are processed (single-shard mode).
            shard_index: 0-based index of this shard (for logging).
            total_shards: Total number of parallel shards (for logging).

        Returns:
            Dict with keys "video_id", "frames_analysed", "shard_index".
        """
        import sys

        sys.path.insert(0, "/app")

        from knowledge_base.postgres.client import get_pool  # type: ignore[import]
        from knowledge_base.postgres.queries import (  # type: ignore[import]
            get_keyframes_for_video,
            get_shots_for_video,
            get_persons_for_video,
            get_transcript_segments_for_video,
            get_face_appearances_for_video_sampled,
            get_existing_frame_analyses,
        )
        from pipeline.level3.context_builder import build_all_frame_contexts  # type: ignore[import]
        from pipeline.level3.qwen_runner import run_qwen_analysis  # type: ignore[import]
        from pipeline.level3.kb_finalizer import finalize_level3_kb  # type: ignore[import]

        import time as _t
        from shared.utils import StepTimer  # type: ignore[import]
        l3_shard_timer = StepTimer()

        pool = await get_pool()

        logger.info(
            "QwenAnalyserModal.analyse_video: shard %d/%d for video_id=%s",
            shard_index + 1, total_shards, video_id,
        )

        with l3_shard_timer.step("db_load"):
            all_keyframes = await get_keyframes_for_video(pool, video_id)
            shots = await get_shots_for_video(pool, video_id)
            transcript_segments = await get_transcript_segments_for_video(pool, video_id)
            persons = await get_persons_for_video(pool, video_id)
            appearances = await get_face_appearances_for_video_sampled(pool, video_id)

        # Filter to this shard's keyframes (already sorted by timestamp_s from DB)
        if keyframe_ids is not None:
            kf_id_set = set(keyframe_ids)
            keyframes = [kf for kf in all_keyframes if kf.id in kf_id_set]
        else:
            keyframes = all_keyframes

        logger.info(
            "Shard %d/%d: processing %d/%d keyframes",
            shard_index + 1, total_shards, len(keyframes), len(all_keyframes),
        )

        # Skip Qwen if all keyframes already have frame_analyses in DB.
        # This happens when re-running L3 to fix Pinecone/Neo4j without re-inferencing.
        existing_analyses = await get_existing_frame_analyses(pool, [kf.id for kf in keyframes])
        if len(existing_analyses) == len(keyframes):
            logger.info(
                "Shard %d/%d: all %d keyframes already analysed — skipping Qwen, "
                "loading from DB for Pinecone/Neo4j write",
                shard_index + 1, total_shards, len(keyframes),
            )
            kf_map = {kf.id: kf for kf in keyframes}
            frame_results = [(kf_map[kid], out) for kid, out in existing_analyses.items()]
            l3_shard_timer.record("qwen_inference", 0.0)
            l3_shard_timer.record("qwen_skipped", float(len(keyframes)))
        else:
            missing = len(keyframes) - len(existing_analyses)
            logger.info(
                "Shard %d/%d: %d/%d keyframes need Qwen inference",
                shard_index + 1, total_shards, missing, len(keyframes),
            )
            contexts = build_all_frame_contexts(
                keyframes, transcript_segments, appearances, persons
            )
            logger.info("Built %d frame contexts for shard %d", len(contexts), shard_index)
            with l3_shard_timer.step("qwen_inference"):
                frame_results = await run_qwen_analysis(contexts, batch_size=16)

        with l3_shard_timer.step("kb_finalize"):
            await finalize_level3_kb(pool, video_id, frame_results, persons, shots)

        valid_count = sum(1 for _, out in frame_results if out is not None)
        l3_shard_timer.log(logger, f"L3 shard {shard_index+1}/{total_shards} — {valid_count} frames")
        _l3_shard = l3_shard_timer.summary()
        logger.info(
            "Shard %d/%d complete: %d/%d frames analysed",
            shard_index + 1, total_shards, valid_count, len(frame_results),
        )
        return {
            "video_id": video_id,
            "frames_analysed": valid_count,
            "shard_index": shard_index,
            "timings": _l3_shard["steps"],
        }


# ---------------------------------------------------------------------------
# Main Modal orchestrator function
# ---------------------------------------------------------------------------


@app.function(
    timeout=21600,  # 6 hours max for the full pipeline
    image=base_image,
    secrets=[modal.Secret.from_name("video-kb-secrets")],
    volumes=_ALL_VOLUMES,
)
async def run_full_pipeline(
    video_path: str,
    video_id: str | None = None,
    cast_db_r2_key: str | None = None,
    cast_list: list[dict] | None = None,
    stop_after: int = 3,
) -> dict:
    """Entry point: run the full L1 → L2 → L3 pipeline on Modal.

    Coordinates level functions, writes all results to PostgreSQL + Pinecone.

    Args:
        stop_after: Stop after this level (1, 2, or 3). Default 3 = full pipeline.

    Args:
        video_path:       Local path to the video OR an R2 key (starting with
                          "videos/").  R2 keys are downloaded to a tmp file.
        video_id:         Stable identifier.  A new nanoid is generated when None.
        cast_db_r2_key:   R2 key for a pre-built cast SQLite DB, or None.
        cast_list:        Inline cast list — takes priority over cast_db_r2_key.
                          Format: [{"name": "Alice", "images": ["https://..."]}, ...]
                          Images are downloaded and embedded on the L40s container.

    Returns:
        Dict with "video_id", "status", and optionally "frames_analysed".
    """
    import concurrent.futures
    import dataclasses
    import sys
    import tempfile
    import os

    sys.path.insert(0, "/app")

    import cv2  # type: ignore[import]

    from shared.utils import gen_id, StepTimer  # type: ignore[import]
    from knowledge_base.postgres.client import get_pool  # type: ignore[import]
    from knowledge_base.postgres.schema import run_migrations  # type: ignore[import]
    from knowledge_base.postgres.queries import (  # type: ignore[import]
        upsert_video,
        upsert_job,
        update_job_status,
        update_job_meta,
        bulk_insert_chunks,
        bulk_insert_shots,
        bulk_insert_keyframes,
        bulk_insert_transcript_segments,
        get_shots_for_video,
        get_keyframes_for_video,
        get_level_status,
    )
    from knowledge_base.r2.uploader import download_bytes, upload_frame as r2_upload_frame  # type: ignore[import]
    from shared.types import (  # type: ignore[import]
        VideoMeta,
        ProcessingJob,
        LevelStatus,
        TranscriptSegment,
        PersonRecord,
        FaceAppearanceRecord,
        FaceTimelineEvent,
        ColorGradeRecord,
        SpeakerTurnRecord,
    )
    from shared.utils import frame_to_png_bytes  # type: ignore[import]

    pool = await get_pool()
    await run_migrations(pool)

    import time as _t

    if not video_id:
        import hashlib as _hl
        video_id = _hl.sha256(video_path.encode()).hexdigest()[:32]
        logger.info("run_full_pipeline: video_id derived from URL hash: %s", video_id)
    logger.info("run_full_pipeline: video_id=%s path=%s", video_id, video_path)

    # ------------------------------------------------------------------
    # Download video ONCE → write to shared volume at /vol/videos/{video_id}.mp4
    # All sub-containers (L1, L2, L3) read from this path — zero re-downloads.
    # ------------------------------------------------------------------
    import urllib.request as _urllib_req

    os.makedirs("/vol/videos", exist_ok=True)
    staged_video_path = f"/vol/videos/{video_id}.mp4"

    if video_path.startswith("http://") or video_path.startswith("https://"):
        logger.info("Downloading video from URL → volume: %s", video_path)
        req = _urllib_req.Request(
            video_path,
            headers={"User-Agent": "Mozilla/5.0 (compatible; video-kb-pipeline/1.0)"},
        )
        with _urllib_req.urlopen(req) as resp, open(staged_video_path, "wb") as f:
            f.write(resp.read())
    elif video_path.startswith("videos/"):
        logger.info("Downloading video from R2 → volume: %s", video_path)
        video_bytes = download_bytes(video_path)
        with open(staged_video_path, "wb") as f:
            f.write(video_bytes)
        logger.info("R2 download complete: %d bytes", len(video_bytes))
    elif os.path.exists(video_path):
        # Local path inside the orchestrator container — copy to volume
        import shutil
        logger.info("Copying local video to volume: %s", video_path)
        shutil.copy2(video_path, staged_video_path)
    else:
        raise FileNotFoundError(
            f"Video not found: {video_path!r}. "
            "Pass a URL, an R2 key starting with 'videos/', or a local path."
        )

    # Commit so sub-containers can see the file immediately
    await video_staging.commit.aio()
    local_video_path = staged_video_path
    logger.info("Video staged at volume path: %s", staged_video_path)

    # Probe video metadata
    if not os.path.exists(local_video_path):
        raise FileNotFoundError(
            f"Video file not found: {local_video_path!r}. "
            "Pass an R2 key (starting with 'videos/') or a path that "
            "already exists inside the Modal container."
        )
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open video: {local_video_path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / fps if fps > 0 else 0.0
    cap.release()

    logger.info(
        "Video probed: fps=%.2f frames=%d size=%dx%d duration=%.1fs",
        fps, frame_count, width, height, duration_s,
    )

    await upsert_video(
        pool,
        VideoMeta(
            id=video_id,
            path=video_path,
            r2_key=video_path if video_path.startswith("videos/") else None,
            duration_s=duration_s,
            fps=fps,
            width=width,
            height=height,
        ),
    )

    # ------------------------------------------------------------------
    # Level 1 — ASHFS (L1A) + Whisper (L1B) in parallel via Modal.spawn
    # Skip entirely if already DONE (allows cheap resume from --stop-after 2+)
    # ------------------------------------------------------------------
    _l1_existing = await get_level_status(pool, video_id, 1)
    if _l1_existing == LevelStatus.DONE:
        logger.info("[L1] Already DONE for video_id=%s — skipping re-run.", video_id)
    else:
        await upsert_job(
            pool,
            ProcessingJob(
                id=gen_id(),
                video_id=video_id,
                level=1,
                status=LevelStatus.RUNNING,
                error_msg=None,
                meta={},
            ),
        )
        try:
            l1_timer = StepTimer()

            # Single A100 container runs ASHFS + Whisper in parallel threads.
            # Reads video from shared volume — no re-download.
            logger.info("[L1] Dispatching ASHFS + Whisper to single A100 container")
            import time as _t
            _t0 = _t.perf_counter()
            l1_result = await run_l1_modal.remote.aio(staged_video_path, video_id)
            l1_timer.record("ashfs_and_whisper", _t.perf_counter() - _t0)

            manifest = l1_result["manifest"]
            segments_data: list[dict] = l1_result["segments"]
            # Propagate sub-timings from the L1 container if available
            for k, v in (l1_result.get("timings") or {}).items():
                l1_timer.record(k, v)

            # Parse ASHFS manifest into typed records
            from pipeline.level1.ashfs_runner import parse_ashfs_manifest  # type: ignore[import]

            chunks, shots, keyframes = parse_ashfs_manifest(manifest, video_id)

            # Upload keyframes to R2 in parallel (one cv2.VideoCapture per thread)
            def _read_and_upload_frame(args: tuple) -> tuple[str, str | None]:
                kf_id, frame_index, video_path_arg, vid_id = args
                _cap = cv2.VideoCapture(video_path_arg)
                _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = _cap.read()
                _cap.release()
                if not ret:
                    return kf_id, None
                try:
                    r2_key = r2_upload_frame(frame_to_png_bytes(frame), vid_id, frame_index)
                    return kf_id, r2_key
                except Exception as e:
                    logger.warning("[L1] R2 upload failed for frame %d: %s", frame_index, e)
                    return kf_id, None

            upload_args = [
                (kf.id, kf.frame_index, local_video_path, video_id)
                for kf in keyframes
            ]

            logger.info("[L1] Uploading %d keyframes to R2 (parallel, max_workers=16)...", len(keyframes))
            kf_id_to_r2: dict[str, str] = {}

            from shared.utils import ProgressTracker  # type: ignore[import]
            r2_progress = ProgressTracker("R2 frame uploads", total=len(keyframes), logger=logger, log_every=10)

            with l1_timer.step("r2_upload"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    for kf_id, r2_key in executor.map(_read_and_upload_frame, upload_args):
                        if r2_key:
                            kf_id_to_r2[kf_id] = r2_key
                        r2_progress.update(1)

            r2_progress.done()
            for kf in keyframes:
                kf.r2_key = kf_id_to_r2.get(kf.id)

            logger.info(
                "[L1] R2 uploads complete: %d/%d successful",
                len(kf_id_to_r2), len(keyframes),
            )

            # Write L1 data to DB — all bulk, no per-row round-trips
            with l1_timer.step("db_write"):
                await bulk_insert_chunks(pool, chunks)
                await bulk_insert_shots(pool, shots)
                await bulk_insert_keyframes(pool, keyframes)

                segments = [
                    TranscriptSegment(
                        id=s["id"],
                        video_id=s["video_id"],
                        chunk_id=s.get("chunk_id"),
                        text=s["text"],
                        start_time=float(s["start_time"]),
                        end_time=float(s["end_time"]),
                        confidence=s.get("confidence"),
                        words=s.get("words") or [],
                    )
                    for s in segments_data
                ]

                from pipeline.level1.aligner import align_transcript_to_chunks  # type: ignore[import]

                segments = align_transcript_to_chunks(segments, chunks)
                if segments:
                    await bulk_insert_transcript_segments(pool, segments)

            await update_job_status(pool, video_id, 1, LevelStatus.DONE)
            _l1_summary = l1_timer.summary()
            await update_job_meta(pool, video_id, 1, {
                **_l1_summary,
                "chunks": len(chunks),
                "shots": len(shots),
                "keyframes": len(keyframes),
                "segments": len(segments),
                "r2_uploaded": len(kf_id_to_r2),
            })
            logger.info(
                "[L1] Complete: chunks=%d shots=%d keyframes=%d segments=%d",
                len(chunks), len(shots), len(keyframes), len(segments),
            )
            l1_timer.log(logger, f"L1 — ASHFS + Whisper + R2 upload  (video_id={video_id[:8]})")

        except Exception as exc:
            err_str = str(exc)
            logger.error("[L1] FAILED: %s", err_str, exc_info=True)
            await update_job_status(pool, video_id, 1, LevelStatus.FAILED, err_str)
            await _cleanup_staged_video(staged_video_path)
            return {"video_id": video_id, "status": "failed_at_level1", "error": err_str}

    if stop_after == 1:
        await _cleanup_staged_video(staged_video_path)
        logger.info("stop_after=1 — pipeline halted after L1. video_id=%s", video_id)
        return {"video_id": video_id, "status": "done_level1"}

    # ------------------------------------------------------------------
    # Level 2 — Face analysis + Color grading via single Modal call
    # Skip if already DONE
    # ------------------------------------------------------------------
    _l2_existing = await get_level_status(pool, video_id, 2)
    if _l2_existing == LevelStatus.DONE:
        logger.info("[L2] Already DONE for video_id=%s — skipping re-run.", video_id)
    else:
        await upsert_job(
            pool,
            ProcessingJob(
                id=gen_id(),
                video_id=video_id,
                level=2,
                status=LevelStatus.RUNNING,
                error_msg=None,
                meta={},
            ),
        )

        try:
            l2_timer = StepTimer()
            shots = await get_shots_for_video(pool, video_id)
            shots_data = [dataclasses.asdict(s) for s in shots]

            # Fetch keyframe timestamps so face_runner samples at ASHFS keyframes
            # rather than 1fps — aligns appearances exactly with L3 Qwen input frames.
            _kfs = await get_keyframes_for_video(pool, video_id)
            kf_timestamps = [kf.timestamp_s for kf in _kfs if kf.timestamp_s is not None]
            logger.info("[L2] Passing %d keyframe timestamps to face_runner", len(kf_timestamps))

            logger.info("[L2] Dispatching face + color grading to Modal")
            _t2 = _t.perf_counter()
            l2_result = await run_level2_modal.remote.aio(
                staged_video_path, video_id, shots_data, cast_db_r2_key, cast_list,
                kf_timestamps,
            )
            l2_timer.record("face_and_color", _t.perf_counter() - _t2)
            for k, v in (l2_result.get("timings") or {}).items():
                l2_timer.record(k, v)

            persons = [
                PersonRecord(
                    id=p["id"],
                    video_id=p["video_id"],
                    pid=p["pid"],
                    display_name=p.get("display_name"),
                    arcface_embedding=(
                        list(p["arcface_embedding"]) if p.get("arcface_embedding") else None
                    ),
                )
                for p in l2_result["persons"]
            ]
            appearances = [
                FaceAppearanceRecord(
                    id=a["id"],
                    video_id=a["video_id"],
                    frame_index=int(a["frame_index"]),
                    timestamp_s=float(a["timestamp_s"]),
                    person_id=a.get("person_id"),
                    track_id=int(a["track_id"]) if a.get("track_id") is not None else None,
                    bbox=a.get("bbox"),
                    emotion=a.get("emotion"),
                    emotion_conf=float(a["emotion_conf"]) if a.get("emotion_conf") is not None else None,
                )
                for a in l2_result["appearances"]
            ]
            timeline_events = [
                FaceTimelineEvent(
                    id=e["id"],
                    video_id=e["video_id"],
                    person_id=e.get("person_id"),
                    emotion=e["emotion"],
                    start_time=float(e["start_time"]),
                    end_time=float(e["end_time"]),
                    confidence=float(e["confidence"]) if e.get("confidence") is not None else None,
                )
                for e in l2_result["timeline_events"]
            ]
            color_grades = [
                ColorGradeRecord(
                    id=c["id"],
                    video_id=c["video_id"],
                    shot_id=c.get("shot_id"),
                    frame_index=int(c["frame_index"]) if c.get("frame_index") is not None else None,
                    timestamp_s=float(c["timestamp_s"]) if c.get("timestamp_s") is not None else None,
                    parameters=c["parameters"],
                    style_tags=list(c.get("style_tags") or []),
                )
                for c in l2_result["color_grades"]
            ]
            speaker_turns = [
                SpeakerTurnRecord(
                    id=t["id"],
                    video_id=t["video_id"],
                    cluster_label=t["cluster_label"],
                    person_id=t.get("person_id"),
                    start_time=float(t["start_time"]),
                    end_time=float(t["end_time"]),
                    confidence=float(t["confidence"]) if t.get("confidence") is not None else None,
                    resolution_method=t.get("resolution_method", "unresolved"),
                )
                for t in l2_result.get("speaker_turns", [])
            ]

            from pipeline.level2.updater import write_level2_to_kb  # type: ignore[import]

            with l2_timer.step("db_write"):
                await write_level2_to_kb(
                    pool, video_id, persons, appearances, timeline_events, color_grades,
                    speaker_turns,
                )

            await update_job_status(pool, video_id, 2, LevelStatus.DONE)
            _l2_summary = l2_timer.summary()
            await update_job_meta(pool, video_id, 2, {
                **_l2_summary,
                "persons": len(persons),
                "appearances": len(appearances),
                "timeline_events": len(timeline_events),
                "color_grades": len(color_grades),
                "speaker_turns": len(speaker_turns),
            })
            logger.info(
                "[L2] Complete: persons=%d appearances=%d timeline=%d color=%d speaker_turns=%d",
                len(persons), len(appearances), len(timeline_events), len(color_grades), len(speaker_turns),
            )
            l2_timer.log(logger, f"L2 — Face + Color grading  (video_id={video_id[:8]})")

        except Exception as exc:
            err_str = str(exc)
            logger.error("[L2] FAILED (non-fatal): %s", err_str, exc_info=True)
            await update_job_status(pool, video_id, 2, LevelStatus.FAILED, err_str)
            # L2 failure is non-fatal — L3 continues with whatever is in DB

    if stop_after == 2:
        await _cleanup_staged_video(staged_video_path)
        logger.info("stop_after=2 — pipeline halted after L2. video_id=%s", video_id)
        return {"video_id": video_id, "status": "done_level2"}

    # ------------------------------------------------------------------
    # Level 3 — Qwen VL analysis via QwenAnalyserModal class
    # Skip if already DONE
    # ------------------------------------------------------------------
    _l3_existing = await get_level_status(pool, video_id, 3)
    if _l3_existing == LevelStatus.DONE:
        logger.info("[L3] Already DONE for video_id=%s — skipping re-run.", video_id)
        await _cleanup_staged_video(staged_video_path)
        return {"video_id": video_id, "status": "complete", "frames_analysed": 0}

    await upsert_job(
        pool,
        ProcessingJob(
            id=gen_id(),
            video_id=video_id,
            level=3,
            status=LevelStatus.RUNNING,
            error_msg=None,
            meta={},
        ),
    )

    frames_analysed = 0
    try:
        l3_timer = StepTimer()
        # Split keyframes across N_SHARDS parallel L40s containers.
        # Each shard loads the full KB (transcript, persons, appearances) but
        # only runs Qwen inference + writes for its subset of keyframes.
        # Writes are idempotent (ON CONFLICT DO UPDATE) so shards can overlap safely.
        N_SHARDS = 1
        # Always read keyframes from DB — L1 may have been skipped this run
        from knowledge_base.postgres.queries import get_keyframes_for_video  # type: ignore[import]
        keyframes = await get_keyframes_for_video(pool, video_id)
        if not keyframes:
            raise RuntimeError(f"No keyframes in DB for video_id={video_id}. Run L1 first.")
        # Sort by timestamp so each shard gets a contiguous time window
        kf_sorted = sorted(keyframes, key=lambda k: k.timestamp_s)
        shard_size = -(-len(kf_sorted) // N_SHARDS)  # ceil
        shards: list[list[str]] = [
            [kf.id for kf in kf_sorted[i : i + shard_size]]
            for i in range(0, len(kf_sorted), shard_size)
        ]

        logger.info(
            "[L3] Dispatching %d parallel Qwen shards for video_id=%s (%d keyframes total)",
            len(shards), video_id, len(kf_sorted),
        )

        analyser = QwenAnalyserModal()
        shard_calls = [
            analyser.analyse_video.remote.aio(
                video_id,
                keyframe_ids=shard_ids,
                shard_index=i,
                total_shards=len(shards),
            )
            for i, shard_ids in enumerate(shards)
        ]
        with l3_timer.step("qwen_inference_and_kb"):
            shard_results = await asyncio.gather(*shard_calls)
        frames_analysed = sum(r.get("frames_analysed", 0) for r in shard_results)

        # Collect per-shard sub-timings (model_load, inference, kb_write, etc.)
        merged_timings: dict[str, float] = {}
        for sr in shard_results:
            for k, v in (sr.get("timings") or {}).items():
                merged_timings[k] = merged_timings.get(k, 0) + v
        for k, v in merged_timings.items():
            l3_timer.record(k, v)

        await update_job_status(pool, video_id, 3, LevelStatus.DONE)
        _l3_summary = l3_timer.summary()
        await update_job_meta(pool, video_id, 3, {
            **_l3_summary,
            "frames_analysed": frames_analysed,
            "shards": len(shards),
            "keyframes_total": len(kf_sorted),
        })
        logger.info("[L3] Complete: %d frames analysed across %d shards", frames_analysed, len(shards))
        l3_timer.log(logger, f"L3 — Qwen VL + KB finalize  (video_id={video_id[:8]})")

    except Exception as exc:
        err_str = str(exc)
        logger.error("[L3] FAILED: %s", err_str, exc_info=True)
        await update_job_status(pool, video_id, 3, LevelStatus.FAILED, err_str)
        await _cleanup_staged_video(staged_video_path)
        return {"video_id": video_id, "status": "failed_at_level3", "error": err_str}

    await _cleanup_staged_video(staged_video_path)

    logger.info(
        "run_full_pipeline complete for video_id=%s — frames_analysed=%d",
        video_id, frames_analysed,
    )
    return {
        "video_id": video_id,
        "status": "complete",
        "frames_analysed": frames_analysed,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _cleanup_staged_video(path: str) -> None:
    """Delete the staged video from the shared volume and commit the deletion."""
    import os
    try:
        os.unlink(path)
        await video_staging.commit.aio()
        logger.debug("Staged video deleted from volume: %s", path)
    except OSError as exc:
        logger.warning("Could not remove staged video %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Local entrypoint for ad-hoc testing
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    input_json: str,
    video_id: str = "",
    stop_after: int = 3,
) -> None:
    """Run the full pipeline from a local terminal using a single input JSON file.

    Usage::

        modal run modal_app.py --input-json input.json
        modal run modal_app.py --input-json input.json --video-id my-stable-id

    Input JSON format::

        {
          "video": {
            "url": "https://example.com/video.mp4",
            "filename": "video.mp4"
          },
          "cast": [
            {
              "id": "person_slug",
              "name": "Person Name",
              "reference_image": "https://example.com/face.jpg"
            }
          ]
        }

    Args:
        input_json:  Path to the input JSON file (see format above).
        video_id:    Optional stable identifier.  A new nanoid is generated when empty.
    """
    import json as _json

    with open(input_json, encoding="utf-8") as f:
        data = _json.load(f)

    # Parse video URL
    video_info = data.get("video", {})
    video_url: str = video_info.get("url", "")
    if not video_url:
        raise ValueError("input JSON missing video.url")

    # Normalise cast list — map reference_image → images array
    raw_cast: list[dict] = data.get("cast", [])
    cast_list: list[dict] | None = None
    if raw_cast:
        cast_list = [
            {
                "name": entry["name"],
                "images": [entry["reference_image"]]
                if "reference_image" in entry
                else entry.get("images", []),
            }
            for entry in raw_cast
            if entry.get("name")
        ]
        print(f"[entrypoint] {len(cast_list)} cast members: {[c['name'] for c in cast_list]}")

    print(f"[entrypoint] Video URL: {video_url}")
    print(f"[entrypoint] stop_after=L{stop_after}")
    result = run_full_pipeline.remote(
        video_url,
        video_id or None,
        None,       # cast_db_r2_key — not used when cast_list provided
        cast_list,
        stop_after,
    )
    print(result)
    if result.get("video_id"):
        print(f"\n[entrypoint] video_id: {result['video_id']}\n"
              f"  Next run: modal run modal_app.py::main --input-json input.json "
              f"--video-id {result['video_id']} --stop-after {stop_after + 1}")


@app.function(
    image=base_image,
    secrets=[modal.Secret.from_name("video-kb-secrets")],
)
async def reset_level(video_id: str, level: int = 3) -> str:
    """Reset a pipeline level status to 'queued' so it re-runs.

    Usage::

        modal run modal_app.py::reset_level --video-id c053f8f5-... --level 3
    """
    import sys
    sys.path.insert(0, "/app")
    from knowledge_base.postgres.client import get_pool  # type: ignore[import]
    pool = await get_pool()
    result = await pool.execute(
        "UPDATE processing_jobs SET status = 'queued', error_msg = NULL "
        "WHERE video_id = $1::uuid AND level = $2",
        video_id, level,
    )
    msg = f"Reset level {level} for video_id={video_id}: {result}"
    print(msg)
    return msg


@app.function(
    image=base_image,
    secrets=[modal.Secret.from_name("video-kb-secrets")],
)
async def run_migration(migration_file: str = "002_dedupe_and_unique_constraints.sql") -> str:
    """Run a SQL migration file against the database.

    Usage::

        modal run modal_app.py::run_migration
        modal run modal_app.py::run_migration --migration-file 002_dedupe_and_unique_constraints.sql
    """
    import sys, pathlib
    sys.path.insert(0, "/app")
    from knowledge_base.postgres.client import get_pool

    migration_path = pathlib.Path("/app/knowledge_base/postgres/migrations") / migration_file
    sql = migration_path.read_text()

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(sql)

    msg = f"Migration {migration_file} applied successfully."
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Single-GPU pipeline — A100 80GB, all L1+L2+L3 in one container
#
# Design rationale (senior ML team):
#   - One cold start instead of three.
#   - Qwen3-VL-8B loaded in @modal.enter() so Triton JIT compiles during
#     container init — by the time L3 starts, inference is instant.
#   - L1 (DINOv2 ~330MB + Whisper ~3GB) and L2 (onnxruntime ~2GB) run while
#     Qwen occupies 16GB: total peak VRAM ~21GB < 80GB, zero pressure.
#   - max_num_seqs=48: KV budget = 80×0.80-16 = 48GB → 48 seqs * 1GB each,
#     capped at 48 to keep decode efficient (diminishing returns above ~64).
#   - Video stays in /tmp — no inter-container volume staging.
#
# Simulates: A100 80GB VRAM / 16 vCPU / 112 GB RAM
# ---------------------------------------------------------------------------

single_gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install(["ffmpeg", "libgl1", "libglib2.0-0", "git"])
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install([
        # L1: ASHFS + Whisper
        "faster-whisper>=1.0",
        "scenedetect>=0.6",
        "timm>=1.0",
        # L2: Face analysis + Color grading
        # 1.19.2 not 1.20.0: 1.20.0 needs transform_tensor_ex_fp from cuDNN 9.3+
        # which vLLM's bundled nvidia-cudnn-cu12==9.1.x doesn't export.
        "onnxruntime-gpu==1.19.2",
        "onnx>=1.16",
        "onnxconverter-common>=1.14",
        "av>=12.0",
        "requests>=2.32",
        "scipy>=1.13",
        "scikit-learn>=1.5",
        "scikit-image>=0.24",
        "matplotlib>=3.9",
        "numba>=0.60",
        # L3: Qwen vLLM + local embedder
        "vllm>=0.6",
        "transformers>=4.45",
        "accelerate>=0.34",
        "pillow>=10.4",
        "sentence-transformers>=3.0,<4.0",
        # shared
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "pyannote.audio>=3.1",   # Stage 0: speaker diarization
    ] + _BASE_PKGS)
    .run_commands("pip uninstall -y torchcodec 2>/dev/null || true")
    # NOTE: unlike face_color_image, do NOT uninstall nvidia-cudnn-cu12 here —
    # this image's onnxruntime-gpu==1.19.2 pin exists specifically to match
    # vLLM's bundled nvidia-cudnn-cu12==9.1.x pip wheel (see comment above).
    # Removing it would break vLLM/L3 in this single-container image.
    #
    # pyannote.audio's dependency resolution can still silently upgrade
    # torchvision past the pin above, breaking its compiled-op ABI match with
    # torch==2.5.1 ("operator torchvision::nms does not exist"). Force-pin
    # the matched trio back as the final step.
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({
        "HF_HOME": "/vol/models/huggingface",
        "TORCH_HOME": "/vol/models/torch",
        "INSIGHTFACE_HOME": "/vol/models/insightface",
        "ORT_CACHE_DIR": "/vol/models/ort",
        "VLLM_CACHE_ROOT": "/vol/models/vllm",
        "TRANSFORMERS_CACHE": "/vol/models/huggingface",
        "SENTENCE_TRANSFORMERS_HOME": "/vol/models/sentence_transformers",
    })
    .add_local_dir(str(_THIS_DIR), remote_path="/app")
    .add_local_dir(str(_SIBLING_ROOT / "ashfs"), remote_path="/app_modules/ashfs", ignore=[".git"])
    .add_local_dir(str(_SIBLING_ROOT / "face_analysis-model"), remote_path="/app_modules/face_analysis", ignore=[".git"])
    .add_local_dir(str(_SIBLING_ROOT / "color-grading"), remote_path="/app_modules/color_grading", ignore=[".git"])
)


@app.cls(
    gpu="a100-80gb",
    cpu=16,
    memory=112 * 1024,  # 112 GB RAM in MiB
    image=single_gpu_image,
    timeout=21600,
    secrets=[modal.Secret.from_name("video-kb-secrets")],
    volumes=_ALL_VOLUMES,
)
class SingleGPUPipeline:
    """Full L1+L2+L3 pipeline in a single A100-80GB container.

    Qwen3-VL-8B loads lazily at L3 start so L1 (DINOv2) and L2 (ONNX face)
    run with full GPU VRAM — no 45 GB KV-cache competing for bandwidth.
    """

    @modal.enter()
    def load_models(self) -> None:
        import sys
        sys.path.insert(0, "/app")
        from models.qwen_vllm import QwenVLLM  # type: ignore[import]
        logger.info("[SingleGPU] Pre-loading Qwen3-VL-8B during container init ...")
        self.qwen = QwenVLLM.get()
        self.qwen.load()  # warms GPU clocks, initialises CUDA context, compiles Triton kernels
        logger.info("[SingleGPU] Qwen ready. Container accepting L1→L2→L3 jobs.")

    @modal.method()
    async def run(
        self,
        video_path: str,
        video_id: str | None = None,
        cast_db_r2_key: str | None = None,
        cast_list: list[dict] | None = None,
    ) -> dict:
        """Run L1 → L2 → L3 sequentially inside this container.

        Args:
            video_path:     URL, R2 key (videos/...), or local path.
            video_id:       Stable ID — generated if None.
            cast_db_r2_key: R2 key for pre-built cast SQLite DB.
            cast_list:      Inline cast [{"name": ..., "images": [...]}].

        Returns:
            Dict with video_id, status, frames_analysed, wall_timings.
        """
        import asyncio
        import concurrent.futures as _cf
        import dataclasses
        import os
        import sys
        import tempfile
        import time as _t
        import urllib.request as _req

        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app_modules/ashfs")
        sys.path.insert(0, "/app_modules/color_grading")

        import cv2  # type: ignore[import]

        from shared.utils import gen_id, StepTimer, frame_to_png_bytes, ProgressTracker  # type: ignore[import]
        from knowledge_base.postgres.client import get_pool  # type: ignore[import]
        from knowledge_base.postgres.schema import run_migrations  # type: ignore[import]
        from knowledge_base.postgres.queries import (  # type: ignore[import]
            upsert_video, upsert_job, update_job_status, update_job_meta,
            bulk_insert_chunks, bulk_insert_shots, bulk_insert_keyframes,
            bulk_insert_transcript_segments, bulk_insert_frame_analyses,
            get_shots_for_video, get_keyframes_for_video, get_level_status,
            get_persons_for_video, get_transcript_segments_for_video,
            get_face_appearances_for_video_sampled, get_existing_frame_analyses,
        )
        from knowledge_base.r2.uploader import download_bytes, upload_frame as r2_upload_frame  # type: ignore[import]
        from shared.types import (  # type: ignore[import]
            VideoMeta, ProcessingJob, LevelStatus,
            TranscriptSegment, PersonRecord, FaceAppearanceRecord,
            FaceTimelineEvent, ColorGradeRecord, SpeakerTurnRecord,
        )
        from pipeline.level1.ashfs_runner import run_ashfs, parse_ashfs_manifest  # type: ignore[import]
        from pipeline.level1.whisper_runner import run_whisper  # type: ignore[import]
        from pipeline.level1.aligner import align_transcript_to_chunks  # type: ignore[import]
        from pipeline.level2.face_runner import run_face_analysis  # type: ignore[import]
        from pipeline.level2.color_runner import run_color_grading  # type: ignore[import]
        from pipeline.level2.diarization_runner import run_diarization  # type: ignore[import]
        from pipeline.level2.fusion import fuse_diarization_with_faces  # type: ignore[import]
        from pipeline.level2.updater import write_level2_to_kb  # type: ignore[import]
        from pipeline.level3.context_builder import build_all_frame_contexts  # type: ignore[import]
        from pipeline.level3.qwen_runner import run_qwen_analysis  # type: ignore[import]
        from pipeline.level3.kb_finalizer import finalize_level3_kb, _build_frame_analysis_records  # type: ignore[import]

        wall = StepTimer()
        if not video_id:
            import hashlib as _hl
            video_id = _hl.sha256(video_path.encode()).hexdigest()[:32]
            logger.info("[SingleGPU] video_id derived from URL hash: %s", video_id)
        logger.info("[SingleGPU] video_id=%s  path=%s", video_id, video_path)

        # ── 0. Download video to /tmp ──────────────────────────────────────
        suffix = ".mp4" if ".mp4" in video_path.lower() else ".video"
        _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        _tmp.close()
        local_video_path = _tmp.name

        with wall.step("video_download"):
            if video_path.startswith(("http://", "https://")):
                req = _req.Request(video_path, headers={"User-Agent": "Mozilla/5.0"})
                with _req.urlopen(req) as resp, open(local_video_path, "wb") as f:
                    f.write(resp.read())
            elif video_path.startswith("videos/"):
                with open(local_video_path, "wb") as f:
                    f.write(download_bytes(video_path))
            elif os.path.exists(video_path):
                import shutil
                shutil.copy2(video_path, local_video_path)
            else:
                raise FileNotFoundError(f"Video not found: {video_path!r}")

        # Probe
        cap = cv2.VideoCapture(local_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = frame_count / fps if fps > 0 else 0.0
        cap.release()
        logger.info("[SingleGPU] Video: fps=%.2f frames=%d %dx%d %.1fs", fps, frame_count, width, height, duration_s)

        # ── DB init ────────────────────────────────────────────────────────
        pool = await get_pool()
        await run_migrations(pool)
        await upsert_video(pool, VideoMeta(
            id=video_id, path=video_path,
            r2_key=video_path if video_path.startswith("videos/") else None,
            duration_s=duration_s, fps=fps, width=width, height=height,
        ))

        # ══════════════════════════════════════════════════════════════════
        # L1 — ASHFS + Whisper parallel
        # ══════════════════════════════════════════════════════════════════
        _l1_status = await get_level_status(pool, video_id, 1)
        if _l1_status == LevelStatus.DONE:
            logger.info("[L1] Already DONE — skipping.")
        else:
            await upsert_job(pool, ProcessingJob(
                id=gen_id(), video_id=video_id, level=1,
                status=LevelStatus.RUNNING, error_msg=None, meta={},
            ))
            try:
                l1_t = StepTimer()

                def _ashfs():
                    t0 = _t.perf_counter()
                    m = run_ashfs(local_video_path, video_id, "/tmp")
                    l1_t.record("ashfs", _t.perf_counter() - t0)
                    return m

                def _whisper():
                    t0 = _t.perf_counter()
                    segs = run_whisper(local_video_path, video_id, "/tmp")
                    l1_t.record("whisper", _t.perf_counter() - t0)
                    return [dataclasses.asdict(s) for s in segs]

                # Sequential: ASHFS/DINOv2 first, then Whisper.
                # Parallel causes Whisper CUDA OOM because vLLM reserves 61 GB
                # (16 GB model + 45 GB KV cache), leaving ~19 GB. DINOv2 retains
                # activation blocks in PyTorch's CUDA caching allocator after
                # processing, so CTranslate2 (Whisper) sees only ~9 GB free at
                # the driver level → OOM. gc.collect + empty_cache flushes those
                # blocks back to the driver before Whisper loads.
                with wall.step("l1_ashfs"):
                    manifest = await asyncio.to_thread(_ashfs)

                # Flush PyTorch CUDA allocator cache so CTranslate2 (faster-whisper)
                # can see the freed DINOv2 activation memory via cudaMalloc.
                try:
                    import gc as _gc
                    import torch as _torch
                    _gc.collect()
                    reserved_before = _torch.cuda.memory_reserved() / 1024**3
                    _torch.cuda.empty_cache()
                    reserved_after = _torch.cuda.memory_reserved() / 1024**3
                    logger.info(
                        "[GPU] Post-ASHFS cache flush: %.1f GiB → %.1f GiB reserved (freed %.1f GiB)",
                        reserved_before, reserved_after, reserved_before - reserved_after,
                    )
                except Exception as _flush_exc:
                    logger.warning("[GPU] Post-ASHFS cache flush failed (non-fatal): %s", _flush_exc)

                with wall.step("l1_whisper"):
                    segments_data = await asyncio.to_thread(_whisper)

                chunks, shots, keyframes = parse_ashfs_manifest(manifest, video_id)

                def _upload_frame(args: tuple) -> tuple[str, str | None]:
                    kf_id, fidx, vpath, vid = args
                    _c = cv2.VideoCapture(vpath)
                    _c.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret, frame = _c.read()
                    _c.release()
                    if not ret:
                        return kf_id, None
                    try:
                        return kf_id, r2_upload_frame(frame_to_png_bytes(frame), vid, fidx)
                    except Exception as _e:
                        logger.warning("[L1] R2 upload failed frame %d: %s", fidx, _e)
                        return kf_id, None

                upload_args = [(kf.id, kf.frame_index, local_video_path, video_id) for kf in keyframes]
                kf_id_to_r2: dict[str, str] = {}
                prog = ProgressTracker("R2 uploads", total=len(keyframes), logger=logger, log_every=50)
                with wall.step("l1_r2_upload"):
                    with _cf.ThreadPoolExecutor(max_workers=16) as _ex:
                        for kid, rk in _ex.map(_upload_frame, upload_args):
                            if rk:
                                kf_id_to_r2[kid] = rk
                            prog.update(1)
                prog.done()
                for kf in keyframes:
                    kf.r2_key = kf_id_to_r2.get(kf.id)

                segments = [
                    TranscriptSegment(
                        id=s["id"], video_id=s["video_id"], chunk_id=s.get("chunk_id"),
                        text=s["text"], start_time=float(s["start_time"]),
                        end_time=float(s["end_time"]), confidence=s.get("confidence"),
                        words=s.get("words") or [],
                    )
                    for s in segments_data
                ]
                segments = align_transcript_to_chunks(segments, chunks)

                with wall.step("l1_db_write"):
                    await bulk_insert_chunks(pool, chunks)
                    await bulk_insert_shots(pool, shots)
                    await bulk_insert_keyframes(pool, keyframes)
                    if segments:
                        await bulk_insert_transcript_segments(pool, segments)

                await update_job_status(pool, video_id, 1, LevelStatus.DONE)
                await update_job_meta(pool, video_id, 1, {
                    **l1_t.summary(), "chunks": len(chunks),
                    "shots": len(shots), "keyframes": len(keyframes),
                    "segments": len(segments), "r2_uploaded": len(kf_id_to_r2),
                })
                logger.info("[L1] Done: %d chunks %d shots %d kf %d segs", len(chunks), len(shots), len(keyframes), len(segments))

            except Exception as exc:
                logger.error("[L1] FAILED: %s", exc, exc_info=True)
                await update_job_status(pool, video_id, 1, LevelStatus.FAILED, str(exc))
                try:
                    os.unlink(local_video_path)
                except OSError:
                    pass
                return {"video_id": video_id, "status": "failed_at_level1", "error": str(exc)}

        # Release any PyTorch CUDA cache left by DINOv2/Whisper before L2 ONNX models load.
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
            logger.info("[GPU] CUDA cache cleared after L1")
        except Exception:
            pass

        # ══════════════════════════════════════════════════════════════════
        # L2 — Face + Color parallel
        # ══════════════════════════════════════════════════════════════════
        _l2_status = await get_level_status(pool, video_id, 2)
        if _l2_status == LevelStatus.DONE:
            logger.info("[L2] Already DONE — skipping.")
        else:
            await upsert_job(pool, ProcessingJob(
                id=gen_id(), video_id=video_id, level=2,
                status=LevelStatus.RUNNING, error_msg=None, meta={},
            ))
            try:
                l2_t = StepTimer()
                db_shots = await get_shots_for_video(pool, video_id)
                _kfs = await get_keyframes_for_video(pool, video_id)
                kf_ts = [kf.timestamp_s for kf in _kfs if kf.timestamp_s is not None]

                cast_db_path: str | None = None
                cast_db_tmp = None
                if cast_list:
                    from pipeline.level2.face_runner import build_cast_db_from_cast_list  # type: ignore[import]
                    cast_db_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
                    cast_db_tmp.close()
                    cast_db_path = cast_db_tmp.name
                    build_cast_db_from_cast_list(cast_list, cast_db_path)
                elif cast_db_r2_key:
                    cast_db_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
                    cast_db_tmp.write(download_bytes(cast_db_r2_key))
                    cast_db_tmp.close()
                    cast_db_path = cast_db_tmp.name

                # Pre-import torch/torchvision here to avoid the concurrent
                # first-import race between face_analysis and diarization
                # threads below — see comment in run_level2_modal.
                import torch as _torch_prewarm  # type: ignore[import]  # noqa: F401
                try:
                    import torchvision as _torchvision_prewarm  # type: ignore[import]  # noqa: F401
                except ImportError:
                    pass

                face_result: dict = {}
                color_result_list: list = []
                diar_turns: list[dict] = []
                num_speakers_hint = len(cast_list) if cast_list else None

                def _face():
                    t0 = _t.perf_counter()
                    r = run_face_analysis(
                        local_video_path, video_id, cast_db_path,
                        keyframe_timestamps=kf_ts,
                        shots=[dataclasses.asdict(s) for s in db_shots],
                    )
                    l2_t.record("face_analysis", _t.perf_counter() - t0)
                    return r

                def _color():
                    t0 = _t.perf_counter()
                    r = run_color_grading(local_video_path, video_id, db_shots)
                    l2_t.record("color_grading", _t.perf_counter() - t0)
                    return r

                def _diarize():
                    t0 = _t.perf_counter()
                    try:
                        r = run_diarization(
                            local_video_path, video_id, "/tmp", num_speakers=num_speakers_hint,
                        )
                    except Exception as _diar_exc:
                        logger.warning("[L2] Diarization failed (non-fatal): %s", _diar_exc)
                        r = []
                    l2_t.record("diarization", _t.perf_counter() - t0)
                    return r

                with wall.step("l2_face_color"):
                    try:
                        with _cf.ThreadPoolExecutor(max_workers=3) as _pool2:
                            _ff = _pool2.submit(_face)
                            _fc = _pool2.submit(_color)
                            _fd = _pool2.submit(_diarize)
                            face_result = _ff.result()
                            color_result_list = _fc.result()
                            diar_turns = _fd.result()
                    finally:
                        if cast_db_tmp:
                            for _ext in (".db", ".pkl"):
                                try:
                                    os.unlink(cast_db_tmp.name.removesuffix(".db") + _ext)
                                except OSError:
                                    pass

                # run_face_analysis and run_color_grading return dataclass objects
                # directly (unlike run_level2_modal which serialises to dicts).
                persons: list[PersonRecord] = face_result["persons"]
                appearances: list[FaceAppearanceRecord] = face_result["appearances"]
                timeline_events: list[FaceTimelineEvent] = face_result["timeline_events"]
                color_grades: list[ColorGradeRecord] = color_result_list
                speaker_turns: list[SpeakerTurnRecord] = fuse_diarization_with_faces(
                    video_id, diar_turns, appearances
                )

                with wall.step("l2_db_write"):
                    await write_level2_to_kb(
                        pool, video_id, persons, appearances, timeline_events, color_grades,
                        speaker_turns,
                    )

                await update_job_status(pool, video_id, 2, LevelStatus.DONE)
                await update_job_meta(pool, video_id, 2, {
                    **l2_t.summary(), "persons": len(persons),
                    "appearances": len(appearances), "color_grades": len(color_grades),
                    "speaker_turns": len(speaker_turns),
                })
                logger.info(
                    "[L2] Done: %d persons %d appearances %d color grades %d speaker turns",
                    len(persons), len(appearances), len(color_grades), len(speaker_turns),
                )

            except Exception as exc:
                logger.error("[L2] FAILED (non-fatal): %s", exc, exc_info=True)
                await update_job_status(pool, video_id, 2, LevelStatus.FAILED, str(exc))

        # Release ONNX/OpenCV CUDA allocations before Qwen inference begins.
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
            logger.info("[GPU] CUDA cache cleared after L2")
        except Exception:
            pass

        # ══════════════════════════════════════════════════════════════════
        # L3 — Qwen VL (pre-loaded in @modal.enter) + KB write
        # ══════════════════════════════════════════════════════════════════
        _l3_status = await get_level_status(pool, video_id, 3)
        if _l3_status == LevelStatus.DONE:
            logger.info("[L3] Already DONE — skipping.")
            try:
                os.unlink(local_video_path)
            except OSError:
                pass
            return {"video_id": video_id, "status": "complete", "frames_analysed": 0, "wall_timings": wall.summary()["steps"]}

        await upsert_job(pool, ProcessingJob(
            id=gen_id(), video_id=video_id, level=3,
            status=LevelStatus.RUNNING, error_msg=None, meta={},
        ))
        frames_analysed = 0
        try:
            l3_t = StepTimer()

            with wall.step("l3_db_load"):
                keyframes = await get_keyframes_for_video(pool, video_id)
                shots = await get_shots_for_video(pool, video_id)
                transcript_segs = await get_transcript_segments_for_video(pool, video_id)
                persons = await get_persons_for_video(pool, video_id)
                appearances = await get_face_appearances_for_video_sampled(pool, video_id)

            if not keyframes:
                raise RuntimeError(f"No keyframes in DB for video_id={video_id}. Run L1 first.")

            existing = await get_existing_frame_analyses(pool, [kf.id for kf in keyframes])
            kf_map = {kf.id: kf for kf in keyframes}

            if len(existing) == len(keyframes):
                logger.info("[L3] All %d keyframes already analysed — loading from DB", len(keyframes))
                frame_results = [(kf_map[kid], out) for kid, out in existing.items()]
            else:
                already_done = set(existing.keys())
                missing_kfs = [kf for kf in keyframes if kf.id not in already_done]
                logger.info(
                    "[L3] %d/%d keyframes need Qwen inference (%d already in DB)",
                    len(missing_kfs), len(keyframes), len(already_done),
                )
                contexts = build_all_frame_contexts(missing_kfs, transcript_segs, appearances, persons)

                # Chunk inference so very long videos (1000+ keyframes) write
                # progress to DB periodically.  If the container is interrupted,
                # the next run skips already-completed frames via existing check.
                _L3_CHUNK = 300
                chunk_results: list[tuple] = []
                with wall.step("l3_qwen_inference"):
                    for _ci in range(0, len(contexts), _L3_CHUNK):
                        _chunk_ctx = contexts[_ci: _ci + _L3_CHUNK]
                        logger.info(
                            "[L3] Qwen chunk %d-%d / %d",
                            _ci, min(_ci + _L3_CHUNK, len(contexts)) - 1, len(contexts),
                        )
                        _chunk_res = await run_qwen_analysis(_chunk_ctx)
                        chunk_results.extend(_chunk_res)

                        # Write frame_analyses for this chunk immediately so
                        # progress survives container interruption.
                        _partial_analyses = _build_frame_analysis_records(_chunk_res)
                        if _partial_analyses:
                            try:
                                await bulk_insert_frame_analyses(pool, _partial_analyses)
                                logger.info("[L3] Checkpoint: wrote %d frame_analyses to DB", len(_partial_analyses))
                            except Exception as _ce:
                                logger.warning("[L3] Checkpoint write failed (non-fatal): %s", _ce)

                # Combine newly inferred + already-existing results for finalize.
                frame_results = chunk_results + [(kf_map[kid], out) for kid, out in existing.items()]

            with wall.step("l3_kb_finalize"):
                await finalize_level3_kb(pool, video_id, frame_results, persons, shots)

            frames_analysed = sum(1 for _, out in frame_results if out is not None)
            await update_job_status(pool, video_id, 3, LevelStatus.DONE)
            await update_job_meta(pool, video_id, 3, {
                **l3_t.summary(), "frames_analysed": frames_analysed,
                "keyframes_total": len(keyframes),
            })
            logger.info("[L3] Done: %d frames analysed", frames_analysed)

        except Exception as exc:
            logger.error("[L3] FAILED: %s", exc, exc_info=True)
            await update_job_status(pool, video_id, 3, LevelStatus.FAILED, str(exc))
            try:
                os.unlink(local_video_path)
            except OSError:
                pass
            return {"video_id": video_id, "status": "failed_at_level3", "error": str(exc)}

        try:
            os.unlink(local_video_path)
        except OSError:
            pass

        wall_summary = wall.summary()
        logger.info(
            "[SingleGPU] Pipeline complete video_id=%s frames=%d total=%.0fs",
            video_id, frames_analysed, wall_summary["total_s"],
        )
        return {
            "video_id": video_id,
            "status": "complete",
            "frames_analysed": frames_analysed,
            "wall_timings": wall_summary["steps"],
            "total_s": wall_summary["total_s"],
        }


@app.local_entrypoint()
def main_single_gpu(
    input_json: str,
    video_id: str = "",
) -> None:
    """Run the full pipeline in a single A100-80GB container.

    Usage::

        modal run modal_app.py::main_single_gpu --input-json input.json
        modal run modal_app.py::main_single_gpu --input-json input.json --video-id <id>
    """
    import json as _json

    with open(input_json, encoding="utf-8") as f:
        data = _json.load(f)

    video_url: str = data.get("video", {}).get("url", "")
    if not video_url:
        raise ValueError("input JSON missing video.url")

    raw_cast: list[dict] = data.get("cast", [])
    cast_list: list[dict] | None = (
        [
            {
                "name": e["name"],
                "images": [e["reference_image"]] if "reference_image" in e else e.get("images", []),
            }
            for e in raw_cast if e.get("name")
        ]
        or None
    )

    if not video_id:
        import hashlib as _hl
        video_id = _hl.sha256(video_url.encode()).hexdigest()[:32]

    print(f"[single-gpu] Video    : {video_url}")
    print(f"[single-gpu] video_id : {video_id}  (stable hash — safe to re-run with --video-id {video_id})")
    print(f"[single-gpu] Cast     : {[c['name'] for c in cast_list] if cast_list else 'none'}")

    pipeline = SingleGPUPipeline()
    result = pipeline.run.remote(video_url, video_id, None, cast_list)
    print(result)

    timings = result.get("wall_timings") or {}
    if timings:
        print("\n[single-gpu] Timing breakdown:")
        for step, secs in timings.items():
            print(f"  {step:40s} {secs:6.0f}s  ({secs/60:.1f} min)")
        total = result.get("total_s", 0)
        print(f"  {'TOTAL':40s} {total:6.0f}s  ({total/60:.1f} min)")


