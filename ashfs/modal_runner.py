"""ASH-FS on Modal — NVIDIA L4 GPU (Ada Lovelace, 24 GB).

No TensorFlow. No TransNetV2. Pure PyTorch CUDA stack.

Setup (one-time):
    pip install modal
    modal setup   # authenticate with feedbackteam177 workspace

Upload video (one-time):
    modal volume create ashfs-data
    modal volume put ashfs-data output/podcast.mp4 /podcast.mp4

Run:
    modal run modal_runner.py
"""
from __future__ import annotations

import modal
from pathlib import Path

ASHFS_DIR = Path(__file__).parent

app = modal.App("ashfs-pipeline")

# Persistent volumes
data_vol = modal.Volume.from_name("ashfs-data", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("ashfs-model-cache", create_if_missing=True)

# CUDA 12.4.1 base — matches torch 2.4.1+cu124 exactly, no CUDA conflicts.
# xFormers accelerates DINOv2 attention on Ada (L4 gains ~30% throughput).
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
    )
    .pip_install(
        "torch==2.4.1+cu124",
        "torchvision==0.19.1+cu124",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("xformers==0.0.27.post2")  # matches torch 2.4.1, auto-detects CUDA
    .pip_install(
        "scenedetect[opencv]",
        "numpy>=2.0,<2.8",
        "scipy",
        "pyyaml",
        "tqdm",
        "Pillow",
        "opencv-python-headless",
    )
    .add_local_dir(str(ASHFS_DIR / "ashfs"), remote_path="/root/ashfs")
    .add_local_file(str(ASHFS_DIR / "config.yaml"), remote_path="/root/config.yaml")
)


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/data": data_vol,
        "/root/.cache/torch/hub": model_cache_vol,  # cache DINOv2 weights across runs
    },
    timeout=3600,
    memory=32768,  # L4 has 124 GB host RAM available — 32 GB safe headroom
)
def run_pipeline(video_filename: str = "podcast.mp4") -> dict:
    import sys
    import json
    import logging
    import time

    sys.path.insert(0, "/root")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    import torch

    torch_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if torch_gpu else "N/A"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch_gpu else 0
    print(f"\n[GPU] PyTorch CUDA : {torch_gpu} — {device_name}  ({vram_gb:.1f} GB VRAM)")

    # Verify xFormers loaded
    try:
        import xformers
        print(f"[GPU] xFormers     : {xformers.__version__} (memory-efficient attention enabled)")
    except ImportError:
        print("[GPU] xFormers     : not available (will use standard attention)")

    from ashfs.sampler_pipeline import SamplerPipeline

    video_path = f"/data/{video_filename}"
    config_path = "/root/config.yaml"

    t_start = time.time()
    pipeline = SamplerPipeline(config_path)
    manifest = pipeline.run(video_path)
    elapsed = time.time() - t_start

    out_path = f"/data/manifest_l4_{int(t_start)}.json"
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    data_vol.commit()

    cand = manifest.get("total_candidate_frames", 0)
    final = manifest.get("total_selected_frames", 0)
    true_cr = cand / final if final else 0.0

    print(f"\n{'='*56}")
    print("  ASH-FS Results (Modal L4 — Ada Lovelace 24 GB)")
    print(f"{'='*56}")
    print(f"  Shots detected  : {manifest['total_shots']}")
    print(f"  Candidate frames: {cand}")
    print(f"  Keyframes       : {final}")
    print(f"  Compression     : {true_cr:.2f}x  (candidate→final)")
    print(f"  Wall-clock      : {elapsed:.1f}s")
    if manifest.get("fallback"):
        print(f"  WARNING: fallback used — {manifest.get('fallback_error')}")
    print(f"\n  Timing breakdown:")
    for stage, t in manifest.get("timing", {}).items():
        print(f"    {stage:<35}: {t:.2f}s")
    print(f"\n  Manifest → {out_path}")
    print(f"{'='*56}\n")

    return {
        "wall_clock_s": round(elapsed, 2),
        "total_shots": manifest["total_shots"],
        "total_candidate_frames": cand,
        "total_selected_frames": final,
        "compression_ratio": round(true_cr, 2),
        "timing": manifest.get("timing", {}),
        "fallback": manifest.get("fallback", False),
    }


@app.local_entrypoint()
def main(
    video: str = "output/podcast.mp4",
    filename: str = "podcast.mp4",
):
    """Upload video to Modal volume then run pipeline on L4."""
    video_path = Path(video)
    if not video_path.exists():
        print(f"ERROR: {video_path} not found locally.")
        return

    print(f"Video : {video_path}  ({video_path.stat().st_size / 1e6:.0f} MB)")
    print(f"Target: /data/{filename} on volume ashfs-data")
    print()
    print("If not yet uploaded, run:")
    print(f"  modal volume put ashfs-data {video} /{filename}")
    print()
    print("Starting pipeline on L4...")
    result = run_pipeline.remote(filename)
    print("\nResult:")
    for k, v in result.items():
        if k != "timing":
            print(f"  {k}: {v}")
