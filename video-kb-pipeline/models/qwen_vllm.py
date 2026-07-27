from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# vLLM conditional import
# All vLLM symbols are imported inside the try block so that the module can be
# imported on machines without a GPU / vLLM installation (e.g. the orchestrator
# container or a developer laptop running tests).
# ---------------------------------------------------------------------------
try:
    from vllm import LLM, SamplingParams  # type: ignore[import]

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed — QwenVLLM will not function locally")


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _make_sampling_params() -> "SamplingParams":
    return SamplingParams(
        temperature=0.0,  # greedy — fastest decoding, fully deterministic
        max_tokens=2048,  # some frames hit 1400-1500 tokens; 2048 is safe ceiling
        stop=None,
    )


def _strip_markdown_fences(raw: str) -> str:
    """Strip markdown code fences from Qwen output before JSON parse.

    Handles cases where the model wraps its JSON in ```json ... ``` or ``` ... ```.
    """
    stripped = raw.strip()
    # Remove opening fence variants
    stripped = stripped.removeprefix("```json").removeprefix("```")
    # Remove closing fence
    stripped = stripped.removesuffix("```")
    return stripped.strip()


class QwenVLLM:
    """Singleton vLLM engine for Qwen VL inference.

    Uses ``vllm.LLM.chat()`` which natively handles multimodal conversations
    (text + PIL images) for Qwen2.5-VL without requiring manual tokenisation
    or pixel-value preparation.  Inference runs synchronously on the calling
    thread; async callers should use ``analyse_batch`` which delegates to
    ``asyncio.to_thread``.
    """

    _instance: "QwenVLLM | None" = None
    _llm: "LLM | None" = None  # type annotation as string to avoid NameError

    @classmethod
    def get(cls) -> "QwenVLLM":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        """Initialize the LLM engine.  Call once at Modal container startup."""
        if self._llm is not None:
            logger.debug("QwenVLLM.load() called but engine already loaded — skipping.")
            return
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed")

        self._llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.92,
            max_model_len=16384,
            max_num_seqs=16,           # L40s 48GB: model=16GB, 32GB free → 16 concurrent seqs
            enable_prefix_caching=True,
            enable_chunked_prefill=True,  # better GPU utilisation for mixed-length batches
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "max_pixels": 602112,   # ~800×750 — faster tokenisation, still high detail
                "min_pixels": 200704,   # 256 * 28 * 28
            },
            trust_remote_code=True,
        )
        logger.info("Qwen VL LLM engine loaded: %s", MODEL_ID)

        # Triton warmup: run a dummy inference to pre-compile JIT kernels
        # (_compute_slot_mapping_kernel, _bilinear_pos_embed_kernel) so they
        # don't spike latency on the first real inference batch.
        self._run_triton_warmup()

        # Load the local embedding model on the same GPU so kb_finalizer can
        # embed searchable facts and scene captions without any external API calls.
        from models.embed_model import LocalEmbedder

        LocalEmbedder.get().load()
        logger.info("QwenVLLM and LocalEmbedder both loaded")

    def _run_triton_warmup(self) -> None:
        """Submit one tiny dummy request to pre-compile all Triton JIT kernels.

        Without this, _compute_slot_mapping_kernel and _bilinear_pos_embed_kernel
        compile during the first real inference batch, causing a 30-60s latency spike.
        Running a dummy call here moves that cost into load() where it belongs.
        """
        import io as _io
        import base64 as _b64
        import numpy as _np
        try:
            from PIL import Image as _PIL
        except ImportError:
            logger.warning("PIL not available — skipping Triton warmup")
            return

        # Tiny 64×64 blank image — minimum pixels that exercises the vision encoder
        blank = _PIL.fromarray(_np.zeros((64, 64, 3), dtype=_np.uint8))
        buf = _io.BytesIO()
        blank.save(buf, format="JPEG", quality=70)
        data_url = "data:image/jpeg;base64," + _b64.b64encode(buf.getvalue()).decode()

        warmup_messages = [[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Reply with one word."},
            ]},
        ]]
        warmup_params = SamplingParams(temperature=0.0, max_tokens=3, stop=None)
        logger.info("QwenVLLM: running Triton warmup inference ...")
        try:
            self._llm.chat(warmup_messages, sampling_params=warmup_params, use_tqdm=False)
            logger.info("QwenVLLM: Triton warmup complete — kernels compiled and cached")
        except Exception as exc:
            logger.warning("QwenVLLM: Triton warmup failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Synchronous batch inference — safe to call from a thread
    # ------------------------------------------------------------------

    def analyse_batch_sync(self, batch: list[dict]) -> list[str]:
        """Run inference synchronously over a batch of frames.

        Each item in *batch* must have:
        - ``request_id`` (str): unique identifier for logging / ordering.
        - ``messages``   (list[dict]): OpenAI-compatible messages list.
          The user message content must already contain a PIL ``Image``
          object embedded via the ``"image"`` type content block, OR the
          image URL is present as an ``image_url`` type block.  vLLM's
          ``LLM.chat()`` accepts both formats for Qwen2.5-VL.
        - ``image``      (PIL.Image.Image | None): PIL image for this frame.
          When provided it is injected as ``multi_modal_data`` so vLLM
          can process pixel values correctly alongside the chat template.

        Returns:
            A list of raw model output strings (markdown fences stripped),
            in the same order as *batch*.  Failed frames return an empty
            string — callers should skip those.
        """
        if self._llm is None:
            raise RuntimeError("LLM engine not loaded — call load() first")

        sampling_params = _make_sampling_params()

        # Build conversation list.  For vLLM's LLM.chat() the messages are
        # passed directly in OpenAI chat format; PIL images embedded in the
        # content list are handled natively by Qwen2.5-VL's processor.
        conversations: list[list[dict]] = []
        for item in batch:
            conversations.append(item["messages"])

        # Submit the full batch — if the whole batch call fails, fall back to
        # per-frame single-item calls so one bad frame doesn't zero out the rest.
        try:
            outputs = self._llm.chat(
                conversations,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except Exception as exc:
            logger.error(
                "LLM.chat() batch failed (%d frames) — retrying per-frame: %s",
                len(batch),
                exc,
                exc_info=True,
            )
            # Per-frame fallback: attempt each conversation individually so that
            # one bad frame does not cause every frame in the batch to return "".
            results: list[str] = []
            for item in batch:
                try:
                    single_out = self._llm.chat(
                        [item["messages"]],
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                    raw = single_out[0].outputs[0].text
                    results.append(_strip_markdown_fences(raw))
                except Exception as single_exc:
                    logger.error(
                        "Per-frame LLM.chat() failed for request_id=%s: %s",
                        item.get("request_id", "?"),
                        single_exc,
                    )
                    results.append("")
            return results

        results = []
        for i, output in enumerate(outputs):
            try:
                raw = output.outputs[0].text
                results.append(_strip_markdown_fences(raw))
            except Exception as exc:
                logger.error(
                    "Failed to extract text for request_id=%s: %s",
                    batch[i].get("request_id", "?"),
                    exc,
                )
                results.append("")

        return results

    # ------------------------------------------------------------------
    # Async wrapper — delegates to thread so the event loop is not blocked
    # ------------------------------------------------------------------

    async def analyse_batch(
        self,
        batch: list[dict],  # list of {request_id, messages, image}
    ) -> list[str]:
        """Async wrapper around analyse_batch_sync.

        Runs synchronous vLLM inference on a thread-pool worker so the
        asyncio event loop is not blocked during GPU inference.

        Returns:
            List of raw JSON strings (markdown fences stripped) in the same
            order as *batch*.  Empty strings indicate failed frames.
        """
        return await asyncio.to_thread(self.analyse_batch_sync, batch)
