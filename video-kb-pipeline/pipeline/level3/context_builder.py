from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from shared.types import (
    FaceAppearanceRecord,
    KeyframeRecord,
    PersonRecord,
    TranscriptSegment,
)
from shared.utils import get_transcript_snippet

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)


@dataclass
class FrameContext:
    keyframe: KeyframeRecord
    messages: list[dict]
    request_id: str
    person_ids: list[str]
    # PIL image downloaded from R2; None means the frame was skipped or the
    # image could not be fetched.  analyse_batch_sync uses this for inference.
    image: "PILImage.Image | None" = field(default=None)


def _download_frame_image(keyframe: KeyframeRecord) -> "PILImage.Image":
    """Download a keyframe PNG from R2 and return it as a PIL RGB Image.

    Args:
        keyframe: The KeyframeRecord whose r2_key identifies the object in R2.

    Raises:
        ValueError: If the keyframe has no r2_key.
        Exception:  Propagates any R2 / network errors.
    """
    from PIL import Image
    from knowledge_base.r2.uploader import download_bytes

    if keyframe.r2_key is None:
        raise ValueError(
            f"Keyframe {keyframe.id} has no r2_key — upload frames to R2 first"
        )

    raw = download_bytes(keyframe.r2_key)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def build_frame_image_url(keyframe: KeyframeRecord) -> str:
    """Get presigned R2 URL for the keyframe PNG.

    Kept for backwards-compatibility with callers that still need a URL (e.g.
    non-vLLM inference paths).  Prefer ``_download_frame_image`` for the
    primary vLLM pipeline.

    Raises:
        ValueError: If the keyframe has no r2_key (frame not yet uploaded to R2).
    """
    from knowledge_base.r2.uploader import get_presigned_url

    if keyframe.r2_key is None:
        raise ValueError(
            f"Keyframe {keyframe.id} has no r2_key — upload frames to R2 first"
        )
    return get_presigned_url(keyframe.r2_key, expires_in=3600)


def get_persons_in_frame(
    frame_index: int,
    appearances: list[FaceAppearanceRecord],
    persons_by_id: dict[str, PersonRecord],
) -> tuple[list[str], list[dict]]:
    """Return (pid_list, emotion_list) for faces visible in this frame.

    Args:
        frame_index: The integer frame index to filter appearances on.
        appearances: All face appearance records for the video.
        persons_by_id: Map from PersonRecord.id (DB uuid) to PersonRecord.

    Returns:
        A tuple of:
        - List of pid strings (e.g. ["P1", "P2"]) for unique persons in this frame.
        - List of emotion dicts: [{"pid": str, "emotion": str, "confidence": float}]
          only for persons that have an emotion label in this appearance.
    """
    frame_appearances = [a for a in appearances if a.frame_index == frame_index]
    pids: list[str] = []
    emotions: list[dict] = []
    seen_pids: set[str] = set()

    for app in frame_appearances:
        person = persons_by_id.get(app.person_id) if app.person_id else None
        if person and person.pid not in seen_pids:
            pids.append(person.pid)
            seen_pids.add(person.pid)
            if app.emotion:
                emotions.append(
                    {
                        "pid": person.pid,
                        "emotion": app.emotion,
                        "confidence": app.emotion_conf or 0.0,
                    }
                )

    return pids, emotions


def build_all_frame_contexts(
    keyframes: list[KeyframeRecord],
    transcript_segments: list[TranscriptSegment],
    appearances: list[FaceAppearanceRecord],
    persons: list[PersonRecord],
    rolling_summary: str = "",
    prev_scene_id: str | None = None,
) -> list[FrameContext]:
    """Build context for every keyframe in timestamp order.

    Downloads each keyframe's PNG from R2 as a PIL Image for vLLM multimodal
    inference.  Frames that do not have an r2_key, or whose image cannot be
    fetched, are skipped with a warning.

    The rolling summary and prev_scene_id are static for this batch — callers
    are responsible for updating them between shots/scenes when calling this
    function for successive shot groups.

    Args:
        keyframes: All keyframes to process.
        transcript_segments: Transcript segments used to build per-frame snippets.
        appearances: Face appearance records used to resolve person identity.
        persons: All PersonRecord instances for this video.
        rolling_summary: Compressed narrative of the previous 1-2 scenes.
        prev_scene_id: scene_id of the most recently completed scene, if any.

    Returns:
        List of FrameContext in ascending timestamp order, skipping frames
        whose image could not be obtained.
    """
    # Build lookup by DB id (UUID) → PersonRecord
    persons_by_id: dict[str, PersonRecord] = {p.id: p for p in persons}

    sorted_kfs = sorted(keyframes, key=lambda k: k.timestamp_s)

    # Pre-compute per-frame text context (fast, no I/O)
    kf_meta: dict[str, dict] = {}
    for kf in sorted_kfs:
        snippet = get_transcript_snippet(transcript_segments, kf.timestamp_s, window=10.0)
        pids, emotions = get_persons_in_frame(kf.frame_index, appearances, persons_by_id)
        kf_meta[kf.id] = {"snippet": snippet, "pids": pids, "emotions": emotions}

    # Download all keyframe PNGs from R2 in parallel (16 threads)
    def _fetch(kf: "KeyframeRecord"):
        try:
            return kf, _download_frame_image(kf)
        except ValueError as exc:
            logger.warning("Skipping frame %s (no r2_key): %s", kf.id, exc)
            return kf, None
        except Exception as exc:
            logger.warning("Skipping frame %s (R2 download failed): %s", kf.id, exc)
            return kf, None

    images: dict[str, "PILImage.Image"] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for kf, img in pool.map(_fetch, sorted_kfs):
            if img is not None:
                images[kf.id] = img

    logger.info(
        "Downloaded %d/%d keyframe images from R2", len(images), len(sorted_kfs)
    )

    contexts: list[FrameContext] = []
    for kf in sorted_kfs:
        pil_image = images.get(kf.id)
        if pil_image is None:
            continue
        meta = kf_meta[kf.id]
        messages = _build_messages_with_pil(
            frame_id=kf.id,
            timestamp=kf.timestamp_s,
            pil_image=pil_image,
            person_ids=meta["pids"],
            person_emotions=meta["emotions"],
            transcript_snippet=meta["snippet"],
            prev_scene_summary=rolling_summary,
            prev_scene_id=prev_scene_id,
        )
        contexts.append(FrameContext(
            keyframe=kf,
            messages=messages,
            request_id=kf.id,
            person_ids=meta["pids"],
            image=pil_image,
        ))

    return contexts


def _pil_to_data_url(img: "PILImage.Image") -> str:
    """Encode a PIL Image as a JPEG base64 data URL for vLLM image_url content."""
    import base64
    import io as _io

    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _build_messages_with_pil(
    frame_id: str,
    timestamp: float,
    pil_image: "PILImage.Image",
    person_ids: list[str],
    person_emotions: list[dict],
    transcript_snippet: str,
    prev_scene_summary: str,
    prev_scene_id: str | None,
) -> list[dict]:
    """Build OpenAI-compatible messages list with an image_url content block.

    vLLM 0.23+ dropped the bare ``"image"`` content type.  We encode the PIL
    image as a JPEG base64 data URL and pass it as ``image_url``, which is
    supported by all vLLM versions and Qwen3-VL's processor.
    """
    from prompts.qwen_v2 import QWEN_SYSTEM_PROMPT, build_user_message

    user_text = build_user_message(
        frame_id=frame_id,
        timestamp=timestamp,
        person_ids=person_ids,
        person_emotions=person_emotions,
        transcript_snippet=transcript_snippet,
        prev_scene_summary=prev_scene_summary,
        prev_scene_id=prev_scene_id,
    )
    data_url = _pil_to_data_url(pil_image)
    return [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
