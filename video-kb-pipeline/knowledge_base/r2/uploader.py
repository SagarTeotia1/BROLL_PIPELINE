from __future__ import annotations

import io
import json
import logging

from shared.utils import retry
from knowledge_base.r2.client import get_r2_client, get_bucket_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


@retry(max_attempts=3, wait_seconds=2.0)
def upload_frame(frame_bytes: bytes, video_id: str, frame_index: int) -> str:
    """Upload a PNG frame to R2 and return its object key.

    The key follows the convention::

        videos/{video_id}/frames/{frame_index:08d}.png

    Args:
        frame_bytes: Raw PNG bytes (e.g. from :func:`shared.utils.frame_to_png_bytes`).
        video_id: The video's unique identifier.
        frame_index: Zero-based frame number.

    Returns:
        The R2 object key for the uploaded frame.
    """
    key = f"videos/{video_id}/frames/{frame_index:08d}.png"
    client = get_r2_client()
    bucket = get_bucket_name()

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=frame_bytes,
        ContentType="image/png",
    )
    logger.debug("Uploaded frame → %s (%d bytes)", key, len(frame_bytes))
    return key


@retry(max_attempts=3, wait_seconds=2.0)
def upload_manifest(manifest: dict, video_id: str) -> str:
    """Upload the video manifest JSON to R2 and return its object key.

    The key is ``videos/{video_id}/manifest.json``.

    Args:
        manifest: A dict that will be JSON-serialised.
        video_id: The video's unique identifier.

    Returns:
        The R2 object key for the uploaded manifest.
    """
    key = f"videos/{video_id}/manifest.json"
    body = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    client = get_r2_client()
    bucket = get_bucket_name()

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    logger.debug("Uploaded manifest → %s (%d bytes)", key, len(body))
    return key


@retry(max_attempts=3, wait_seconds=2.0)
def upload_json(data: dict, video_id: str, name: str) -> str:
    """Upload an arbitrary JSON payload to R2 and return its object key.

    The key is ``videos/{video_id}/{name}.json``.

    Args:
        data: A dict that will be JSON-serialised.
        video_id: The video's unique identifier.
        name: File stem (without ``.json`` extension).

    Returns:
        The R2 object key for the uploaded file.
    """
    key = f"videos/{video_id}/{name}.json"
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    client = get_r2_client()
    bucket = get_bucket_name()

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    logger.debug("Uploaded JSON → %s (%d bytes)", key, len(body))
    return key


@retry(max_attempts=3, wait_seconds=2.0)
def download_bytes(key: str) -> bytes:
    """Download an object from R2 and return its raw bytes.

    Args:
        key: The R2 object key to download.

    Returns:
        The raw byte content of the object.
    """
    client = get_r2_client()
    bucket = get_bucket_name()

    response = client.get_object(Bucket=bucket, Key=key)
    data: bytes = response["Body"].read()
    logger.debug("Downloaded %s (%d bytes)", key, len(data))
    return data


@retry(max_attempts=3, wait_seconds=2.0)
def get_presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate a presigned GET URL for an R2 object.

    Args:
        key: The R2 object key.
        expires_in: URL expiry time in seconds (default: 3600 = 1 hour).

    Returns:
        A time-limited presigned HTTPS URL that allows anyone to GET the object.
    """
    client = get_r2_client()
    bucket = get_bucket_name()

    url: str = client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
    logger.debug("Generated presigned URL for %s (expires_in=%ds)", key, expires_in)
    return url
