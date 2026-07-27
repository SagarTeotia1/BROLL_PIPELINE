from __future__ import annotations

import logging

import boto3
import boto3.session
from botocore.client import BaseClient

from shared.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_r2_client: BaseClient | None = None


def get_r2_client() -> BaseClient:
    """Return the module-level boto3 S3-compatible client for Cloudflare R2.

    The client is created on the first call and cached for the lifetime of the
    process.  R2 uses S3-compatible APIs, so we use ``boto3.client("s3")`` with
    a custom ``endpoint_url``.

    Returns:
        A boto3 ``S3Client`` configured for Cloudflare R2.
    """
    global _r2_client

    if _r2_client is not None:
        return _r2_client

    logger.info("Creating R2 client → endpoint: %s", settings.R2_ENDPOINT_URL)
    _r2_client = boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=boto3.session.Config(signature_version="s3v4"),
    )
    return _r2_client


def get_bucket_name() -> str:
    """Return the R2 bucket name from settings."""
    return settings.R2_BUCKET_NAME
