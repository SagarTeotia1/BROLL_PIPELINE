"""Resumable HTTP downloads + archive extraction for the model zoo.

Model weights are not vendored in the repository; :func:`fetch` pulls them once into
``models/weights`` and every later run hits the local copy. Downloads are atomic
(``.part`` file then rename) so an interrupted run never leaves a corrupt ONNX file.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Optional, Sequence

import requests

from utils.logging_utils import get_logger

log = get_logger(__name__)

ProgressFn = Callable[[int, int], None]

_CHUNK = 1 << 20  # 1 MiB


def sha256sum(path: Path, chunk: int = _CHUNK) -> str:
    """Streamed SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    destination: Path,
    expected_sha256: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    timeout: int = 60,
) -> Path:
    """Download ``url`` to ``destination`` atomically.

    Args:
        url: source URL.
        destination: final file path (parent directories are created).
        expected_sha256: verified after download when provided.
        progress: called as ``progress(downloaded_bytes, total_bytes)``.
        timeout: per-request connect/read timeout in seconds.

    Raises:
        RuntimeError: on HTTP failure or checksum mismatch.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    log.info("Downloading %s -> %s", url, destination.name)
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with part.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {url}: {exc}") from exc

    if expected_sha256:
        digest = sha256sum(part)
        if digest.lower() != expected_sha256.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {url}: expected {expected_sha256}, got {digest}"
            )

    part.replace(destination)
    log.info("Saved %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
    return destination


def download_first_available(
    urls: Sequence[str],
    destination: Path,
    expected_sha256: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Try mirrors in order; return on the first success."""
    errors: list[str] = []
    for url in urls:
        try:
            return download(url, destination, expected_sha256, progress)
        except RuntimeError as exc:
            log.warning("Mirror failed: %s", exc)
            errors.append(str(exc))
    raise RuntimeError(
        "all mirrors failed for " + destination.name + ":\n  " + "\n  ".join(errors)
    )


def extract_member(
    archive: Path, member_suffix: str, destination: Path
) -> Path:
    """Pull a single file out of a zip archive by filename suffix.

    Args:
        archive: path to the ``.zip``.
        member_suffix: e.g. ``"det_10g.onnx"``; matched case-insensitively against
            the end of each archive member name.
        destination: where the extracted file is written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        target = None
        for member in zf.namelist():
            if member.lower().endswith(member_suffix.lower()):
                target = member
                break
        if target is None:
            raise RuntimeError(
                f"{member_suffix} not found in {archive.name}; members={zf.namelist()}"
            )
        with zf.open(target) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=_CHUNK)
    log.info("Extracted %s from %s", destination.name, archive.name)
    return destination


def human_bytes(n: float) -> str:
    """Format a byte count for logs/progress bars."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


__all__ = [
    "sha256sum",
    "download",
    "download_first_available",
    "extract_member",
    "human_bytes",
]
