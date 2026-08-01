"""Pexels stock-video API client (A2 — knowledge_base/stock_assets/).

Standalone infrastructure: build once, query later by L6's Compositing
Agent (built separately — see CLAUDE.md "PIPELINE ADDENDUM" → A2/A3).

Uses the Pexels Video Search API (free tier):
    GET https://api.pexels.com/videos/search?query=<q>&per_page=<n>&page=<p>
    Header: Authorization: <PEXELS_API_KEY>

Docs: https://www.pexels.com/api/documentation/#videos-search

ASSUMPTION (flagged per task instructions — no live API call was made to
verify this): the response shape modeled below (`PexelsVideoFile`,
`PexelsVideo`, `PexelsSearchResponse`) matches Pexels' public API
documentation as of this writing. A real `PEXELS_API_KEY` and a live call
should be used to confirm field names/nesting (particularly `video_files[]`
entry shape and whether `tags` is ever populated — Pexels' documented
schema does not guarantee a populated `tags` list on every video) before
this client is trusted in production.

Pexels' free tier license: videos are free to use (including for
commercial purposes) without attribution required, per Pexels' license —
but see CLAUDE.md "Before building A3: stock licensing": confirm the exact
tier/redistribution terms cover a paid client's final delivered video
before enforcement is wired up. `license_type` is recorded on every
`StockAssetRecord`/`stock_assets` row (currently `"pexels_free"`, a
placeholder label) precisely so this can be filtered/audited later without
re-ingesting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import requests

from shared.config import settings
from shared.utils import retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.pexels.com/videos/search"
_DEFAULT_TIMEOUT_S = 15
_DEFAULT_PER_PAGE = 20

# Placeholder license label for everything ingested from Pexels' free tier.
# See CLAUDE.md "Before building A3: stock licensing" — enforcement (i.e.
# actually gating what the Compositing Agent may pick based on this value)
# is deferred; this constant only records provenance.
PEXELS_LICENSE_TYPE = "pexels_free"


# ---------------------------------------------------------------------------
# Normalized asset shape returned by this client
# ---------------------------------------------------------------------------


@dataclass
class NormalizedStockAsset:
    """Normalized stock-video metadata, source-agnostic.

    This is the shape `indexer.py` consumes to build `StockAssetRecord`
    rows — kept separate from the raw Pexels response so a future
    Storyblocks/internal-library client can produce the same shape without
    `indexer.py` needing to know which source it came from.
    """

    source: str                    # "pexels" | "storyblocks" | "internal"
    external_id: str
    description: str               # Pexels calls this "alt" text — see note below
    tags: list[str] = field(default_factory=list)
    download_url: str | None = None   # best-quality `video_files[].link`
    license_type: str = PEXELS_LICENSE_TYPE
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None


# ---------------------------------------------------------------------------
# Pexels client
# ---------------------------------------------------------------------------


class PexelsClient:
    """Thin client over the Pexels Video Search API.

    `PEXELS_API_KEY` is validated as optional/nullable at import time in
    `shared/config.py` (the "internal" source needs no API key) — this
    class raises at *call* time, not at import, if a search is attempted
    without a configured key, so importing this module never crashes a
    process that only ever ingests internal/curated assets.
    """

    def __init__(self, api_key: str | None = None, timeout_s: int = _DEFAULT_TIMEOUT_S) -> None:
        self._api_key = api_key if api_key is not None else settings.PEXELS_API_KEY
        self._timeout_s = timeout_s

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise RuntimeError(
                "PEXELS_API_KEY is not configured. Set it in the environment "
                "(see .env.example) before calling PexelsClient.search()."
            )
        return self._api_key

    @retry(max_attempts=3, wait_seconds=2.0)
    def search(
        self,
        query: str,
        per_page: int = _DEFAULT_PER_PAGE,
        page: int = 1,
    ) -> list[NormalizedStockAsset]:
        """Search Pexels for stock videos matching *query*.

        Args:
            query: Free-text search query (e.g. scene mood/tags/keywords).
            per_page: Results per page (Pexels max is 80).
            page: 1-indexed page number.

        Returns:
            Normalized asset metadata — id, description, tags (if any),
            best-available download URL, and license info. Never raises on
            malformed individual entries — a malformed entry is skipped and
            logged rather than failing the whole search.

        Raises:
            RuntimeError: If no API key is configured.
            requests.HTTPError: On a non-2xx response, after retries.
        """
        api_key = self._require_api_key()
        response = requests.get(
            _BASE_URL,
            headers={"Authorization": api_key},
            params={"query": query, "per_page": per_page, "page": page},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        payload = response.json()

        results: list[NormalizedStockAsset] = []
        for video in payload.get("videos", []):
            try:
                results.append(self._normalize_video(video))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "PexelsClient.search: skipping malformed video entry (id=%s): %s",
                    video.get("id") if isinstance(video, dict) else "?",
                    exc,
                )
        return results

    @staticmethod
    def _normalize_video(video: dict) -> NormalizedStockAsset:
        """Normalize one raw Pexels `videos[]` entry.

        Pexels does not return free-text "tags" on video search results the
        way it does for photos in some API versions — `user.name` and the
        query itself are common substitutes callers use for tagging. This
        normalizer looks for a `tags` field defensively (some Pexels API
        responses/collections do include one) and falls back to an empty
        list rather than guessing. UNVERIFIED against a live response — see
        module docstring.
        """
        external_id = str(video["id"])
        description = video.get("alt") or video.get("url") or f"pexels video {external_id}"
        tags = list(video.get("tags") or [])

        video_files = video.get("video_files") or []
        download_url = PexelsClient._best_download_url(video_files)

        return NormalizedStockAsset(
            source="pexels",
            external_id=external_id,
            description=description,
            tags=tags,
            download_url=download_url,
            license_type=PEXELS_LICENSE_TYPE,
            width=video.get("width"),
            height=video.get("height"),
            duration_s=float(video["duration"]) if video.get("duration") is not None else None,
        )

    @staticmethod
    def _best_download_url(video_files: list[dict]) -> str | None:
        """Pick the highest-resolution `.mp4` link from `video_files[]`.

        Falls back to the first entry with a `link` if no `.mp4`/resolution
        data is present (defensive — response shape unverified live).
        """
        if not video_files:
            return None
        mp4_files = [f for f in video_files if f.get("file_type") == "video/mp4" and f.get("link")]
        candidates = mp4_files or [f for f in video_files if f.get("link")]
        if not candidates:
            return None
        best = max(candidates, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))
        return best.get("link")


_client: PexelsClient | None = None


def get_pexels_client() -> PexelsClient:
    """Return the module-level singleton `PexelsClient`."""
    global _client
    if _client is None:
        _client = PexelsClient()
    return _client
