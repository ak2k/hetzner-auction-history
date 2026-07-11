"""Download Server Radar's public DuckDB snapshot of the Hetzner auction board.

Radar publishes a rolling 90-day snapshot at a static URL behind Cloudflare,
which rejects requests without a browser-like ``User-Agent`` (403). Transport
and HTTP failures are wrapped as domain errors; ``httpx`` exceptions never leak
to callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import stamina
import structlog

from hetzner_auction_history.errors import ExternalServiceError

if TYPE_CHECKING:
    from pathlib import Path

    from structlog.stdlib import BoundLogger

# structlog.get_logger() returns Any; the annotation narrows it for callers.
log: BoundLogger = structlog.get_logger(__name__)  # pyright: ignore[reportAny]

# Server Radar's public snapshot (elsbrock/hetzner-radar). ~50-200 MB DuckDB.
SNAPSHOT_URL = "https://static.radar.iodev.org/sb.duckdb"

# Cloudflare fronts the static host and 403s a default httpx/python User-Agent;
# a real browser UA passes. This is a public file, not an auth bypass.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# The snapshot is only ~2-4 MB, so timeouts are tight. The whole retry budget is
# bounded to stay well under the CI job's ceiling: at most 4 attempts x 120s read
# + exponential backoff (2+4+8 ~ 14s) ~ 9 min worst case. Keep this in sync with
# .github/workflows/sync.yml `timeout-minutes: 15`.
_TIMEOUT = httpx.Timeout(30.0, read=120.0)
_FORBIDDEN = 403
_MAX_ATTEMPTS = 4

# Cloudflare / the origin can serve transient 5xx or rate-limit bursts; back off
# and retry those (but not 403/404 — a UA or path fix, not a wait).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


@stamina.retry(on=_is_retryable, attempts=_MAX_ATTEMPTS, wait_initial=2.0, wait_max=30.0)
def _stream_to_file(client: httpx.Client, url: str, dest: Path) -> None:
    with client.stream("GET", url) as response:
        _ = response.raise_for_status()
        with dest.open("wb") as fh:  # truncates on each (re)try — clean restart
            for chunk in response.iter_bytes(1 << 20):
                _ = fh.write(chunk)


def make_client() -> httpx.Client:
    """Build an httpx client carrying a browser User-Agent (Cloudflare-gated)."""
    return httpx.Client(
        headers={"User-Agent": _BROWSER_UA}, timeout=_TIMEOUT, follow_redirects=True
    )


def download_snapshot(client: httpx.Client, dest: Path) -> Path:
    """Download the Radar snapshot to ``dest`` (atomic via a temp file).

    Raises:
        ExternalServiceError: on transport or HTTP failure (retries exhausted).

    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    log.info("radar.download.start", url=SNAPSHOT_URL)
    try:
        _stream_to_file(client, SNAPSHOT_URL, tmp)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        hint = ""
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == _FORBIDDEN:
            hint = " (403 — Cloudflare rejects a non-browser User-Agent)"
        raise ExternalServiceError(f"failed to download Radar snapshot: {exc}{hint}") from exc
    _ = tmp.replace(dest)
    log.info("radar.download.done", mb=round(dest.stat().st_size / 1e6))
    return dest
