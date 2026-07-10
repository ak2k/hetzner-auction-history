"""Download Server Radar's public DuckDB snapshot of the Hetzner auction board.

Radar publishes a rolling 90-day snapshot at a static URL behind Cloudflare,
which rejects requests without a browser-like ``User-Agent`` (403). Transport
and HTTP failures are wrapped as domain errors; ``httpx`` exceptions never leak
to callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

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
_TIMEOUT = httpx.Timeout(60.0, read=600.0)
_NOT_MODIFIED = 304
_FORBIDDEN = 403

# Cloudflare / the origin can serve transient 5xx or rate-limit bursts; back off
# and retry those (but not 403/404 — a UA or path fix, not a wait).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class SnapshotResult(NamedTuple):
    """Outcome of a snapshot download.

    ``not_modified`` is True when a conditional GET matched the previous ETag
    (HTTP 304): the file on disk is untouched and the caller can short-circuit.
    ``etag`` is the current snapshot's ETag (for the next conditional GET), or
    ``None`` if the server sent none.
    """

    not_modified: bool
    etag: str | None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


@stamina.retry(on=_is_retryable, attempts=6, wait_initial=2.0, wait_max=120.0)
def _stream_to_file(
    client: httpx.Client, url: str, dest: Path, prev_etag: str | None
) -> SnapshotResult:
    headers = {"If-None-Match": prev_etag} if prev_etag else {}
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == _NOT_MODIFIED:
            return SnapshotResult(not_modified=True, etag=prev_etag)
        _ = response.raise_for_status()
        with dest.open("wb") as fh:  # truncates on each (re)try — clean restart
            for chunk in response.iter_bytes(1 << 20):
                _ = fh.write(chunk)
        return SnapshotResult(not_modified=False, etag=response.headers.get("etag"))


def make_client() -> httpx.Client:
    """Build an httpx client carrying a browser User-Agent (Cloudflare-gated)."""
    return httpx.Client(
        headers={"User-Agent": _BROWSER_UA}, timeout=_TIMEOUT, follow_redirects=True
    )


def download_snapshot(
    client: httpx.Client, dest: Path, *, prev_etag: str | None = None
) -> SnapshotResult:
    """Download the Radar snapshot to ``dest`` (atomic via a temp file).

    When ``prev_etag`` is supplied, sends a conditional GET (``If-None-Match``);
    an HTTP 304 short-circuits with ``not_modified=True`` and leaves ``dest``
    untouched. At daily cadence this rarely triggers — Radar rewrites the file
    every ~5 minutes so the ETag almost always differs — so it is cheap
    insurance, not a load-bearing optimization.

    Raises:
        ExternalServiceError: on transport or HTTP failure.

    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    log.info("radar.download.start", url=SNAPSHOT_URL, conditional=prev_etag is not None)
    try:
        result = _stream_to_file(client, SNAPSHOT_URL, tmp, prev_etag)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        hint = ""
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == _FORBIDDEN:
            hint = " (403 — Cloudflare rejects a non-browser User-Agent)"
        raise ExternalServiceError(f"failed to download Radar snapshot: {exc}{hint}") from exc
    if result.not_modified:
        tmp.unlink(missing_ok=True)
        log.info("radar.download.not_modified")
        return result
    _ = tmp.replace(dest)
    log.info("radar.download.done", mb=round(dest.stat().st_size / 1e6))
    return result
