"""Tests for the Radar snapshot download — hermetic via httpx.MockTransport."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from hetzner_auction_history import radar
from hetzner_auction_history.errors import ExternalServiceError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_make_client_sets_browser_user_agent():
    client = radar.make_client()
    assert "Mozilla/5.0" in client.headers["User-Agent"]
    client.close()


def test_download_snapshot_writes_file_and_returns_dest(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"DUCKDB-BYTES")

    dest = tmp_path / "sb.duckdb"
    with _client(handler) as client:
        result = radar.download_snapshot(client, dest)
    assert result == dest
    assert dest.read_bytes() == b"DUCKDB-BYTES"


def test_download_retries_on_503_then_succeeds(tmp_path: Path):
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503)  # transient once, then OK
        return httpx.Response(200, content=b"OK")

    dest = tmp_path / "sb.duckdb"
    with _client(handler) as client:
        _ = radar.download_snapshot(client, dest)
    assert attempts["n"] == 2
    assert dest.read_bytes() == b"OK"


def test_download_retries_exhausted_raises(tmp_path: Path):
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503)  # always down

    dest = tmp_path / "sb.duckdb"
    with _client(handler) as client, pytest.raises(ExternalServiceError):
        _ = radar.download_snapshot(client, dest)
    assert attempts["n"] > 1  # retried before giving up
    assert not dest.exists()


def test_download_wraps_error_and_cleans_temp(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    dest = tmp_path / "sb.duckdb"
    with _client(handler) as client, pytest.raises(ExternalServiceError):
        _ = radar.download_snapshot(client, dest)
    assert not dest.exists()
    assert not dest.with_suffix(".duckdb.tmp").exists()


def test_403_error_mentions_cloudflare(tmp_path: Path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    dest = tmp_path / "sb.duckdb"
    with (
        _client(handler) as client,
        pytest.raises(ExternalServiceError, match="Cloudflare"),
    ):
        _ = radar.download_snapshot(client, dest)
