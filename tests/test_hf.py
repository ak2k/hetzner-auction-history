"""Tests for the HuggingFace reconcile boundary — hermetic via stubs.

The download/upload seams are exercised by stubbing the ``huggingface_hub`` calls
(``hf_hub_download``, ``create_repo``, ``HfApi``) so this module's own not-found
handling, spurious-404 guard, and error-wrapping run without the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import httpx
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from hetzner_auction_history import hf
from hetzner_auction_history.errors import ExternalServiceError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# huggingface_hub's HTTP errors require a response object; a 404 stands in.
_RESP = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))
_SHARD = "data/month=2026-07/data.parquet"


def test_shard_path():
    assert hf.shard_path("2026-07") == _SHARD


def _fake_api(list_files: Callable[[], list[str]]) -> type:
    """An HfApi stub whose list_repo_files is driven by ``list_files``."""

    class _Api:
        def __init__(self, token: str | None = None) -> None:
            self._token = token

        def list_repo_files(self, repo_id: str, repo_type: str | None = None) -> list[str]:
            assert (repo_id, repo_type) == ("me/ds", "dataset")
            return list_files()

    return _Api


def test_download_shard_returns_local_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    shard = tmp_path / "data.parquet"
    _ = shard.write_bytes(b"parquet")

    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        assert (repo_id, filename, repo_type) == ("me/ds", _SHARD, "dataset")
        return str(shard)

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    assert hf.download_shard("me/ds", "2026-07") == shard


def test_download_shard_none_when_genuinely_absent(monkeypatch: pytest.MonkeyPatch):
    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        raise EntryNotFoundError(f"no {filename}")

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    monkeypatch.setattr(hf, "HfApi", _fake_api(list))  # repo lists no such shard
    assert hf.download_shard("me/ds", "2026-07") is None


def test_download_shard_none_when_repo_absent(monkeypatch: pytest.MonkeyPatch):
    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        raise RepositoryNotFoundError(f"no repo {repo_id}", response=_RESP)

    def _raise() -> list[str]:
        raise RepositoryNotFoundError("no repo", response=_RESP)

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    monkeypatch.setattr(hf, "HfApi", _fake_api(_raise))  # repo genuinely does not exist
    assert hf.download_shard("me/ds", "2026-07") is None  # first run for the repo


def test_download_shard_spurious_404_on_listed_shard_raises(monkeypatch: pytest.MonkeyPatch):
    # The finding-3 guard: a 404 that contradicts the repo file list must NOT be
    # treated as "month is new" — that would truncate aged-out history.
    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        raise EntryNotFoundError(f"transient 404 on {filename}")

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    monkeypatch.setattr(hf, "HfApi", _fake_api(lambda: [_SHARD]))  # shard IS listed
    with pytest.raises(ExternalServiceError, match="refusing to treat it as absent"):
        _ = hf.download_shard("me/ds", "2026-07")


def test_download_shard_wraps_http_error(monkeypatch: pytest.MonkeyPatch):
    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        raise HfHubHTTPError("500 boom", response=_RESP)

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    with pytest.raises(ExternalServiceError, match="failed to download"):
        _ = hf.download_shard("me/ds", "2026-07")


def test_download_shard_wraps_raw_transport_error(monkeypatch: pytest.MonkeyPatch):
    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        raise httpx.ConnectError("connection reset")  # raw httpx, not HfHubHTTPError

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    with pytest.raises(ExternalServiceError, match="failed to download"):
        _ = hf.download_shard("me/ds", "2026-07")


def test_ensure_repo_wraps_http_error(monkeypatch: pytest.MonkeyPatch):
    def fake_create(*_args: object, **_kwargs: object) -> str:
        raise HfHubHTTPError("403 nope", response=_RESP)

    monkeypatch.setattr(hf, "create_repo", fake_create)
    with pytest.raises(ExternalServiceError, match="failed to ensure"):
        hf.ensure_repo("me/ds")


class _FakeApi:
    calls: ClassVar[list[tuple[str, str, tuple[str, ...]]]] = []

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    def upload_large_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        allow_patterns: list[str],
        ignore_patterns: list[str],
    ) -> None:
        assert repo_type == "dataset"
        assert ignore_patterns  # some ignore set is always passed
        _FakeApi.calls.append((repo_id, folder_path, tuple(allow_patterns)))


def test_upload_folder_passes_folder_and_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _FakeApi.calls = []
    monkeypatch.setattr(hf, "HfApi", _FakeApi)

    hf.upload_folder("me/ds", tmp_path)
    assert len(_FakeApi.calls) == 1
    repo_id, folder, allow = _FakeApi.calls[0]
    assert repo_id == "me/ds"
    assert folder == str(tmp_path)
    assert "*.parquet" in allow
    assert "README.md" in allow


def test_upload_folder_wraps_transport_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _BoomApi:
        def __init__(self, token: str | None = None) -> None:
            self._token = token

        def upload_large_folder(self, **_kwargs: object) -> None:
            raise httpx.ConnectError("connection reset")  # raw httpx

    monkeypatch.setattr(hf, "HfApi", _BoomApi)
    with pytest.raises(ExternalServiceError, match="failed to upload"):
        hf.upload_folder("me/ds", tmp_path)
