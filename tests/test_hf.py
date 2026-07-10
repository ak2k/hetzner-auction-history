"""Tests for the HuggingFace reconcile boundary — hermetic via stubs.

The download/upload seams are exercised by stubbing the ``huggingface_hub`` calls
(``hf_hub_download``, ``create_repo``, ``HfApi.upload_file``) so this module's own
not-found handling, parquet read, and error-wrapping run without the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pandas as pd
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from hetzner_auction_history import hf
from hetzner_auction_history.errors import ExternalServiceError

if TYPE_CHECKING:
    from pathlib import Path

# huggingface_hub's HTTP errors require a response object; a 404 stands in.
_RESP = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))


def test_shard_path():
    assert hf.shard_path("2026-07") == "data/2026-07.parquet"


def test_download_shard_reads_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    shard = tmp_path / "2026-07.parquet"
    pd.DataFrame({"id": [1], "day": ["2026-07-01"], "price": [90.0]}).to_parquet(shard, index=False)

    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        assert (repo_id, filename, repo_type) == ("me/ds", "data/2026-07.parquet", "dataset")
        return str(shard)

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)

    frame = hf.download_shard("me/ds", "2026-07")
    assert frame is not None
    assert list(frame["id"]) == [1]


def test_download_shard_none_when_entry_absent(monkeypatch: pytest.MonkeyPatch):
    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        raise EntryNotFoundError(f"no {filename} in {repo_id} ({repo_type})")

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    assert hf.download_shard("me/ds", "2026-07") is None  # first run for the month


def test_download_shard_none_when_repo_absent(monkeypatch: pytest.MonkeyPatch):
    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        raise RepositoryNotFoundError(f"no repo {repo_id}", response=_RESP)

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)
    assert hf.download_shard("me/ds", "2026-07") is None  # first run for the repo


def test_download_shard_wraps_http_error(monkeypatch: pytest.MonkeyPatch):
    def fake_download(
        repo_id: str, filename: str, repo_type: str | None = None, token: str | None = None
    ) -> str:
        raise HfHubHTTPError("500 boom", response=_RESP)

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
    calls: list[tuple[str, str, tuple[str, ...]]] = []  # noqa: RUF012

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
        _FakeApi.calls.append((repo_id, folder_path, tuple(allow_patterns)))


def test_upload_folder_passes_folder_and_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _FakeApi.calls = []
    monkeypatch.setattr(hf, "HfApi", _FakeApi)

    hf.upload_folder("me/ds", tmp_path)
    assert len(_FakeApi.calls) == 1
    repo_id, folder, allow = _FakeApi.calls[0]
    assert repo_id == "me/ds"
    assert folder == str(tmp_path)
    assert "data/*.parquet" in allow
    assert "README.md" in allow


def test_upload_folder_wraps_http_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _BoomApi:
        def __init__(self, token: str | None = None) -> None:
            self._token = token

        def upload_large_folder(self, **_kwargs: object) -> None:
            raise HfHubHTTPError("502 boom", response=_RESP)

    monkeypatch.setattr(hf, "HfApi", _BoomApi)
    with pytest.raises(ExternalServiceError, match="failed to upload"):
        hf.upload_folder("me/ds", tmp_path)
