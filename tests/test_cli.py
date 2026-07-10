"""Tests for the sync CLI orchestration — network + HF seams stubbed."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pandas as pd

from hetzner_auction_history import cli, convert, hf, radar
from hetzner_auction_history.cli import sync
from hetzner_auction_history.config import Settings
from hetzner_auction_history.radar import SnapshotResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _snapshot_frame() -> pd.DataFrame:
    """A two-month snapshot, one row per (id, day), as read_auction_frame returns."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "day": ["2026-06-01", "2026-07-15"],
            "seen": pd.to_datetime(["2026-06-01 20:00", "2026-07-15 12:00"]),
            "price": [90.0, 50.0],
            "month": ["2026-06", "2026-07"],
        }
    )


def _settings(tmp_path: Path, *, token: str | None) -> Settings:
    return Settings(
        hf_repo_id="me/ds",
        hf_token=token,
        data_dir=tmp_path / "_local",
        work_dir=tmp_path / ".work",
    )


def _patch_source(monkeypatch: pytest.MonkeyPatch, settings: Settings, frame: pd.DataFrame) -> None:
    def fake_download(
        _client: httpx.Client, _dest: Path, *, prev_etag: str | None = None
    ) -> SnapshotResult:
        return SnapshotResult(not_modified=False, etag='"new"')

    def fake_read(_db: Path) -> pd.DataFrame:
        return frame

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(radar, "make_client", httpx.Client)
    monkeypatch.setattr(radar, "download_snapshot", fake_download)
    monkeypatch.setattr(convert, "read_auction_frame", fake_read)


def test_sync_dry_run_writes_local_shards_and_skips_hf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _settings(tmp_path, token=None)
    _patch_source(monkeypatch, settings, _snapshot_frame())

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("HF must not be touched in dry-run")

    monkeypatch.setattr(hf, "ensure_repo", _boom)
    monkeypatch.setattr(hf, "download_shard", _boom)
    monkeypatch.setattr(hf, "upload_folder", _boom)

    sync(repo=None, no_upload=True)

    data = settings.data_dir / "data"
    assert (data / "2026-06.parquet").exists()
    assert (data / "2026-07.parquet").exists()
    assert (settings.data_dir / "README.md").exists()  # card written locally too


def test_sync_upload_merges_and_uploads_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path, token="tok")
    _patch_source(monkeypatch, settings, _snapshot_frame())

    downloaded: list[str] = []
    ensured: list[str] = []
    uploaded: list[str] = []

    def fake_download_shard(_repo: str, month: str, _tok: str | None = None) -> pd.DataFrame | None:
        downloaded.append(month)
        return None  # nothing published yet

    def fake_ensure(repo: str, _tok: str | None = None) -> None:
        ensured.append(repo)

    def fake_upload_folder(repo: str, folder: Path, _tok: str | None = None) -> None:
        uploaded.append(f"{repo}:{folder}")

    monkeypatch.setattr(hf, "download_shard", fake_download_shard)
    monkeypatch.setattr(hf, "ensure_repo", fake_ensure)
    monkeypatch.setattr(hf, "upload_folder", fake_upload_folder)

    sync(repo=None, no_upload=False)

    assert sorted(downloaded) == ["2026-06", "2026-07"]  # merged against existing (absent) shards
    assert ensured == ["me/ds"]
    assert uploaded == [f"me/ds:{settings.data_dir}"]  # content-addressed, whole folder
    assert (settings.data_dir / "data" / "2026-06.parquet").exists()
    assert (settings.data_dir / "README.md").exists()


def test_sync_short_circuits_on_not_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path, token="tok")

    def fake_download(
        _client: httpx.Client, _dest: Path, *, prev_etag: str | None = None
    ) -> SnapshotResult:
        return SnapshotResult(not_modified=True, etag='"same"')

    def _boom_read(_db: Path) -> pd.DataFrame:
        raise AssertionError("must not read the snapshot when 304")

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(radar, "make_client", httpx.Client)
    monkeypatch.setattr(radar, "download_snapshot", fake_download)
    monkeypatch.setattr(convert, "read_auction_frame", _boom_read)

    sync(repo=None, no_upload=False)

    assert not (settings.data_dir / "data").exists()  # nothing written
