"""Tests for the sync CLI orchestration.

The Radar download is stubbed to hand back a small real DuckDB snapshot, so the
whole DuckDB read -> merge -> write path runs for real; only the network (radar
client) and HuggingFace seams are replaced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import httpx

from hetzner_auction_history import cli, hf, radar
from hetzner_auction_history.cli import sync
from hetzner_auction_history.config import Settings

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _make_radar_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE server (id BIGINT, seen TIMESTAMP, price INTEGER, "
        "server_type VARCHAR, cpu VARCHAR)"
    )
    con.execute(
        """
        INSERT INTO server VALUES
          (1, TIMESTAMP '2026-06-01 20:00:00', 90, 'auction', 'Xeon'),
          (2, TIMESTAMP '2026-07-15 12:00:00', 50, 'auction', 'Ryzen')
        """
    )
    con.close()


def _settings(tmp_path: Path, *, token: str | None) -> Settings:
    return Settings(
        hf_repo_id="me/ds",
        hf_token=token,
        data_dir=tmp_path / "_local",
        work_dir=tmp_path / ".work",
    )


def _patch_source(monkeypatch: pytest.MonkeyPatch, settings: Settings, radar_db: Path) -> None:
    def fake_download(_client: httpx.Client, dest: Path) -> Path:
        _ = dest.write_bytes(radar_db.read_bytes())  # hand back the real snapshot
        return dest

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(radar, "make_client", httpx.Client)
    monkeypatch.setattr(radar, "download_snapshot", fake_download)


def test_sync_dry_run_writes_hive_shards_and_skips_hf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    radar_db = tmp_path / "radar.duckdb"
    _make_radar_db(radar_db)
    settings = _settings(tmp_path, token=None)
    _patch_source(monkeypatch, settings, radar_db)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("HF must not be touched in dry-run")

    monkeypatch.setattr(hf, "ensure_repo", _boom)
    monkeypatch.setattr(hf, "download_shard", _boom)
    monkeypatch.setattr(hf, "upload_folder", _boom)

    sync(repo=None, no_upload=True)

    data = settings.data_dir / "data"
    assert (data / "month=2026-06" / "data.parquet").exists()
    assert (data / "month=2026-07" / "data.parquet").exists()
    assert (settings.data_dir / "README.md").exists()


def test_sync_upload_merges_and_uploads_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    radar_db = tmp_path / "radar.duckdb"
    _make_radar_db(radar_db)
    settings = _settings(tmp_path, token="tok")
    _patch_source(monkeypatch, settings, radar_db)

    downloaded: list[str] = []
    ensured: list[str] = []
    uploaded: list[str] = []

    def fake_download_shard(_repo: str, month: str, _tok: str | None = None) -> Path | None:
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
    assert (settings.data_dir / "data" / "month=2026-06" / "data.parquet").exists()
    assert (settings.data_dir / "README.md").exists()
