"""Tests for environment-driven settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hetzner_auction_history.config import get_settings

if TYPE_CHECKING:
    import pytest


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    monkeypatch.setenv("HF_REPO_ID", "me/ds")
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = get_settings()
    assert settings.hf_repo_id == "me/ds"
    assert settings.hf_token == "tok"
    assert settings.log_level == "debug"


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    for var in ("HF_TOKEN", "HF_REPO_ID", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)

    settings = get_settings()
    assert settings.hf_repo_id == "trader298/hetzner-auction-history"
    assert settings.hf_token is None  # optional — only needed for upload
    assert settings.log_level == "info"
    assert settings.data_dir == Path("_local")
