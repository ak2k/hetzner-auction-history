"""Typed configuration sourced from the environment.

Follows the canonical pydantic-settings shape: `extra="forbid"` + `frozen=True`
so accidental env vars and mid-run mutation fail loudly. Field names map to
upper-case env vars (case-insensitive), so `hf_token` reads `HF_TOKEN` — the
conventional, un-prefixed name the HuggingFace ecosystem already uses.

`hf_token` is optional because a dry run (write shards locally, no upload)
needs no token; the upload path validates its presence via the HuggingFace
client and raises a domain error if missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["debug", "info", "warning", "error"]


class Settings(BaseSettings):
    """Application settings sourced from environment + `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    hf_token: str | None = Field(
        default=None,
        description="HuggingFace write token (only needed for upload).",
    )
    hf_repo_id: str = Field(
        default="trader298/hetzner-auction-history",
        description="Target HuggingFace dataset repo id.",
    )
    data_dir: Path = Field(
        default=Path("_local"),
        description="Local output directory for month-partitioned Parquet shards.",
    )
    work_dir: Path = Field(
        default=Path(".work"),
        description="Scratch directory for the downloaded Radar DuckDB snapshot.",
    )
    log_level: LogLevel = Field(default="info", description="Application log level.")
    log_json: bool = Field(default=False, description="Emit JSON logs (production).")


def get_settings() -> Settings:
    """Construct Settings; BaseSettings populates fields from the environment."""
    return Settings()
