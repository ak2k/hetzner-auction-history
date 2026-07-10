"""Reconcile month-partitioned Parquet shards with the HuggingFace dataset repo.

Shards live at ``data/YYYY-MM.parquet`` on the dataset. Each run downloads the
existing shard for every month in the current Radar snapshot (the trailing ~4
months), merges the new rows in, and writes the merged shards deterministically
to a local folder. :func:`upload_folder` then hands the folder to
``upload_large_folder``, which hashes each file and uploads only those whose
bytes differ from the remote — so months whose data did not change (and older
months, which are never re-written locally) are left untouched. Change detection
is the content hash; there is no hand-rolled diffing here.

``huggingface_hub`` errors are wrapped as domain errors at this boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import structlog
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from hetzner_auction_history.errors import ExternalServiceError

if TYPE_CHECKING:
    from pathlib import Path

    from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger(__name__)  # pyright: ignore[reportAny]


def shard_path(month: str) -> str:
    """Repo-relative path of a month's shard, e.g. ``data/2026-07.parquet``."""
    return f"data/{month}.parquet"


def ensure_repo(repo_id: str, token: str | None = None) -> None:
    """Create the (public) dataset repo if it does not already exist.

    Raises:
        ExternalServiceError: on transport/HTTP failure.

    """
    try:
        _ = create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False, token=token)
    except HfHubHTTPError as exc:
        raise ExternalServiceError(f"failed to ensure dataset repo {repo_id}: {exc}") from exc


def download_shard(repo_id: str, month: str, token: str | None = None) -> pd.DataFrame | None:
    """Download and read the existing shard for ``month``.

    Returns ``None`` when the repo or the shard does not exist yet (first run for
    the month) — treated as an empty starting point, not an error.

    Raises:
        ExternalServiceError: on transport/HTTP failure other than not-found.

    """
    try:
        local = hf_hub_download(repo_id, shard_path(month), repo_type="dataset", token=token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except HfHubHTTPError as exc:
        raise ExternalServiceError(
            f"failed to download {shard_path(month)} from {repo_id}: {exc}"
        ) from exc
    return pd.read_parquet(local)


def upload_folder(repo_id: str, local_dir: Path, token: str | None = None) -> None:
    """Upload the shard tree + dataset card, content-addressed (skips unchanged).

    ``upload_large_folder`` hashes every file and uploads only those absent or
    changed on the remote, so byte-identical shards are skipped for free.

    Raises:
        ExternalServiceError: on transport/HTTP failure.

    """
    log.info("hf.upload_folder", repo_id=repo_id, local_dir=str(local_dir))
    try:
        HfApi(token=token).upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
            allow_patterns=["data/*.parquet", "README.md"],
            ignore_patterns=["._*", ".tmp-*"],
        )
    except HfHubHTTPError as exc:
        raise ExternalServiceError(f"failed to upload to {repo_id}: {exc}") from exc
