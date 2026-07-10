"""Reconcile month-partitioned Parquet shards with the HuggingFace dataset repo.

Shards live at ``data/month=YYYY-MM/data.parquet`` (Hive layout, one deterministic
file per partition). Each run downloads the existing shard for every month in the
current Radar snapshot (the trailing ~4 months), merges the new rows in, and writes
the merged shards deterministically to a local folder mirroring that layout.
:func:`upload_folder` then hands the folder to ``upload_large_folder``, which hashes
each file and uploads only those whose bytes differ from the remote — so months
whose data did not change (and older months, which are never re-written locally)
are left untouched. Change detection is the content hash; there is no hand-rolled
diffing here.

A spurious HTTP 404 on an *existing* shard must never be mistaken for "this month
is new": that would collapse the merge to trailing rows only and truncate aged-out
history. So :func:`download_shard` confirms a not-found against the repo file list
before treating a shard as absent.

``huggingface_hub`` and raw ``httpx`` errors are wrapped as domain errors here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import structlog
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from hetzner_auction_history.errors import ExternalServiceError

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger(__name__)  # pyright: ignore[reportAny]


def shard_path(month: str) -> str:
    """Repo-relative path of a month's shard, e.g. ``data/month=2026-07/data.parquet``."""
    return f"data/month={month}/data.parquet"


def _list_files(repo_id: str, token: str | None) -> set[str]:
    """Repo file list, or empty set when the repo does not exist yet."""
    try:
        return set(HfApi(token=token).list_repo_files(repo_id, repo_type="dataset"))
    except RepositoryNotFoundError:
        return set()
    except (HfHubHTTPError, httpx.HTTPError) as exc:
        raise ExternalServiceError(f"failed to list files of {repo_id}: {exc}") from exc


def ensure_repo(repo_id: str, token: str | None = None) -> None:
    """Create the (public) dataset repo if it does not already exist.

    Raises:
        ExternalServiceError: on transport/HTTP failure.

    """
    try:
        _ = create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False, token=token)
    except (HfHubHTTPError, httpx.HTTPError) as exc:
        raise ExternalServiceError(f"failed to ensure dataset repo {repo_id}: {exc}") from exc


def download_shard(repo_id: str, month: str, token: str | None = None) -> Path | None:
    """Download the existing shard for ``month`` and return its local path.

    Returns ``None`` only when the shard is genuinely absent (first run for the
    month, or the repo does not exist yet). A not-found that *contradicts* the
    repo file list — a spurious 404 on a shard that is listed — is raised, not
    swallowed: treating it as absent would truncate published history.

    Raises:
        ExternalServiceError: on a contradictory 404, or transport/HTTP failure.

    """
    path_in_repo = shard_path(month)
    try:
        local = hf_hub_download(repo_id, path_in_repo, repo_type="dataset", token=token)
    except (EntryNotFoundError, RepositoryNotFoundError) as nf:
        if path_in_repo in _list_files(repo_id, token):
            raise ExternalServiceError(
                f"{path_in_repo} is listed on {repo_id} but 404'd on download; refusing "
                f"to treat it as absent (a merge would truncate aged-out history)"
            ) from nf
        return None
    except (HfHubHTTPError, httpx.HTTPError) as exc:
        raise ExternalServiceError(
            f"failed to download {path_in_repo} from {repo_id}: {exc}"
        ) from exc
    return Path(local)


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
            allow_patterns=["*.parquet", "README.md"],
            ignore_patterns=["._*", "*.tmp"],
        )
    except (HfHubHTTPError, httpx.HTTPError) as exc:
        raise ExternalServiceError(f"failed to upload to {repo_id}: {exc}") from exc
