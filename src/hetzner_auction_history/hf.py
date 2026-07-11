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
history. So :func:`download_shard` returns ``None`` (absent) **only** when it has
positively proven absence — either the repo genuinely does not exist (a true first
run) or the repo file list was obtained and does not contain the shard. Any
ambiguity (a shard 404 contradicted by the listing, or a listing that could not be
obtained) raises instead of being treated as absent.

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
    """Positively obtain the repo file list.

    Lets ``RepositoryNotFoundError`` propagate (so the caller can treat a
    genuinely-absent repo as a first run) and wraps any other failure as
    ``ExternalServiceError`` — an ambiguous listing is never silently treated as
    an empty set.

    Raises:
        RepositoryNotFoundError: if the repo does not exist.
        ExternalServiceError: on any other transport/HTTP failure.

    """
    try:
        return set(HfApi(token=token).list_repo_files(repo_id, repo_type="dataset"))
    except (HfHubHTTPError, httpx.HTTPError) as exc:
        if isinstance(exc, RepositoryNotFoundError):
            raise
        raise ExternalServiceError(f"failed to list files of {repo_id}: {exc}") from exc


def _prove_absent_or_raise(
    repo_id: str, path_in_repo: str, token: str | None, not_found: Exception
) -> None:
    """Confirm a downloaded shard is genuinely absent, or raise on any ambiguity.

    Returns (None) only when absence is proven: the repo does not exist, or the
    file list was obtained and lacks the shard. A shard the listing *does* contain
    (a spurious 404), or a listing that could not be obtained, raises — so a
    transient 404 is never mistaken for "absent" (which would truncate history).
    """
    try:
        files = _list_files(repo_id, token)
    except RepositoryNotFoundError:
        return  # repo does not exist at all -> genuine first run
    if path_in_repo in files:
        raise ExternalServiceError(
            f"{path_in_repo} is listed on {repo_id} but 404'd on download; refusing to "
            f"treat it as absent (a merge would truncate aged-out history)"
        ) from not_found
    # repo exists and the shard is not listed -> positively proven absent


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

    Returns ``None`` only when absence is *positively proven* (see
    :func:`_prove_absent_or_raise`); any ambiguity raises rather than risk a
    truncating overwrite.

    Raises:
        ExternalServiceError: on a contradictory/ambiguous 404, or transport/HTTP
            failure.

    """
    path_in_repo = shard_path(month)
    try:
        local = hf_hub_download(repo_id, path_in_repo, repo_type="dataset", token=token)
    except (EntryNotFoundError, RepositoryNotFoundError) as nf:
        _prove_absent_or_raise(repo_id, path_in_repo, token, nf)
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
