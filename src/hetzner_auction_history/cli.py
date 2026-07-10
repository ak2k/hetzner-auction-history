"""Command-line entry point (typer).

hetzner-auction-history sync                 # upsert today's board into the HF dataset
hetzner-auction-history sync --no-upload     # write shards to a local dir, skip HF
hetzner-auction-history sync --dry-run       # alias for --no-upload
hetzner-auction-history sync --repo me/ds    # override the target dataset
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import structlog
import typer

from hetzner_auction_history import card, convert, hf, radar
from hetzner_auction_history.config import get_settings
from hetzner_auction_history.log import configure

if TYPE_CHECKING:
    from pathlib import Path

    from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger("hetzner_auction_history")  # pyright: ignore[reportAny]

app = typer.Typer(
    add_completion=False,
    help="Accumulate the Hetzner server auction into a HuggingFace dataset.",
)

_SNAPSHOT_FILE = "sb.duckdb"
_ETAG_FILE = "sb.etag"


@app.callback()
def _root() -> None:  # pyright: ignore[reportUnusedFunction]  # registered via decorator, not called directly
    """Group callback — keeps ``sync`` a named subcommand (not the bare root)."""


def _read_prev_etag(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() or None if path.exists() else None


@app.command()
def sync(
    repo: Annotated[
        str | None, typer.Option("--repo", help="Target dataset repo id (overrides HF_REPO_ID).")
    ] = None,
    no_upload: Annotated[
        bool,
        typer.Option(
            "--no-upload",
            "--dry-run",
            help="Write shards to the local data dir and skip HuggingFace entirely.",
        ),
    ] = False,
) -> None:
    """Download the Radar snapshot and upsert it into the HF dataset's shards.

    Change detection is delegated to HuggingFace's content-addressed upload:
    shards are written deterministically, so unchanged months serialise to
    identical bytes and are skipped by ``upload_large_folder``.
    """
    settings = get_settings()
    configure(settings.log_level, settings.log_json)
    repo_id = repo or settings.hf_repo_id
    token = settings.hf_token

    settings.work_dir.mkdir(parents=True, exist_ok=True)
    db_path = settings.work_dir / _SNAPSHOT_FILE
    etag_path = settings.work_dir / _ETAG_FILE
    with radar.make_client() as client:
        result = radar.download_snapshot(client, db_path, prev_etag=_read_prev_etag(etag_path))
    if result.not_modified:
        log.info("sync.snapshot_unchanged")
        return
    if result.etag:
        _ = etag_path.write_text(result.etag + "\n", encoding="utf-8")

    frame = convert.read_auction_frame(db_path)
    db_path.unlink(missing_ok=True)

    shards = convert.partition_by_month(frame)
    log.info(
        "sync.snapshot",
        rows=len(frame),
        months=len(shards),
        span=[shards[0][0], shards[-1][0]] if shards else [],
    )
    if not shards:
        log.warning("sync.empty_snapshot")
        return

    # Shards live under <data_dir>/data/ so the local tree mirrors the repo
    # layout (data/YYYY-MM.parquet) that upload_folder + the card config expect.
    shard_dir = settings.data_dir / "data"
    for month, new in shards:
        existing = None if no_upload else hf.download_shard(repo_id, month, token)
        merged = convert.merge_shard(existing, new)
        _ = convert.write_shard(merged, shard_dir / f"{month}.parquet")
        log.info("sync.shard", month=month, rows=len(merged))

    _ = card.write_card(repo_id, settings.data_dir)

    if not no_upload:
        hf.ensure_repo(repo_id, token)
        hf.upload_folder(repo_id, settings.data_dir, token)
    log.info("sync.done", months=len(shards), uploaded=not no_upload)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
