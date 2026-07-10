"""Read the Radar auction board and merge it into month-partitioned Parquet.

Pure DuckDB — no pandas. Radar's ``server`` table holds every 5-minute
observation of the Hetzner auction board over a rolling 90-day window. We keep
**one row per auction per calendar day** — ``(id, day)`` — taking the latest
observation that day, and partition the result by month (``YYYY-MM``). Each month
is one Parquet shard (``data/YYYY-MM.parquet``).

A snapshot only ever contains the trailing ~90 days, i.e. ~4 months. Months
fully inside that window are present in every snapshot; the **oldest** month is
truncated a little more each day as its early days age out. So the merge is
essential, not cosmetic: for each month we read the shard already published for
that month (if any) together with the new rows in one DuckDB query, ``UNION`` and
de-duplicate on ``(id, day)`` keeping the latest ``seen``, and ``COPY`` the result
to a Parquet shard. That preserves the full history of the aging-out month (its
early days survive on the shard even after they vanish from Radar); months that
have fully aged out never appear in a snapshot, so their shards are never
rewritten — that retained tail is the whole point.

Two safeguards protect the retained tail:

- **Monotonic-tail guard** (:func:`merge_month`): when a shard already exists for
  the month, the merged shard must not have fewer ``(id, day)`` rows nor a later
  ``min(day)`` than the existing one — a regression means the existing history was
  dropped, so we refuse to write it and fail the run loudly.
- **Byte-determinism**: the COPY ends with ``ORDER BY id, day`` (a total order —
  the pair is unique after dedup), and DuckDB's Parquet writer emits no run-varying
  metadata (no pandas blob, no timestamps). DuckDB does not guarantee row order
  without an explicit ``ORDER BY`` (parallel scans + hash aggregation reorder), so
  the clause is load-bearing: with it, pinned zstd, and the pinned DuckDB version
  (uv.lock), an unchanged month serialises to identical bytes and HuggingFace's
  content-addressed upload skips it. There is no hand-rolled "did this shard
  change?" bookkeeping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import structlog
from pydantic import BaseModel, ConfigDict

from hetzner_auction_history.errors import ExternalServiceError

if TYPE_CHECKING:
    from pathlib import Path

    from duckdb import DuckDBPyConnection
    from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger(__name__)  # pyright: ignore[reportAny]

# One row per auction (id) per calendar day (day) — the dataset's primary key.
KEY: tuple[str, str] = ("id", "day")

# Keep only real, purchasable auction rows: Radar tags non-auction inventory and
# emits price=0 placeholder rows we don't want; a NULL `seen` cannot be bucketed
# into a day and is excluded explicitly (counted + logged, never silently lost).
# Build the de-duplicated auction set as a temp table `auction`: one row per
# (id, seen::DATE), latest `seen` wins, with string `day`/`month` partitions.
# A plain (non-f) constant string — no interpolation, so no injection surface.
_AUCTION_SQL = """
CREATE TEMP TABLE auction AS
SELECT
    * EXCLUDE (rn),
    strftime(seen, '%Y-%m-%d') AS day,
    strftime(seen, '%Y-%m') AS month
FROM (
    SELECT *, row_number() OVER (
        PARTITION BY id, seen::DATE ORDER BY seen DESC
    ) AS rn
    FROM radar.server
    WHERE server_type = 'auction' AND price > 0 AND seen IS NOT NULL
) WHERE rn = 1
"""


class ShardStats(BaseModel):
    """Row count + earliest day of a shard, used by the monotonic-tail guard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int
    min_day: str | None  # None only when the shard is empty


def attach_snapshot(db_path: Path) -> DuckDBPyConnection:
    """Attach the Radar snapshot and build the de-duplicated ``auction`` temp table.

    Returns an in-memory connection carrying ``auction`` (one row per
    ``(id, day)``). Auction rows whose ``seen`` is NULL cannot be bucketed into a
    day; they are excluded and the count is logged rather than silently dropped.
    """
    con = duckdb.connect()
    con.execute(f"ATTACH '{db_path}' AS radar (READ_ONLY)")
    seen_null = con.execute(
        "SELECT count(*) FROM radar.server "
        "WHERE server_type = 'auction' AND price > 0 AND seen IS NULL"
    ).fetchone()
    con.execute(_AUCTION_SQL)
    kept = con.execute("SELECT count(*) FROM auction").fetchone()
    log.info(
        "convert.auction",
        kept=int(kept[0]) if kept else 0,
        dropped_null_seen=int(seen_null[0]) if seen_null else 0,
    )
    return con


def months(con: DuckDBPyConnection) -> list[str]:
    """Distinct months present in the ``auction`` table, ascending."""
    rows = con.execute("SELECT DISTINCT month FROM auction ORDER BY month").fetchall()
    return [str(r[0]) for r in rows]


def total_rows(con: DuckDBPyConnection) -> int:
    """Total de-duplicated auction rows across all months."""
    row = con.execute("SELECT count(*) FROM auction").fetchone()
    return int(row[0]) if row else 0


def shard_stats(con: DuckDBPyConnection, path: Path) -> ShardStats:
    """Row count + earliest ``day`` of a Parquet shard on disk."""
    row = con.execute(
        f"SELECT count(*), min(day) FROM read_parquet('{path}')"  # noqa: S608  # path is ours
    ).fetchone()
    if not row:
        return ShardStats(rows=0, min_day=None)
    return ShardStats(rows=int(row[0]), min_day=row[1])


def _check_monotonic(month: str, merged: ShardStats, prev: ShardStats) -> None:
    """Raise if ``merged`` dropped rows or its tail moved forward vs ``prev``.

    A shrink or a later ``min(day)`` means published history was lost — most
    likely a spurious 404 that made an existing shard look absent, collapsing the
    merge to trailing rows only. We refuse to overwrite and fail the run.
    """
    shrank = merged.rows < prev.rows
    tail_moved = prev.min_day is not None and (
        merged.min_day is None or merged.min_day > prev.min_day
    )
    if shrank or tail_moved:
        raise ExternalServiceError(
            f"refusing to write shard {month}: merged shard regressed vs the "
            f"published one (rows {merged.rows} vs {prev.rows}, min_day "
            f"{merged.min_day!r} vs {prev.min_day!r}) — this would truncate "
            f"aged-out history, likely from a spurious 404 on an existing shard"
        )


def merge_month(
    con: DuckDBPyConnection, month: str, existing_path: Path | None, out_path: Path
) -> ShardStats:
    """Merge ``month``'s new rows with its existing shard and write the result.

    Reads ``auction`` (new rows for the month) and, if present, the existing
    shard, ``UNION``s them, de-duplicates on ``(id, day)`` keeping the latest
    ``seen``, and ``COPY``s to ``out_path`` as zstd Parquet ordered by ``(id, day)``
    (deterministic bytes). Enforces the monotonic-tail guard against
    ``existing_path`` before committing the write.

    Args:
        con: Connection carrying the ``auction`` temp table.
        month: The ``YYYY-MM`` partition to build.
        existing_path: Local path to the shard already published for this month,
            or ``None`` on the first run for the month.
        out_path: Where to write the merged shard.

    Returns:
        Stats of the written shard.

    Raises:
        ExternalServiceError: if the merged shard regresses vs ``existing_path``.

    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = f"SELECT * FROM auction WHERE month = '{month}'"  # noqa: S608  # month is derived
    if existing_path is None:
        source = new_rows
    else:
        source = f"{new_rows} UNION ALL BY NAME SELECT * FROM read_parquet('{existing_path}')"  # noqa: S608
    merged_sql = (
        "SELECT * EXCLUDE (rn) FROM ("  # noqa: S608  # SQL from our own month/path, not user input
        "  SELECT *, row_number() OVER (PARTITION BY id, day ORDER BY seen DESC) AS rn"
        f"  FROM ({source})"
        ") WHERE rn = 1 ORDER BY id, day"
    )
    tmp = out_path.with_name(out_path.name + ".tmp")
    con.execute(f"COPY ({merged_sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION zstd)")
    merged = shard_stats(con, tmp)
    if existing_path is not None:
        _check_monotonic(month, merged, shard_stats(con, existing_path))
    _ = tmp.replace(out_path)
    log.info("convert.merge_month", month=month, rows=merged.rows, min_day=merged.min_day)
    return merged
