"""Read the Radar auction board and merge it into month-partitioned Parquet.

Pure DuckDB — no pandas. Radar's ``server`` table holds every 5-minute
observation of the Hetzner auction board over a rolling 90-day window. We keep
**one row per auction per calendar day** — ``(id, day)`` — taking the latest
observation that day, and partition the result by month (``YYYY-MM``). Each month
is one Parquet shard, Hive-partitioned as ``data/month=YYYY-MM/data.parquet``.

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

- **Monotonic-tail guard** (:func:`merge_month`): a cheap regression tripwire. When
  a shard already exists for the month, the merged shard must not have fewer
  **distinct ``(id, day)``** keys nor a later ``min(day)`` than the existing one.
  On honest input it can never fire (the merge ``UNION``s the existing shard in, so
  the result is always a superset); it exists so a future merge-logic regression
  can't silently shrink a shard vs the one it merged from. Distinct-key counts
  (not raw rows) keep a legacy duplicate-key shard from false-tripping it.
- **Byte-determinism**: the COPY ends with ``ORDER BY id, day`` (a total order —
  the pair is unique after dedup), the ``row_number`` dedup breaks ``seen`` ties
  deterministically (``seen DESC, price DESC, id``), the connection is pinned to
  UTC (so day-bucketing is stable against a future upstream ``TIMESTAMPTZ``), and
  DuckDB's Parquet writer emits no run-varying metadata (no pandas blob, no
  timestamps). With that plus pinned zstd and the pinned DuckDB version (uv.lock),
  an unchanged month serialises to identical bytes and HuggingFace's
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

# Deterministic winner within a (id, day): latest observation, ties broken by
# price then id so the survivor is a pure function of the data (no byte churn).
_DEDUP_ORDER = "seen DESC, price DESC, id"

# Build the de-duplicated auction set as a temp table `auction`: one row per
# (id, seen::DATE), deterministic winner, with string `day`/`month` partitions.
# `seen::TIMESTAMP` pins bucketing against a future upstream TIMESTAMPTZ (the
# connection is also SET to UTC). Plain (non-f) string — no injection surface.
_AUCTION_SQL = f"""
CREATE TEMP TABLE auction AS
SELECT
    * EXCLUDE (rn),
    strftime(seen::TIMESTAMP, '%Y-%m-%d') AS day,
    strftime(seen::TIMESTAMP, '%Y-%m') AS month
FROM (
    SELECT *, row_number() OVER (
        PARTITION BY id, seen::TIMESTAMP::DATE ORDER BY {_DEDUP_ORDER}
    ) AS rn
    FROM radar.server
    WHERE server_type = 'auction' AND price > 0 AND seen IS NOT NULL
) WHERE rn = 1
"""  # noqa: S608  # only _DEDUP_ORDER (our own constant) is interpolated, not user input


class ShardStats(BaseModel):
    """Distinct-key count + earliest day of a shard, for the monotonic-tail guard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keys: int  # distinct (id, day) pairs
    min_day: str | None  # None only when the shard is empty


def attach_snapshot(db_path: Path) -> DuckDBPyConnection:
    """Attach the Radar snapshot and build the de-duplicated ``auction`` temp table.

    Returns an in-memory connection (pinned to UTC) carrying ``auction`` (one row
    per ``(id, day)``). Auction rows whose ``seen`` is NULL cannot be bucketed into
    a day; they are excluded and the count is logged rather than silently dropped.
    """
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")  # stable day-bucketing regardless of upstream tz
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
    """Distinct ``(id, day)`` count + earliest ``day`` of a Parquet shard on disk.

    Counts DISTINCT keys (not raw rows) so a legacy duplicate-key shard cannot
    false-trip the monotonic guard.
    """
    row = con.execute(
        f"SELECT count(*), min(day) FROM (SELECT DISTINCT id, day FROM read_parquet('{path}'))"  # noqa: S608  # path is ours
    ).fetchone()
    if not row:
        return ShardStats(keys=0, min_day=None)
    return ShardStats(keys=int(row[0]), min_day=row[1])


def _check_monotonic(month: str, merged: ShardStats, prev: ShardStats) -> None:
    """Raise if ``merged`` dropped keys or its tail moved forward vs ``prev``.

    Fewer distinct ``(id, day)`` keys, or a later ``min(day)``, means published
    history was lost — the merge should always be a superset of the shard it merged
    from, so a regression is a bug (or a truncating overwrite). We refuse to
    overwrite and fail the run.
    """
    shrank = merged.keys < prev.keys
    tail_moved = prev.min_day is not None and (
        merged.min_day is None or merged.min_day > prev.min_day
    )
    if shrank or tail_moved:
        raise ExternalServiceError(
            f"refusing to write shard {month}: merged shard regressed vs the "
            f"published one (keys {merged.keys} vs {prev.keys}, min_day "
            f"{merged.min_day!r} vs {prev.min_day!r}) — this would truncate "
            f"aged-out history"
        )


def merge_month(
    con: DuckDBPyConnection, month: str, existing_path: Path | None, out_path: Path
) -> ShardStats:
    """Merge ``month``'s new rows with its existing shard and write the result.

    Reads ``auction`` (new rows for the month) and, if present, the existing
    shard, ``UNION``s them, de-duplicates on ``(id, day)`` keeping the latest
    ``seen``, and ``COPY``s to ``out_path`` as zstd Parquet ordered by ``(id, day)``
    (deterministic bytes). Runs the monotonic-tail guard against ``existing_path``
    before committing the write; the scratch ``.tmp`` is always cleaned up.

    Args:
        con: Connection carrying the ``auction`` temp table.
        month: The ``YYYY-MM`` partition to build.
        existing_path: Local path to the shard already published for this month,
            or ``None`` on the first run for the month.
        out_path: Where to write the merged shard.

    Returns:
        Stats of the written shard.

    Raises:
        ExternalServiceError: if the merged shard regresses vs ``existing_path``
            (the ``.tmp`` is removed and ``out_path`` left untouched).

    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = f"SELECT * FROM auction WHERE month = '{month}'"  # noqa: S608  # month is derived
    if existing_path is None:
        source = new_rows
    else:
        # UNION ALL BY NAME tolerates schema drift between the stored shard and a
        # newer snapshot: a column present in only one side is NULL-filled, not an
        # error, so a Radar column add/remove degrades gracefully.
        source = f"{new_rows} UNION ALL BY NAME SELECT * FROM read_parquet('{existing_path}')"  # noqa: S608
    merged_sql = (
        "SELECT * EXCLUDE (rn) FROM ("  # noqa: S608  # SQL from our own month/path, not user input
        f"  SELECT *, row_number() OVER (PARTITION BY id, day ORDER BY {_DEDUP_ORDER}) AS rn"
        f"  FROM ({source})"
        ") WHERE rn = 1 ORDER BY id, day"
    )
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        con.execute(f"COPY ({merged_sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION zstd)")
        merged = shard_stats(con, tmp)
        if existing_path is not None:
            _check_monotonic(month, merged, shard_stats(con, existing_path))
        _ = tmp.replace(out_path)
        log.info("convert.merge_month", month=month, keys=merged.keys, min_day=merged.min_day)
        return merged
    finally:
        # No-op after a successful replace (tmp is gone); cleans up on any raise.
        tmp.unlink(missing_ok=True)
