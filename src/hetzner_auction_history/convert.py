"""Read the Radar auction board and merge it into month-partitioned Parquet.

Radar's ``server`` table holds every 5-minute observation of the Hetzner
auction board over a rolling 90-day window. We keep **one row per auction per
calendar day** — ``(id, day)`` — taking the latest observation that day, and
partition the result by month (``YYYY-MM``). Each month is one Parquet shard
(``data/YYYY-MM.parquet``).

A snapshot only ever contains the trailing ~90 days, i.e. ~4 months. Months
fully inside that window are present in every snapshot; the **oldest** month in
the window is truncated a little more each day as its early days age out. So the
merge is essential, not cosmetic: for each month in the snapshot we union the
new rows onto the shard already published for that month and de-duplicate on
``(id, day)`` keeping the latest ``seen``. That preserves the full history of
the aging-out month (its early days survive on the shard even after they vanish
from Radar), while months that have fully aged out never appear in a snapshot,
so their shards are never rewritten — that retained tail is the whole point.

Change detection is delegated to HuggingFace's content-addressed upload: shards
are written **deterministically** (:func:`write_shard`), so a month whose data
did not change serialises to identical bytes and the upload is skipped. There is
therefore no hand-rolled "did this shard change?" bookkeeping here.

DuckDB reads and de-duplicates the snapshot; pandas carries the frames through
the merge. This module isolates both libraries' loosely-typed surfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger(__name__)  # pyright: ignore[reportAny]

# One row per auction (id) per calendar day (day) — the dataset's primary key.
KEY: list[str] = ["id", "day"]

# Radar records each observation's wall-clock time in `seen`; the latest
# observation on a given day wins when de-duplicating.
_SORT: str = "seen"

# Keep only real, purchasable auction rows: Radar tags non-auction inventory and
# emits price=0 placeholder rows we don't want.
_READ_SQL = """
    SELECT * EXCLUDE (rn) FROM (
        SELECT *, row_number() OVER (
            PARTITION BY id, seen::DATE ORDER BY seen DESC
        ) AS rn
        FROM server
        WHERE server_type = 'auction' AND price > 0
    ) WHERE rn = 1
"""


def read_auction_frame(db_path: Path) -> pd.DataFrame:
    """Read Radar's auction rows, de-duplicated to one row per ``(id, day)``.

    Adds string ``day`` (``YYYY-MM-DD``) and ``month`` (``YYYY-MM``) columns
    derived from ``seen``; ``day`` is the dedup/merge key and ``month`` is the
    shard partition.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        frame: pd.DataFrame = con.execute(_READ_SQL).df()
    finally:
        con.close()
    return _with_partitions(frame)


def _with_partitions(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized ``seen`` plus ``day``/``month`` partition columns."""
    frame = frame.copy()
    frame["seen"] = pd.to_datetime(frame["seen"])
    frame["day"] = frame["seen"].dt.strftime("%Y-%m-%d")
    frame["month"] = frame["seen"].dt.strftime("%Y-%m")
    return frame


def partition_by_month(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split a frame into ``(month, rows)`` pairs, ascending by month."""
    out: list[tuple[str, pd.DataFrame]] = []
    for month, group in frame.groupby("month", sort=True):
        out.append((str(month), group.reset_index(drop=True)))
    return out


def _dedup(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per ``(id, day)``, keeping the latest ``seen``.

    Sorts by ``(id, day)`` at the end so the row order is a deterministic
    function of the data alone — a prerequisite for byte-stable shards.
    """
    return (
        frame.sort_values(_SORT)
        .drop_duplicates(subset=KEY, keep="last")
        .sort_values(KEY)
        .reset_index(drop=True)
    )


def merge_shard(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union ``new`` rows onto an ``existing`` shard, de-duplicated by ``(id, day)``.

    Args:
        existing: The shard already published for this month, or ``None`` on the
            first run for the month.
        new: This run's rows for the month (already one per ``(id, day)``).

    Returns:
        The merged, de-duplicated shard, sorted deterministically by ``(id, day)``.

    """
    if existing is None:
        return _dedup(new)
    return _dedup(pd.concat([existing, new], ignore_index=True))


def write_shard(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``frame`` to ``path`` as Parquet, **byte-deterministically**.

    Identical data must serialise to identical bytes so HuggingFace's
    content-addressed upload skips an unchanged shard. To that end: rows are
    sorted by ``(id, day)`` (a total order — the pair is unique), compression is
    pinned to ``zstd``, and pyarrow's writer emits no run-varying metadata
    (``created_by`` is the constant library-version string; no timestamps). The
    same frame written twice yields identical files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(KEY).reset_index(drop=True)
    table = pa.Table.from_pandas(ordered, preserve_index=False)
    pq.write_table(table, str(path), compression="zstd")
    return path
