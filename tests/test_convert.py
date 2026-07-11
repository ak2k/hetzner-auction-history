"""Tests for the pure-DuckDB (id, day) dedup + month-partitioned merge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hetzner_auction_history import convert
from hetzner_auction_history.convert import (
    ShardStats,
    _check_monotonic,  # pyright: ignore[reportPrivateUsage]  # unit-test the guard directly
)
from hetzner_auction_history.errors import ExternalServiceError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from duckdb import DuckDBPyConnection

# Canonical shard schema shared by the `auction` temp table and existing shards,
# so `UNION ALL BY NAME` aligns in merge_month.
_SCHEMA = "id BIGINT, seen TIMESTAMP, price INTEGER, day VARCHAR, month VARCHAR"

_Spec = tuple[int | None, str, int]  # (id, seen, price)


def _rows(specs: Sequence[_Spec]) -> list[tuple[int | None, str, int, str, str]]:
    """Expand (id, seen, price) into full shard rows with derived day/month."""
    return [(i, s, p, s[:10], s[:7]) for i, s, p in specs]


def _con_with_auction(specs: Sequence[_Spec]) -> DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE TEMP TABLE auction ({_SCHEMA})")
    con.executemany("INSERT INTO auction VALUES (?, ?, ?, ?, ?)", _rows(specs))
    return con


def _write_shard(path: Path, specs: Sequence[_Spec]) -> None:
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t ({_SCHEMA})")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", _rows(specs))
    con.execute(
        f"COPY (SELECT * FROM t ORDER BY id, day) TO '{path}' (FORMAT PARQUET, COMPRESSION zstd)"
    )
    con.close()


def _read(path: Path) -> list[tuple[object, ...]]:
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT id, day, price FROM read_parquet('{path}') ORDER BY id, day"
    ).fetchall()
    con.close()
    return rows


def _make_radar_db(path: Path) -> None:
    """A tiny Radar-shaped `server` table exercising dedup + the filters."""
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE server ("
        "  id BIGINT, seen TIMESTAMP, price INTEGER, server_type VARCHAR, cpu VARCHAR"
        ")"
    )
    con.execute(
        """
        INSERT INTO server VALUES
          (1, TIMESTAMP '2026-06-01 08:00:00', 100, 'auction', 'Xeon'),  -- same day, earlier
          (1, TIMESTAMP '2026-06-01 20:00:00',  90, 'auction', 'Xeon'),  -- same day, kept
          (1, TIMESTAMP '2026-06-02 08:00:00',  85, 'auction', 'Xeon'),  -- next day
          (2, TIMESTAMP '2026-07-15 12:00:00',  50, 'auction', 'Ryzen'), -- other month
          (3, TIMESTAMP '2026-06-01 09:00:00',   0, 'auction', 'Xeon'),  -- price 0: filtered
          (4, TIMESTAMP '2026-06-01 09:00:00',  40, 'ram',     'Xeon'),  -- non-auction: gone
          (5, NULL,                             30, 'auction', 'Xeon')   -- null seen: excluded
        """
    )
    con.close()


def test_attach_snapshot_dedups_filters_and_excludes_null_seen(tmp_path: Path):
    db = tmp_path / "sb.duckdb"
    _make_radar_db(db)

    con = convert.attach_snapshot(db)
    try:
        assert convert.months(con) == ["2026-06", "2026-07"]
        assert convert.total_rows(con) == 3  # price=0, non-auction, null-seen all dropped
        keep = con.execute("SELECT id, day, price FROM auction ORDER BY id, day").fetchall()
        assert keep == [(1, "2026-06-01", 90), (1, "2026-06-02", 85), (2, "2026-07-15", 50)]
    finally:
        con.close()


def test_merge_first_run_dedups_new(tmp_path: Path):
    con = _con_with_auction(
        [
            (1, "2026-06-01 08:00:00", 100),
            (1, "2026-06-01 20:00:00", 90),  # same (id, day), later -> kept
        ]
    )
    out = tmp_path / "data.parquet"
    stats = convert.merge_month(con, "2026-06", None, out)
    con.close()
    assert stats.keys == 1
    assert _read(out) == [(1, "2026-06-01", 90)]


def test_merge_price_change_same_day_keeps_latest(tmp_path: Path):
    existing = tmp_path / "existing.parquet"
    _write_shard(existing, [(1, "2026-06-01 08:00:00", 100)])
    con = _con_with_auction([(1, "2026-06-01 20:00:00", 90)])  # later, cheaper
    out = tmp_path / "data.parquet"

    stats = convert.merge_month(con, "2026-06", existing, out)
    con.close()

    assert stats.keys == 1
    assert _read(out) == [(1, "2026-06-01", 90)]  # latest observation wins


def test_merge_new_day_appends(tmp_path: Path):
    existing = tmp_path / "existing.parquet"
    _write_shard(existing, [(1, "2026-06-01 20:00:00", 90)])
    con = _con_with_auction([(1, "2026-06-02 20:00:00", 85)])
    out = tmp_path / "data.parquet"

    convert.merge_month(con, "2026-06", existing, out)
    con.close()
    assert _read(out) == [(1, "2026-06-01", 90), (1, "2026-06-02", 85)]


def test_merge_preserves_aging_out_days_dropped_from_new(tmp_path: Path):
    # Trailing-month scenario: `existing` holds days that have aged out of Radar's
    # window (so the new snapshot no longer carries them). The union must keep them.
    existing = tmp_path / "existing.parquet"
    _write_shard(
        existing,
        [(1, "2026-04-01 20:00:00", 100), (1, "2026-04-20 20:00:00", 90)],
    )
    con = _con_with_auction([(1, "2026-04-20 21:00:00", 88)])  # only the tail day, updated
    out = tmp_path / "data.parquet"

    convert.merge_month(con, "2026-04", existing, out)
    con.close()
    assert _read(out) == [(1, "2026-04-01", 100), (1, "2026-04-20", 88)]  # early day survived


def test_unchanged_month_rerun_is_byte_identical(tmp_path: Path):
    # The property HF content-addressing relies on: re-running an already-merged
    # month (existing shard + the same new rows) reproduces identical bytes.
    con = _con_with_auction([(2, "2026-06-02 09:00:00", 50), (1, "2026-06-01 20:00:00", 90)])
    first = tmp_path / "first.parquet"
    convert.merge_month(con, "2026-06", None, first)

    second = tmp_path / "second.parquet"
    convert.merge_month(con, "2026-06", first, second)  # real path: existing + new -> COPY
    con.close()
    assert first.read_bytes() == second.read_bytes()


def test_merge_is_byte_deterministic_across_calls(tmp_path: Path):
    # Two full merges of identical (existing + new) inputs -> identical bytes,
    # regardless of input row order (ORDER BY id, day is the total order).
    existing = tmp_path / "existing.parquet"
    _write_shard(existing, [(3, "2026-06-03 10:00:00", 70), (1, "2026-06-01 10:00:00", 90)])
    con = _con_with_auction([(2, "2026-06-02 09:00:00", 50), (1, "2026-06-01 20:00:00", 88)])

    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    convert.merge_month(con, "2026-06", existing, a)
    convert.merge_month(con, "2026-06", existing, b)
    con.close()
    assert a.read_bytes() == b.read_bytes()


def test_null_id_stays_bigint_not_upcast(tmp_path: Path):
    # Pure DuckDB keeps `id` BIGINT even with NULLs — no pandas int->float upcast.
    con = _con_with_auction([(None, "2026-06-01 10:00:00", 90), (1, "2026-06-02 10:00:00", 85)])
    out = tmp_path / "data.parquet"
    convert.merge_month(con, "2026-06", None, out)
    types = {
        name: typ
        for name, typ, *_ in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()
    }
    con.close()
    assert types["id"] == "BIGINT"


@given(prices=st.lists(st.integers(min_value=1, max_value=100000), min_size=1, max_size=8))
def test_merge_keeps_latest_across_repeated_same_day_observations(
    tmp_path_factory: pytest.TempPathFactory, prices: list[int]
):
    specs = [(1, f"2026-06-01 {h:02d}:00:00", p) for h, p in enumerate(prices)]
    con = _con_with_auction(specs)
    out = tmp_path_factory.mktemp("shard") / "data.parquet"
    stats = convert.merge_month(con, "2026-06", None, out)
    con.close()
    assert stats.keys == 1
    assert _read(out) == [(1, "2026-06-01", prices[-1])]  # latest seen wins


def test_check_monotonic_allows_growth():
    _check_monotonic(  # no raise: more distinct keys, same tail
        "2026-04",
        merged=ShardStats(keys=10, min_day="2026-04-01"),
        prev=ShardStats(keys=8, min_day="2026-04-01"),
    )


def test_check_monotonic_blocks_row_shrink():
    with pytest.raises(ExternalServiceError, match="truncate aged-out history"):
        _check_monotonic(
            "2026-04",
            merged=ShardStats(keys=3, min_day="2026-04-01"),
            prev=ShardStats(keys=8, min_day="2026-04-01"),
        )


def test_check_monotonic_blocks_tail_moving_forward():
    # Same key count but the earliest day advanced — early history was dropped.
    with pytest.raises(ExternalServiceError, match="truncate aged-out history"):
        _check_monotonic(
            "2026-04",
            merged=ShardStats(keys=8, min_day="2026-04-15"),
            prev=ShardStats(keys=8, min_day="2026-04-01"),
        )


def test_merge_month_guard_raises_and_cleans_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The guard can't fire on honest input (the merge is always a superset), so we
    # drive it via the real merge_month entry point by making the existing shard's
    # stats look larger than the merged output — simulating a merge-logic regression.
    existing = tmp_path / "existing.parquet"
    _write_shard(existing, [(1, "2026-04-02 10:00:00", 90)])
    con = _con_with_auction([(1, "2026-04-02 20:00:00", 88)])
    out = tmp_path / "data.parquet"

    real_stats = convert.shard_stats

    def fake_stats(c: DuckDBPyConnection, path: Path) -> ShardStats:
        if path == existing:  # pretend the published shard had far more history
            return ShardStats(keys=99, min_day="2026-01-01")
        return real_stats(c, path)

    monkeypatch.setattr(convert, "shard_stats", fake_stats)

    with pytest.raises(ExternalServiceError, match="truncate aged-out history"):
        _ = convert.merge_month(con, "2026-04", existing, out)
    con.close()
    assert not out.exists()  # never committed
    assert not out.with_name(out.name + ".tmp").exists()  # scratch cleaned up


def test_merge_tolerates_schema_drift(tmp_path: Path):
    # An existing shard carrying a column the new snapshot lacks must merge (the
    # extra column NULL-fills), not raise a raw DuckDB binder error.
    existing = tmp_path / "existing.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t ({_SCHEMA}, extra VARCHAR)")
    con.executemany(
        "INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)",
        [(*r, "legacy") for r in _rows([(1, "2026-06-01 10:00:00", 100)])],
    )
    con.execute(f"COPY (SELECT * FROM t) TO '{existing}' (FORMAT PARQUET)")
    con.close()

    aucon = _con_with_auction([(1, "2026-06-02 10:00:00", 85)])  # no `extra` column
    out = tmp_path / "data.parquet"
    stats = convert.merge_month(aucon, "2026-06", existing, out)
    verify = duckdb.connect()
    cols = {
        n for n, *_ in verify.execute(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()
    }
    verify.close()
    aucon.close()

    assert stats.keys == 2  # both days retained
    assert "extra" in cols  # drifted column preserved, NULL-filled for the new row


def test_seen_tie_broken_deterministically_and_byte_identical(tmp_path: Path):
    # Two rows with the SAME (id, day, seen) but different price: the dedup tie is
    # broken by price DESC, and repeated merges are byte-identical (no churn).
    con = _con_with_auction(
        [
            (1, "2026-06-01 12:00:00", 90),
            (1, "2026-06-01 12:00:00", 100),  # identical seen, higher price
        ]
    )
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    convert.merge_month(con, "2026-06", None, a)
    convert.merge_month(con, "2026-06", None, b)
    con.close()
    assert _read(a) == [(1, "2026-06-01", 100)]  # price DESC wins the tie
    assert a.read_bytes() == b.read_bytes()


class _CapLog:
    """Minimal capturing logger; sidesteps structlog's first-use logger cache."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **kw: object) -> None:
        self.events.append((event, kw))

    def warning(self, event: str, **kw: object) -> None:
        self.events.append((event, kw))


def test_attach_snapshot_logs_dropped_null_seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "sb.duckdb"
    _make_radar_db(db)  # contains exactly one auction row with NULL seen
    cap = _CapLog()
    monkeypatch.setattr(convert, "log", cap)

    con = convert.attach_snapshot(db)
    con.close()

    auction = [kw for event, kw in cap.events if event == "convert.auction"]
    assert auction and auction[0]["dropped_null_seen"] == 1
    assert auction[0]["kept"] == 3
