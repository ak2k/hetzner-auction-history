"""Tests for the (id, day) dedup + month-partitioned merge — the core invariant."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from hetzner_auction_history import convert

if TYPE_CHECKING:
    from pathlib import Path


def _make_radar_db(path: Path) -> None:
    """A tiny Radar-shaped `server` table exercising dedup + the filters."""
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE server ("
        "  id BIGINT, seen TIMESTAMP, price DOUBLE, server_type VARCHAR, cpu VARCHAR"
        ")"
    )
    con.execute(
        """
        INSERT INTO server VALUES
          (1, TIMESTAMP '2026-06-01 08:00:00', 100.0, 'auction', 'Xeon'),  -- same day, earlier
          (1, TIMESTAMP '2026-06-01 20:00:00',  90.0, 'auction', 'Xeon'),  -- same day, kept
          (1, TIMESTAMP '2026-06-02 08:00:00',  85.0, 'auction', 'Xeon'),  -- next day
          (2, TIMESTAMP '2026-07-15 12:00:00',  50.0, 'auction', 'Ryzen'), -- other month
          (3, TIMESTAMP '2026-06-01 09:00:00',   0.0, 'auction', 'Xeon'),  -- price 0: filtered
          (4, TIMESTAMP '2026-06-01 09:00:00',  40.0, 'ram',     'Xeon')   -- non-auction: gone
        """
    )
    con.close()


def _keys(frame: pd.DataFrame) -> set[tuple[int, str]]:
    return {(int(i), str(d)) for i, d in zip(frame["id"], frame["day"], strict=True)}


def test_read_auction_frame_dedups_and_filters(tmp_path: Path):
    db = tmp_path / "sb.duckdb"
    _make_radar_db(db)

    frame = convert.read_auction_frame(db)

    assert _keys(frame) == {(1, "2026-06-01"), (1, "2026-06-02"), (2, "2026-07-15")}
    assert len(frame) == 3  # price=0 and non-auction rows dropped

    # The later observation on 2026-06-01 wins (price 90, not 100).
    same_day = frame[(frame["id"] == 1) & (frame["day"] == "2026-06-01")]
    assert float(same_day["price"].iloc[0]) == 90.0
    assert set(frame["month"]) == {"2026-06", "2026-07"}


def test_partition_by_month_splits_and_sorts(tmp_path: Path):
    db = tmp_path / "sb.duckdb"
    _make_radar_db(db)
    frame = convert.read_auction_frame(db)

    parts = convert.partition_by_month(frame)
    assert [m for m, _ in parts] == ["2026-06", "2026-07"]
    assert [len(g) for _, g in parts] == [2, 1]


def _frame(rows: list[tuple[int, str, str, float]]) -> pd.DataFrame:
    """Build a shard-shaped frame from (id, day, seen, price) tuples."""
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "day": [r[1] for r in rows],
            "seen": pd.to_datetime([r[2] for r in rows]),
            "price": [r[3] for r in rows],
        }
    )


def _price(frame: pd.DataFrame, id_: int, day: str) -> float:
    sel = frame[(frame["id"] == id_) & (frame["day"] == day)]
    return float(sel["price"].iloc[0])


def test_merge_first_run_dedups_new():
    new = _frame(
        [
            (1, "2026-06-01", "2026-06-01 08:00", 100.0),
            (1, "2026-06-01", "2026-06-01 20:00", 90.0),  # same key, later
        ]
    )
    merged = convert.merge_shard(None, new)
    assert len(merged) == 1
    assert _price(merged, 1, "2026-06-01") == 90.0  # latest seen kept


def test_merge_is_idempotent():
    shard = _frame(
        [
            (1, "2026-06-01", "2026-06-01 20:00", 90.0),
            (2, "2026-06-01", "2026-06-01 12:00", 50.0),
        ]
    )
    once = convert.merge_shard(None, shard)
    twice = convert.merge_shard(once, shard.copy())  # re-merging the same rows is a no-op
    assert _keys(twice) == _keys(once)
    assert len(twice) == len(once) == 2
    assert twice.reset_index(drop=True).equals(once.reset_index(drop=True))


def test_merge_price_change_same_day_keeps_latest():
    existing = _frame([(1, "2026-06-01", "2026-06-01 08:00", 100.0)])
    new = _frame([(1, "2026-06-01", "2026-06-01 20:00", 90.0)])  # later, cheaper

    merged = convert.merge_shard(existing, new)

    assert len(merged) == 1  # still one row for the (id, day)
    assert _price(merged, 1, "2026-06-01") == 90.0  # latest observation wins


def test_merge_new_day_appends():
    existing = _frame([(1, "2026-06-01", "2026-06-01 20:00", 90.0)])
    new = _frame([(1, "2026-06-02", "2026-06-02 20:00", 85.0)])

    merged = convert.merge_shard(existing, new)

    assert _keys(merged) == {(1, "2026-06-01"), (1, "2026-06-02")}


def test_merge_preserves_aging_out_days_dropped_from_new():
    # The trailing month scenario: `existing` holds days that have since aged out
    # of Radar's window (so `new` no longer carries them). The union must keep
    # them — this is why the download+merge is load-bearing, not cosmetic.
    existing = _frame(
        [
            (1, "2026-04-01", "2026-04-01 20:00", 100.0),  # aged out of `new`
            (1, "2026-04-20", "2026-04-20 20:00", 90.0),
        ]
    )
    new = _frame([(1, "2026-04-20", "2026-04-20 21:00", 88.0)])  # only the tail day, updated

    merged = convert.merge_shard(existing, new)

    assert _keys(merged) == {(1, "2026-04-01"), (1, "2026-04-20")}
    assert _price(merged, 1, "2026-04-01") == 100.0  # survived
    assert _price(merged, 1, "2026-04-20") == 88.0  # latest wins


@given(
    prices=st.lists(
        st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_merge_keeps_latest_across_repeated_same_day_observations(prices: list[float]):
    # A day observed N times (increasing `seen`) must collapse to the last price,
    # and re-merging that result must leave the same single row.
    rows = [(1, "2026-06-01", f"2026-06-01 {h:02d}:00", p) for h, p in enumerate(prices)]
    observed = _frame(rows)

    merged = convert.merge_shard(None, observed)
    assert len(merged) == 1
    assert _price(merged, 1, "2026-06-01") == prices[-1]

    remerged = convert.merge_shard(merged, merged.copy())
    assert len(remerged) == 1
    assert _price(remerged, 1, "2026-06-01") == prices[-1]


def test_write_shard_roundtrips(tmp_path: Path):
    shard = _frame([(1, "2026-06-01", "2026-06-01 20:00", 90.0)])
    path = convert.write_shard(shard, tmp_path / "2026-06.parquet")
    assert path.exists()
    back = pd.read_parquet(path)
    assert list(back["id"]) == [1]
    assert _price(back, 1, "2026-06-01") == 90.0


def test_write_shard_is_byte_deterministic(tmp_path: Path):
    # Identical data -> identical bytes, so HuggingFace's content-addressed
    # upload skips an unchanged shard. Row order in the input must not matter.
    rows = [
        (2, "2026-06-02", "2026-06-02 09:00", 50.0),
        (1, "2026-06-01", "2026-06-01 20:00", 90.0),
    ]
    a = convert.write_shard(_frame(rows), tmp_path / "a.parquet")
    b = convert.write_shard(_frame(list(reversed(rows))), tmp_path / "b.parquet")
    assert a.read_bytes() == b.read_bytes()
