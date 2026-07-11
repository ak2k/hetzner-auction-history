# hetzner-auction-history

A daily-updated, unbounded-history dataset of the [Hetzner server
auction](https://www.hetzner.com/sb) (Serverbörse), published to Hugging Face as
Hive-partitioned Parquet — queryable directly from DuckDB.

Hetzner sells returned/refurbished dedicated servers via a live auction whose
prices drift down until each box sells. [Server Radar](https://radar.iodev.org/)
scrapes the board every 5 minutes and publishes a rolling **90-day** DuckDB
snapshot — but it purges anything older. This project downloads that snapshot
daily and **upserts** it into a growing Parquet archive keyed on `(id, day)`, so
the long-run history (base drift, seasonality, per-spec trends) stays queryable
long after Radar forgets it.

**Dataset:** <https://huggingface.co/datasets/trader298/hetzner-auction-history>

## How it works

```
Radar DuckDB snapshot  ->  DuckDB: dedup (id,day), merge existing partition  ->  Hugging Face
   sb.duckdb                                                    data/month=YYYY-MM/data.parquet
```

- **One row per `(id, day)`.** Radar's `server` table holds every 5-minute
  observation; we filter to `server_type='auction' AND price>0` and keep the
  latest observation per auction per calendar day.
- **Hive-partitioned by month.** Each month is one Parquet file at
  `data/month=YYYY-MM/data.parquet`, written deterministically (rows sorted by
  `(id, day)`, pinned zstd) so an unchanged month is byte-identical run to run.
- **Idempotent upsert.** Each run merges the existing partition with the new
  rows and de-duplicates on `(id, day)`, keeping the latest `seen`. Hugging
  Face's content-addressed upload skips byte-identical partitions, so only
  months that actually changed move. Months aged out of Radar's 90-day window
  are never in a snapshot again and are never rewritten — that retained tail is
  the whole point, and a monotonic-tail guard refuses any write that would
  shrink or advance the start of a published partition (so a transient upstream
  404 can't truncate history).

> **Prices are gross** (incl. 19% German VAT), matching Radar. Divide by `1.19`
> for the net price a non-EU buyer pays.

## Usage

Query the published dataset in place — no download, with partition pruning on
`month`:

```sql
-- DuckDB
SELECT month, count(*) AS rows
FROM read_parquet(
  'hf://datasets/trader298/hetzner-auction-history/data/**/*.parquet',
  hive_partitioning = true
)
GROUP BY month ORDER BY month;
```

or load it with 🤗 `datasets`:

```python
from datasets import load_dataset

ds = load_dataset("trader298/hetzner-auction-history", split="train")
```

Run the sync yourself:

```bash
# Merge the month partitions to ./_local and skip Hugging Face entirely:
uv run hetzner-auction-history sync --dry-run     # (alias: --no-upload)

# Upsert into the dataset (needs a write token):
export HF_TOKEN=...
export HF_REPO_ID="<user>/hetzner-auction-history"
uv run hetzner-auction-history sync
```

The Radar snapshot is fetched with a browser `User-Agent` — Cloudflare fronts
the static host and 403s a default client. The file is public; this is not an
auth bypass.

## Stack

Built on the [python-starter](https://github.com/ak2k/python-starter) template:
`uv`, `ruff` + `basedpyright` strict, Pydantic v2 at boundaries, `httpx`,
`structlog`, `typer`, domain errors (`hetzner_auction_history.errors.AppError`).
The read -> dedup -> merge -> write is pure DuckDB, isolated in `convert.py`. See
`AGENTS.md` for the full contract.

## Develop

```bash
make check    # ruff + basedpyright strict + pytest (CI runs this)
make fix      # autofix + format, then check
nix develop   # optional: uv + make in a pinned shell
```

## CI

`.github/workflows/sync.yml` runs daily (and on dispatch): a single `sync` job
downloads the Radar snapshot, upserts the changed month partitions, and
regenerates + uploads the dataset card.

Configure:
- var `HF_REPO_ID` — the target dataset, e.g. `trader298/hetzner-auction-history`.
- secret `HF_TOKEN` — Hugging Face write token.

## Layout

| Path | |
|------|--|
| `src/hetzner_auction_history/radar.py` | download Radar's DuckDB snapshot (httpx, browser-UA, retry) |
| `src/hetzner_auction_history/convert.py` | read + `(id, day)`-dedup + per-partition merge/write, all in DuckDB |
| `src/hetzner_auction_history/hf.py` | Hugging Face partition download/upload + card upload + the tail guard |
| `src/hetzner_auction_history/card.py` | generate the dataset card |
| `src/hetzner_auction_history/cli.py` | `hetzner-auction-history` CLI (typer) |
| `src/hetzner_auction_history/config.py` | typed settings (pydantic-settings) |

## Attribution

The underlying facts are Hetzner's public auction listings. The 5-minute scrape
and the CPU-spec enrichment (benchmarks, generation, structured disks) are the
work of **[Server Radar](https://radar.iodev.org/) by @elsbrock**
([elsbrock/hetzner-radar](https://github.com/elsbrock/hetzner-radar)) — this
project is a downstream archive of that feed and would not exist without it.

Not affiliated with or endorsed by Hetzner or Server Radar. "Hetzner" is a
trademark of its owner. Data provided as-is, no correctness guarantee.

## License

- **Code** (this repo): MIT.
- **Data** (the HF dataset): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
  — use freely, credit Server Radar and this dataset.
