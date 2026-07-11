# hetzner-auction-history

Accumulate the [Hetzner server auction](https://www.hetzner.com/sb) (Serverbörse)
into a public HuggingFace dataset, so the price history survives past [Server
Radar](https://radar.iodev.org/)'s rolling **90-day** purge — as month-partitioned
Parquet, queryable directly from DuckDB.

Hetzner sells returned/refurbished dedicated servers via a live auction whose
prices drift down until each box sells. Server Radar scrapes the board every 5
minutes and publishes a rolling-90-day DuckDB snapshot; this tool downloads that
snapshot daily and **upserts** it into a growing archive, keyed on `(id, day)`,
so the long-run history (base drift, seasonality, per-spec trends) stays
queryable long after Radar forgets it.

## How it works

```
Radar DuckDB snapshot  ──►  DuckDB (dedup to one row per id/day)  ──►  merge  ──►  HuggingFace
   sb.duckdb                                                     data/YYYY-MM.parquet
```

- **One row per `(id, day)`.** Radar's `server` table holds every 5-minute
  observation; we filter to `server_type='auction' AND price>0` and keep the
  latest observation per auction per calendar day.
- **Month-partitioned shards.** Rows are partitioned by `YYYY-MM`; each month is
  one shard `data/YYYY-MM.parquet`.
- **Idempotent merge.** Each run concatenates the existing shard with the new
  rows and de-duplicates on `(id, day)` keeping the latest `seen`. Re-running
  never duplicates; a later-in-the-day price change on a stored day replaces it;
  and only shards that actually changed are re-uploaded — so months aged out of
  Radar's 90-day window (never in a snapshot again) are never touched. That
  retained tail is the whole point.

> **Prices are gross** (incl. 19% German VAT), matching Radar. Divide by `1.19`
> for the net price a non-EU buyer pays.

## Stack

Built on the [python-starter](https://github.com/ak2k/python-starter) template:
`uv`, `ruff` + `basedpyright` strict, Pydantic v2 at boundaries, `httpx`,
`structlog`, `typer`, domain errors (`hetzner_auction_history.errors.AppError`).
DuckDB + pandas do the read/merge, isolated in `convert.py`. See `AGENTS.md` for
the full contract.

## Usage

```bash
# Write the merged month shards to ./_local and skip HuggingFace entirely:
uv run hetzner-auction-history sync --dry-run     # (alias: --no-upload)

# Upsert into the HuggingFace dataset (needs a write token):
export HF_TOKEN=...
export HF_REPO_ID="<user>/hetzner-auction-history"
uv run hetzner-auction-history sync

uv run hetzner-auction-history sync --repo me/ds  # override the target dataset
```

The Radar snapshot is fetched with a browser `User-Agent` — Cloudflare fronts
the static host and 403s a default python client. The file is public; this is
not an auth bypass.

## Develop

```bash
make check    # ruff + basedpyright strict + pytest (CI runs this)
make fix      # autofix + format, then check
nix develop   # optional: uv + make in a pinned shell
```

## CI

`.github/workflows/sync.yml` runs daily (and on dispatch): a single `sync` job
downloads the Radar snapshot, upserts the changed shards, and regenerates +
uploads the dataset card.

Configure:
- var `HF_REPO_ID` — the target dataset, e.g. `ak2k/hetzner-auction-history`.
- secret `HF_TOKEN` — HuggingFace write token.

## Layout

| Path | |
|------|--|
| `src/hetzner_auction_history/radar.py` | download Radar's DuckDB snapshot (httpx, browser-UA, retry) |
| `src/hetzner_auction_history/convert.py` | read + `(id, day)`-dedup + month-partitioned merge (DuckDB/pandas) |
| `src/hetzner_auction_history/hf.py` | HuggingFace shard download/upload + card upload |
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
