"""Generate the HuggingFace dataset card (README.md).

The YAML frontmatter declares a single ``default`` config globbing the
Hive-partitioned shards (``data/**/*.parquet``) so the dataset viewer and
``datasets.load_dataset`` see one train split spanning all months. The body
carries the schema summary, usage, the VAT note, and — prominently — Server
Radar attribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger(__name__)  # pyright: ignore[reportAny]

_FRONTMATTER = """\
---
license: cc-by-4.0
pretty_name: Hetzner Server Auction — Price History
tags:
- hetzner
- servers
- dedicated-servers
- auction
- pricing
- time-series
- infrastructure
configs:
- config_name: default
  data_files:
  - split: train
    path: data/**/*.parquet
---
"""

_BODY = """\
# Hetzner Server Auction — Price History

A daily-growing, **unbounded-history** archive of the [Hetzner server
auction](https://www.hetzner.com/sb) (Serverbörse): every listing observed on
every day, kept long past the rolling 90-day window that the upstream feed
retains. One row per auction listing (`id`) per calendar day (`day`), taking the
latest observation that day.

Hetzner sells returned/refurbished dedicated servers via a live auction whose
prices drift down until each box sells. This dataset preserves the long-run
price history — base drift, seasonality, per-spec trends — as month-partitioned
Parquet, queryable directly from DuckDB or 🤗 `datasets`.

## Attribution — built on Server Radar

The underlying facts are Hetzner's public auction listings. The 5-minute scrape
**and** the CPU-spec enrichment (Geekbench-style benchmarks, CPU generation,
structured disk layout) are the work of **[Server
Radar](https://radar.iodev.org/) by @elsbrock**
([elsbrock/hetzner-radar](https://github.com/elsbrock/hetzner-radar)), which
publishes the rolling-90-day DuckDB snapshot this archive accumulates. This
dataset is a downstream, long-term mirror of that feed and would not exist
without it — please credit Server Radar.

Not affiliated with or endorsed by Hetzner or Server Radar. "Hetzner" is a
trademark of its owner. Data provided as-is, with no correctness guarantee.

## Structure

```
data/month=YYYY-MM/data.parquet   # one Hive-partitioned shard per calendar month
```

`month` is a Hive partition column (also stored in the file). Each row is one
listing observed on one day. Columns follow Radar's `server` schema (~39
columns), including:

- **identity / time**: `id`, `seen` (observation timestamp), `day` (`YYYY-MM-DD`,
  the per-day key), `month` (`YYYY-MM`, the shard partition)
- **CPU**: `cpu`, `cpu_generation`, `cpu_cores`, `cpu_threads`, `cpu_score`
  (single-core), `cpu_multicore_score`
- **memory / disk**: `ram_size`, `is_ecc`, `nvme_drives` / `nvme_size`,
  `sata_drives` / `sata_size`, `hdd_drives` / `hdd_size`
- **misc / auction**: `datacenter`, `is_highio`, `with_gpu`, `price`,
  `setup_price`

> **Prices are gross** (include 19% German VAT), matching Radar. Divide `price`
> by `1.19` for the net price a non-EU buyer pays.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
```

Or query the shards in place with DuckDB — no download, `hive_partitioning=true`
exposes `month` as a column and prunes by it over the whole history:

```sql
-- cheapest listing seen each day in a given month
SELECT day, min(price / 1.19) AS min_net_eur
FROM read_parquet('hf://datasets/{repo_id}/data/**/*.parquet', hive_partitioning=true)
WHERE month = '2026-06'
GROUP BY day ORDER BY day;

-- price trajectory of one auction id across its life
SELECT day, price, price / 1.19 AS net_eur
FROM read_parquet('hf://datasets/{repo_id}/data/**/*.parquet', hive_partitioning=true)
WHERE id = 1234567 ORDER BY day;
```

## License

- **Data** (this dataset): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
  — use freely, credit Server Radar and this dataset.
- **Code** (the updater): MIT — <https://github.com/ak2k/hetzner-auction-history>.
"""


def render_card(repo_id: str) -> str:
    """Return the full dataset-card text (frontmatter + body) for ``repo_id``."""
    return _FRONTMATTER + "\n" + _BODY.replace("{repo_id}", repo_id)


def write_card(repo_id: str, out_dir: Path) -> Path:
    """Write ``<out_dir>/README.md`` (the dataset card) and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "README.md"
    _ = out.write_text(render_card(repo_id), encoding="utf-8")
    log.info("card.written", repo_id=repo_id)
    return out
