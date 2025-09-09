## Historical DB One‑Click Workflow

This guide shows how to prepare a separate historical data folder for personal/local backtesting in one shot, including deduplication, gap detection/filling, and an optional backtest run.

This is intended for local use. Do not expose it on public networks when combined with disabled authentication.

### Prerequisites

- A working Freqtrade installation (virtualenv recommended)
- A configuration file (e.g., `user_data/config.json`)
- Historical data folder (e.g., `historical data/`) containing OHLCV data, or accessible exchange credentials for gap filling

### Helper script

The helper script `scripts/hdb_backtest.py` wraps the following steps:

1. Deduplicate OHLCV data (optional `--fix`)
2. Detect gaps (timeframe boundary scan)
3. Fill gaps precisely per missing window (optional `--fill-gaps`, requires exchange access)
4. Optional backtesting run (`--backtest`)

Common examples:

- Deduplicate + scan:

```
python scripts/hdb_backtest.py \
  --config user_data/config.json \
  --datadir "historical data" \
  --pairs BTC/USDT ETH/USDT \
  --timeframes 5m 1h \
  --timerange 20240101-20240701 \
  --fix --jobs 4
```

- Deduplicate + scan + fill gaps (requires exchange access):

```
python scripts/hdb_backtest.py \
  --config user_data/config.json \
  --datadir "historical data" \
  --pairs BTC/USDT \
  --timeframes 5m \
  --timerange 20240101-20240701 \
  --fix --fill-gaps --jobs 4
```

- Prep + backtest:

```
python scripts/hdb_backtest.py \
  --config user_data/config.json \
  --datadir "historical data" \
  --pairs BTC/USDT \
  --timeframes 5m \
  --timerange 20240110-20240111 \
  --fix --fill-gaps --jobs 4 \
  --backtest --strategy SampleStrategy
```

#### Export backtest reports (JSON/CSV)

You can export backtest trades as JSON or CSV via `--report-out` and `--report-format`:

```
python scripts/hdb_backtest.py \
  --config user_data/config.json \
  --datadir "historical data" \
  --pairs BTC/USDT \
  --timeframes 5m \
  --timerange 20240110-20240111 \
  --fix --jobs 4 \
  --backtest --strategy SampleStrategy \
  --report-out user_data/backtest_results/report.csv \
  --report-format csv
```

Alternatively, you can provide a directory to auto‑create a timestamped filename (UTC):

```
python scripts/hdb_backtest.py \
  --config user_data/config.json \
  --datadir "historical data" \
  --pairs BTC/USDT \
  --timeframes 5m \
  --timerange 20240110-20240111 \
  --backtest --strategy SampleStrategy \
  --report-dir user_data/backtest_results \
  --report-base mybt \
  --report-format json
```

Notes:

- Use `--timerange` to limit both scanning and filling windows.
- Use `--jobs` to parallelize scanning (adjust for disk and CPU; SSD: higher, HDD/network share: lower).
- Gap filling requests historical OHLCV from your configured exchange for each missing window only (not full re-download).

I/O profile hint:

```
# SSD
--io-profile ssd  # sets jobs=4 if --jobs not provided

# HDD / network share
--io-profile hdd  # sets jobs=2 if --jobs not provided
--io-profile net  # sets jobs=1 if --jobs not provided
```

### Makefile shortcuts

You can use Makefile targets for convenience (override variables as needed):

```
make hdb-prep \
  HDB_DATADIR="historical data" \
  HDB_PAIRS="BTC/USDT ETH/USDT" \
  HDB_TFS="5m 1h" \
  HDB_TIMERANGE=20240101-20240701

make hdb-backtest \
  HDB_DATADIR="historical data" \
  HDB_PAIRS="BTC/USDT" \
  HDB_TFS="5m" \
  HDB_TIMERANGE=20240110-20240111 \
  HDB_STRATEGY=SampleStrategy \
  HDB_REPORT_OUT=user_data/backtest_results/report.json \
  HDB_REPORT_FORMAT=json

Or use a timestamped report into a directory:

```
make hdb-backtest \
  HDB_REPORT_DIR=user_data/backtest_results \
  HDB_REPORT_BASE=mybt \
  HDB_REPORT_FORMAT=csv
```
```

#### Custom profile

You can create a local profile to persist your preferred variables:

```
cp Makefile.hdb.example Makefile.local
# then edit Makefile.local to set HDB_* variables.

# or generate via make target:
make hdb-create-profile
```

`Makefile.local` is included automatically and is not tracked by git.

### Import existing data (CSV/JSON) into historical folder

If you already have data in CSV/JSON and want to migrate it to your historical folder:

- Convert OHLCV formats:

```
freqtrade convert-data \
  --config user_data/config.json \
  --datadir "historical data" \
  --format-from json \
  --format-to feather \
  --timeframes 5m 1h \
  --erase
```

- Convert trades formats:

```
freqtrade convert-trade-data \
  --config user_data/config.json \
  --datadir "historical data" \
  --format-from json \
  --format-to feather \
  --erase
```

- Convert trades to OHLCV (if exchange lacks historical klines):

```
freqtrade trades-to-ohlcv \
  --config user_data/config.json \
  --datadir "historical data" \
  --timeframes 5m 1h \
  --pairs BTC/USDT ETH/USDT
```

See also: the sections on Data Downloading and the `convert-data` / `trades-to-ohlcv` commands for more details.

### FAQ / Troubleshooting

- Gap filling is slow
  - Use `--timerange` to reduce the window.
  - Limit pairs/timeframes to those you actually need.
  - Ensure network and exchange rate limits are not throttling you.

- Gaps still present after fill
  - Some exchanges have inconsistent historical coverage. Run the script again or expand the timerange slightly.
  - Verify data boundaries align with the timeframe (5m candles must be 00/05/10...).

- Too many I/O operations during scan
  - Reduce `--jobs` (e.g. 2) for HDD/network shares; higher values (4–8) are fine for SSD.

- I want to disable API auth for personal/local mode
  - Set in your config:
    ```
    "api_server": {
      "auth_disabled": true,
      "enable_rate_limit": false
    }
    ```
  - Do not expose this to public networks.

### Personal/Local mode (optional)

For purely personal/local setups, you can disable API authentication by setting:

```
"api_server": {
  "auth_disabled": true,
  "enable_rate_limit": false
}
```

This must not be used on public networks or shared environments.
