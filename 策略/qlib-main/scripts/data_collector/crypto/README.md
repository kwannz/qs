Crypto Data Collector (Experimental)

Overview
- Fetches OHLCV from centralized exchanges (via ccxt) for symbols like BTC/USDT, ETH/USDT.
- Normalizes to Qlib’s expected CSV schema and dumps to Qlib binary format using scripts/dump_bin.py.
- Generates calendars automatically from data (24/7 supported) and a single instrument list.

Quick Start
- Install deps: `pip install -r scripts/data_collector/crypto/requirements.txt`
- Download data (1d):
  - `python scripts/data_collector/crypto/collector.py download_data --source_dir ~/.qlib/crypto/source --exchange binance --symbols BTC/USDT,ETH/USDT --interval 1d --start 2018-01-01 --end 2099-01-01`
- Optional: Download data (1min):
  - `python scripts/data_collector/crypto/collector.py download_data --source_dir ~/.qlib/crypto/source_1m --exchange binance --symbols BTC/USDT --interval 1min --start 2021-01-01 --end 2099-01-01`
- Normalize data to Qlib-style CSV (adds change, factor):
  - `python scripts/data_collector/crypto/collector.py normalize_data --source_dir ~/.qlib/crypto/source --normalize_dir ~/.qlib/crypto/normalize --interval 1d`
- Dump to Qlib binary dataset:
  - `python scripts/dump_bin.py dump_all --data_path ~/.qlib/crypto/normalize --qlib_dir ~/.qlib/qlib_data/crypto_data --freq day --date_field_name date --symbol_field_name symbol --file_suffix .csv`
  - For 1min: `python scripts/dump_bin.py dump_all --data_path ~/.qlib/crypto/normalize_1m --qlib_dir ~/.qlib/qlib_data/crypto_data_1min --freq 1min`

Import Local CoinGlass/Freqtrade Data
- Normalize Parquet/CSV to Qlib CSVs:
  - `python scripts/data_collector/crypto/import_freqtrade.py import_coinglass --source_root "/Users/you/Downloads/freqtrade-develop/historical data" --normalize_dir ~/.qlib/crypto/normalize`
- Then dump to Qlib format (day):
  - `python scripts/dump_bin.py dump_all --data_path ~/.qlib/crypto/normalize --qlib_dir ~/.qlib/qlib_data/crypto_data --freq day --date_field_name date --symbol_field_name symbol --file_suffix .csv`

Use in Qlib
- Init:
  - `qlib.init(provider_uri={"day": "~/.qlib/qlib_data/crypto_data", "1min": "~/.qlib/qlib_data/crypto_data_1min"}, dataset_cache=None, expression_cache=None, region="us")`
- Query:
  - `D.features(D.instruments("all"), ["$open","$high","$low","$close","$volume","$change"], start_time="2019-01-01", end_time="2024-01-01", freq="day")`

Notes
- Symbols are normalized to UPPERCASE codes like BINANCE_BTCUSDT.
- change is computed as close.pct_change, factor is fixed at 1.0 for crypto (no splits).
- Limit thresholds are not set by default; set `limit_threshold=None` in backtests or via `C.limit_threshold=None`.
- Fees/lot sizes differ per exchange; adjust backtest costs and `trade_unit=None` as needed.
