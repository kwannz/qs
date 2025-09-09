#!/usr/bin/env python3
"""
Fill gaps in a historical data folder by scanning for missing candles
and fetching only the missing windows from the configured exchange.

Examples:
  python scripts/fill_hdb_gaps.py \
    --config user_data/config.json \
    --datadir "/Users/zhaoleon/Downloads/freqtrade-develop/historical data" \
    --pairs BTC/USDT ETH/USDT \
    --timeframes 5m 1h \
    --timerange 20240101-20240701 \
    --jobs 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from freqtrade.configuration import setup_utils_configuration
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.data.quality import scan_ohlcv_quality
from freqtrade.enums import TradingMode, CandleType, RunMode
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.data.converter import clean_ohlcv_dataframe


def fill_gaps(report: dict[str, dict[str, Any]], config: dict) -> int:
    exchange = ExchangeResolver.load_exchange(config, validate=False)
    dh = get_datahandler(config["datadir"], config.get("dataformat_ohlcv", "feather"))
    candle_type = CandleType.get_default(config.get("trading_mode", TradingMode.SPOT))
    total = 0
    for pair, tf_map in report.items():
        for timeframe, res in tf_map.items():
            for gap in res.get("gaps", []):
                start_s, end_s = gap["start"], gap["end"]
                try:
                    df_new = exchange.get_historic_ohlcv(
                        pair=pair,
                        timeframe=timeframe,
                        since_ms=(start_s + 1) * 1000,
                        is_new_pair=False,
                        candle_type=candle_type,
                        until_ms=end_s * 1000,
                    )
                    df_old = dh.ohlcv_load(
                        pair,
                        timeframe=timeframe,
                        timerange=None,
                        fill_missing=False,
                        drop_incomplete=False,
                        warn_no_data=False,
                        candle_type=candle_type,
                    )
                    if df_new is not None and not df_new.empty:
                        merged = clean_ohlcv_dataframe(
                            pd.concat([df_old, df_new], axis=0),
                            timeframe,
                            pair,
                            fill_missing=False,
                            drop_incomplete=False,
                        )
                        dh.ohlcv_store(pair, timeframe, data=merged, candle_type=candle_type)
                        fetched = len(df_new)
                        total += fetched
                        print(f"Filled {pair} {timeframe}: {start_s}-{end_s} (+{fetched} candles)")
                except Exception as e:  # noqa: BLE001
                    print(f"WARN: Failed to fill {pair} {timeframe} gap {start_s}-{end_s}: {e}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill gaps in historical data folder")
    ap.add_argument("--config", "-c", required=True, help="Path to config.json")
    ap.add_argument("--datadir", "-d", required=True, help="Historical data folder")
    ap.add_argument("--pairs", "-p", nargs="+", required=True, help="Pairs, e.g. BTC/USDT ETH/USDT")
    ap.add_argument("--timeframes", "-t", nargs="+", required=True, help="Timeframes, e.g. 5m 1h")
    ap.add_argument("--timerange", help="Time window for scanning/filling, e.g. 20240101-20240701")
    ap.add_argument("--jobs", "-j", type=int, default=1, help="Parallel jobs for scanning")
    ap.add_argument(
        "--io-profile",
        choices=["ssd", "hdd", "net"],
        help="I/O profile hint (sets recommended jobs if --jobs not provided)",
    )
    args = ap.parse_args()

    cfg_args = {"config": [args.config], "datadir": args.datadir}
    config = setup_utils_configuration(cfg_args, RunMode.UTIL_EXCHANGE)
    config["datadir"] = Path(args.datadir)

    # I/O hint
    if args.io_profile and args.jobs == 1:
        if args.io_profile == "ssd":
            args.jobs = 4
        elif args.io_profile == "hdd":
            args.jobs = 2
        else:
            args.jobs = 1

    print("Scanning OHLCV quality (gap detection)...")
    report = scan_ohlcv_quality(
        config["datadir"],
        config.get("dataformat_ohlcv", "feather"),
        config.get("trading_mode", TradingMode.SPOT),
        pairs=args.pairs,
        timeframes=args.timeframes,
        fix=False,
        jobs=args.jobs,
        timerange=args.timerange,
    )
    total_gaps = sum(len(r.get("gaps", [])) for m in report.values() for r in m.values())
    print(f"Gap detection complete. Gaps found: {total_gaps}")
    if total_gaps == 0:
        print("No gaps to fill.")
        return

    filled = fill_gaps(report, config)
    print(f"Gap filling done. Total candles fetched: {filled}")


if __name__ == "__main__":
    main()
