#!/usr/bin/env python3
"""
One-click Historical DB prep + optional backtest runner.

Features:
- Deduplicate OHLCV data (per pair/timeframe)
- Detect and optionally fill gaps (requires exchange access)
- Optional: run backtesting after prep

Examples:
  python scripts/hdb_backtest.py \
    --config user_data/config.json \
    --datadir "historical data" \
    --pairs BTC/USDT ETH/USDT \
    --timeframes 5m 1h \
    --timerange 20240101-20240701 \
    --fix --fill-gaps --jobs 4 \
    --backtest

Note: Designed for local/personal use. Avoid running with public exposure.
"""
from __future__ import annotations

import argparse
import subprocess
from typing import Any
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.data.quality import scan_ohlcv_quality
from freqtrade.enums import TradingMode, CandleType
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.data.converter import clean_ohlcv_dataframe
import pandas as pd


def fill_gaps_for_report(report: dict[str, dict[str, Any]], config: dict) -> None:
    """
    Fill gaps reported by scan_ohlcv_quality using exchange.get_historic_ohlcv.
    """
    gaps_to_fill: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for pair, tf_map in report.items():
        for tf, res in tf_map.items():
            for gap in res.get("gaps", []):
                gaps_to_fill.setdefault((pair, tf), []).append((gap["start"], gap["end"]))

    if not gaps_to_fill:
        print("No gaps detected.")
        return

    exchange = ExchangeResolver.load_exchange(config, validate=False)
    dh = get_datahandler(config["datadir"], config["dataformat_ohlcv"])
    candle_type = CandleType.get_default(config.get("trading_mode", TradingMode.SPOT))

    total_filled = 0
    for (pair, timeframe), ranges in gaps_to_fill.items():
        for start_s, end_s in ranges:
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
                    total_filled += len(df_new)
                    print(
                        f"Filled gap for {pair} {timeframe}: {start_s}-{end_s} (+{len(df_new)} candles)"
                    )
            except Exception as e:  # noqa: BLE001
                print(f"WARN: Failed to fill gap for {pair} {timeframe}: {e}")

    print(f"Gap fill complete. Total candles fetched: {total_filled}")


def run_backtest(
    config_path: str,
    datadir: str,
    timeframes: list[str] | None,
    timerange: str | None,
    strategy: str | None,
    report_out: str | None,
    report_format: str | None,
    summary_out: str | None,
    summary_format: str | None,
) -> int:
    cmd = ["freqtrade", "backtesting", "--config", config_path, "--datadir", datadir]
    # Backtesting expects a single timeframe via -i/--timeframe
    if timeframes:
        tf = timeframes[0]
        cmd.extend(["-i", tf])
    if timerange:
        cmd.extend(["--timerange", timerange])
    if strategy:
        cmd.extend(["--strategy", strategy])
    tmpdir = None
    tmpjson = None
    try:
        if report_out:
            tmpdir = tempfile.mkdtemp(prefix="ft_bt_")
            tmpjson = "report_tmp.json"
            cmd.extend(["--export", "trades", "--export-directory", tmpdir, "--export-filename", tmpjson])
        print("Running backtest:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc == 0 and tmpdir:
            src = f"{tmpdir}/{tmpjson}"
            trades_df = pd.DataFrame()
            try:
                if Path(src).exists():
                    with open(src, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    trades = data if isinstance(data, list) else data.get("trades", [])
                    trades_df = pd.DataFrame(trades)
                else:
                    # Try to read from zip artifact produced by backtest
                    zips = list(Path(tmpdir).glob("backtest-result-*.zip"))
                    if zips:
                        import zipfile

                        with zipfile.ZipFile(zips[0]) as zf:
                            # Common path inside zip: trades.json
                            cand = [n for n in zf.namelist() if n.endswith("trades.json")]
                            if cand:
                                with zf.open(cand[0]) as f:
                                    data = json.load(f)
                                trades = data if isinstance(data, list) else data.get("trades", [])
                                trades_df = pd.DataFrame(trades)
            except Exception:
                pass
            # Write detailed report
            if report_out:
                if report_format == "csv" and not trades_df.empty:
                    trades_df.to_csv(report_out, index=False)
                elif report_format == "json":
                    if Path(src).exists():
                        shutil.move(src, report_out)
                    elif not trades_df.empty:
                        # Write reconstructed trades JSON
                        with open(report_out, "w", encoding="utf-8") as f:
                            json.dump(trades_df.to_dict(orient="records"), f, ensure_ascii=False)
            # Write summary
            if summary_out:
                summary = {}
                try:
                    n = len(trades_df) if not trades_df.empty else 0
                    wins = (trades_df.get("profit_ratio", 0) > 0).sum()
                    losses = (trades_df.get("profit_ratio", 0) <= 0).sum()
                    sum_profit_abs = trades_df.get("profit_abs", pd.Series(dtype=float)).sum()
                    sum_profit_ratio = trades_df.get("profit_ratio", pd.Series(dtype=float)).sum()
                    mean_profit_ratio = trades_df.get("profit_ratio", pd.Series(dtype=float)).mean()
                    # PF
                    pos_sum = trades_df.loc[trades_df.get("profit_abs", 0) > 0, "profit_abs"].sum()
                    neg_sum = -trades_df.loc[trades_df.get("profit_abs", 0) <= 0, "profit_abs"].sum()
                    pf = float(pos_sum / neg_sum) if neg_sum else None
                    summary = {
                        "trades": int(n),
                        "wins": int(wins),
                        "losses": int(losses),
                        "win_rate": float(wins / n) if n else None,
                        "sum_profit_abs": float(sum_profit_abs),
                        "sum_profit_ratio": float(sum_profit_ratio) if pd.notna(sum_profit_ratio) else None,
                        "mean_profit_ratio": float(mean_profit_ratio) if pd.notna(mean_profit_ratio) else None,
                        "profit_factor": pf,
                    }
                    if summary_format == "csv":
                        pd.DataFrame([summary]).to_csv(summary_out, index=False)
                    else:
                        with open(summary_out, "w", encoding="utf-8") as f:
                            json.dump(summary, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        return rc
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Historical DB one-click prep + backtest")
    ap.add_argument("--config", "-c", required=True, help="Path to config.json")
    ap.add_argument("--datadir", "-d", required=True, help="Historical data folder")
    ap.add_argument("--pairs", "-p", nargs="+", help="Pairs list e.g. BTC/USDT ETH/USDT")
    ap.add_argument("--timeframes", "-t", nargs="+", help="Timeframes e.g. 5m 1h")
    ap.add_argument("--timerange", help="Time window e.g. 20240101-20240701")
    ap.add_argument("--fix", action="store_true", help="Deduplicate candles")
    ap.add_argument("--fill-gaps", action="store_true", help="Fill detected gaps via exchange")
    ap.add_argument("--jobs", "-j", type=int, default=1, help="Parallel jobs for scanning")
    ap.add_argument(
        "--io-profile",
        choices=["ssd", "hdd", "net"],
        help="I/O profile hint (sets recommended jobs if --jobs not provided)",
    )
    ap.add_argument("--backtest", action="store_true", help="Run backtesting after prep")
    ap.add_argument("--strategy", help="Strategy class to use for backtest (e.g. SampleStrategy)")
    ap.add_argument("--report-out", help="Path to save backtest report (json or csv)")
    ap.add_argument("--report-dir", help="Directory to save timestamped report into (auto-created)")
    ap.add_argument("--report-base", default="report", help="Base filename for report when using --report-dir")
    ap.add_argument("--report-format", choices=["json", "csv"], default="json", help="Backtest report format (default: json)")
    ap.add_argument("--summary-out", help="Path to save backtest summary (json or csv)")
    ap.add_argument("--summary-format", choices=["json", "csv"], default="json", help="Backtest summary format (default: json)")
    args = ap.parse_args()

    # Build config via freqtrade config loader to ensure defaults
    cfg_args = {
        "config": [args.config],
        "datadir": args.datadir,
    }
    config = setup_utils_configuration(cfg_args, RunMode.UTIL_EXCHANGE)
    config["datadir"] = Path(args.datadir)

    pairs = args.pairs
    tfs = args.timeframes
    tr = args.timerange

    # I/O hint
    if args.io_profile and args.jobs == 1:
        if args.io_profile == "ssd":
            args.jobs = 4
        elif args.io_profile == "hdd":
            args.jobs = 2
        else:
            args.jobs = 1

    # Scan & dedup
    print("Scanning OHLCV quality (duplicates/gaps)...")
    report = scan_ohlcv_quality(
        config["datadir"],
        config.get("dataformat_ohlcv", "feather"),
        config.get("trading_mode", TradingMode.SPOT),
        pairs=pairs,
        timeframes=tfs,
        fix=bool(args.fix),
        jobs=args.jobs,
        timerange=tr,
    )
    total_dups = sum(
        int(r.get("duplicate_count", 0)) for m in report.values() for r in m.values()
    )
    total_gaps = sum(len(r.get("gaps", [])) for m in report.values() for r in m.values())
    print(f"Scan complete. Duplicates: {total_dups}, gaps: {total_gaps}")

    if args.fill_gaps:
        fill_gaps_for_report(report, config)

    if args.backtest:
        # Determine report/summary paths
        report_out = args.report_out
        summary_out = args.summary_out
        if args.report_dir:
            rpt_dir = Path(args.report_dir)
            rpt_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            ext = args.report_format
            report_out = str(rpt_dir / f"{args.report_base}_{ts}.{ext}")
            # summary default next to report unless explicitly provided
            if not args.summary_out:
                sext = args.summary_format
                summary_out = str(rpt_dir / f"{args.report_base}_{ts}_summary.{sext}")
        rc = run_backtest(
            args.config,
            str(config["datadir"]),
            tfs,
            tr,
            args.strategy,
            report_out,
            args.report_format,
            summary_out,
            args.summary_format,
        )
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
