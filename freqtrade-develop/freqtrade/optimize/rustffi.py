import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from freqtrade.optimize.backtesting import HEADERS


def enabled() -> bool:
    return os.environ.get("FT_USE_RUST_FFI_BACKTEST", "").lower() in ("1", "true", "yes")


def _dt_to_epoch_s(dt: datetime) -> int:
    return int(dt.timestamp())


def _prepare_data_for_rust(
    self_bt, data: Dict[str, list[list[Any]]], start_date, end_date
) -> Dict[str, list[list[Any]]]:
    """
    Convert pandas Timestamps (or datetimes) in first column to epoch seconds for Rust.
    Keep the remaining columns as-is.
    """
    out: Dict[str, list[list[Any]]] = {}
    timeframe_secs = getattr(self_bt, "timeframe_secs", None)
    tf_detail = getattr(self_bt, "timeframe_detail", "")
    detail_data: dict[str, pd.DataFrame] = getattr(self_bt, "detail_data", {}) or {}

    for pair, rows in data.items():
        conv_rows: list[list[Any]] = []
        # Detail timeframe expansion if available
        if tf_detail and pair in detail_data and timeframe_secs:
            df_det = detail_data[pair]
            for i, row in enumerate(rows):
                if not row:
                    continue
                r = list(row)
                ts = r[HEADERS.index("date")]
                if hasattr(ts, "timestamp"):
                    ts_s = int(ts.timestamp())
                else:
                    ts_s = int(ts)
                # window end: next main candle or ts + timeframe
                next_ts_s = (
                    int(rows[i + 1][0].timestamp())
                    if i + 1 < len(rows) and hasattr(rows[i + 1][0], "timestamp")
                    else (
                        int(rows[i + 1][0]) if i + 1 < len(rows) else ts_s + int(timeframe_secs)
                    )
                )
                # slice detail rows
                dfw = df_det[(df_det["date"] >= pd.to_datetime(ts, utc=True)) & (df_det["date"] < pd.to_datetime(next_ts_s, unit="s", utc=True))]
                if dfw.empty:
                    # fallback to main row
                    if r[9] is None:
                        r[9] = ""
                    if r[10] is None:
                        r[10] = ""
                    r[0] = ts_s
                    conv_rows.append(r)
                else:
                    for _, drow in dfw.iterrows():
                        rr = [
                            int(pd.to_datetime(drow["date"]).timestamp()),
                            float(drow["open"]),
                            float(drow["high"]),
                            float(drow["low"]),
                            float(drow["close"]),
                            r[5],
                            r[6],
                            r[7],
                            r[8],
                            r[9] or "",
                            r[10] or "",
                        ]
                        conv_rows.append(rr)
        else:
            for row in rows:
                if not row:
                    continue
                r = list(row)
                ts = r[HEADERS.index("date")]
                if hasattr(ts, "timestamp"):
                    r[0] = int(ts.timestamp())
                else:
                    try:
                        r[0] = int(ts)
                    except Exception:
                        raise ValueError("Unsupported date type in backtest rows for Rust FFI")
                if len(r) > 9 and r[9] is None:
                    r[9] = ""
                if len(r) > 10 and r[10] is None:
                    r[10] = ""
                conv_rows.append(r)
        out[pair] = conv_rows
    return out


def run(self_bt, processed: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> dict:
    """
    Execute Rust FFI backtest. Returns a dict compatible with BacktestContentTypeIcomplete.
    """
    try:
        import ft_rust_backtest
    except Exception as e:
        raise RuntimeError(
            "ft_rust_backtest module not found. Build it via 'maturin develop' in rust/ft_rust_backtest'."
        ) from e

    # Reuse optimized conversion from Backtesting to lists
    data = self_bt._get_ohlcv_as_lists(processed)
    data_conv = _prepare_data_for_rust(self_bt, data, start_date, end_date)

    # Build params
    strat_cfg = getattr(self_bt, "strategy", None)
    cfg = strat_cfg.config if strat_cfg else self_bt.config
    fee = getattr(self_bt, "fee", 0.0) or 0.0
    stoploss = cfg.get("stoploss", -1.0)
    minimal_roi = cfg.get("minimal_roi", {})
    slippage = cfg.get("backtest_slippage", 0.0)

    params = {
        "fee": float(fee),
        "stoploss": float(stoploss),
        "minimal_roi": minimal_roi,
        "slippage": float(slippage),
        "can_short": bool(getattr(self_bt, "_can_short", False)),
        # trailing params if present on strategy
        "trailing_stop": bool(getattr(self_bt.strategy, "trailing_stop", False)),
        "trailing_stop_positive": getattr(self_bt.strategy, "trailing_stop_positive", None),
        "trailing_stop_positive_offset": getattr(
            self_bt.strategy, "trailing_stop_positive_offset", 0.0
        ),
        "trailing_only_offset_is_reached": bool(
            getattr(self_bt.strategy, "trailing_only_offset_is_reached", False)
        ),
        # minimal protections (optional): cooldown after trade in minutes
        "cooldown_minutes": int(self_bt.config.get("ffi_cooldown_minutes", 0)),
        # frequent stoploss protection (optional)
        "freqsl_threshold": int(self_bt.config.get("ffi_freqsl_threshold", 0)),
        "freqsl_lookback_minutes": int(self_bt.config.get("ffi_freqsl_lookback_minutes", 0)),
        "freqsl_cooldown_minutes": int(self_bt.config.get("ffi_freqsl_cooldown_minutes", 0)),
        # account-level max drawdown (fraction, e.g. 0.2 for 20%)
        "max_drawdown_pct": float(self_bt.config.get("ffi_max_drawdown_pct", 0.0)),
        # account-level max drawdown (fraction, e.g. 0.2 for 20%)
        "max_drawdown_pct": float(self_bt.config.get("ffi_max_drawdown_pct", 0.0)),
    }

    # Futures funding fees (optional)
    try:
        futures_data = getattr(self_bt, "futures_data", {}) or {}
    except Exception:
        futures_data = {}
    if futures_data:
        funding_map: dict[str, list[list[float | int]]] = {}
        mark_map: dict[str, list[list[float | int]]] = {}
        for pair, df in futures_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            rate_col = None
            for c in ("funding_rate", "funding", "fr"):
                if c in df.columns:
                    rate_col = c
                    break
            if rate_col is None:
                continue
            lst: list[list[float | int]] = []
            mark_col = None
            for c in ("mark", "mark_price", "mark_close", "index_close", "close"):
                if c in df.columns:
                    mark_col = c
                    break
            for _, row in df.iterrows():
                ts = int(pd.to_datetime(row["date"]).timestamp()) if "date" in df.columns else None
                if ts is None:
                    continue
                try:
                    rate = float(row[rate_col])
                except Exception:
                    continue
                lst.append([ts, rate])
                if mark_col is not None:
                    try:
                        mp = float(row[mark_col])
                        mark_map.setdefault(pair, []).append([ts, mp])
                    except Exception:
                        pass
            if lst:
                funding_map[pair] = lst
        if funding_map:
            params["funding"] = funding_map
            fint = getattr(self_bt, "funding_fee_timeframe_secs", None)
            if fint:
                params["funding_interval"] = int(fint)
        if mark_map:
            params["funding_mark"] = mark_map

    trades_pylist = ft_rust_backtest.simulate_trades(
        data_conv, int(start_date.timestamp()), int(end_date.timestamp()), params
    )

    # Build DataFrame with expected columns
    from freqtrade.data.btanalysis.bt_fileutils import BT_DATA_COLUMNS

    results_df = pd.DataFrame(list(trades_pylist), columns=BT_DATA_COLUMNS)
    if len(results_df) > 0:
        results_df["open_date"] = pd.to_datetime(results_df["open_date"], unit="s", utc=True)
        results_df["close_date"] = pd.to_datetime(results_df["close_date"], unit="s", utc=True)

    # Rust already netted fees and applied slippage in profit_abs and profit_ratio

    return {
        "results": results_df,
        "config": self_bt.strategy.config,
        "locks": [],
        "rejected_signals": 0,
        "timedout_entry_orders": 0,
        "timedout_exit_orders": 0,
        "canceled_trade_entries": 0,
        "canceled_entry_orders": 0,
        "replaced_entry_orders": 0,
        "final_balance": self_bt.wallets.get_total(self_bt.strategy.config["stake_currency"]),
    }
