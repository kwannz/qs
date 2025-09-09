import os
from datetime import datetime, timezone

import pandas as pd


def _py_ref_sim(rows, fee=0.0, slippage=0.0, stoploss=-1.0, minimal_roi=None):
    if minimal_roi is None:
        minimal_roi = {}
    # normalize ROI dict keys to int minutes
    roi_items = sorted([(int(k), float(v)) for k, v in minimal_roi.items()], key=lambda x: x[0])

    def roi_target(elapsed_min):
        ret = None
        for m, r in roi_items:
            if m <= elapsed_min:
                ret = r
            else:
                break
        return ret

    trades = []
    in_pos = False
    entry_ts = 0
    entry_price = 0.0
    for r in rows:
        ts, o, h, l, c, ent_l, ex_l, *_ = r
        if not in_pos and ent_l:
            in_pos = True
            entry_ts = int(ts)
            entry_price = float(o) * (1.0 + slippage)
            continue
        if in_pos:
            reason = "exit_signal"
            exit_rate = None
            stop_price = entry_price * (1.0 + stoploss)
            if l <= stop_price:
                exit_rate = stop_price
                reason = "stoploss"
            else:
                elapsed_min = max(0, int((ts - entry_ts) // 60))
                roi_req = roi_target(elapsed_min)
                if roi_req is not None:
                    roi_price = entry_price * (1.0 + roi_req)
                    if h >= roi_price:
                        exit_rate = roi_price
                        reason = "roi"
            if exit_rate is None and ex_l:
                exit_rate = o
                reason = "exit_signal"
            if exit_rate is not None:
                exit_rate = exit_rate * (1.0 - slippage)
                amount = 1.0
                fee_open = entry_price * amount * fee
                fee_close = exit_rate * amount * fee
                profit_abs = (exit_rate - entry_price) * amount - (fee_open + fee_close)
                trades.append({
                    "open_date": datetime.fromtimestamp(entry_ts, tz=timezone.utc),
                    "close_date": datetime.fromtimestamp(int(ts), tz=timezone.utc),
                    "open_rate": entry_price,
                    "close_rate": exit_rate,
                    "profit_abs": profit_abs,
                    "exit_reason": reason,
                })
                in_pos = False
                entry_ts = 0
                entry_price = 0.0
    return pd.DataFrame(trades)


def test_rustffi_core_alignment():
    ft_rust_backtest = __import__("ft_rust_backtest") if __import__("importlib").util.find_spec("ft_rust_backtest") else __import__("pytest").importorskip("ft_rust_backtest")

    # Synthetic one-pair rows: [ts, open, high, low, close, enter_long, exit_long, enter_short, exit_short, enter_tag, exit_tag]
    base = 1_700_000_000  # arbitrary epoch seconds
    rows = [
        [base + 0*300, 100.0, 101.0,  99.5, 100.5, 1, 0, 0, 0, None, None],  # enter at 100
        [base + 1*300, 100.5, 104.5, 100.4, 104.0, 0, 0, 0, 0, None, None],  # ROI could trigger here
        [base + 2*300, 104.0, 104.2,  99.0, 100.0, 0, 1, 0, 0, None, None],  # exit signal (fallback)
    ]

    data = {"BTC/USDT": rows}
    params = {"fee": 0.001, "slippage": 0.0005, "stoploss": -0.2, "minimal_roi": {"0": 0.04}}
    rust = ft_rust_backtest.simulate_trades(data, rows[0][0], rows[-1][0], params)
    rust_df = pd.DataFrame(list(rust))

    py_df = _py_ref_sim(rows, fee=0.001, slippage=0.0005, stoploss=-0.2, minimal_roi={"0": 0.04})

    # Basic alignment checks
    assert len(rust_df) == len(py_df) == 1
    # Compare close_rate within small epsilon
    assert abs(rust_df.iloc[0]["close_rate"] - py_df.iloc[0]["close_rate"]) < 1e-9
    # Compare profit within small epsilon
    assert abs(rust_df.iloc[0]["profit_abs"] - py_df.iloc[0]["profit_abs"]) < 1e-9
    assert rust_df.iloc[0]["exit_reason"] == py_df.iloc[0]["exit_reason"]
