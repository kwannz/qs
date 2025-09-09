import os
from datetime import datetime, timezone

import pandas as pd


def _py_ref_sim_short(rows, fee=0.0, slippage=0.0, stoploss=-0.1, minimal_roi=None):
    if minimal_roi is None:
        minimal_roi = {}
    items = sorted([(int(k), float(v)) for k, v in minimal_roi.items()], key=lambda x: x[0])

    def roi_target(elapsed_min):
        ret = None
        for m, r in items:
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
        ts, o, h, l, c, ent_l, ex_l, ent_s, ex_s, *_ = r
        if not in_pos and ent_s:
            in_pos = True
            entry_ts = int(ts)
            # short entry: worse at lower price but not below low
            ep = o * (1.0 - slippage)
            if ep < l:
                ep = l
            entry_price = ep
            continue
        if in_pos:
            reason = "exit_signal"
            exit_rate = None
            # SL for short: price rises against us
            stop_price = entry_price * (1.0 - stoploss)
            if h >= stop_price:
                exit_rate = stop_price
                reason = "stoploss"
            else:
                elapsed = max(0, int((ts - entry_ts) // 60))
                roi_req = roi_target(elapsed)
                if roi_req is not None:
                    roi_price = entry_price * (1.0 - roi_req)
                    if l <= roi_price:
                        exit_rate = roi_price
                        reason = "roi"
            if exit_rate is None and ex_s:
                er = o
                if er > h:
                    er = h
                exit_rate = er
                reason = "exit_signal"
            if exit_rate is not None:
                # short exit: worse at higher price, apply +slippage
                exit_rate = exit_rate * (1.0 + slippage)
                amount = 1.0
                fee_open = entry_price * amount * fee
                fee_close = exit_rate * amount * fee
                profit_abs = (entry_price - exit_rate) * amount - (fee_open + fee_close)
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


def test_rustffi_short_alignment():
    ft = __import__("pytest").importorskip("ft_rust_backtest")
    base = 1_700_000_000
    rows = [
        [base + 0*300, 100.0, 101.0, 99.5, 100.5, 0, 0, 1, 0, None, None],  # enter short
        [base + 1*300,  99.8, 100.8, 98.0,  98.5, 0, 0, 0, 0, None, None],   # move lower (ROI candidate)
        [base + 2*300,  98.6,  99.0, 97.9,  98.0, 0, 0, 0, 1, None, None],  # exit signal fallback
    ]
    data = {"BTC/USDT": rows}
    params = {"fee": 0.001, "slippage": 0.0005, "stoploss": -0.2, "minimal_roi": {"0": 0.04}, "can_short": True}
    rust = ft.ft_rust_backtest.simulate_trades(data, rows[0][0], rows[-1][0], params)
    rust_df = pd.DataFrame(list(rust))
    py_df = _py_ref_sim_short(rows, fee=0.001, slippage=0.0005, stoploss=-0.2, minimal_roi={"0": 0.04})
    assert len(rust_df) == len(py_df) == 1
    assert abs(rust_df.iloc[0]["close_rate"] - py_df.iloc[0]["close_rate"]) < 1e-9
    assert abs(rust_df.iloc[0]["profit_abs"] - py_df.iloc[0]["profit_abs"]) < 1e-9
    assert rust_df.iloc[0]["exit_reason"] == py_df.iloc[0]["exit_reason"]

