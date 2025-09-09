import os
from datetime import datetime, timezone

import pandas as pd


def _pnl_with_funding_long(open_rate, close_rate, funding_events):
    # amount = 1.0, no fees, no slippage
    gross = close_rate - open_rate
    funding = sum(notional * rate for (_, rate, notional) in funding_events)
    return gross - funding


def _pnl_with_funding_short(open_rate, close_rate, funding_events):
    gross = open_rate - close_rate
    funding = sum(notional * rate for (_, rate, notional) in funding_events)
    # short receives positive funding
    return gross + funding


def test_rustffi_funding_mark_precedence():
    ft = __import__("pytest").importorskip("ft_rust_backtest")

    base = 1_700_000_000
    # Rows layout: [ts, open, high, low, close, enter_long, exit_long, enter_short, exit_short, enter_tag, exit_tag]
    rows = [
        [base + 0*300, 100.0, 101.0,  99.5, 100.0, 1, 0, 0, 0, None, None],
        [base + 1*300, 102.0, 103.0, 101.5, 102.5, 0, 1, 0, 0, None, None],
    ]
    data = {"BTC/USDT": rows}

    # funding events at ts1; funding rate +0.001. Mark price should be used over open price.
    ts1 = rows[1][0]
    params = {
        "fee": 0.0,
        "slippage": 0.0,
        "stoploss": -1.0,
        "minimal_roi": {},
        "funding": {"BTC/USDT": [[ts1, 0.001]]},
        "funding_mark": {"BTC/USDT": [[ts1, 200.0]]},  # mark price
    }

    rust = ft.ft_rust_backtest.simulate_trades(data, rows[0][0], rows[-1][0], params)
    rust_df = pd.DataFrame(list(rust))
    assert len(rust_df) == 1
    # Expected: pnl = (close-open) - funding(MARK)
    open_rate = rows[0][1]
    close_rate = rows[1][1]  # exit at next open by our model
    funding_events = [(ts1, 0.001, 200.0)]
    exp = _pnl_with_funding_long(open_rate, close_rate, funding_events)
    assert abs(rust_df.iloc[0]["profit_abs"] - exp) < 1e-9


def test_rustffi_funding_open_fallback_and_short():
    ft = __import__("pytest").importorskip("ft_rust_backtest")
    base = 1_700_100_000
    # short path
    rows = [
        [base + 0*300, 100.0, 101.0,  99.5, 100.0, 0, 0, 1, 0, None, None],
        [base + 1*300,  98.0,  98.5,  97.0,  97.5, 0, 0, 0, 1, None, None],
    ]
    data = {"ETH/USDT": rows}
    ts1 = rows[1][0]
    # No mark provided -> fallback to open price at ts1 (98.0)
    params = {
        "fee": 0.0,
        "slippage": 0.0,
        "stoploss": -1.0,
        "minimal_roi": {},
        "can_short": True,
        "funding": {"ETH/USDT": [[ts1, 0.001]]},
    }
    rust = ft.ft_rust_backtest.simulate_trades(data, rows[0][0], rows[-1][0], params)
    rust_df = pd.DataFrame(list(rust))
    assert len(rust_df) == 1
    open_rate = rows[0][1]
    close_rate = rows[1][1]
    funding_events = [(ts1, 0.001, rows[1][1])]  # fallback notional=open at ts1
    exp = _pnl_with_funding_short(open_rate, close_rate, funding_events)
    assert abs(rust_df.iloc[0]["profit_abs"] - exp) < 1e-9

