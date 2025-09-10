#!/usr/bin/env python3
"""
离线回测 CLI：
- 从 Mongo 读取日线或分钟数据（通过 panda_data），支持多标的；
- 支持成本（fee/slippage/maker/taker）与执行约束（max_weight/max_turnover/min_trade_weight）；
- 输出关键指标（stdout）与 equity.csv（可选）。

示例：
python tools/crypto_backtest_cli.py \
  --symbols BINANCE:BTCUSDT,BINANCE:ETHUSDT \
  --freq 1d --start 20240101 --end 20240201 \
  --fee 1 --slip 2 --maker 0 --taker 5 \
  --max_weight 0.6 --max_turnover 0.5 --min_trade 0.01 \
  --out equity.csv
"""

from __future__ import annotations

import argparse
import pandas as pd
import sys, os, logging

# 保证可导入 panda_plugins（位于 sibling 'src'）
CUR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(CUR, '..', 'src'))
if SRC not in sys.path:
    sys.path.append(SRC)

import panda_data
from panda_plugins.internal.crypto_backtest_engine import (
    simulate_long_flat,
    simulate_portfolio_equal_weight,
    simulate_portfolio_equal_weight_exec,
    calc_metrics,
)
from common.logging.log_context import set_backtest_id, set_request_id, reset_backtest_id, reset_request_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Crypto 回测 CLI')
    p.add_argument('--symbols', required=True, help='逗号分隔，多标的如 BINANCE:BTCUSDT,BINANCE:ETHUSDT')
    p.add_argument('--freq', required=True, choices=['1d', '1m'])
    p.add_argument('--resample', default='', help='当 freq=1m 时，可选重采样规则：5min/15min/30min/1h')
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--fee', type=float, default=1.0)
    p.add_argument('--slip', type=float, default=0.0)
    p.add_argument('--maker', type=float, default=0.0)
    p.add_argument('--taker', type=float, default=0.0)
    p.add_argument('--max_weight', type=float, default=1.0)
    p.add_argument('--max_turnover', type=float, default=1.0)
    p.add_argument('--min_trade', type=float, default=0.0)
    p.add_argument('--out', default='')
    p.add_argument('--req-id', default='')
    p.add_argument('--bt-id', default='')
    p.add_argument('-v','--verbose', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger(__name__).debug('Verbose logging enabled')
    t_req = None
    t_bt = None
    if args.req_id:
        t_req = set_request_id(args.req_id)
    if args.bt_id:
        t_bt = set_backtest_id(args.bt_id)
    syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    panda_data.init()

    if args.freq == '1m':
        dfs = []
        for s in syms:
            if args.resample and args.resample != '1m':
                dfi = panda_data.get_crypto_min_data_resampled(
                    start_date=args.start,
                    end_date=args.end,
                    symbol=s,
                    rule=args.resample,
                )
            else:
                dfi = panda_data.get_crypto_min_data(
                    start_date=args.start, end_date=args.end,
                    symbol=s,
                    fields=["datetime","symbol","open","high","low","close","volume"],
                )
            if dfi is not None and not dfi.empty:
                dfs.append(dfi)
        df = None if not dfs else pd.concat(dfs, ignore_index=True)
    else:
        df = panda_data.get_crypto_market_data(
            start_date=args.start, end_date=args.end,
            symbols=syms,
            fields=["date","symbol","open","high","low","close","volume"],
        )

    if df is None or df.empty:
        print('未查询到行情数据')
        return 1

    if len(syms) == 1:
        df = df[df['symbol'] == syms[0]].copy()
        bt = simulate_long_flat(
            df, None,
            fee_bps=args.fee,
            slippage_bps=args.slip,
            maker_bps=args.maker,
            taker_bps=args.taker,
        )
    else:
        use_exec = (args.max_weight < 1.0) or (args.max_turnover < 1.0) or (args.min_trade > 0.0)
        if use_exec:
            bt = simulate_portfolio_equal_weight_exec(
                df, None,
                fee_bps=args.fee,
                slippage_bps=args.slip,
                maker_bps=args.maker,
                taker_bps=args.taker,
                max_weight=args.max_weight,
                max_turnover=args.max_turnover,
                min_trade_weight=args.min_trade,
            )
        else:
            bt = simulate_portfolio_equal_weight(
                df, None,
                fee_bps=args.fee,
                slippage_bps=args.slip,
                maker_bps=args.maker,
                taker_bps=args.taker,
            )

    metrics = calc_metrics(bt)
    print('回测指标:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.6f}')

    if args.out:
        bt.reset_index().to_csv(args.out, index=False)
        print(f'已输出 {args.out}')
    # cleanup contextvars
    try:
        if t_req:
            reset_request_id(t_req)
        if t_bt:
            reset_backtest_id(t_bt)
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
