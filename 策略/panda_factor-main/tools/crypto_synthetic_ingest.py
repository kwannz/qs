#!/usr/bin/env python3
"""
合成加密数据写入工具（本地快速演示）

功能
- 生成两只币（BINANCE:BTCUSDT, BINANCE:ETHUSDT）的日线/分钟随机游走数据，写入 Mongo：
  - 日线 -> crypto_market（date, symbol, open, high, low, close, volume）
  - 分钟 -> crypto_market_1m（datetime[UTC], symbol, open, high, low, close, volume）

用法
  python tools/crypto_synthetic_ingest.py --timeframe 1d --start 20240101 --end 20240131
  python tools/crypto_synthetic_ingest.py --timeframe 1m --start 20240101 --end 20240102
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from panda_common.config import get_config
from panda_common.handlers.database_handler import DatabaseHandler


SYMS = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT"]


def gen_daily(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq='D')
    out = []
    rng = np.random.default_rng(123)
    for sym in SYMS:
        price = 100.0 if 'BTC' in sym else 50.0
        for d in dates:
            ret = rng.normal(0.001, 0.03)
            o = price
            c = max(0.1, price * (1.0 + ret))
            h = max(o, c) * (1.0 + abs(rng.normal(0, 0.01)))
            l = min(o, c) * (1.0 - abs(rng.normal(0, 0.01)))
            v = float(abs(rng.normal(1000, 200)))
            out.append({
                'date': d.strftime('%Y%m%d'),
                'symbol': sym,
                'open': round(o, 6),
                'high': round(h, 6),
                'low': round(l, 6),
                'close': round(c, 6),
                'volume': v,
            })
            price = c
    return pd.DataFrame(out)


def gen_minute(start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start, '%Y%m%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, '%Y%m%d').replace(tzinfo=timezone.utc) + timedelta(days=1)
    times = pd.date_range(start_dt, end_dt, freq='1min', inclusive='left')
    out = []
    rng = np.random.default_rng(456)
    for sym in SYMS:
        price = 100.0 if 'BTC' in sym else 50.0
        for t in times:
            ret = rng.normal(0.0, 0.001)
            o = price
            c = max(0.1, price * (1.0 + ret))
            h = max(o, c) * (1.0 + abs(rng.normal(0, 0.0005)))
            l = min(o, c) * (1.0 - abs(rng.normal(0, 0.0005)))
            v = float(abs(rng.normal(100, 20)))
            out.append({
                'datetime': t.to_pydatetime(),
                'symbol': sym,
                'open': round(o, 6),
                'high': round(h, 6),
                'low': round(l, 6),
                'close': round(c, 6),
                'volume': v,
            })
            price = c
    return pd.DataFrame(out)


def upsert_many(dbh: DatabaseHandler, db: str, coll: str, rows: List[Dict[str, Any]], keys: List[str]) -> int:
    from pymongo import UpdateOne
    if not rows:
        return 0
    collection = dbh.get_mongo_collection(db, coll)
    ops = []
    for d in rows:
        ops.append(UpdateOne({k: d[k] for k in keys}, {'$set': d}, upsert=True))
    res = collection.bulk_write(ops, ordered=False)
    return (res.upserted_count or 0) + (res.modified_count or 0)


def ensure_indexes(dbh: DatabaseHandler, db: str):
    daily = dbh.get_mongo_collection(db, "crypto_market")
    minute = dbh.get_mongo_collection(db, "crypto_market_1m")
    try:
        daily.create_index([("symbol", 1), ("date", 1)], unique=True)
    except Exception:
        pass
    try:
        minute.create_index([("symbol", 1), ("datetime", 1)], unique=True)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='合成加密数据写入工具')
    p.add_argument('--timeframe', required=True, choices=['1d', '1m'])
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if os.getenv('LOG_LEVEL','').upper() == 'DEBUG':
        logging.getLogger().setLevel(logging.DEBUG)
    cfg = get_config()
    dbh = DatabaseHandler(cfg)
    db = cfg['MONGO_DB']
    ensure_indexes(dbh, db)

    if args.timeframe == '1d':
        df = gen_daily(args.start, args.end)
        n = upsert_many(dbh, db, 'crypto_market', df.to_dict('records'), ['symbol', 'date'])
        logging.getLogger(__name__).info(f'写入/更新 {n} 条到 crypto_market')
    else:
        df = gen_minute(args.start, args.end)
        n = upsert_many(dbh, db, 'crypto_market_1m', df.to_dict('records'), ['symbol', 'datetime'])
        logging.getLogger(__name__).info(f'写入/更新 {n} 条到 crypto_market_1m')


if __name__ == '__main__':
    main()
