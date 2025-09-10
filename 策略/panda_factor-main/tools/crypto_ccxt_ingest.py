#!/usr/bin/env python3
"""
CCXT 加密货币行情导入脚本（MongoDB）

功能
- 从交易所通过 CCXT 抓取 OHLCV，导入至 MongoDB：
  - 日线 -> 集合 `crypto_market`（字段：date[YYYYMMDD], symbol, open, high, low, close, volume）
  - 1m   -> 集合 `crypto_market_1m`（字段：datetime[UTC], symbol, open, high, low, close, volume）

约定
- 统一符号格式：`EXCHANGE:BASEQUOTE`，示例：`BINANCE:BTCUSDT`。
- 从 CCXT 读取使用 `BASE/QUOTE`（如 `BTC/USDT`）。
- 分钟级以 UTC 存储 `datetime`，日线以 `YYYYMMDD` 存储 `date`。

使用示例
- 先安装依赖：`pip install ccxt pandas pymongo`（panda_factor-main 子项目请按 AGENTS.md 进行可编辑安装）
- 导入 BINANCE BTC/ETH 日线：
  python tools/crypto_ccxt_ingest.py \
    --exchange binance \
    --symbols BTC/USDT,ETH/USDT \
    --timeframe 1d \
    --start 20240101 --end 20240201

- 导入 BINANCE BTC/ETH 1m：
  python tools/crypto_ccxt_ingest.py \
    --exchange binance \
    --symbols BTC/USDT,ETH/USDT \
    --timeframe 1m \
    --start 20240101 --end 20240103

注意
- 不同交易所对交易对命名可能稍有差异；脚本会尝试在 `exchange.symbols` 中模糊匹配。
- 首次运行会自动创建必要索引；如历史数据中存在重复键，唯一索引可能创建失败，请清理后重试。
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Any

import pandas as pd

try:
    import ccxt  # type: ignore
except Exception as _e:  # pragma: no cover - 运行时提示更友好
    raise SystemExit("未安装 ccxt，请先执行: pip install ccxt")

from panda_common.config import get_config
from panda_common.handlers.database_handler import DatabaseHandler


def timeframe_to_ms(tf: str) -> int:
    tf = tf.lower()
    if tf.endswith('m'):
        return int(tf[:-1]) * 60_000
    if tf.endswith('h'):
        return int(tf[:-1]) * 60 * 60_000
    if tf.endswith('d'):
        return int(tf[:-1]) * 24 * 60 * 60_000
    raise ValueError(f"不支持的 timeframe: {tf}")


def yyyymmdd_to_ms(yyyymmdd: str, end_of_day: bool = False) -> int:
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    if end_of_day:
        # 包含 end_date 当天：设为次日 00:00:00 再减 1ms
        dt = dt + timedelta(days=1)
        ts = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000) - 1
    else:
        ts = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return ts


def ccxt_symbol_from_pair(exchange, base_quote: str) -> str:
    """将 'BTC/USDT' 或 'BTCUSDT' 映射为该交易所在用 symbol。
    优先匹配精确 'BTC/USDT'，否则在 exchange.symbols 中模糊匹配。
    """
    # 标准化输入
    s = base_quote.strip().upper()
    if '/' not in s and len(s) > 3:
        # 尝试插入 '/'
        # 简单拆分：末尾以 USDT/USDC/BUSD/USD/USDD/FDUSD 等为常见报价货币
        for q in [
            "USDT", "USDC", "BUSD", "USD", "USDD", "FDUSD", "EUR", "BTC", "ETH"
        ]:
            if s.endswith(q):
                s = s[:-len(q)] + '/' + q
                break

    # 加载市场，尝试直接匹配
    markets = exchange.load_markets()
    if s in exchange.symbols:
        return s

    # 模糊匹配（忽略分隔符差异）
    cand = s.replace('/', '')
    for sym in exchange.symbols:
        if sym.replace('/', '').upper() == cand:
            return sym
    raise ValueError(f"在交易所 {exchange.id} 未找到交易对: {base_quote}")


def normalize_repo_symbol(exchange_id: str, base_quote_ccxt: str) -> str:
    base, quote = base_quote_ccxt.upper().split('/')
    return f"{exchange_id.upper()}:{base}{quote}"


def fetch_ohlcv_all(
    exchange,
    symbol_ccxt: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 1000,
) -> List[List[Any]]:
    """分页抓取 OHLCV（包含 since_ms~until_ms）。"""
    tf_ms = timeframe_to_ms(timeframe)
    all_rows: List[List[Any]] = []
    cursor = since_ms
    while True:
        # 防止超出 until
        if cursor > until_ms:
            break
        try:
            batch = exchange.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, since=cursor, limit=limit)
        except Exception as e:
            # 网络抖动 / 限速
            time.sleep(max(1.0, exchange.rateLimit / 1000.0))
            raise e

        if not batch:
            break

        all_rows.extend(batch)

        last_ts = batch[-1][0]
        # 下一页从最后一根之后开始
        cursor = last_ts + tf_ms

        # 限速保护
        time.sleep(max(0.05, exchange.rateLimit / 1000.0))

        # 如果已经到尽头，退出
        if last_ts >= until_ms:
            break
    # 去重（有的交易所可能会返回重叠）
    seen = set()
    dedup = []
    for row in all_rows:
        ts = row[0]
        if ts in seen:
            continue
        seen.add(ts)
        dedup.append(row)
    return dedup


def ensure_indexes(dbh: DatabaseHandler, db: str):
    daily = dbh.get_mongo_collection(db, "crypto_market")
    minute = dbh.get_mongo_collection(db, "crypto_market_1m")
    try:
        daily.create_index([("symbol", 1), ("date", 1)], unique=True, background=True)
    except Exception:
        pass
    try:
        minute.create_index([("symbol", 1), ("datetime", 1)], unique=True, background=True)
    except Exception:
        pass


def upsert_docs(dbh: DatabaseHandler, db: str, coll: str, docs: List[Dict[str, Any]], key_fields: List[str]):
    from pymongo import UpdateOne  # 延迟导入
    if not docs:
        return 0
    collection = dbh.get_mongo_collection(db, coll)
    ops = []
    for d in docs:
        key = {k: d[k] for k in key_fields}
        ops.append(UpdateOne(key, {"$set": d}, upsert=True))
    if not ops:
        return 0
    res = collection.bulk_write(ops, ordered=False)
    return (res.upserted_count or 0) + (res.modified_count or 0)


def ingest(
    exchange_id: str,
    symbols_input: List[str],
    timeframe: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> None:
    log = logging.getLogger(__name__)
    config = get_config()
    dbh = DatabaseHandler(config)
    db = config["MONGO_DB"]
    ensure_indexes(dbh, db)

    # 构造 CCXT 交易所
    ex_id = exchange_id.lower()
    if not hasattr(ccxt, ex_id):
        raise ValueError(f"未知交易所: {exchange_id}")
    exchange = getattr(ccxt, ex_id)({"enableRateLimit": True})
    exchange.load_markets()

    # 解析 symbol 列表
    # 支持：
    #  - 'BTC/USDT'
    #  - 'BINANCE:BTCUSDT'（会忽略冒号前部分并按 --exchange 解析）
    basequotes: List[str] = []
    for s in symbols_input:
        s = s.strip()
        if not s:
            continue
        if ':' in s:
            s = s.split(':', 1)[1]  # 去掉 EXCHANGE:
        basequotes.append(s)

    since_ms = yyyymmdd_to_ms(start_yyyymmdd)
    until_ms = yyyymmdd_to_ms(end_yyyymmdd, end_of_day=(timeframe.lower().endswith('d')))

    for bq in basequotes:
        sym_ccxt = ccxt_symbol_from_pair(exchange, bq)  # 如 'BTC/USDT'
        repo_symbol = normalize_repo_symbol(exchange.id, sym_ccxt)  # 'BINANCE:BTCUSDT'
        log.info(f"抓取 {exchange.id} {sym_ccxt} -> 存储符号 {repo_symbol} | {timeframe} | {start_yyyymmdd}~{end_yyyymmdd}")

        rows = fetch_ohlcv_all(exchange, sym_ccxt, timeframe, since_ms, until_ms)
        if not rows:
            log.warning(f"无数据: {exchange.id} {sym_ccxt} {timeframe} {start_yyyymmdd}~{end_yyyymmdd}")
            continue

        if timeframe.lower().endswith('d'):
            # 日线：转为 date 字段
            docs = []
            for ts, o, h, l, c, v in rows:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                ymd = dt.strftime("%Y%m%d")
                docs.append({
                    "date": ymd,
                    "symbol": repo_symbol,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                })
            n = upsert_docs(dbh, db, "crypto_market", docs, ["symbol", "date"])
            log.info(f"写入/更新 {n} 条到 crypto_market ({repo_symbol})")
        else:
            # 分钟：转为 datetime（UTC）
            docs = []
            for ts, o, h, l, c, v in rows:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                docs.append({
                    "datetime": dt,
                    "symbol": repo_symbol,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                })
            n = upsert_docs(dbh, db, "crypto_market_1m", docs, ["symbol", "datetime"])
            log.info(f"写入/更新 {n} 条到 crypto_market_1m ({repo_symbol})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCXT 加密行情导入 MongoDB")
    p.add_argument("--exchange", required=True, help="交易所 ID（如 binance, okx, kucoin）")
    p.add_argument("--symbols", required=True, help="逗号分隔的交易对：如 BTC/USDT,ETH/USDT 或 BINANCE:BTCUSDT")
    p.add_argument("--timeframe", required=True, choices=["1m", "1d"], help="时间粒度：1m 或 1d")
    p.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    p.add_argument("--end", required=True, help="结束日期 YYYYMMDD（含当日）")
    p.add_argument("-v", "--verbose", action="store_true", help="启用详细日志(DEBUG)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger(__name__).debug("Verbose logging enabled")
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    ingest(args.exchange, symbols, args.timeframe, args.start, args.end)


if __name__ == "__main__":
    main()
