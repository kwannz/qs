# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import time
from pathlib import Path
from typing import Iterable, List, Optional

import fire
import numpy as np
import pandas as pd
from loguru import logger

from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize


def _now_ts_ms() -> int:
    return int(time.time() * 1000)


def _parse_symbols(symbols: Optional[str]) -> List[str]:
    if not symbols:
        return []
    return [s.strip() for s in symbols.split(",") if s.strip()]


class CryptoCollector(BaseCollector):
    """Collect OHLCV via ccxt for crypto symbols

    - Supports 1d and 1min
    - Default exchange: binance (spot)
    """

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval: str = "1d",
        max_workers: int = 1,
        max_collector_count: int = 2,
        delay: float = 0.0,
        check_data_length: int = None,
        limit_nums: int = None,
        symbols: Optional[str] = None,
        exchange: str = "binance",
    ):
        import ccxt  # local import to avoid hard dep for non-crypto users

        self._ccxt = ccxt
        self._exchange_name = exchange.lower()
        self._exchange = getattr(ccxt, self._exchange_name)({"enableRateLimit": True})
        self._user_symbols = _parse_symbols(symbols)
        super().__init__(
            save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_instrument_list(self) -> List[str]:
        if self._user_symbols:
            return [s.upper() for s in self._user_symbols]

        # fallback: top few USD/USDT pairs
        markets = self._exchange.load_markets()
        symbols = []
        for m in markets.values():
            sym = m.get("symbol") or ""
            quote = m.get("quote") or ""
            if quote in ("USDT", "USD") and "/" in sym:
                symbols.append(sym)
        # de-dup and sort
        return sorted(set(s.upper() for s in symbols))

    def normalize_symbol(self, symbol: str) -> str:
        # Turn "BTC/USDT" into "BINANCE_BTCUSDT"
        return f"{self._exchange_name.upper()}_" + symbol.replace("/", "").upper()

    @staticmethod
    def _timeframe(interval: str) -> str:
        if interval in ("1d", "1D", "day"):
            return "1d"
        if interval in ("1min", "1m", "1M"):
            return "1m"
        raise ValueError(f"Unsupported interval: {interval}")

    def _fetch_ohlcv_all(self, symbol: str, timeframe: str, since_ms: int, end_ms: int) -> List[List]:
        # ccxt typical max limit is 1000 per call
        limit = 1000
        ohlcv: List[List] = []
        fetch_since = since_ms
        while fetch_since < end_ms:
            batch = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
            if not batch:
                break
            ohlcv.extend(batch)
            # move forward by last timestamp; add 1ms to avoid duplicates
            fetch_since = batch[-1][0] + 1
            # Respect rate limits
            time.sleep(self.delay)
            # Guard to avoid infinite loops
            if len(batch) < limit:
                # no more data for this window
                if fetch_since >= end_ms:
                    break
        return ohlcv

    def get_data(self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp) -> pd.DataFrame:
        timeframe = self._timeframe(interval)
        # NOTE: Use UTC timestamps
        start_ms = int(pd.Timestamp(start_datetime, tz="UTC").timestamp() * 1000)
        # ccxt end is exclusive; we fetch until now/end
        end_ms = int(pd.Timestamp(end_datetime, tz="UTC").timestamp() * 1000)
        try:
            raw = self._fetch_ohlcv_all(symbol, timeframe, start_ms, end_ms)
        except Exception as e:
            logger.warning(f"fetch error: {self._exchange_name} {symbol} {interval}: {e}")
            return pd.DataFrame()
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"]).dropna()
        # ccxt timestamps are in ms; use UTC then drop tz
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.drop(columns=["ts"]).sort_values("date").reset_index(drop=True)
        df["symbol"] = symbol.upper()
        # Deduplicate if any
        df = df.drop_duplicates(subset=["date"]).reset_index(drop=True)
        # Coerce dtypes
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        return df


class CryptoNormalize(BaseNormalize):
    """Normalize raw OHLCV to Qlib CSV schema

    - Adds change (pct change of close)
    - Adds factor (fixed 1.0 for crypto)
    - Keeps columns: date, symbol, open, high, low, close, volume, change, factor
    """

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # Not strictly needed for crypto; dump_bin will aggregate calendar across files.
        return []

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        # Expected source columns: date, symbol, open, high, low, close, volume
        cols = set(df.columns)
        required = {self._date_field_name, self._symbol_field_name, "open", "high", "low", "close"}
        missing = required - cols
        if missing:
            logger.warning(f"normalize: missing columns {missing}")
        df = df.copy()
        df[self._date_field_name] = pd.to_datetime(df[self._date_field_name])
        df[self._symbol_field_name] = df[self._symbol_field_name].astype(str).str.upper()
        df = df.sort_values(self._date_field_name).drop_duplicates(self._date_field_name)
        # Compute change; first value -> 0.0
        df["change"] = df["close"].pct_change().fillna(0.0)
        # Crypto has no splits/dividends -> factor 1.0
        df["factor"] = 1.0
        keep = [self._date_field_name, self._symbol_field_name, "open", "high", "low", "close", "volume", "change", "factor"]
        return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


class Run(BaseRun):
    @property
    def collector_class_name(self):
        return "CryptoCollector"

    @property
    def normalize_class_name(self):
        return "CryptoNormalize"

    @property
    def default_base_dir(self) -> [Path, str]:
        return Path("~/.qlib/crypto").expanduser().resolve()


if __name__ == "__main__":
    fire.Fire(Run)

