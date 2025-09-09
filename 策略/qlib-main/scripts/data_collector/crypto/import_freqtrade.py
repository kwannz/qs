# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

import fire
import pandas as pd
from loguru import logger


def _find_first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def _normalize_one(
    src: Path,
    out_dir: Path,
    exchange: str,
    pair: str,
    time_col_candidates=(
        "date",
        "datetime",
        "time",
        "timestamp",
        "open_time",
        "candle_begin_time",
        "candle_begin_time_utc",
        "ts",
    ),
    open_col_candidates=("open",),
    high_col_candidates=("high",),
    low_col_candidates=("low",),
    close_col_candidates=("close", "last", "close_price"),
    vol_col_candidates=("volume", "vol", "quote_volume", "basevolume"),
) -> Optional[Path]:
    try:
        if src.suffix.lower() == ".parquet":
            df = pd.read_parquet(src)
        elif src.suffix.lower() == ".csv":
            df = pd.read_csv(src)
        else:
            logger.warning(f"skip unsupported file: {src}")
            return None
    except Exception as e:
        logger.warning(f"failed reading {src}: {e}")
        return None

    tcol = _find_first_col(df, time_col_candidates)
    ocol = _find_first_col(df, open_col_candidates)
    hcol = _find_first_col(df, high_col_candidates)
    lcol = _find_first_col(df, low_col_candidates)
    ccol = _find_first_col(df, close_col_candidates)
    vcol = _find_first_col(df, vol_col_candidates)

    miss = [n for n, v in [("time", tcol), ("open", ocol), ("high", hcol), ("low", lcol), ("close", ccol)] if v is None]
    if miss:
        logger.warning(f"{src.name}: missing columns {miss}, skip")
        return None

    df = df.copy()
    # 处理时间列：若为整型/浮点型的 epoch 值，按毫秒解析；否则走常规解析
    tseries = df[tcol]
    if pd.api.types.is_numeric_dtype(tseries):
        # 简单判断数量级，>1e12 多为毫秒或纳秒；这里按毫秒处理（CoinGlass/Freqtrade 常见为 ms）
        df["date"] = pd.to_datetime(tseries.astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
    else:
        df["date"] = pd.to_datetime(tseries, errors="coerce", utc=True).dt.tz_convert(None)
    df["open"] = pd.to_numeric(df[ocol], errors="coerce")
    df["high"] = pd.to_numeric(df[hcol], errors="coerce")
    df["low"] = pd.to_numeric(df[lcol], errors="coerce")
    df["close"] = pd.to_numeric(df[ccol], errors="coerce")
    df["volume"] = pd.to_numeric(df[vcol], errors="coerce") if vcol is not None else pd.NA

    df = df.dropna(subset=["date", "open", "high", "low", "close"])  # volume may be NA
    df = df.sort_values("date").drop_duplicates("date")

    symbol = f"{exchange.upper()}_{pair.upper()}"
    df["symbol"] = symbol
    df["change"] = df["close"].pct_change().fillna(0.0)
    df["factor"] = 1.0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.csv"
    keep = ["date", "symbol", "open", "high", "low", "close", "volume", "change", "factor"]
    try:
        df[keep].to_csv(out_path, index=False)
    except Exception as e:
        logger.warning(f"failed writing {out_path}: {e}")
        return None
    return out_path


def import_coinglass(
    source_root: str,
    normalize_dir: str,
    pattern: str = "processed_data/futures_price_history/futures_price_history_*_1d.parquet",
):
    """Import CoinGlass/ freqtrade historical data into Qlib-normalized CSVs.

    Parameters
    ----------
    source_root : str
        Path to "historical data" root directory.
    normalize_dir : str
        Target directory to write normalized CSVs.
    pattern : str
        Glob pattern relative to source_root to match OHLCV files.
        Default imports 1d futures price history parquet.

    After normalization, run dump_bin to convert to Qlib format, e.g.:
        python scripts/dump_bin.py dump_all \
            --data_path <normalize_dir> \
            --qlib_dir ~/.qlib/qlib_data/crypto_data \
            --freq day --date_field_name date --symbol_field_name symbol --file_suffix .csv
    """
    src_root = Path(source_root).expanduser().resolve()
    out_dir = Path(normalize_dir).expanduser().resolve()

    files = sorted(src_root.glob(pattern))
    if not files:
        logger.warning(f"no files matched: {src_root}/{pattern}")
        return

    ok, fail = 0, 0
    for f in files:
        m = re.match(r".*futures_price_history_(?P<ex>[A-Za-z0-9]+)_(?P<pair>[A-Za-z0-9-]+)_(?P<tf>\w+)\.(parquet|csv)$", f.as_posix())
        if not m:
            logger.warning(f"skip unrecognized filename: {f.name}")
            fail += 1
            continue
        ex = m.group("ex")
        pair = m.group("pair").replace("-", "")
        out = _normalize_one(f, out_dir, ex, pair)
        if out is None:
            fail += 1
        else:
            ok += 1
            logger.info(f"normalized -> {out}")

    logger.info(f"done. success={ok}, failed={fail}")


if __name__ == "__main__":
    fire.Fire({"import_coinglass": import_coinglass})
