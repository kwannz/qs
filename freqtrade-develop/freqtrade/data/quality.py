import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from pandas import DataFrame

from freqtrade.data.history.datahandlers import IDataHandler, get_datahandler
from freqtrade.configuration import TimeRange
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_seconds


logger = logging.getLogger(__name__)


@dataclass
class GapItem:
    start: int
    end: int
    missing_candles: int


def detect_ohlcv_gaps_df(df: DataFrame, timeframe: str) -> dict[str, Any]:
    """
    Detect duplicate timestamps and gaps in a single OHLCV dataframe.
    Returns summary dict with duplicate_count and gaps list.
    """
    if df.empty:
        return {"duplicate_count": 0, "gaps": []}

    # Ensure sorted
    df = df.sort_values("date").reset_index(drop=True)

    # Duplicates by timestamp
    dups = int(df["date"].duplicated(keep=False).sum())

    # Gaps by timestamp delta > expected step
    step = timeframe_to_seconds(timeframe)
    # to seconds
    ts = df["date"].astype("int64") // 10**9
    delta = ts.diff().fillna(0).astype(int)

    gaps: List[GapItem] = []
    for i in range(1, len(ts)):
        dt = int(ts.iloc[i] - ts.iloc[i - 1])
        if dt > step:
            missing = int(dt // step - 1)
            gaps.append(GapItem(start=int(ts.iloc[i - 1]), end=int(ts.iloc[i]), missing_candles=missing))

    return {
        "duplicate_count": dups,
        "gaps": [gap.__dict__ for gap in gaps],
    }


def scan_ohlcv_quality(
    datadir,
    data_format: str,
    trading_mode: TradingMode,
    *,
    pairs: list[str] | None = None,
    timeframes: list[str] | None = None,
    fix: bool = False,
    jobs: int | None = None,
    timerange: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Scan stored OHLCV data for duplicates and gaps per pair/timeframe.
    If fix=True, removes duplicate timestamps and re-stores data.
    """
    dh: IDataHandler = get_datahandler(datadir, data_format)
    # Discover available data
    available = dh.ohlcv_get_available_data(datadir, trading_mode)
    # Filter by pairs/timeframes if provided
    if pairs:
        available = [x for x in available if x[0] in pairs]
    if timeframes:
        available = [x for x in available if x[1] in timeframes]

    tasks = available

    def _process(item):
        p, tf, ct = item
        tr: TimeRange | None = TimeRange.parse_timerange(timerange) if timerange else None
        data = dh.ohlcv_load(
            p,
            timeframe=tf,
            timerange=tr,
            fill_missing=False,
            drop_incomplete=False,
            warn_no_data=False,
            candle_type=ct,
        )
        res = detect_ohlcv_gaps_df(data, tf)
        if fix and not data.empty:
            before = len(data)
            data = data.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            after = len(data)
            if after != before:
                logger.info(f"Deduplicated {p} {tf}: {before - after} duplicate candles removed.")
                dh.ohlcv_store(p, tf, data=data, candle_type=ct)
        return p, tf, res

    results: list[tuple[str, str, dict[str, Any]]]
    if jobs and jobs > 1:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=jobs)(delayed(_process)(t) for t in tasks)
        except Exception:
            results = [_process(t) for t in tasks]
    else:
        results = [_process(t) for t in tasks]

    report: Dict[str, Dict[str, Any]] = {}
    for p, tf, res in results:
        report.setdefault(p, {})[tf] = res
    return report


def scan_trades_quality(
    datadir,
    data_format: str,
    trading_mode: TradingMode,
    *,
    pairs: list[str] | None = None,
    fix: bool = False,
    jobs: int | None = None,
) -> Dict[str, Any]:
    """
    Scan trades data for duplicates per pair. Optionally de-duplicate.
    """
    from freqtrade.data.converter import trades_df_remove_duplicates

    dh: IDataHandler = get_datahandler(datadir, data_format)
    available = dh.trades_get_available_data(datadir, trading_mode)
    if pairs:
        available = [p for p in available if p in pairs]

    def _process(pair: str):
        df = dh.trades_load(pair, trading_mode, timerange=None, warn_no_data=False)
        if df.empty:
            return pair, {"duplicate_count": 0}
        before = len(df)
        df_clean = trades_df_remove_duplicates(df)
        dups = before - len(df_clean)
        if fix and dups > 0:
            logger.info(f"Deduplicated trades for {pair}: {dups} rows removed.")
            dh.trades_store(pair, df_clean, trading_mode)
        return pair, {"duplicate_count": int(dups)}

    if jobs and jobs > 1:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=jobs)(delayed(_process)(p) for p in available)
        except Exception:
            results = [_process(p) for p in available]
    else:
        results = [_process(p) for p in available]

    return {p: res for p, res in results}
