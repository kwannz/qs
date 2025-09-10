import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method='bfill')


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return pd.DataFrame({'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist})


def compute_basic_factors_daily(
    ohlcv: pd.DataFrame,
    requested: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    基于日线 OHLCV 计算加密基础因子。返回 (date, symbol, factors...).
    支持因子：
      - ret_1d, ret_5d, ret_20d
      - vol_20 (收益波动率) / mom_20 (动量 = close/close.shift(20) -1)
      - rsi14
      - macd, macd_signal, macd_hist
      - hhv_20, llv_20
    """
    if requested is None:
        requested = []
    if isinstance(requested, str):
        requested = [requested]
    req = set([f.lower() for f in requested])

    df = ohlcv.copy()
    if 'date' not in df.columns or 'symbol' not in df.columns or 'close' not in df.columns:
        return pd.DataFrame()

    df = df.sort_values(['symbol', 'date'])
    out_list = []
    for sym, g in df.groupby('symbol', sort=False):
        g2 = g.copy()
        close = pd.to_numeric(g2['close'], errors='coerce')
        # returns
        if not req or 'ret_1d' in req:
            g2['ret_1d'] = close.pct_change()
        if not req or 'ret_5d' in req:
            g2['ret_5d'] = close.pct_change(5)
        if not req or 'ret_20d' in req:
            g2['ret_20d'] = close.pct_change(20)
        # volatility / momentum
        if not req or 'vol_20' in req:
            g2['vol_20'] = close.pct_change().rolling(20).std()
        if not req or 'mom_20' in req:
            g2['mom_20'] = (close / close.shift(20)) - 1.0
        # RSI
        if not req or 'rsi14' in req:
            g2['rsi14'] = _rsi(close, 14)
        # MACD
        if not req or {'macd','macd_signal','macd_hist'} & req:
            macd_df = _macd(close)
            g2 = pd.concat([g2.reset_index(drop=True), macd_df.reset_index(drop=True)], axis=1)
        # rolling high/low
        if not req or 'hhv_20' in req:
            g2['hhv_20'] = pd.to_numeric(g2['high'], errors='coerce').rolling(20).max()
        if not req or 'llv_20' in req:
            g2['llv_20'] = pd.to_numeric(g2['low'], errors='coerce').rolling(20).min()

        out_list.append(g2)

    out = pd.concat(out_list, ignore_index=True)
    # 选择输出列
    base_cols = ['date', 'symbol']
    all_cols = list(out.columns)
    factor_cols = [c for c in all_cols if c not in base_cols and c not in ['open','high','low','close','volume','_id']]
    # 若指明 requested，则保留相交部分
    if req:
        factor_cols = [c for c in factor_cols if c.lower() in req or any(c.lower().startswith(x) for x in req)]
    return out[base_cols + factor_cols].dropna(how='all', subset=factor_cols)

