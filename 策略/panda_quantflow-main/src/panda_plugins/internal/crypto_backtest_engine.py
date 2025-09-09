from __future__ import annotations

import pandas as pd
from typing import Optional


def _ensure_time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if 'datetime' in df.columns:
        df = df.copy()
        df['__ts__'] = pd.to_datetime(df['datetime'])
        return df.sort_values('__ts__').drop(columns=['__ts__'])
    elif 'date' in df.columns:
        return df.sort_values('date')
    else:
        return df


def simulate_long_flat(
    price_df: pd.DataFrame,
    signal_df: Optional[pd.DataFrame] = None,
    fee_bps: float = 10.0,
) -> pd.DataFrame:
    """
    简易多/空仓位（仅多仓/空仓=0）回测：
    - 使用 close 列计算收益
    - signal_df 的 signal 列取值 {0,1} 表示持仓/空仓（默认全部持仓=1）
    - 手续费按仓位切换当期收取：cost = fee_bps/10000 * |pos - pos_pre|

    返回含 columns: ['close','ret','pos','ret_strategy','ret_net','equity'] 的 DataFrame。
    """
    if price_df is None or price_df.empty:
        raise ValueError('price_df is empty')

    df = price_df.copy()
    df = _ensure_time_sorted(df)
    if 'close' not in df.columns:
        raise ValueError('price_df must contain close column')

    # 价格收益
    df['ret'] = pd.to_numeric(df['close'], errors='coerce').pct_change().fillna(0.0)

    # 仓位
    if signal_df is None or signal_df.empty or 'signal' not in signal_df.columns:
        df['pos'] = 1.0
    else:
        s = signal_df.copy()
        # 对齐索引
        if 'datetime' in df.columns and 'datetime' in s.columns:
            s = s[['datetime', 'signal']]
            s['__key__'] = pd.to_datetime(s['datetime'])
            df['__key__'] = pd.to_datetime(df['datetime'])
            df = df.merge(s[['__key__', 'signal']], on='__key__', how='left')
            df = df.drop(columns=['__key__'])
        elif 'date' in df.columns and 'date' in s.columns:
            s = s[['date', 'signal']]
            df = df.merge(s, on='date', how='left')
        else:
            # 兜底：不对齐则全持仓
            df['signal'] = 1.0
        df['pos'] = df['signal'].fillna(method='ffill').fillna(0.0).clip(0, 1)
        df = df.drop(columns=['signal'])

    # 策略毛收益（T+1 执行，使用前一时刻仓位）
    df['ret_strategy'] = df['pos'].shift(1).fillna(0.0) * df['ret']

    # 手续费（当期仓位变化收取一次）
    fee_rate = fee_bps / 10000.0
    df['trade_cost'] = - fee_rate * (df['pos'].fillna(0.0) - df['pos'].shift(1).fillna(0.0)).abs()

    # 净收益与权益
    df['ret_net'] = df['ret_strategy'] + df['trade_cost']
    df['equity'] = (1.0 + df['ret_net']).cumprod()

    return df

