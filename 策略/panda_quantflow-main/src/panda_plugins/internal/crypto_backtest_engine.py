from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

log = logging.getLogger(__name__)


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
    slippage_bps: float = 0.0,
    maker_bps: float = 0.0,
    taker_bps: float = 0.0,
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

    # 手续费（当期仓位变化收取一次）+ 滑点 + maker/taker（近似）
    pos_change = (df['pos'].fillna(0.0) - df['pos'].shift(1).fillna(0.0))
    l1 = pos_change.abs()
    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0
    maker_rate = maker_bps / 10000.0
    taker_rate = taker_bps / 10000.0
    # 加仓与减仓分量
    inc = pos_change.clip(lower=0)
    dec = (-pos_change.clip(upper=0))
    # 基础手续费 + 滑点
    df['trade_cost'] = -(fee_rate + slip_rate) * l1
    # maker/taker：加仓视为 taker，减仓视为 maker（简化）
    if maker_bps or taker_bps:
        df['trade_cost'] += -(taker_rate * inc + maker_rate * dec)

    # 净收益与权益
    df['ret_net'] = df['ret_strategy'] + df['trade_cost']
    df['equity'] = (1.0 + df['ret_net']).cumprod()
    df['turnover'] = l1

    log.debug(f"simulate_long_flat: rows={len(df)} fee_bps={fee_bps} slip_bps={slippage_bps} maker={maker_bps} taker={taker_bps}")
    return df


def simulate_portfolio_equal_weight(
    price_df: pd.DataFrame,
    signal_df: Optional[pd.DataFrame] = None,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    maker_bps: float = 0.0,
    taker_bps: float = 0.0,
) -> pd.DataFrame:
    """
    组合等权回测：
    - 要求 price_df 至少包含 ['close','symbol']，以及 'date' 或 'datetime' 之一。
    - 若提供 signal_df，需包含 ['symbol','signal']（0/1）；同一时间点等权分配到 signal==1 的标的；
      未提供则对 price_df 中每个时间点的所有标的等权分配。
    - 成本：对组合权重矩阵逐时刻计算 L1 变化量之和乘以 fee_bps/10000。

    返回含时序索引（date 或 datetime）的 DataFrame：
    ['ret_net','equity','ret_strategy','trade_cost']
    """
    if price_df is None or price_df.empty:
        raise ValueError('price_df is empty')

    df = price_df.copy()
    time_col = 'datetime' if 'datetime' in df.columns else 'date'
    # 价格与收益
    df['_ret_'] = pd.to_numeric(df['close'], errors='coerce').groupby(df['symbol']).pct_change().fillna(0.0)

    # 透视为 [time x symbol] 的收益矩阵
    ret_wide = df.pivot(index=time_col, columns='symbol', values='_ret_').fillna(0.0)

    # 仓位/权重矩阵 W：等权或按信号等权
    if signal_df is None or signal_df.empty or 'signal' not in signal_df.columns:
        # 全部标的等权
        n = (ret_wide.notna()).sum(axis=1).clip(lower=1)
        W = ret_wide.copy()
        W[W.notna()] = 1.0
        W = W.div(n, axis=0)
    else:
        s = signal_df.copy()
        # 使用相同的时间列对齐
        if time_col not in s.columns and 'date' in s.columns and time_col == 'datetime':
            # 简单兜底：若传入的是 date 信号而价格是 datetime，则用日期对齐到天（将 datetime 取日期）
            df_idx = ret_wide.index
            # 将 ret_wide 的 datetime 索引转换为 date 字符串
            ret_date = pd.to_datetime(df_idx).tz_localize(None, nonexistent='shift_forward', ambiguous='NaT', errors='coerce') if hasattr(pd.to_datetime(df_idx), 'tz_localize') else pd.to_datetime(df_idx)
            ret_date = ret_date.strftime('%Y%m%d')
            ret_wide_by_date = ret_wide.copy()
            ret_wide_by_date.index = ret_date
            s_use = s[['date','symbol','signal']].copy()
            Sig = s_use.pivot(index='date', columns='symbol', values='signal')
            Sig = Sig.reindex(index=ret_wide_by_date.index).fillna(method='ffill').fillna(0.0)
            # 将信号映射回原时间索引
            Sig.index = ret_wide.index
        else:
            # 直接按时间列与 symbol 透视
            s_use = s[[time_col, 'symbol', 'signal']].copy()
            Sig = s_use.pivot(index=time_col, columns='symbol', values='signal')
            Sig = Sig.reindex(index=ret_wide.index).fillna(method='ffill').fillna(0.0)

        Sig = Sig.where(Sig > 0, 0.0)
        n_active = Sig.sum(axis=1).replace(0, pd.NA).fillna(1.0)
        W = Sig.div(n_active, axis=0)

    # 策略毛收益（T+1 执行）
    ret_strategy = (W.shift(1).fillna(0.0) * ret_wide).sum(axis=1)

    # 手续费/滑点/双边费率：基于权重变化
    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0
    maker_rate = maker_bps / 10000.0
    taker_rate = taker_bps / 10000.0
    dW = W.fillna(0.0) - W.shift(1).fillna(0.0)
    weight_change = dW.abs().sum(axis=1)
    trade_cost = -(fee_rate + slip_rate) * weight_change
    inc = dW.clip(lower=0).sum(axis=1)
    dec = (-dW.clip(upper=0)).sum(axis=1)
    if maker_bps or taker_bps:
        trade_cost = trade_cost - (taker_rate * inc + maker_rate * dec)

    ret_net = ret_strategy + trade_cost
    equity = (1.0 + ret_net).cumprod()

    out = pd.DataFrame({
        'ret_strategy': ret_strategy,
        'trade_cost': trade_cost,
        'ret_net': ret_net,
        'equity': equity,
        'turnover': weight_change,
    })
    out.index.name = time_col
    log.debug(f"simulate_portfolio_equal_weight: n_steps={len(out)} fee={fee_bps} slip={slippage_bps}")
    return out


def _infer_freq_per_year(index: pd.Index) -> float:
    """根据时间索引推断年化步数。"""
    if len(index) < 3:
        return 365.0
    try:
        dt = pd.to_datetime(index)
    except Exception:
        try:
            dt = pd.to_datetime(index.astype(str))
        except Exception:
            return 365.0
    diffs = (dt[1:] - dt[:-1]).to_series().dt.total_seconds().clip(lower=1)
    med = diffs.median()
    m = med / 60.0
    if m <= 0:
        return 365.0
    per_year = (365.0 * 24.0 * 60.0) / m
    if abs(m - 24*60) < 1e-6:
        return 365.0
    return float(per_year)


def calc_metrics(bt: pd.DataFrame) -> Dict[str, float]:
    """从回测结果计算关键指标。要求包含列 ['ret_net','equity']。"""
    out: Dict[str, float] = {}
    if bt is None or bt.empty or 'ret_net' not in bt.columns or 'equity' not in bt.columns:
        return out
    r = bt['ret_net'].fillna(0.0)
    eq = bt['equity'].fillna(method='ffill').fillna(1.0)
    total_return = float(eq.iloc[-1] - 1.0)
    steps = max(1, len(r))
    per_year = _infer_freq_per_year(bt.index)
    ann_return = float(np.power(1.0 + total_return, per_year / steps) - 1.0)
    std = float(r.std(ddof=0))
    ann_vol = float(std * np.sqrt(per_year)) if std > 0 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    roll_max = eq.cummax()
    dd = (eq / roll_max - 1.0).min()
    max_dd = float(dd)
    out.update(
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )
    log.debug(f"metrics: {out}")
    return out


def simulate_portfolio_equal_weight_exec(
    price_df: pd.DataFrame,
    signal_df: Optional[pd.DataFrame] = None,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    maker_bps: float = 0.0,
    taker_bps: float = 0.0,
    max_weight: float = 1.0,
    max_turnover: float = 1.0,
    min_trade_weight: float = 0.0,
) -> pd.DataFrame:
    """
    组合回测（等权目标 + 执行约束）：
    - 目标：每期对活跃标的等权分配，总权重不超过 1.0；每标的不超过 max_weight。
    - 执行：对目标与上期权重差额按 turnover 约束（L1<=max_turnover）与最小交易权重阈值裁剪。
    - 成本：与 simulate_portfolio_equal_weight 一致（费用+滑点+maker/taker）。
    返回含列：['ret_strategy','trade_cost','ret_net','equity','turnover']
    """
    if price_df is None or price_df.empty:
        raise ValueError('price_df is empty')

    df = price_df.copy()
    time_col = 'datetime' if 'datetime' in df.columns else 'date'
    df['_ret_'] = pd.to_numeric(df['close'], errors='coerce').groupby(df['symbol']).pct_change().fillna(0.0)
    ret_wide = df.pivot(index=time_col, columns='symbol', values='_ret_').fillna(0.0)

    # 信号矩阵（用于定义活跃标的）
    Sig = None
    if signal_df is not None and not signal_df.empty and 'signal' in signal_df.columns:
        s = signal_df.copy()
        if time_col in s.columns:
            Sig = s[[time_col, 'symbol', 'signal']].pivot(index=time_col, columns='symbol', values='signal')
        elif time_col == 'datetime' and 'date' in s.columns:
            # 将 datetime 映射为 date 对齐
            idx = pd.to_datetime(ret_wide.index)
            idx_date = idx.strftime('%Y%m%d')
            Sig = s[['date', 'symbol', 'signal']].pivot(index='date', columns='symbol', values='signal')
            Sig = Sig.reindex(index=idx_date)
            Sig.index = ret_wide.index
        if Sig is not None:
            Sig = Sig.reindex(index=ret_wide.index).fillna(method='ffill').fillna(0.0)

    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0
    maker_rate = maker_bps / 10000.0
    taker_rate = taker_bps / 10000.0

    eq = 1.0
    w_prev = pd.Series(0.0, index=ret_wide.columns)
    ret_strategy_list = []
    trade_cost_list = []
    ret_net_list = []
    equity_list = []
    turnover_list = []

    for t, row in ret_wide.iterrows():
        # 构造目标等权权重
        if Sig is not None:
            active = Sig.loc[t].fillna(0.0)
            act_symbols = active[active > 0].index.tolist()
        else:
            act_symbols = row.index.tolist()

        if len(act_symbols) == 0:
            w_target = pd.Series(0.0, index=ret_wide.columns)
        else:
            gross = min(1.0, max_weight * len(act_symbols))
            per = gross / len(act_symbols)
            w_target = pd.Series(0.0, index=ret_wide.columns)
            w_target.loc[act_symbols] = per

        # delta 权重并施加 turnover 与最小交易阈值
        delta = (w_target - w_prev).copy()
        # 最小交易阈值
        if min_trade_weight > 0:
            delta[delta.abs() < min_trade_weight] = 0.0
        l1 = float(delta.abs().sum())
        if l1 > 0 and l1 > max_turnover:
            scale = max_turnover / l1
            delta *= scale
            l1 = float(delta.abs().sum())

        w_exec = w_prev + delta

        # T+1 执行：使用上期权重计算收益
        r = float((w_prev * row).sum())

        # 成本（本期调仓成本）
        inc = delta.clip(lower=0).sum()
        dec = (-delta.clip(upper=0)).sum()
        cost = -(fee_rate + slip_rate) * l1
        if maker_bps or taker_bps:
            cost -= (taker_rate * inc + maker_rate * dec)

        ret_strategy = r
        ret_net = ret_strategy + cost
        eq *= (1.0 + ret_net)

        ret_strategy_list.append(ret_strategy)
        trade_cost_list.append(cost)
        ret_net_list.append(ret_net)
        equity_list.append(eq)
        turnover_list.append(l1)

        w_prev = w_exec

    out = pd.DataFrame({
        'ret_strategy': ret_strategy_list,
        'trade_cost': trade_cost_list,
        'ret_net': ret_net_list,
        'equity': equity_list,
        'turnover': turnover_list,
    }, index=ret_wide.index)
    out.index.name = time_col
    log.debug(f"simulate_portfolio_equal_weight_exec: n_steps={len(out)} fee={fee_bps} slip={slippage_bps} max_w={max_weight} max_turnover={max_turnover} min_trade={min_trade_weight}")
    return out
