from typing import Optional, Type

from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import uuid
from common.logging.log_context import set_backtest_id, reset_backtest_id

# 依赖于策略项目中的 panda_data（请确保已在该子项目下 editable 安装）
try:
    import panda_data
except Exception:  # 懒加载失败时提示
    panda_data = None


@ui(
    factors={"input_type": "None"},
    code={"input_type": "text_field", "placeholder": "策略代码（当前支持 Buy&Hold / 信号等权）"},
    frequency={"input_type": "combobox", "options": ["1d", "1m"], "placeholder": "回测频率", "allow_link": False},
    start_capital={"input_type": "number_field", "placeholder": "请输入初始资金", "allow_link": False},
    base_symbol={"input_type": "text_field", "placeholder": "单标的(如 BINANCE:BTCUSDT)"},
    symbols={"input_type": "text_field", "placeholder": "多标的(逗号分隔，如 BINANCE:BTCUSDT,BINANCE:ETHUSDT)"},
    resample_rule={"input_type": "combobox", "options": ["1m","5min","15min","30min","1h"], "placeholder": "分钟重采样(仅在1m频率有效)", "allow_link": False},
    fee_rate={"input_type": "number_field", "placeholder": "基础手续费(基点)", "allow_link": False},
    slippage_bps={"input_type": "number_field", "placeholder": "滑点(基点)", "allow_link": False},
    maker_bps={"input_type": "number_field", "placeholder": "Maker费率(基点)", "allow_link": False},
    taker_bps={"input_type": "number_field", "placeholder": "Taker费率(基点)", "allow_link": False},
    start_date={"input_type":"date_field","placeholder": "开始日期", "allow_link": False},
    end_date={"input_type":"date_field","placeholder": "结束日期", "allow_link": False}
)
class CryptoBacktestInputModel(BaseModel):
    code: str = Field(default="", title="策略代码")
    factors: pd.DataFrame = Field(default_factory=pd.DataFrame, title="因子值")
    start_capital: float = Field(default=100000.0, title="初始资金")
    base_symbol: str = Field(default="BINANCE:BTCUSDT", title="单标的（兼容保留）")
    symbols: str = Field(default="", title="多标的（逗号分隔）")
    resample_rule: str = Field(default="1m", title="分钟重采样(仅1m有效)")
    fee_rate: float = Field(default=1.0, title="手续费(基点)")
    slippage_bps: float = Field(default=0.0, title="滑点(基点)")
    maker_bps: float = Field(default=0.0, title="Maker费率(基点)")
    taker_bps: float = Field(default=0.0, title="Taker费率(基点)")
    max_weight: float = Field(default=1.0, title="单标的最大权重")
    max_turnover: float = Field(default=1.0, title="单期最大换手(L1)")
    min_trade_weight: float = Field(default=0.0, title="最小交易权重阈值")
    frequency: str = Field(default="1d", title="回测频率")
    start_date: str = Field(default="20240101", title="开始日期")
    end_date: str = Field(default="20241231", title="结束日期")
    model_config = {"arbitrary_types_allowed": True}

    @field_validator('factors')
    def validate_df_factor(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('factors must be a pandas DataFrame')
        return v


class CryptoBacktestOutputModel(BaseModel):
    backtest_id: str = Field(default="error", title="回测id")
    total_return: float = Field(default=0.0, title="总收益率")
    message: str = Field(default="", title="说明")
    metrics: dict = Field(default_factory=dict, title="关键指标")
    equity_preview: list = Field(default_factory=list, title="权益预览(最多500点)")


@work_node(name="加密回测", group="05-回测相关", type="general", box_color="yellow")
class CryptoBacktestControl(BaseWorkNode):
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return CryptoBacktestInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return CryptoBacktestOutputModel

    def run(self, input: BaseModel) -> BaseModel:
        # 简易回测：支持 Buy&Hold 或基于外部 signal 的多/空位（仅多/空=0）
        backtest_id = uuid.uuid4().hex[:12]
        try:
            if panda_data is None:
                raise RuntimeError("未找到 panda_data，请先在 策略/panda_factor-main 下 editable 安装")
            panda_data.init()
            token_bt = set_backtest_id(backtest_id)
            self.log_debug(f"params: freq={input.frequency} symbols={getattr(input,'symbols','')} base={input.base_symbol} start={input.start_date} end={input.end_date} fee={input.fee_rate} slip={getattr(input,'slippage_bps',0)} maker={getattr(input,'maker_bps',0)} taker={getattr(input,'taker_bps',0)} resample={getattr(input,'resample_rule','1m')}")
            # 解析多标的
            symbols = []
            if getattr(input, 'symbols', None):
                symbols = [s.strip() for s in input.symbols.split(',') if s.strip()]
            if not symbols:
                symbols = [input.base_symbol]

            if input.frequency == '1m':
                # 逐标的查询分钟数据并合并
                dfs = []
                for sym in symbols:
                    # 若选择了重采样规则且非1m，则进行重采样
                    if getattr(input, 'resample_rule', '1m') and input.resample_rule != '1m':
                        dfi = panda_data.get_crypto_min_data_resampled(
                            start_date=input.start_date,
                            end_date=input.end_date,
                            symbol=sym,
                            rule=input.resample_rule,
                        )
                    else:
                        dfi = panda_data.get_crypto_min_data(
                            start_date=input.start_date,
                            end_date=input.end_date,
                            symbol=sym,
                            fields=["datetime", "symbol", "open", "high", "low", "close", "volume"],
                        )
                    if dfi is not None and not dfi.empty:
                        dfs.append(dfi)
                df = None if not dfs else pd.concat(dfs, ignore_index=True)
            else:
                df = panda_data.get_market_data(
                    start_date=input.start_date,
                    end_date=input.end_date,
                    symbols=symbols,
                    fields=["date", "symbol", "open", "high", "low", "close", "volume"],
                    type='crypto'
                )
            if df is None or df.empty:
                raise RuntimeError("未查询到加密行情数据")
            self.log_debug(f"loaded rows: {len(df)}; unique symbols: {df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'}")
            # 若为单标的，与旧行为一致；多标的走组合
            single = len(symbols) == 1
            # 组装信号：若 factors 带有 signal 列（0/1），则使用；否则等同 Buy&Hold/等权
            sig_df = None
            if input.factors is not None and not input.factors.empty and 'signal' in input.factors.columns:
                sig_df = input.factors.copy()
                if 'symbol' in sig_df.columns and single:
                    sig_df = sig_df[sig_df['symbol'] == symbols[0]]

            from .crypto_backtest_engine import (
                simulate_long_flat,
                simulate_portfolio_equal_weight,
                simulate_portfolio_equal_weight_exec,
                calc_metrics,
            )
            if single:
                # 过滤到单一标的
                df = df[df['symbol'] == symbols[0]].copy()
                bt = simulate_long_flat(
                    df, sig_df,
                    fee_bps=input.fee_rate,
                    slippage_bps=getattr(input, 'slippage_bps', 0.0) or 0.0,
                    maker_bps=getattr(input, 'maker_bps', 0.0) or 0.0,
                    taker_bps=getattr(input, 'taker_bps', 0.0) or 0.0,
                )
            else:
                # 若提供了执行约束参数，则使用带执行约束的版本
                use_exec = (
                    (getattr(input, 'max_weight', 1.0) or 1.0) < 1.0 or
                    (getattr(input, 'max_turnover', 1.0) or 1.0) < 1.0 or
                    (getattr(input, 'min_trade_weight', 0.0) or 0.0) > 0.0
                )
                if use_exec:
                    bt = simulate_portfolio_equal_weight_exec(
                        df, sig_df,
                        fee_bps=input.fee_rate,
                        slippage_bps=getattr(input, 'slippage_bps', 0.0) or 0.0,
                        maker_bps=getattr(input, 'maker_bps', 0.0) or 0.0,
                        taker_bps=getattr(input, 'taker_bps', 0.0) or 0.0,
                        max_weight=getattr(input, 'max_weight', 1.0) or 1.0,
                        max_turnover=getattr(input, 'max_turnover', 1.0) or 1.0,
                        min_trade_weight=getattr(input, 'min_trade_weight', 0.0) or 0.0,
                    )
                else:
                    bt = simulate_portfolio_equal_weight(
                        df, sig_df,
                        fee_bps=input.fee_rate,
                        slippage_bps=getattr(input, 'slippage_bps', 0.0) or 0.0,
                        maker_bps=getattr(input, 'maker_bps', 0.0) or 0.0,
                        taker_bps=getattr(input, 'taker_bps', 0.0) or 0.0,
                    )

            total_return = float(bt['equity'].iloc[-1] - 1.0)
            metrics = calc_metrics(bt)
            self.log_debug(f"metrics: {metrics}")

            # 组装预览点（最多500点）
            eq_points = []
            idx = bt.index
            eq = bt['equity']
            step = max(1, len(eq) // 500)
            for i in range(0, len(eq), step):
                t = idx[i]
                if hasattr(t, 'isoformat'):
                    t_str = t.isoformat()
                else:
                    t_str = str(t)
                eq_points.append({"t": t_str, "equity": float(eq.iloc[i])})
            msg = f"{input.frequency} 回测 {input.base_symbol} | {input.start_date}~{input.end_date} | 总收益率: {total_return:.4f}"
            self.log_info(msg)
            return CryptoBacktestOutputModel(
                backtest_id=backtest_id,
                total_return=total_return,
                message=msg,
                metrics=metrics,
                equity_preview=eq_points,
            )
        except Exception as e:
            self.log_error(str(e))
            return CryptoBacktestOutputModel(
                backtest_id=backtest_id,
                total_return=0.0,
                message=f"回测失败: {e}"
            )
        finally:
            try:
                reset_backtest_id(token_bt)
            except Exception:
                pass
