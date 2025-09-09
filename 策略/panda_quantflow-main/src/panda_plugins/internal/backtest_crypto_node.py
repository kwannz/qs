from typing import Optional, Type

from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import uuid

# 依赖于策略项目中的 panda_data（请确保已在该子项目下 editable 安装）
try:
    import panda_data
except Exception:  # 懒加载失败时提示
    panda_data = None


@ui(
    factors={"input_type": "None"},
    code={"input_type": "text_field", "placeholder": "策略代码（当前仅支持 Buy&Hold 演示）"},
    frequency={"input_type": "combobox", "options": ["1d", "1m"], "placeholder": "回测频率", "allow_link": False},
    start_capital={"input_type": "number_field", "placeholder": "请输入初始资金", "allow_link": False},
    base_symbol={"input_type": "text_field", "placeholder": "基准(如 BINANCE:BTCUSDT)"},
    fee_rate={"input_type": "number_field", "placeholder": "手续费率(基点)", "allow_link": False},
    start_date={"input_type":"date_field","placeholder": "开始日期", "allow_link": False},
    end_date={"input_type":"date_field","placeholder": "结束日期", "allow_link": False}
)
class CryptoBacktestInputModel(BaseModel):
    code: str = Field(default="", title="策略代码")
    factors: pd.DataFrame = Field(default_factory=pd.DataFrame, title="因子值")
    start_capital: float = Field(default=100000.0, title="初始资金")
    base_symbol: str = Field(default="BINANCE:BTCUSDT", title="基准品种")
    fee_rate: float = Field(default=1.0, title="手续费(基点)")
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
            if input.frequency == '1m':
                df = panda_data.get_crypto_min_data(
                    start_date=input.start_date,
                    end_date=input.end_date,
                    symbol=input.base_symbol,
                    fields=["datetime", "symbol", "open", "high", "low", "close", "volume"],
                )
            else:
                df = panda_data.get_market_data(
                    start_date=input.start_date,
                    end_date=input.end_date,
                    symbols=[input.base_symbol],
                    fields=["date", "symbol", "open", "high", "low", "close", "volume"],
                    type='crypto'
                )
            if df is None or df.empty:
                raise RuntimeError("未查询到加密行情数据")
            # 只取目标标的
            df = df[df['symbol'] == input.base_symbol].copy()
            # 组装信号：若 factors 带有 signal 列（0/1），则使用；否则等同 Buy&Hold
            sig_df = None
            if input.factors is not None and not input.factors.empty and 'signal' in input.factors.columns:
                # 仅取对应标的
                sig_df = input.factors.copy()
                if 'symbol' in sig_df.columns:
                    sig_df = sig_df[sig_df['symbol'] == input.base_symbol]

            from .crypto_backtest_engine import simulate_long_flat
            bt = simulate_long_flat(df, sig_df, fee_bps=input.fee_rate)
            total_return = float(bt['equity'].iloc[-1] - 1.0)
            msg = f"{input.frequency} 回测 {input.base_symbol} | {input.start_date}~{input.end_date} | 总收益率: {total_return:.4f}"
            self.log_info(msg)
            return CryptoBacktestOutputModel(
                backtest_id=backtest_id,
                total_return=total_return,
                message=msg,
            )
        except Exception as e:
            self.log_error(str(e))
            return CryptoBacktestOutputModel(
                backtest_id=backtest_id,
                total_return=0.0,
                message=f"回测失败: {e}"
            )
