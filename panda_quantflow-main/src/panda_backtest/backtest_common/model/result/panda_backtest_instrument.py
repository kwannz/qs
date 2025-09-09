from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class PandaBacktestInstrument:
    type = EMPTY_INT                                        # 类型（0：股票  1：期货）
    symbol = EMPTY_STRING
    name = EMPTY_STRING                                     # 合约名称
    market = EMPTY_STRING                                   # 交易所
    contractmul = EMPTY_FLOAT                               # 合约乘数
    min_price_unit = EMPTY_FLOAT                            # 最小交易单位
    long_margin = EMPTY_FLOAT                               # 费用
    short_margin_ = EMPTY_FLOAT                             # 交易日期
    margin_type = EMPTY_INT
    open_cost = EMPTY_FLOAT
    close_cost = EMPTY_FLOAT
    cost_type = EMPTY_INT