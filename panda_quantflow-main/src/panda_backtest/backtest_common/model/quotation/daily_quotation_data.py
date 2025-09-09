#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-22 下午12:37
# @Author : wlb
# @File   : daily_quotation_data.py
# @desc   : 日线数据
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class DailyQuotationData:
    symbol = EMPTY_STRING                  # 标的代码
    code = EMPTY_STRING                    # 交易所原始代码
    trade_date = EMPTY_INT                 # YYYYMMDD格式，如20170823
    open = EMPTY_FLOAT                     # 开盘价
    high = EMPTY_FLOAT                     # 最高价
    low = EMPTY_FLOAT                      # 最低价
    close = EMPTY_FLOAT                    # 收盘价
    volume = EMPTY_FLOAT                   # 成交量
    turnover = EMPTY_FLOAT                 # 成交金额
    vwap = EMPTY_FLOAT                     # 成交均价
    settlement = EMPTY_FLOAT                   # 结算价
    oi = EMPTY_FLOAT                       # 持仓量
    trade_status = EMPTY_STRING            # 交易状态（”停牌”或者”交易”）
    last = EMPTY_FLOAT
    price_limit_range = EMPTY_FLOAT
    unit_nav = EMPTY_FLOAT        # 基金单位净值

