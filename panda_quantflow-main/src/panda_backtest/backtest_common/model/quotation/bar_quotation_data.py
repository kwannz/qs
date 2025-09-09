#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-22 下午5:08
# @Author : wlb
# @File   : bar_quotation_data.py
# @desc   :
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class BarQuotationData:
    symbol = EMPTY_STRING              # 标的代码
    code = EMPTY_STRING                # 交易所原始代码
    date = EMPTY_INT                   # 自然日, YYYYMMDD格式，如20170823
    time = EMPTY_INT                   # 时间，精确到秒，如14:21:05记为142105
    trade_date = EMPTY_INT             # YYYYMMDD格式，如20170823
    freq = EMPTY_INT                    # bar类型
    open = EMPTY_FLOAT                 # bar内开盘价
    high = EMPTY_FLOAT                 # bar内最高价
    low = EMPTY_FLOAT                  # bar内最低价
    close = EMPTY_FLOAT                # bar内收盘价
    volume = EMPTY_FLOAT               # bar内成交量
    turnover = EMPTY_FLOAT             # bar内成交金额
    vwap = EMPTY_FLOAT                 # bar内成交均价
    oi = EMPTY_FLOAT                   # 当前持仓量
    settlement = EMPTY_FLOAT               # 结算价
    last = EMPTY_FLOAT
    preclose = EMPTY_FLOAT
    limit_up = EMPTY_FLOAT
    limit_down = EMPTY_FLOAT

    askprice1 = EMPTY_FLOAT
    bidprice1 = EMPTY_FLOAT
    askvolume1 = EMPTY_FLOAT
    bidvolume1 = EMPTY_FLOAT

    askprice2 = EMPTY_FLOAT
    bidprice2 = EMPTY_FLOAT
    askvolume2 = EMPTY_FLOAT
    bidvolume2 = EMPTY_FLOAT

    askprice3 = EMPTY_FLOAT
    bidprice3 = EMPTY_FLOAT
    askvolume3 = EMPTY_FLOAT
    bidvolume3 = EMPTY_FLOAT

    askprice4 = EMPTY_FLOAT
    bidprice4 = EMPTY_FLOAT
    askvolume4 = EMPTY_FLOAT
    bidvolume4 = EMPTY_FLOAT

    askprice5 = EMPTY_FLOAT
    bidprice5 = EMPTY_FLOAT
    askvolume5 = EMPTY_FLOAT
    bidvolume5 = EMPTY_FLOAT

