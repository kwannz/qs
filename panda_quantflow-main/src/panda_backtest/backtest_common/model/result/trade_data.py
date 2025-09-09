#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 19-4-5 下午8:31
# @Author : wlb
# @File   : trade_data.py
# @desc   :
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class TradeData:

    def __init__(self):
        # 代码编号相关
        self.symbol = EMPTY_STRING  # 合约代码
        self.symbolName = EMPTY_STRING  # 合约名称
        self.exchange = EMPTY_STRING  # 交易所代码

        self.tradeID = EMPTY_STRING  # 成交编号

        self.orderID = EMPTY_STRING  # 订单编号

        # 成交相关
        self.direction = EMPTY_UNICODE  # 成交方向
        self.offset = EMPTY_UNICODE  # 成交开平仓
        self.price = EMPTY_FLOAT  # 成交价格
        self.volume = EMPTY_INT  # 成交数量
        self.tradeTime = EMPTY_STRING  # 成交时间
        self.tradeDate = EMPTY_STRING  # 成交日期
        self.cost = EMPTY_FLOAT  # 成交费用
