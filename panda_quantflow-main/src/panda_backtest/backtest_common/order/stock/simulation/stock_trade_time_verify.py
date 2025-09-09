#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/5/18 9:42
# @Author : wlb
# @File   : stock_trade_time_verify.py
# @desc   :
import datetime
import logging

import pickle
import re
import time

from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, REJECTED
from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, SYMBOL_NOT_TRADE_IN_THIS_TIME
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.system.context.core_context import CoreContext

class StockTradeTimeVerify(object):
    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):
        strategy_context = self.context.strategy_context
        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        if strategy_context.is_stock_trade():
            return True

        message = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                 account, order_effect, order_side, order_result.order_id,
                                                 SYMBOL_NOT_TRADE_IN_THIS_TIME)
        order_result.message = message
        return False
