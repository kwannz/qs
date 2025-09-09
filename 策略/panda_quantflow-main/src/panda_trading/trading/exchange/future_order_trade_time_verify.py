#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/3/26 10:08
# @Author : wlb
# @File   : future_order_trade_time_verify.py
# @desc   :
from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, REJECTED
from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, SYMBOL_NOT_TRADE_IN_THIS_TIME
from panda_backtest.backtest_common.order.order_verify import OrderVerify
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory


class FutureOrderTradeTimeVerify(OrderVerify):

    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):

        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        if not self.context.strategy_context.judge_future_trade(order_result.order_book_id):
            order_result.status = REJECTED
            message = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                     account, order_effect, order_side, order_result.order_id,
                                                     SYMBOL_NOT_TRADE_IN_THIS_TIME)
            order_result.message = message
            return False
        return True
