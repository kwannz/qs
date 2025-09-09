#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/11/10 11:06
# @Author : wlb
# @File   : order_risk_control_verify.py
# @desc   :
from panda_backtest.backtest_common.constant.strategy_constant import REJECTED, SIDE_BUY, CLOSE
import logging

from panda_backtest.backtest_common.constant.string_constant import STOCK_ORDER_FAILED_MESSAGE, \
    FUTURE_ORDER_FAILED_MESSAGE, FUND_ORDER_FAILED_MESSAGE
from panda_backtest.backtest_common.order.order_verify import OrderVerify
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent

class OrderRiskControlVerify(OrderVerify):
    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):
        event_bus = self.context.event_bus

        event = Event(ConstantEvent.RISK_CONTROL_ORDER_VERIFY, conext=self.context.strategy_context, order=order_result)
        event_bus.publish_event(event)

        if order_result.status == REJECTED:
            order_result.is_reject_by_risk = 1
            if order_result.side == SIDE_BUY:
                order_side = '买入'
            else:
                order_side = '卖出'
            if order_result.effect == CLOSE:
                order_effect = '平仓'
            else:
                order_effect = '开仓'

            if order_result.order_type == 0:
                order_failed_message_constant = STOCK_ORDER_FAILED_MESSAGE
            elif order_result.order_type == 1:
                order_failed_message_constant = FUTURE_ORDER_FAILED_MESSAGE
            else:
                order_failed_message_constant = FUND_ORDER_FAILED_MESSAGE

            order_result.message = order_failed_message_constant % (
                order_result.order_book_id, str(order_result.quantity),
                account, order_effect, order_side, order_result.order_id,
                order_result.message)
            return False
        return True
