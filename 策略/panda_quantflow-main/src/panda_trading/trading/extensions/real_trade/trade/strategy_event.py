#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/26 上午11:37
# @Author : wlb
# @File   : strategy_event.py
# @desc   :
from panda_backtest import SRLogger
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.exception.error_exception import ErrorException
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent


class StrategyEvent(object):

    def __init__(self, event_bus, strategy_context):
        self.event_bus = event_bus
        self.strategy_context = strategy_context

    def push_trading_before(self, strategy_context):
        try:
            event = Event(
                ConstantEvent.STRATEGY_TRADING_BEFORE,
                context=strategy_context)
            self.event_bus.publish_event(event)
        except ErrorException as e:
            SRLogger.error(e.message)

    def push_day_before(self, strategy_context):
        try:
            event = Event(
                ConstantEvent.STRATEGY_DAY_BEFORE,
                context=strategy_context)
            self.event_bus.publish_event(event)
        except ErrorException as e:
            SRLogger.error(e.message)

    def push_trading_after(self, strategy_context):
        try:
            event = Event(
                ConstantEvent.STRATEGY_TRADING_AFTER,
                context=strategy_context)
            self.event_bus.publish_event(event)
        except ErrorException as e:
            SRLogger.error(e.message)

    def push_handle_bar(self, strategy_context, data):
        try:
            event = Event(
                ConstantEvent.STRATEGY_HANDLE_BAR,
                data=data,
                context=strategy_context)
            self.event_bus.publish_event(event)
        except ErrorException as e:
            SRLogger.error(e.message)

    def push_order_cancel_system(self, order):
        try:
            quotation_data = QuotationData.get_instance()
            data = quotation_data.bar_dict
            event = Event(
                ConstantEvent.ORDER_CANCEL_SYSTEM,
                order=order,
                context=self.strategy_context,
                data=data)
            self.event_bus.publish_event(event)
        except ErrorException as e:
            SRLogger.error(e.message)