# -*- coding: utf-8 -*-
"""
File: strategy.py
Author: peiqi
Date: 2025/5/14
Description: 
"""
from panda_backtest.backtest_common.system.event.event import ConstantEvent
import logging

from panda_backtest.backtest_common.exception.error_exception import ErrorException
from panda_backtest.backtest_common.exception.code import error_code

class Strategy(object):

    def __init__(self, global_args, event_bus):
        self._init = global_args.get('initialize', None)
        self._before_trading = global_args.get('before_trading', None)
        self._after_trading = global_args.get('after_trading', None)
        self._handle_data = global_args.get('handle_data', None)
        if self._init is None:
            raise ErrorException(
                '缺少必要方法：initialize',
                error_code.FUNCTION_NOT_FOUND,
                'initialize')
        else:
            event_bus.register_handle(
                ConstantEvent.STRATEGY_INIT,
                global_args.get(
                    'initialize',
                    None))

        if global_args.get('handle_data', None) is None:
            raise ErrorException(
                '缺少必要方法：handle_data',
                error_code.FUNCTION_NOT_FOUND,
                'handle_data')

        event_bus.register_handle(
            ConstantEvent.STRATEGY_HANDLE_BAR,
            global_args.get(
                'handle_data',
                None))

        if global_args.get('before_trading', None) is not None:
            event_bus.register_handle(
                ConstantEvent.STRATEGY_TRADING_BEFORE,
                global_args.get(
                    'before_trading',
                    None))

        if global_args.get('after_trading', None) is not None:
            event_bus.register_handle(
                ConstantEvent.STRATEGY_TRADING_AFTER,
                global_args.get(
                    'after_trading',
                    None))

        if global_args.get('trade_error', None) is not None:
            event_bus.register_handle(
                ConstantEvent.STRATEGY_TRADE_ERROR,
                global_args.get(
                    'trade_error',
                    None))

        if global_args.get('day_before', None) is not None:
            event_bus.register_handle(
                ConstantEvent.STRATEGY_DAY_BEFORE,
                global_args.get(
                    'day_before',
                    None))

        if global_args.get('stock_order_cancel', None) is not None:
            event_bus.register_handle(
                ConstantEvent.STOCK_ORDER_CANCEL,
                global_args.get(
                    'stock_order_cancel',
                    None))

        if global_args.get('future_order_cancel', None) is not None:
            event_bus.register_handle(
                ConstantEvent.FUTURE_ORDER_CANCEL,
                global_args.get(
                    'future_order_cancel',
                    None))

        if global_args.get('on_future_trade_rtn', None) is not None:
            event_bus.register_handle(
                ConstantEvent.ON_FUTURE_TRADE_RTN,
                global_args.get(
                    'on_future_trade_rtn',
                    None))

        if global_args.get('on_stock_trade_rtn', None) is not None:
            event_bus.register_handle(
                ConstantEvent.ON_STOCK_TRADE_RTN,
                global_args.get(
                    'on_stock_trade_rtn',
                    None))

        if global_args.get('handle_tick', None) is not None:
            event_bus.register_handle(
                ConstantEvent.STRATEGY_HANDLE_TICK,
                global_args.get(
                    'handle_tick',
                    None))
