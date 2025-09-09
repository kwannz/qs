#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-7 下午3:42
# @Author : wlb
# @File   : bar_map.py
# @desc   : 行情自定义字典
import six
import logging

from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.data.quotation.back_test.base_bar_data_source import BaseBarDataSource

class BarMap(object):
    def __init__(self, bar_data_source: BaseBarDataSource):
        self.context = CoreContext.get_instance()
        self.bar_data_source = bar_data_source
        
    def get(self, key):
        return self.__getitem__(key)

    def __getitem__(self, key):
        """
        合约行情数据
        :param key:
        :return:
        """
        if not isinstance(key, str):
            # TODO：抛异常
            raise Exception('获取行情数据失败')

        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        frequency = run_info.frequency

        key_list = key.split('.')
        if len(key_list) == 2:
            exchange = key_list[1]
        else:
            exchange = 'SH'

        if exchange == 'OF':
            return self.bar_data_source.get_fund_daily_bar(key, strategy_context.trade_date)
        elif exchange == 'SZ' or exchange == 'SH':
            if frequency == '1d':
                return self.bar_data_source.get_stock_daily_bar(key, strategy_context.trade_date)
            elif frequency == '1M':
                return self.bar_data_source.get_stock_minute_bar(key, strategy_context.trade_date, strategy_context.hms)

        else:
            if frequency == '1d':
                return self.bar_data_source.get_future_daily_bar(key, strategy_context.trade_date)
            elif frequency == '1M':
                return self.bar_data_source.get_future_minute_bar(key, strategy_context.trade_date, strategy_context.hms)

    def init_future_position_quotation(self, position_list):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        frequency = run_info.frequency
        self.bar_data_source.init_future_list_daily_quotation(position_list, strategy_context.trade_date, frequency)

    def init_stock_position_quotation(self, position_list):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        frequency = run_info.frequency
        self.bar_data_source.init_stock_list_daily_quotation(position_list, strategy_context.trade_date, frequency)

    def change_last_field(self, field_type):
        self.bar_data_source.change_last_field(field_type)