#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-7 下午3:42
# @Author : wlb
# @File   : bar_map.py
# @desc   : 行情数据资源
import abc
import logging

from six import with_metaclass

class BaseBarDataSource(with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def get_stock_daily_bar(self, symbol, date):
        """
        获取股票日线行情
        :param symbol:
        :param date:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_stock_minute_bar(self, symbol, date, time):
        """
        获取股票分钟线行情
        :param symbol:
        :param date:
        :param time:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_future_daily_bar(self, symbol, date):
        """
        获取期货日线行情
        :param symbol:
        :param date:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_future_minute_bar(self, symbol, date, time):
        """
        获取期货分钟线行情
        :param symbol:
        :param date:
        :param time:
        :return:
        """
        pass

    @abc.abstractmethod
    def init_future_list_daily_quotation(self, position_list, date, freq='d'):
        pass

    @abc.abstractmethod
    def init_stock_list_daily_quotation(self, position_list, date, freq='d'):
        pass

    @abc.abstractmethod
    def clear_cache_data(self):
        pass

    @abc.abstractmethod
    def get_fund_daily_bar(self, symbol, trade_date):
        pass

    @abc.abstractmethod
    def change_last_field(self, field_type):
        pass
