#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:02
# @Author : wlb
# @File   : base_future_positions_item.py
# @desc   : 上下文对象期货账号持仓item
import abc
import logging

from six import with_metaclass

class BaseFuturePositionsItmes(with_metaclass(abc.ABCMeta)):

    @property
    @abc.abstractmethod
    def order_book_id(self):
        pass

    @property
    @abc.abstractmethod
    def pnl(self):
        pass

    @property
    @abc.abstractmethod
    def daily_pnl(self):
        pass

    @property
    @abc.abstractmethod
    def holding_pnl(self):
        pass

    @property
    @abc.abstractmethod
    def realized_pnl(self):
        pass

    @property
    @abc.abstractmethod
    def transaction_cost(self):
        pass

    @property
    @abc.abstractmethod
    def margin(self):
        pass

    @property
    @abc.abstractmethod
    def market_value(self):
        pass

    @property
    @abc.abstractmethod
    def buy_daily_pnl(self):
        pass

    @property
    @abc.abstractmethod
    def buy_pnl(self):
        pass

    @property
    @abc.abstractmethod
    def buy_transaction_cost(self):
        pass

    @property
    @abc.abstractmethod
    def closable_buy_quantity(self):
        pass

    @property
    @abc.abstractmethod
    def buy_margin(self):
        pass

    @property
    @abc.abstractmethod
    def buy_today_quantity(self):
        pass

    @property
    @abc.abstractmethod
    def buy_quantity(self):
        pass

    @property
    @abc.abstractmethod
    def buy_avg_open_price(self):
        pass

    @property
    @abc.abstractmethod
    def buy_avg_holding_price(self):
        pass

    @property
    @abc.abstractmethod
    def sell_daily_pnl(self):
        pass

    @property
    @abc.abstractmethod
    def sell_pnl(self):
        pass

    @property
    @abc.abstractmethod
    def sell_transaction_cost(self):
        pass

    @property
    @abc.abstractmethod
    def closable_sell_quantity(self):
        pass

    @property
    @abc.abstractmethod
    def sell_margin(self):
        pass

    @property
    @abc.abstractmethod
    def sell_today_quantity(self):
        pass

    @property
    @abc.abstractmethod
    def sell_quantity(self):
        pass

    @property
    @abc.abstractmethod
    def sell_avg_open_price(self):
        pass

    @property
    @abc.abstractmethod
    def sell_avg_holding_price(self):
        pass
