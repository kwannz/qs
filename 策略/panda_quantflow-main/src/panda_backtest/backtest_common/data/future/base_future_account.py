#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-8 下午8:56
# @Author : wlb
# @File   : base_stock_account.py
# @desc   : 股票账户对象
import abc
import logging

from six import with_metaclass

class BaseFutureAccount(with_metaclass(abc.ABCMeta)):

    @property
    @abc.abstractmethod
    def cash(self):
        pass

    @property
    @abc.abstractmethod
    def frozen_cash(self):
        pass

    @property
    @abc.abstractmethod
    def market_value(self):
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
    def total_value(self):
        pass

    @property
    @abc.abstractmethod
    def transaction_cost(self):
        pass

    @property
    @abc.abstractmethod
    def positions(self):
        pass

    @property
    @abc.abstractmethod
    def margin(self):
        pass

    @property
    @abc.abstractmethod
    def buy_margin(self):
        pass

    @property
    @abc.abstractmethod
    def sell_margin(self):
        pass
