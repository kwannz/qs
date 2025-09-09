#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-9-17 下午4:27
# @Author : wlb
# @File   : base_stock_positions_item.py
# @desc   :
import abc
import logging

from six import with_metaclass

class BaseStockPositionsItmes(with_metaclass(abc.ABCMeta)):

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
    def quantity(self):
        pass

    @property
    @abc.abstractmethod
    def sellable(self):
        pass

    @property
    @abc.abstractmethod
    def market_value(self):
        pass

    @property
    @abc.abstractmethod
    def value_percent(self):
        pass

    @property
    @abc.abstractmethod
    def avg_price(self):
        pass
