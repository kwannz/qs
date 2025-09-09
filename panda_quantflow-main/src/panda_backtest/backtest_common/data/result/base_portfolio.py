#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-8 下午7:35
# @Author : wlb
# @File   : base_portfolio.py
# @desc   : 投资组合信息
import abc
import logging

from six import with_metaclass

class BasePortfolio(with_metaclass(abc.ABCMeta)):

    @property
    @abc.abstractmethod
    def cash(self):
        """可用资金"""
        pass

    @property
    @abc.abstractclassmethod
    def frozen_cash(self):
        """冻结资金"""
        pass

    @property
    @abc.abstractmethod
    def total_returns(self):
        """投资组合至今的累积收益率"""
        pass

    @property
    @abc.abstractmethod
    def daily_returns(self):
        """投资组合每日收益率"""
        pass

    @property
    @abc.abstractmethod
    def daily_pnl(self):
        """当日盈亏"""
        pass

    @property
    @abc.abstractmethod
    def market_value(self):
        """投资组合当前的市场价值"""
        pass

    @property
    @abc.abstractmethod
    def total_value(self):
        """总权益"""
        pass

    @property
    @abc.abstractmethod
    def units(self):
        """份额"""
        pass

    @property
    @abc.abstractmethod
    def unit_net_value(self):
        """单位净值"""
        pass

    @property
    @abc.abstractmethod
    def static_unit_net_value(self):
        """静态单位权益"""
        pass

    @property
    @abc.abstractmethod
    def transaction_cost(self):
        """当日费用"""
        pass

    @property
    @abc.abstractmethod
    def pnl(self):
        """当前投资组合的累计盈亏"""
        pass

    @property
    @abc.abstractmethod
    def start_date(self):
        """策略投资组合的回测/实时模拟交易的开始日期"""
        pass

    @property
    @abc.abstractmethod
    def annualized_returns(self):
        """投资组合的年化收益率"""
        pass

