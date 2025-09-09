#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-8 下午8:56
# @Author : wlb
# @File   : fund_account.py
# @desc   : 股票账户对象
from panda_backtest.backtest_common.data.fund.back_test.fund_positions import FundPositions
import logging

from panda_backtest.backtest_common.data.fund.base_fund_account import BaseFundAccount

class FundAccount(BaseFundAccount):

    def __init__(self, strategy_context, account):

        self.strategy_context = strategy_context
        self.account = account

    def __getitem__(self, key):
        """通过属性名获取属性值"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in FundAccount")

    def __contains__(self, key):
        """检查是否包含指定属性"""
        return key in self.keys()

    def get(self, key, default=None):
        """获取指定键的值，如果键不存在则返回默认值"""
        if key in self:
            return self[key]
        return default

    def __iter__(self):
        """返回迭代器，迭代所有属性名"""
        return iter(self.keys())

    def __len__(self):
        """返回属性数量"""
        return len(self.keys())

    def keys(self):
        """返回所有属性名称"""
        return ['cash', 'frozen_cash', 'market_value', 'total_value', 'transaction_cost', 'positions', 'add_profit', 'dividend_receivable']

    def values(self):
        """返回所有属性值"""
        return [getattr(self, key) for key in self.keys()]

    def items(self):
        """返回所有属性名和值的键值对"""
        return [(key, getattr(self, key)) for key in self.keys()]

    @property
    def cash(self):
        return self.strategy_context.all_trade_reverse_result.get_fund_reverse_result(
            self.account).xb_back_test_account.available_funds

    @property
    def frozen_cash(self):
        return self.strategy_context.all_trade_reverse_result.get_fund_reverse_result(
            self.account).xb_back_test_account.frozen_capital

    @property
    def market_value(self):
        return self.strategy_context.all_trade_reverse_result.get_fund_reverse_result(
            self.account).xb_back_test_account.market_value

    @property
    def total_value(self):
        return self.strategy_context.all_trade_reverse_result.get_fund_reverse_result(
            self.account).xb_back_test_account.total_profit

    @property
    def transaction_cost(self):
        return self.strategy_context.all_trade_reverse_result.get_fund_reverse_result(
            self.account).xb_back_test_account.cost

    @property
    def positions(self):
        return FundPositions(self.strategy_context, self.account)

    @property
    def add_profit(self):
        start_capital = self.strategy_context.run_info.stock_starting_cash
        if self.strategy_context.all_trade_reverse_result.get_fund_reverse_result(
                self.account).start_capital:
            start_capital = self.strategy_context.all_trade_reverse_result.get_fund_reverse_result(
                self.account).start_capital
        return (self.total_value - start_capital) / start_capital

    @property
    def dividend_receivable(self):
        return 0
