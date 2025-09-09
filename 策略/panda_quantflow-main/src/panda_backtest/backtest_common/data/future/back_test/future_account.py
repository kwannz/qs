#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-8 下午8:56
# @Author : wlb
# @File   : fund_account.py
# @desc   : 股票账户对象
from panda_backtest.backtest_common.data.future.back_test.future_positions import FuturePositions
import logging

from panda_backtest.backtest_common.data.future.base_future_account import BaseFutureAccount

class FutureAccount(BaseFutureAccount):

    def __init__(self, strategy_context, account):

        self.strategy_context = strategy_context
        self.account = account

    def __getitem__(self, key):
        """通过属性名获取属性值"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in FutureAccount")

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
        return ['cash', 'frozen_cash', 'market_value', 'daily_pnl', 'holding_pnl', 'realized_pnl', 
                'total_value', 'transaction_cost', 'positions', 'margin', 'buy_margin', 'sell_margin', 
                'deposit', 'withdraw', 'add_profit', 'today_deposit', 'today_withdraw']

    def values(self):
        """返回所有属性值"""
        return [getattr(self, key) for key in self.keys()]

    def items(self):
        """返回所有属性名和值的键值对"""
        return [(key, getattr(self, key)) for key in self.keys()]

    @property
    def cash(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.available_funds

    @property
    def frozen_cash(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.frozen_capital

    @property
    def market_value(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.market_value

    @property
    def daily_pnl(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.daily_pnl

    @property
    def holding_pnl(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.holding_pnl

    @property
    def realized_pnl(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).dailyResult.xb_back_test_account.realized_pnl

    @property
    def total_value(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.total_profit

    @property
    def transaction_cost(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.cost

    @property
    def positions(self):
        return FuturePositions(self.strategy_context, self.account)

    @property
    def margin(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.margin

    @property
    def buy_margin(self):
        return self.positions.all_buy_margin()

    @property
    def sell_margin(self):
        return self.positions.all_sell_margin()

    @property
    def deposit(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.deposit

    @property
    def withdraw(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.withdraw

    @property
    def add_profit(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.add_profit

    @property
    def today_deposit(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.today_deposit

    @property
    def today_withdraw(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.today_withdraw
