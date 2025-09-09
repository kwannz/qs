#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-8 下午8:56
# @Author : wlb
# @File   : fund_account.py
# @desc   : 股票账户对象
from panda_backtest.backtest_common.data.future.real_time.future_positions import FuturePositions
import logging

from panda_backtest.backtest_common.data.future.base_future_account import BaseFutureAccount

class FutureAccount(BaseFutureAccount):

    def __init__(self, strategy_context, account):

        self.strategy_context = strategy_context
        self.account = account
        self.init_pos_status = False

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
            self.account).xb_back_test_account.realized_pnl

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
    def today_deposit(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.today_deposit

    @property
    def today_withdraw(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.today_withdraw

    @property
    def add_profit(self):
        return self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).xb_back_test_account.add_profit
