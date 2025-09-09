#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-8 下午7:35
# @Author : wlb
# @File   : portfolio.py
# @desc   : 投资组合信息
import time
import logging

from panda_backtest.backtest_common.data.result.base_portfolio import BasePortfolio

class Portfolio(BasePortfolio):

    def __init__(self, strategy_context):
        self.strategy_context = strategy_context

    @property
    def cash(self):
        """可用资金"""

        cash = 0

        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            cash += stock_daily_result.xb_back_test_account.available_funds

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            cash += future_daily_result.xb_back_test_account.available_funds

        return cash

    @property
    def frozen_cash(self):
        """冻结资金"""

        frozen_cash = 0

        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            frozen_cash += stock_daily_result.xb_back_test_account.frozen_capital

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            frozen_cash += future_daily_result.xb_back_test_account.frozen_capital

        return frozen_cash

    @property
    def total_returns(self):
        """投资组合至今的累积收益率"""

        start_capital = 0
        add_profit = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            add_profit += stock_daily_result.xb_back_test_account.add_profit
            start_capital += stock_daily_result.xb_back_test_account.start_capital

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            add_profit += future_daily_result.xb_back_test_account.add_profit
            start_capital += future_daily_result.xb_back_test_account.start_capital

        return add_profit / start_capital

    @property
    def daily_returns(self):
        """投资组合每日收益率"""

        yes_total_capital = 0
        start_capital = 0
        total_profit = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            xb_back_test_account = stock_daily_result.xb_back_test_account
            yes_total_capital += xb_back_test_account.yes_total_capital
            start_capital += xb_back_test_account.start_capital
            total_profit += xb_back_test_account.total_profit

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            xb_back_test_account = future_daily_result.xb_back_test_account
            yes_total_capital += xb_back_test_account.yes_total_capital
            start_capital += xb_back_test_account.start_capital
            total_profit += xb_back_test_account.total_profit

        return total_profit / yes_total_capital - 1

    @property
    def daily_pnl(self):
        """当日盈亏"""
        daily_pnl = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            xb_back_test_account = stock_daily_result.xb_back_test_account
            daily_pnl += xb_back_test_account.total_profit - xb_back_test_account.yes_total_capital

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            xb_back_test_account = future_daily_result.xb_back_test_account
            daily_pnl += xb_back_test_account.holding_pnl + xb_back_test_account.realized_pnl - \
                xb_back_test_account.cost
        return daily_pnl

    @property
    def market_value(self):
        """投资组合当前的市场价值"""
        market_value = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            xb_back_test_account = stock_daily_result.xb_back_test_account
            market_value += xb_back_test_account.market_value

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            xb_back_test_account = future_daily_result.xb_back_test_account
            market_value += xb_back_test_account.market_value
        return market_value

    @property
    def total_value(self):
        """总权益"""
        total_profit = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            xb_back_test_account = stock_daily_result.xb_back_test_account
            total_profit += xb_back_test_account.total_profit

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            xb_back_test_account = future_daily_result.xb_back_test_account
            total_profit += xb_back_test_account.total_profit
        return total_profit

    @property
    def units(self):
        """份额"""
        start_capital = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            xb_back_test_account = stock_daily_result.xb_back_test_account
            start_capital += xb_back_test_account.start_capital

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            xb_back_test_account = future_daily_result.xb_back_test_account
            start_capital += xb_back_test_account.start_capital
        return start_capital

    @property
    def unit_net_value(self):
        """单位净值"""
        return self.total_value / self.units

    @property
    def static_unit_net_value(self):
        """静态单位权益"""
        yes_total_capital = 0
        start_capital = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            xb_back_test_account = stock_daily_result.xb_back_test_account
            yes_total_capital += xb_back_test_account.yes_total_capital
            start_capital += xb_back_test_account.start_capital

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            xb_back_test_account = future_daily_result.xb_back_test_account
            yes_total_capital += xb_back_test_account.yes_total_capital
            start_capital += xb_back_test_account.start_capital
        return yes_total_capital / self.units

    @property
    def transaction_cost(self):
        """当日费用"""
        cost = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            xb_back_test_account = stock_daily_result.xb_back_test_account
            cost += xb_back_test_account.cost

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            xb_back_test_account = future_daily_result.xb_back_test_account
            cost += xb_back_test_account.cost
        return cost

    @property
    def pnl(self):
        """当前投资组合的累计盈亏"""
        add_profit = 0
        for trade_reverse_reslut in self.strategy_context.all_trade_reverse_result.result_dict.values():
            stock_daily_result = trade_reverse_reslut.dailyResult
            xb_back_test_account = stock_daily_result.xb_back_test_account
            add_profit += xb_back_test_account.add_profit

        for future_reverse_reslut in self.strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_daily_result = future_reverse_reslut.dailyResult
            xb_back_test_account = future_daily_result.xb_back_test_account
            add_profit += xb_back_test_account.add_profit
        return add_profit

    @property
    def start_date(self):
        """策略投资组合的回测/实时模拟交易的开始日期"""
        return self.strategy_context.run_info.start_date

    @property
    def annualized_returns(self):
        """投资组合的年化收益率"""

        end_sec = time.mktime(time.strptime(str(self.strategy_context.now), '%Y%m%d'))
        natural_start_date = time.mktime(time.strptime(
            str(self.start_date), '%Y%m%d'))
        natural_work_days = int((end_sec - natural_start_date) / (24 * 60 * 60)) + 1

        annual_return_value = pow(
            1 + self.total_returns, 1 / (natural_work_days / 365)) - 1
        return annual_return_value
