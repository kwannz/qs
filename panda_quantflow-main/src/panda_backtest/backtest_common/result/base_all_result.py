#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-4-9 上午10:04
# @Author : wlb
# @File   : all_result.py
# @desc   :
from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount
import logging

from panda_backtest.backtest_common.model.result.panda_backtest_profit import PandaBacktestProfit
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.util.annotation.singleton_annotation import singleton

class BaseAllTradeReverseResult(object):

    def __init__(self):
        self.stock_result_dict = dict()
        self.future_result_dict = dict()
        self.fund_result_dict = dict()
        self.standard_symbol_result = None
        self.context = CoreContext.get_instance()
        self.all_account = PandaBacktestAccount()
        self.all_profit = PandaBacktestProfit()
        self.all_account_value_list = list()
        # 保存每日策略当日收益率（今日收益/昨日总权益）
        self.all_strategy_portfolio = list()

    def add_stock_account(self, account):
        pass

    def add_future_account(self, future_account):
        pass

    def add_fund_account(self, fund_account):
        pass

    def add_standard_result(self, standard_symbol_result):
        self.standard_symbol_result = standard_symbol_result

    def init_data(self):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.init_data()

        for future_reverse_result in self.future_result_dict.values():
            future_reverse_result.init_data()

        for fund_reverse_result in self.fund_result_dict.values():
            fund_reverse_result.init_data()

        self.standard_symbol_result.init_data()

    def new_date(self):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.new_date()

        for future_reverse_result in self.future_result_dict.values():
            future_reverse_result.new_date()

        for fund_reverse_result in self.fund_result_dict.values():
            fund_reverse_result.new_date()

        self.standard_symbol_result.new_date()

    def day_start(self):
        self.standard_symbol_result.day_start()

    def end_date(self):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.end_date()

        for future_reverse_result in self.future_result_dict.values():
            future_reverse_result.end_date()

        for fund_reverse_result in self.fund_result_dict.values():
            fund_reverse_result.end_date()

        self.standard_symbol_result.end_date()

    def cash_moving(self, from_account, to_account, cash, move_type):
        # 0:股转期，1：期转股，2：股转基，3：基转股，4：期转基，5：基转期
        if move_type == 0:
            from_account_result = self.stock_result_dict[from_account]
            to_account_result = self.future_result_dict[to_account]
        elif move_type == 1:
            from_account_result = self.future_result_dict[from_account]
            to_account_result = self.stock_result_dict[to_account]
        elif move_type == 2:
            from_account_result = self.stock_result_dict[from_account]
            to_account_result = self.fund_result_dict[to_account]
        elif move_type == 3:
            from_account_result = self.fund_result_dict[from_account]
            to_account_result = self.stock_result_dict[to_account]
        elif move_type == 4:
            from_account_result = self.future_result_dict[from_account]
            to_account_result = self.fund_result_dict[to_account]
        else:
            from_account_result = self.fund_result_dict[from_account]
            to_account_result = self.future_result_dict[to_account]

        res = from_account_result.move_cash(cash, 0)
        if res:
            res = to_account_result.move_cash(cash, 1)
        return res

    def on_stock_rtn_order(self, order):
        trade_reverse_result = self.stock_result_dict[order.account]
        trade_reverse_result.on_stock_rtn_order(order)

    def on_stock_rtn_trade(self, trade):
        trade_reverse_result = self.stock_result_dict[trade.account_id]
        trade_reverse_result.on_stock_rtn_trade(trade)

    def refresh_stock_position(self, bar_data):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.refresh_position(bar_data)

    def on_future_rtn_order(self, order):
        future_reverse_result = self.future_result_dict[order.account]
        future_reverse_result.on_future_rtn_order(order)

    def on_future_rtn_trade(self, trade):
        future_reverse_result = self.future_result_dict[trade.account_id]
        future_reverse_result.on_future_rtn_trade(trade)

    def refresh_future_position(self, bar_data):
        for future_reverse_result in self.future_result_dict.values():
            future_reverse_result.refresh_position(bar_data)

    def on_fund_rtn_order(self, order):
        fund_reverse_result = self.fund_result_dict[order.account]
        fund_reverse_result.on_fund_rtn_order(order)

    def on_fund_rtn_trade(self, trade):
        fund_reverse_result = self.fund_result_dict[trade.account_id]
        fund_reverse_result.on_fund_rtn_trade(trade)

    def refresh_fund_position(self, bar_data):
        for fund_reverse_result in self.fund_result_dict.values():
            fund_reverse_result.refresh_position(bar_data)

    def refresh_standard_symbol(self, bar_data):
        self.standard_symbol_result.refresh_position(bar_data)

    def get_trade_reverse_result(self, account):
        return self.stock_result_dict[account]

    def get_future_reverse_result(self, account):
        return self.future_result_dict[account]

    def get_fund_reverse_result(self, account):
        return self.fund_result_dict[account]

    def on_rtn_dividend(self, dividend):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.on_rtn_dividend(dividend)
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 0:
            self.standard_symbol_result.on_rtn_dividend(dividend)

    def on_etf_rtn_split(self, fund_split):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.on_etf_rtn_split(fund_split)
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 0:
            self.standard_symbol_result.on_etf_rtn_split(fund_split)

    def on_fund_rtn_dividend(self, dividend):
        for fund_reverse_result in self.fund_result_dict.values():
            fund_reverse_result.on_rtn_dividend(dividend)
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 2:
            self.standard_symbol_result.on_rtn_fund_dividend(dividend)

    def on_fund_rtn_split(self, fund_split):
        for fund_reverse_result in self.fund_result_dict.values():
            fund_reverse_result.on_fund_rtn_split(fund_split)
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 2:
            self.standard_symbol_result.on_fund_rtn_split(fund_split)

    def on_future_burned(self, account):
        if account not in self.future_result_dict.keys():
            return
        future_reverse_result = self.future_result_dict[account]
        future_reverse_result.future_burned()

    def on_future_settle(self):
        for future_reverse_result in self.future_result_dict.values():
            future_reverse_result.future_symbol_settle()

    def on_future_delivery(self, future_symbol):
        for future_reverse_result in self.future_result_dict.values():
            future_reverse_result.future_symbol_delivery(future_symbol)
