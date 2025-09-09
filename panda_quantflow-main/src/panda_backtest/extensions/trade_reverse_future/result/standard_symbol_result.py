#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:02
# @Author : wlb
# @File   : standard_symbol_result.py
# @desc   : 基准合约处理
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
import logging

from panda_backtest.backtest_common.data.stock.stock_info_map import StockInfoMap
from panda_backtest.backtest_common.exchange.fund.fund_info_map import FundInfoMap
from panda_backtest.backtest_common.system.context.core_context import CoreContext

from common.connector.mongodb_handler import DatabaseHandler
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from common.config.config import config

class StandSymbolResult(object):

    def __init__(self):
        self.start_capital = 1000000  # 回测时的起始本金（默认100万）
        self.standard_symbol = '603081'  # 基准合约
        self.standard_symbol_start_value = None  # 基准合约开始价值
        self.standard_symbol_position = 0  # 基准合约仓位
        self.standard_symbol_cash = 0  # 基准合约余额
        self.standard_symbol_value = 0  # 基准合约余额

        # 每日统计结果
        self.standard_portfolio = list()

        self.yes_standard_net_value = None

        self.quotation_mongo_db =DatabaseHandler(config)  # mongodb客户端连接

        self.context = CoreContext.get_instance()
        self.symbol_name = '未知'

    def init_data(
            self):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        self.standard_symbol = run_info.benchmark
        if run_info.standard_type == 0:
            self.start_capital = run_info.stock_starting_cash
        else:
            self.start_capital = run_info.future_starting_cash
        self.standard_symbol_value = self.start_capital
        self.standard_symbol_cash = 0
        self.yes_standard_net_value = 1

        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 0:
            stock_info_map = StockInfoMap(self.quotation_mongo_db)
            instrument_info = stock_info_map[self.standard_symbol]
            stock_type = instrument_info['type']
            if stock_type == 1:
                strategy_context.run_info.standard_stock_type = 3
            elif stock_type == 2:
                strategy_context.run_info.standard_stock_type = 2
            else:
                strategy_context.run_info.standard_stock_type = 0
            self.symbol_name = instrument_info['name']
        elif run_info.standard_type == 1:
            fund_info_map = FundInfoMap(self.quotation_mongo_db)
            instrument_info = fund_info_map.get_fund_info(self.standard_symbol)
            self.symbol_name = instrument_info['fund_name']
        else:
            future_info_map = FutureInfoMap(self.quotation_mongo_db)
            instrument_info = future_info_map[self.standard_symbol]
            self.symbol_name = instrument_info['name']

    def new_date(self):
        pass

    def day_start(self):
        pass

    def refresh_position(self, bar_data):
        if bar_data.symbol != self.standard_symbol:
            return

        # 基准合约初始化价值
        if self.standard_symbol_start_value is None and bar_data.last != 0:
            self.standard_symbol_start_value = bar_data.last
            self.standard_symbol_position = self.start_capital / bar_data.last
            self.standard_symbol_cash = self.standard_symbol_value - self.standard_symbol_position * bar_data.last

    def end_date(self):
        bar_dict = QuotationData.get_instance().bar_dict
        if bar_dict[self.standard_symbol].last != 0:
            self.standard_symbol_value = self.standard_symbol_cash + \
                                         self.standard_symbol_position * bar_dict[self.standard_symbol].last

        self.standard_portfolio.append(
            self.standard_symbol_value / self.start_capital / self.yes_standard_net_value - 1)

        self.yes_standard_net_value = self.standard_symbol_value / self.start_capital

    def on_rtn_dividend(self, dividend):
        if dividend.symbol != self.standard_symbol:
            return

        self.standard_symbol_cash += dividend.unit_cash_div_tax * self.standard_symbol_position
        self.standard_symbol_position = int(
            self.standard_symbol_position +
            self.standard_symbol_position * (dividend.share_trans_ratio + dividend.share_ratio))
        bar_dict = QuotationData.get_instance().bar_dict
        if bar_dict[self.standard_symbol].last != 0:
            self.standard_symbol_value = bar_dict[self.standard_symbol].last * self.standard_symbol_position

    def on_rtn_fund_dividend(self, dividend):
        if dividend.symbol != self.standard_symbol:
            return
        self.standard_symbol_cash += float(dividend.fund_unit_ataxdev) * self.standard_symbol_position

    def on_fund_rtn_split(self, fund_split):
        if fund_split.symbol != self.standard_symbol:
            return
        self.standard_symbol_position = self.standard_symbol_position * fund_split.split_ratio

    def on_etf_rtn_split(self, etf_split):
        if etf_split.symbol != self.standard_symbol:
            return
        self.standard_symbol_position = self.standard_symbol_position * etf_split.divcvratio

