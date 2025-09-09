#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-4-9 上午10:04
# @Author : wlb
# @File   : all_result.py
# @desc   :

import math
import logging

import json
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
from panda_backtest.extensions.trade_reverse_future.result.fund_reverse_result import FundReverseResult
from panda_backtest.extensions.trade_reverse_future.result.result_db import ResultDb
from panda_backtest.extensions.trade_reverse_future.result.standard_symbol_result import StandSymbolResult
from panda_backtest.backtest_common.result.base_all_result import BaseAllTradeReverseResult
from panda_backtest.extensions.trade_reverse_future.result.trade_reverse_result import TradeReverseResult
from panda_backtest.extensions.trade_reverse_future.result.future_reverse_result import FutureReverseResult
from panda_backtest.util.annotation.singleton_annotation import singleton
from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount
from panda_backtest.backtest_common.model.result.panda_backtest_profit import PandaBacktestProfit
from panda_backtest.extensions.common_api.index_calculate import IndexCalculate

class AllTradeReverseResult(BaseAllTradeReverseResult):

    def __init__(self):
        super().__init__()
        self.result_db = ResultDb()

    def init_data(self):
        super().add_standard_result(StandSymbolResult())
        super().init_data()

    def add_stock_account(self, account):
        self.stock_result_dict[account] = TradeReverseResult(account)

    def add_future_account(self, future_account):
        self.future_result_dict[future_account] = FutureReverseResult(future_account)

    def add_fund_account(self, fund_account):
        self.fund_result_dict[fund_account] = FundReverseResult(fund_account)

    def save_strategy_account_info(self):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        last_csi_profit = self.all_profit.csi_stock
        last_strategy_profit = self.all_profit.strategy_profit
        last_overful_profit = self.all_profit.overful_profit
        self.all_profit = PandaBacktestProfit()
        self.all_account = PandaBacktestAccount()
        self.all_account.back_id = run_info.run_id
        self.all_account.type = 2
        self.all_account.gmt_create = strategy_context.trade_date
        all_position_list = list()
        all_account_list = list()
        all_trade_list = list()

        for trade_reverse_result in self.stock_result_dict.values():
            all_position_list.extend(list(trade_reverse_result.xb_back_test_position_dict.values()))
            all_trade_list.extend(list(trade_reverse_result.xb_back_test_trade_dict.values()))
            all_account_list.append(trade_reverse_result.xb_back_test_account.__dict__)
            self.all_account.available_funds += trade_reverse_result.xb_back_test_account.available_funds
            self.all_account.total_profit += trade_reverse_result.xb_back_test_account.total_profit
            self.all_account.add_profit += trade_reverse_result.xb_back_test_account.add_profit
            self.all_account.market_value += trade_reverse_result.xb_back_test_account.market_value
            self.all_account.cost += trade_reverse_result.xb_back_test_account.cost
            self.all_account.margin += trade_reverse_result.xb_back_test_account.margin
            self.all_account.start_capital += trade_reverse_result.xb_back_test_account.start_capital
            self.all_account.yes_total_capital += trade_reverse_result.xb_back_test_account.yes_total_capital
            self.all_profit.day_purchase += trade_reverse_result.xb_back_test_profit.day_purchase
            self.all_profit.day_put += trade_reverse_result.xb_back_test_profit.day_put

        for future_reverse_result in self.future_result_dict.values():
            all_position_list.extend(list(future_reverse_result.long_position_dict.values()))
            all_position_list.extend(list(future_reverse_result.short_position_dict.values()))
            all_trade_list.extend(list(future_reverse_result.xb_back_test_trade_dict.values()))
            all_account_list.append(future_reverse_result.xb_back_test_account.__dict__)
            self.all_account.available_funds += future_reverse_result.xb_back_test_account.available_funds
            self.all_account.total_profit += future_reverse_result.xb_back_test_account.total_profit
            self.all_account.add_profit += future_reverse_result.xb_back_test_account.add_profit
            self.all_account.market_value += future_reverse_result.xb_back_test_account.market_value
            self.all_account.cost += future_reverse_result.xb_back_test_account.cost
            self.all_account.margin += future_reverse_result.xb_back_test_account.margin
            self.all_account.start_capital += future_reverse_result.xb_back_test_account.start_capital
            self.all_account.yes_total_capital += future_reverse_result.xb_back_test_account.yes_total_capital
            self.all_profit.day_purchase += future_reverse_result.xb_back_test_profit.day_purchase
            self.all_profit.day_put += future_reverse_result.xb_back_test_profit.day_put

        for fund_reverse_result in self.fund_result_dict.values():
            all_position_list.extend(list(fund_reverse_result.xb_back_test_position_dict.values()))
            all_trade_list.extend(list(fund_reverse_result.xb_back_test_trade_dict.values()))
            all_account_list.append(fund_reverse_result.xb_back_test_account.__dict__)
            self.all_account.available_funds += fund_reverse_result.xb_back_test_account.available_funds
            self.all_account.total_profit += fund_reverse_result.xb_back_test_account.total_profit
            self.all_account.add_profit += fund_reverse_result.xb_back_test_account.add_profit
            self.all_account.market_value += fund_reverse_result.xb_back_test_account.market_value
            self.all_account.cost += fund_reverse_result.xb_back_test_account.cost
            self.all_account.margin += fund_reverse_result.xb_back_test_account.margin
            self.all_account.start_capital += fund_reverse_result.xb_back_test_account.start_capital
            self.all_account.yes_total_capital += fund_reverse_result.xb_back_test_account.yes_total_capital
            self.all_profit.day_purchase += fund_reverse_result.xb_back_test_profit.day_purchase
            self.all_profit.day_put += fund_reverse_result.xb_back_test_profit.day_put
            fund_reverse_result.xb_back_test_trade_dict.clear()

        all_account_list.append(self.all_account.__dict__)
        self.all_account_value_list.append(self.all_account.total_profit)

        if self.all_account.yes_total_capital == 0:
            self.all_strategy_portfolio.append(0)
        else:
            self.all_strategy_portfolio.append(self.all_account.total_profit / self.all_account.yes_total_capital - 1)

        self.all_profit.gmt_create = strategy_context.trade_date
        self.all_profit.gmt_create_time = '153000'
        self.all_profit.back_id = run_info.run_id
        self.all_profit.csi_stock = self.standard_symbol_result.standard_symbol_value / self.standard_symbol_result.start_capital - 1
        self.all_profit.strategy_profit = self.all_account.add_profit / self.all_account.start_capital
        self.all_profit.day_profit = self.all_account.total_profit - self.all_account.yes_total_capital
        if last_strategy_profit + 1 == 0:
            today_strategy_profit = 0
        else:
            today_strategy_profit = ((self.all_profit.strategy_profit + 1) - (last_strategy_profit + 1)) / \
                                    (last_strategy_profit + 1)

        if last_csi_profit + 1 == 0:
            today_csi_profit = 0
        else:
            today_csi_profit = ((self.all_profit.csi_stock + 1) - (last_csi_profit + 1)) / \
                               (last_csi_profit + 1)
        self.all_profit.overful_profit = (today_strategy_profit - today_csi_profit + 1) * (last_overful_profit + 1) - 1

        self.result_db.save_daily_data_to_db(all_account_list, all_position_list, all_trade_list,
                                             self.all_profit.__dict__)

    def save_strategy_result(self):
        strategy_context = self.context.strategy_context
        # 策略年化收益
        ar = IndexCalculate.annual_return(self.all_profit.strategy_profit, strategy_context.trade_date_len)
        last_standard_profit = self.standard_symbol_result.standard_symbol_value / self.standard_symbol_result.start_capital - 1
        # 基准年化收益
        sr = IndexCalculate.new_annual_return(last_standard_profit, strategy_context.trade_date_len)
        # 最大回撤
        md = IndexCalculate.max_drawdown(
            strategy_context.trade_date_list, self.all_account_value_list)
        vol = IndexCalculate.volatility(self.all_strategy_portfolio)
        sharpe = IndexCalculate.sharpe_ratio(ar, 0.04, vol)
        dw = IndexCalculate.downside_risk(
            self.all_strategy_portfolio,
            self.standard_symbol_result.standard_portfolio)

        te = IndexCalculate.tracking_error(
            self.all_strategy_portfolio,
            self.standard_symbol_result.standard_portfolio)

        annual_te = IndexCalculate.annual_tracking_error(te)

        info_ration = IndexCalculate.info_ratio(
            self.all_strategy_portfolio, self.standard_symbol_result.standard_portfolio, annual_te)

        aer = IndexCalculate.avg_excess_return(
            self.all_strategy_portfolio,
            [0.04 / 250] * len(self.all_strategy_portfolio))
        sortino = IndexCalculate.sortino(ar, sr, dw)

        if vol and math.isnan(vol):
            vol = None

        if sharpe and math.isnan(sharpe):
            sharpe = None

        beta = IndexCalculate.new_beta(self.all_strategy_portfolio, self.standard_symbol_result.standard_portfolio)
        if beta and math.isnan(beta):
            beta = None
            alpha = None
        else:
            alpha = IndexCalculate.alpha(ar, 0.04, sr, beta)

        kama_ratio = IndexCalculate.kama_ratio(ar, md)
        self.result_db.save_result_to_db(self.all_profit.strategy_profit, ar,
                                         last_standard_profit, sr, alpha, beta, sharpe, vol,
                                         md, info_ration, sortino, annual_te, kama_ratio, dw,
                                         self.standard_symbol_result.symbol_name)

    def draw(self, data):
        sr_logger = RemoteLogFactory.get_sr_logger()
        if type(data) != list:
            sr_logger.info("绘图数据类型必须为list")
            return
        try:
            res = json.dumps(data)
            self.result_db.save_draw(res)
        except Exception:
            sr_logger.info("绘图数据有误，请检查格式")
