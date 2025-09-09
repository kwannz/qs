#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:02
# @Author : wlb
# @File   : trade_reverse_result.py
# @desc   :
import math
import logging

from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, REJECTED, CANCELLED, ACTIVE, FILLED
from panda_backtest.backtest_common.constant.string_constant import FUND_DIVIDEND_INFO, FUND_ARRIVE_INFO, FUND_SPLIT_INFO
from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount
from panda_backtest.backtest_common.model.result.panda_backtest_position import PandaBacktestPosition
from panda_backtest.backtest_common.model.result.panda_backtest_profit import PandaBacktestProfit
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class BaseFundReverseResult(object):

    def __init__(self, account):
        self.account = account
        self.xb_back_test_account = None
        self.xb_back_test_profit = None
        self.xb_back_test_trade_dict = dict()
        self.xb_back_test_position_dict = dict()  # 持仓
        self.context = CoreContext.get_instance()

    def init_data(self):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        self.xb_back_test_account = XbBacktestAccount()
        self.xb_back_test_account.type = 3
        self.xb_back_test_account.back_id = run_info.run_id
        self.xb_back_test_account.account_id = run_info.fund_account
        self.xb_back_test_account.total_profit = run_info.fund_starting_cash
        self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.total_profit
        self.xb_back_test_account.start_capital = run_info.fund_starting_cash
        self.xb_back_test_account.available_funds = run_info.fund_starting_cash

        self.xb_back_test_profit = XbBacktestProfit()
        self.xb_back_test_profit.back_id = run_info.run_id
        self.xb_back_test_profit.account_id = run_info.stock_account

    def bar_start(self):
        pass

    def bar_end(self):
        pass

    def new_date(self):
        self.xb_back_test_trade_dict.clear()
        strategy_context = self.context.strategy_context
        # 更新市值
        self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.total_profit
        self.xb_back_test_account.gmt_create = strategy_context.trade_date

        self.xb_back_test_profit.day_profit = 0
        self.xb_back_test_profit.day_loss = 0
        self.xb_back_test_profit.day_purchase = 0
        self.xb_back_test_profit.day_put = 0
        self.xb_back_test_profit.gmt_create = strategy_context.trade_date
        self.xb_back_test_profit.gmt_create_time = strategy_context.hms

        for symbol, position in self.xb_back_test_position_dict.items():
            position.gmt_create = strategy_context.trade_date

    def end_date(self):
        # 统计数据
        pass

    def refresh_account(self):
        bar_dict = QuotationData.get_instance().bar_dict
        for symbol, xb_back_test_position in self.xb_back_test_position_dict.items():
            self.refresh_position(bar_dict[symbol])

    def refresh_position(self, bar_data):
        if bar_data is None or bar_data.last == 0:
            return
        if bar_data.symbol in self.xb_back_test_position_dict.keys():
            xb_back_test_position = self.xb_back_test_position_dict[bar_data.symbol]
            xb_back_test_position.last_price = bar_data.last
            old_market_value = xb_back_test_position.market_value
            xb_back_test_position.market_value = xb_back_test_position.last_price * \
                                                 xb_back_test_position.position
            xb_back_test_position.accumulate_profit = xb_back_test_position.position * \
                                                      (xb_back_test_position.last_price - xb_back_test_position.price)

            self.xb_back_test_account.market_value += xb_back_test_position.market_value - old_market_value
            self.xb_back_test_account.total_profit += xb_back_test_position.market_value - old_market_value
            self.xb_back_test_account.add_profit = self.xb_back_test_account.total_profit - \
                                                   self.xb_back_test_account.start_capital + \
                                                   self.xb_back_test_account.withdraw - \
                                                   self.xb_back_test_account.deposit

    def on_fund_rtn_order(self, order):
        if order.status == ACTIVE:
            if order.side == SIDE_BUY:
                # 冻结相关资金和手续费
                self.xb_back_test_account.available_funds -= order.purchase_amount
                self.xb_back_test_account.frozen_capital += order.purchase_amount
            else:
                # 申请赎回
                symbol_position = self.xb_back_test_position_dict[order.order_book_id]
                symbol_position.sellable = math.floor(
                    (symbol_position.sellable - order.quantity) * 10000) / 10000

        elif order.status == CANCELLED:
            if order.side == SIDE_BUY:
                self.xb_back_test_account.frozen_capital -= order.purchase_amount
                self.xb_back_test_account.available_funds += order.purchase_amount
            else:
                symbol_position = self.xb_back_test_position_dict[order.order_book_id]
                symbol_position.sellable = math.floor(
                    (symbol_position.sellable + order.quantity) * 10000) / 10000

        elif order.status == FILLED:
            # 将冻结资金正式扣除
            if order.side == SIDE_BUY:
                self.xb_back_test_account.frozen_capital -= order.purchase_amount
                self.xb_back_test_account.available_funds += order.purchase_amount
            else:
                # 资金到账
                # print('资金到账' + str(order.redeem_frozen_amount) + '合约：' + str(order.order_book_id) + '日期' + str(self.context.strategy_context.trade_date))
                self.xb_back_test_account.frozen_capital -= order.redeem_frozen_amount
                self.xb_back_test_account.available_funds += order.redeem_frozen_amount

                sr_logger = RemoteLogFactory.get_sr_logger()
                sr_logger.info(
                    FUND_ARRIVE_INFO % (str(self.account), order.order_book_id, str(order.redeem_frozen_amount)))

        elif order.status == REJECTED:
            pass

    def on_fund_rtn_trade(self, trade):
        if trade.account_id != self.account:
            return
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        bar_dict = QuotationData.get_instance().bar_dict

        self.xb_back_test_trade_dict[trade.trade_id] = trade

        if trade.business == SIDE_BUY:
            pos_change = trade.volume
            self.xb_back_test_profit.day_purchase += trade.price * trade.volume
            if trade.contract_code in self.xb_back_test_position_dict.keys():
                xb_back_test_position = self.xb_back_test_position_dict[trade.contract_code]
                xb_back_test_position.price = (
                                                      xb_back_test_position.price * xb_back_test_position.position + trade.price * trade.volume) / (
                                                      xb_back_test_position.position + trade.volume)
                xb_back_test_position.position = math.floor(
                    (xb_back_test_position.position + trade.volume) * 10000) / 10000
                xb_back_test_position.sellable = math.floor(
                    (xb_back_test_position.sellable + trade.volume) * 10000) / 10000
                xb_back_test_position.cost += trade.cost
                xb_back_test_position.gmt_create = strategy_context.trade_date
            else:
                xb_back_test_position = XbBacktestPosition()
                xb_back_test_position.back_id = run_info.run_id
                xb_back_test_position.type = 2
                xb_back_test_position.direction = SIDE_BUY
                xb_back_test_position.account_id = self.account
                xb_back_test_position.contract_code = trade.contract_code
                xb_back_test_position.contract_name = trade.contract_name
                xb_back_test_position.price = trade.price
                xb_back_test_position.position = trade.volume
                xb_back_test_position.sellable = trade.volume
                xb_back_test_position.cost = trade.cost
                xb_back_test_position.last_price = bar_dict[trade.contract_code].last
                xb_back_test_position.gmt_create = strategy_context.trade_date

                self.xb_back_test_position_dict[trade.contract_code] = xb_back_test_position

                # 订阅
                event_bus = self.context.event_bus
                event = Event(
                    ConstantEvent.SYSTEM_FUND_QUOTATION_START_SUB,
                    symbol_list=[trade.contract_code],
                    sub_type=0)
                event_bus.publish_event(event)
            # 持仓详情
            xb_back_test_position.position_detail_list.append((strategy_context.trade_date, trade.volume))

        else:
            pos_change = -trade.volume
            self.xb_back_test_profit.day_put += trade.price * trade.volume
            xb_back_test_position = self.xb_back_test_position_dict[trade.contract_code]

            xb_back_test_position.position += pos_change
            trade_volume = trade.volume
            while trade_volume > 0:
                fund_position_date_data = xb_back_test_position.position_detail_list.pop()
                if fund_position_date_data[1] > trade_volume:
                    new_tuple = (fund_position_date_data[0], fund_position_date_data[1] - trade_volume)
                    xb_back_test_position.position_detail_list.append(new_tuple)
                    trade_volume = 0
                else:
                    trade_volume -= fund_position_date_data[1]

            xb_back_test_position.cost += trade.cost
            if xb_back_test_position.position == 0:
                del self.xb_back_test_position_dict[trade.contract_code]
                # 取消合约的订阅
                event_bus = self.context.event_bus
                event = Event(
                    ConstantEvent.SYSTEM_FUND_QUOTATION_START_UN_SUB,
                    symbol_list=[xb_back_test_position.contract_code],
                    sub_type=0)
                event_bus.publish_event(event)

        xb_back_test_position.accumulate_profit = xb_back_test_position.position * \
                                                  (xb_back_test_position.last_price - xb_back_test_position.price)
        xb_back_test_position.market_value = xb_back_test_position.position * \
                                             xb_back_test_position.last_price

        # 个人账号信息
        if trade.business == SIDE_BUY:
            self.xb_back_test_account.available_funds -= trade.price * trade.volume + trade.cost
        else:
            # 冻结赎回资金
            self.xb_back_test_account.frozen_capital += trade.trade_amount
            # print('冻结赎回' + str(trade.trade_amount) + '合约' + str(trade.contract_code) + '到账'+str(trade.fund_arrive_date))
        self.xb_back_test_account.cost += trade.cost
        self.xb_back_test_account.market_value += pos_change * \
                                                  xb_back_test_position.last_price
        self.xb_back_test_account.total_profit = self.xb_back_test_account.available_funds + \
                                                 self.xb_back_test_account.market_value + self.xb_back_test_account.frozen_capital
        self.xb_back_test_account.add_profit = self.xb_back_test_account.total_profit - \
                                               self.xb_back_test_account.start_capital + \
                                               self.xb_back_test_account.withdraw - \
                                               self.xb_back_test_account.deposit

    def on_rtn_dividend(self, dividend):
        if dividend.symbol in self.xb_back_test_position_dict.keys():
            xb_back_test_position = self.xb_back_test_position_dict[dividend.symbol]
            cash_tax = float(dividend.fund_unit_ataxdev) * xb_back_test_position.position
            xb_back_test_position.price = (xb_back_test_position.price * xb_back_test_position.position - cash_tax) / \
                                          xb_back_test_position.position
            self.xb_back_test_account.available_funds += cash_tax
            self.xb_back_test_account.total_profit = self.xb_back_test_account.available_funds + \
                                                     self.xb_back_test_account.market_value + \
                                                     self.xb_back_test_account.frozen_capital
            sr_logger = RemoteLogFactory.get_sr_logger()
            sr_logger.info(FUND_DIVIDEND_INFO % (str(self.account), dividend.symbol, str(cash_tax)))

    def on_fund_rtn_split(self, fund_split):
        if fund_split.symbol in self.xb_back_test_position_dict.keys():
            xb_back_test_position = self.xb_back_test_position_dict[fund_split.symbol]
            old_position = xb_back_test_position.position
            old_price = xb_back_test_position.price
            xb_back_test_position.position = xb_back_test_position.position * fund_split.split_ratio
            xb_back_test_position.price = old_position * xb_back_test_position.price / xb_back_test_position.position
            sr_logger = RemoteLogFactory.get_sr_logger()
            sr_logger.info(FUND_SPLIT_INFO % (
                str(self.account), fund_split.symbol, str(old_position), str(xb_back_test_position.position),
                str(old_price), str(xb_back_test_position.price)))

    def move_cash(self, cash, move_type):
        sr_logger = RemoteLogFactory.get_sr_logger()
        if move_type == 1:
            self.xb_back_test_account.available_funds += cash
            self.xb_back_test_account.total_profit += cash
            self.xb_back_test_account.deposit += cash
            sr_logger.info('基金账号入金成功，账号：【%s】，入金金额：【%s】' % (str(self.account), str(cash)))
            return 1
        else:
            if cash > self.xb_back_test_account.available_funds:
                sr_logger.error('基金账号出金失败，账号:【%s】，出金金额：【%s】，可用资金：【%s】' %
                                (self.account, str(cash), str(self.xb_back_test_account.available_funds)))
                return 0
            self.xb_back_test_account.available_funds -= cash
            self.xb_back_test_account.total_profit -= cash
            self.xb_back_test_account.withdraw += cash
            sr_logger.info('基金账号出金成功，账号：【%s】，出金金额：【%s】' % (str(self.account), str(cash)))
            return 1
