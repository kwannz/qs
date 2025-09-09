#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:02
# @Author : wlb
# @File   : trade_reverse_result.py
# @desc   :
import json
import logging

import math
from collections import OrderedDict, defaultdict
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
from common.connector.redis_client import RedisClient
from panda_backtest.backtest_common.data.future.future_margin_map import FutureMarginMap
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from panda_backtest.backtest_common.util.date_util import DateUtil
from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, ACTIVE, CANCELLED, FILLED, REJECTED, OPEN, \
    PartTradedQueueing, PartTradedNotQueueing
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.model.result.panda_backtest_position import PandaBacktestPosition
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.model.result.panda_backtest_profit import PandaBacktestProfit
from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount
from panda_backtest.backtest_common.system.context.core_context import CoreContext

class _MarginRateCacheMixin:
    """Lightweight cache that mimics the subset of Redis features we used."""

    _margin_rate_cache: dict[str, float] = {}

    @staticmethod
    def _cache_key(trade_date: int | str, symbol: str) -> str:
        return f"margin_rate_{trade_date}_{symbol}"

    # --- Public helpers used by business logic -----------------------------------

    def _cache_set(self, trade_date: int | str, symbol: str, value: float) -> None:
        """Store *value* under the composed cache key."""
        self._margin_rate_cache[self._cache_key(trade_date, symbol)] = value

    def _cache_get(self, trade_date: int | str, symbol: str):
        """Retrieve a value or ``None`` if it does not exist."""
        return self._margin_rate_cache.get(self._cache_key(trade_date, symbol))

    def _cache_delete(self, trade_date: int | str, symbol: str) -> None:
        """Remove a cached entry if it exists (no‑op otherwise)."""
        self._margin_rate_cache.pop(self._cache_key(trade_date, symbol), None)
class BaseFutureReverseResult(_MarginRateCacheMixin):

    def __init__(self, account, quotation_mongo_db):
        self.account = account
        self.xb_back_test_account = None
        self.xb_back_test_profit = None
        self.long_position_dict = dict()
        self.short_position_dict = dict()
        self.xb_back_test_trade_dict = dict()
        self.context = CoreContext.get_instance()
        self.future_info_map = FutureInfoMap(quotation_mongo_db)
        self.future_margin_map = FutureMarginMap()
        self.today_long_symbol_list = defaultdict(list)
        self.today_short_symbol_list = defaultdict(list)

    def init_data(self):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        self.xb_back_test_account = PandaBacktestAccount()
        self.xb_back_test_account.type = 1
        self.xb_back_test_account.back_id = run_info.run_id
        self.xb_back_test_account.account_id = self.account
        self.xb_back_test_account.total_profit = run_info.future_starting_cash
        self.xb_back_test_account.start_capital = run_info.future_starting_cash
        self.xb_back_test_account.available_funds = run_info.future_starting_cash
        self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.total_profit
        self.xb_back_test_account.no_settle_total_capital = self.xb_back_test_account.total_profit

        self.xb_back_test_profit = PandaBacktestProfit()
        self.xb_back_test_profit.back_id = run_info.run_id
        self.xb_back_test_profit.account_id = run_info.stock_account

    def new_date(self):

        self.today_long_symbol_list.clear()
        self.today_short_symbol_list.clear()
        self.xb_back_test_trade_dict.clear()

        strategy_context = self.context.strategy_context

        # self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.total_profit
        self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.no_settle_total_capital
        self.xb_back_test_account.gmt_create = strategy_context.trade_date
        self.xb_back_test_account.today_withdraw = 0
        self.xb_back_test_account.today_deposit = 0

        self.xb_back_test_profit.day_profit = 0
        self.xb_back_test_profit.day_loss = 0
        self.xb_back_test_profit.day_purchase = 0
        self.xb_back_test_profit.day_put = 0
        self.xb_back_test_profit.gmt_create = strategy_context.trade_date
        self.xb_back_test_profit.gmt_create_time = strategy_context.hms

        for symbol, position in self.long_position_dict.items():
            position.sellable = position.position
            position.td_position = 0
            position.gmt_create = strategy_context.trade_date

        for symbol, position in self.short_position_dict.items():
            position.sellable = position.position
            position.td_position = 0
            position.gmt_create = strategy_context.trade_date

    def end_date(self):
        pass

    def refresh_account(self):
        strategy_context = self.context.strategy_context
        if not strategy_context.is_future_trade():
            return
        bar_dict = QuotationData.get_instance().bar_dict
        for symbol, xb_back_test_position in self.long_position_dict.items():
            self.refresh_position(bar_dict[symbol])

        for symbol, xb_back_test_position in self.short_position_dict.items():
            self.refresh_position(bar_dict[symbol])

    def refresh_position(self, bar_data):
        if bar_data.close == 0:
            return
        strategy_context = self.context.strategy_context
        trade_date = int(strategy_context.trade_date)
        if bar_data.symbol in self.long_position_dict.keys():
            xb_back_test_position = self.long_position_dict[bar_data.symbol]
            xb_back_test_position.last_price = bar_data.last
            xb_back_test_position.accumulate_profit = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.last_price - xb_back_test_position.price)
            old_pnl = xb_back_test_position.holding_pnl
            xb_back_test_position.holding_pnl = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.last_price - xb_back_test_position.hold_price)

            xb_back_test_position.settlement = bar_data.settlement
            old_market_value = xb_back_test_position.market_value
            xb_back_test_position.market_value = \
                xb_back_test_position.last_price * xb_back_test_position.position * xb_back_test_position.round_lot

            self.xb_back_test_account.market_value += xb_back_test_position.market_value - old_market_value
            self.xb_back_test_account.holding_pnl += xb_back_test_position.holding_pnl - old_pnl
            self.xb_back_test_account.total_profit = \
                self.xb_back_test_account.available_funds + self.xb_back_test_account.holding_pnl + \
                self.xb_back_test_account.frozen_capital + self.xb_back_test_account.margin
            self.xb_back_test_account.add_profit = \
                self.xb_back_test_account.total_profit - self.xb_back_test_account.start_capital + \
                self.xb_back_test_account.withdraw - \
                self.xb_back_test_account.deposit
            self.xb_back_test_account.daily_pnl = self.xb_back_test_account.total_profit - self.xb_back_test_account.yes_total_capital + \
                                                  self.xb_back_test_account.today_withdraw - \
                                                  self.xb_back_test_account.today_deposit
        if bar_data.symbol in self.short_position_dict.keys():
            xb_back_test_position = self.short_position_dict[bar_data.symbol]
            xb_back_test_position.last_price = bar_data.last
            xb_back_test_position.accumulate_profit = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.price - xb_back_test_position.last_price)
            old_pnl = xb_back_test_position.holding_pnl
            xb_back_test_position.holding_pnl = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.hold_price - xb_back_test_position.last_price)

            xb_back_test_position.settlement = bar_data.settlement
            old_market_value = xb_back_test_position.market_value
            xb_back_test_position.market_value = \
                xb_back_test_position.last_price * xb_back_test_position.position * xb_back_test_position.round_lot

            self.xb_back_test_account.market_value += xb_back_test_position.market_value - old_market_value
            self.xb_back_test_account.holding_pnl += xb_back_test_position.holding_pnl - old_pnl
            self.xb_back_test_account.total_profit = \
                self.xb_back_test_account.available_funds + self.xb_back_test_account.holding_pnl + \
                self.xb_back_test_account.frozen_capital + self.xb_back_test_account.margin
            self.xb_back_test_account.add_profit = \
                self.xb_back_test_account.total_profit - self.xb_back_test_account.start_capital + \
                self.xb_back_test_account.withdraw - \
                self.xb_back_test_account.deposit
            self.xb_back_test_account.daily_pnl = self.xb_back_test_account.total_profit - self.xb_back_test_account.yes_total_capital + \
                                                  self.xb_back_test_account.today_withdraw - \
                                                  self.xb_back_test_account.today_deposit

        print(
            "trade[%s],total_profit[%s],holding_pnl[%s],yes_total_capital[%s],margin[%s],available_funds[%s],frozen_capital【%s】" % (
                trade_date, self.xb_back_test_account.total_profit,
                self.xb_back_test_account.holding_pnl, self.xb_back_test_account.yes_total_capital,
                self.xb_back_test_account.margin, self.xb_back_test_account.available_funds,
                self.xb_back_test_account.frozen_capital))

    def future_symbol_settle(self):
        sr_logger = RemoteLogFactory.get_sr_logger()
        # sr_logger.info('进行期货结算')
        self.xb_back_test_account.no_settle_total_capital = self.xb_back_test_account.total_profit
        sell_margin = 0
        buy_margin = 0
        all_holding_pnl = 0
        capital_change = 0
        margin_wanting_rate = 0
        buy_margin_wanting = 0
        sell_margin_wanting = 0
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        trade_date = int(strategy_context.trade_date)
        pre_trade_date = DateUtil.get_pre_date(trade_date)
        # ypq
        # order_dict_change = self.context.strategy_context.order_dict_change
        # 结算仓位
        for symbol, position in self.long_position_dict.items():
            # 避免模拟盘获取不到结算价
            if position.settlement >= 10000000:
                position.settlement = position.last_price

            margin_info = self.future_margin_map.get_future_margin_info(symbol, trade_date)
            if margin_info['name'] == '未知':
                instrument_info = self.future_info_map[symbol]
                margin_rate = instrument_info['ftfirsttransmargin'] / 100 * run_info.margin_multiplier
            else:
                margin_rate = margin_info['long_margin'] / 100
                if margin_rate == 0:
                    margin_rate = margin_info['margin'] / 100
            self._cache_set(trade_date, symbol, margin_rate)
            pre_margin_rate = self._cache_get(pre_trade_date, symbol)
            self._cache_delete(pre_trade_date, symbol)
            # 获取上一交易日的保证金比率
            # 抵消掉保证金增长
            # self.redis_client.set('margin_rate_' + str(trade_date) + "_" + symbol, margin_rate, int(120))
            # pre_margin_rate = self.redis_client.get('margin_rate_' + str(pre_trade_date) + "_" + symbol)
            # self.redis_client.delete('margin_rate_' + str(pre_trade_date) + "_" + symbol)
            if pre_margin_rate is not None:
                pre_margin_rate = float(pre_margin_rate)
                if pre_margin_rate != margin_rate:
                    margin_wanting_rate = margin_rate - pre_margin_rate
                    buy_margin_wanting += position.hold_price * position.position * position.round_lot * margin_wanting_rate
                    print('保证金差异1', str(trade_date), symbol, margin_rate,
                          pre_margin_rate, buy_margin_wanting)

            capital_change += (position.settlement - position.hold_price) * \
                              position.position * position.round_lot * (1 - margin_rate)
            position.pre_settlement = position.settlement
            position.hold_price = position.settlement
            position.margin = position.hold_price * position.position * position.round_lot * margin_rate
            position.holding_pnl = 0
            position.realized_pnl = 0
            position.market_value = position.last_price * position.position * position.round_lot
            all_holding_pnl += position.holding_pnl
            buy_margin += position.margin
            position.last_price = position.settlement

        for symbol, position in self.short_position_dict.items():
            # 避免模拟盘获取不到结算价
            if position.settlement >= 10000000:
                position.settlement = position.last_price

            margin_info = self.future_margin_map.get_future_margin_info(symbol, trade_date)
            if margin_info['name'] == '未知':
                instrument_info = self.future_info_map[symbol]
                margin_rate = instrument_info['ftfirsttransmargin'] / 100 * run_info.margin_multiplier
            else:
                margin_rate = margin_info['short_margin'] / 100
                if margin_rate == 0:
                    margin_rate = margin_info['margin'] / 100

            # self.redis_client.set('margin_rate_' + str(trade_date) + "_" + symbol, margin_rate, 300)
            self._cache_set(trade_date, symbol, margin_rate)
            pre_margin_rate = self._cache_get(pre_trade_date, symbol)
            self._cache_delete(pre_trade_date, symbol)
            # 获取上一交易日的保证金比率
            # 抵消掉保证金增长
            # pre_margin_rate = self.redis_client.get('margin_rate_' + str(pre_trade_date) + "_" + symbol)
            # self.redis_client.delete('margin_rate_' + str(pre_trade_date) + "_" + symbol)
            if pre_margin_rate is not None:
                pre_margin_rate = float(pre_margin_rate)
                if pre_margin_rate != margin_rate:
                    margin_wanting_rate = margin_rate - pre_margin_rate
                    sell_margin_wanting += position.hold_price * position.position * position.round_lot * margin_wanting_rate
                    print('保证金差异3', str(trade_date), symbol, margin_rate, pre_margin_rate, sell_margin_wanting)

            capital_change += (position.hold_price - position.settlement) * position.position * \
                              position.round_lot * (1 + margin_rate)
            position.pre_settlement = position.settlement
            position.hold_price = position.settlement
            position.holding_pnl = 0
            position.margin = position.hold_price * position.position * position.round_lot * margin_rate
            position.realized_pnl = 0
            position.market_value = position.last_price * position.position * position.round_lot
            position.accumulate_profit = (
                                                 position.last_price - position.price) * position.position * position.round_lot
            all_holding_pnl += position.holding_pnl
            sell_margin += position.margin
            position.last_price = position.settlement

        print(
            "trade[%s],available_funds[%s],buy_margin_wanting[%s],sell_margin_wanting[%s],buy_margin[%s],sell_margin[%s],capital_change【%s】" % (
                trade_date, self.xb_back_test_account.available_funds,
                buy_margin_wanting, sell_margin_wanting, buy_margin, sell_margin, capital_change))
        self.xb_back_test_account.available_funds += capital_change - buy_margin_wanting - sell_margin_wanting
        self.xb_back_test_account.sell_margin = sell_margin
        self.xb_back_test_account.buy_margin = buy_margin
        self.xb_back_test_account.margin = buy_margin + sell_margin
        self.xb_back_test_account.holding_pnl = all_holding_pnl
        # 总权益=可用资金+当日持仓盈亏+冻结资金+保证金
        print(
            "trade[%s],available_funds[%s],holding_pnl[%s],frozen_capital[%s],margin[%s]" % (
                trade_date, self.xb_back_test_account.available_funds,
                self.xb_back_test_account.holding_pnl, self.xb_back_test_account.frozen_capital,
                self.xb_back_test_account.margin))
        self.xb_back_test_account.total_profit = self.xb_back_test_account.available_funds + \
                                                 self.xb_back_test_account.holding_pnl + \
                                                 self.xb_back_test_account.frozen_capital + self.xb_back_test_account.margin
        # 累计盈亏=总权益-当日盈亏+出金金额-入金金额
        self.xb_back_test_account.add_profit = self.xb_back_test_account.total_profit - \
                                               self.xb_back_test_account.start_capital + \
                                               self.xb_back_test_account.withdraw - \
                                               self.xb_back_test_account.deposit
        # 当日盈亏=总权益-yes_total_capital+
        self.xb_back_test_account.daily_pnl = self.xb_back_test_account.total_profit - self.xb_back_test_account.yes_total_capital + \
                                              self.xb_back_test_account.today_withdraw - \
                                              self.xb_back_test_account.today_deposit
        print(
            "trade[%s],available_funds[%s],buy_margin_wanting[%s],sell_margin_wanting[%s],buy_margin[%s],sell_margin[%s],capital_change【%s】" % (
                trade_date, self.xb_back_test_account.available_funds,
                buy_margin_wanting, sell_margin_wanting, buy_margin, sell_margin, capital_change))
        self.xb_back_test_account.realized_pnl = 0

    def future_symbol_delivery(self, future_symbol):
        sr_logger = RemoteLogFactory.get_sr_logger()
        if future_symbol in self.long_position_dict.keys():
            sr_logger.error('触发合约多头交割，合约为:' + str(future_symbol))
            xb_back_test_position = self.long_position_dict[future_symbol]
            self.xb_back_test_account.market_value -= xb_back_test_position.market_value
            self.xb_back_test_account.holding_pnl -= xb_back_test_position.holding_pnl
            self.xb_back_test_account.margin -= xb_back_test_position.margin
            self.xb_back_test_account.buy_margin -= xb_back_test_position.margin
            self.xb_back_test_account.available_funds += xb_back_test_position.holding_pnl + \
                                                         xb_back_test_position.margin
            del self.long_position_dict[future_symbol]

        if future_symbol in self.short_position_dict.keys():
            sr_logger.error('触发合约空头交割，合约为:' + str(future_symbol))
            xb_back_test_position = self.short_position_dict[future_symbol]
            self.xb_back_test_account.market_value -= xb_back_test_position.market_value
            self.xb_back_test_account.holding_pnl -= xb_back_test_position.holding_pnl
            self.xb_back_test_account.margin -= xb_back_test_position.margin
            self.xb_back_test_account.sell_margin -= xb_back_test_position.margin
            self.xb_back_test_account.available_funds += xb_back_test_position.holding_pnl + \
                                                         xb_back_test_position.margin
            del self.short_position_dict[future_symbol]

    def future_burned(self):
        sr_logger = RemoteLogFactory.get_sr_logger()
        sr_logger.error('触发强制平仓，当前总权益为:' + str(self.xb_back_test_account.total_profit))
        self.long_position_dict.clear()
        self.short_position_dict.clear()
        self.xb_back_test_account.available_funds = 0
        self.xb_back_test_account.total_profit = 0
        self.xb_back_test_account.margin = 0
        self.xb_back_test_account.buy_margin = 0
        self.xb_back_test_account.sell_margin = 0

    def on_future_rtn_order(self, order):
        if order.account != self.account:
            return
        if order.status == ACTIVE:
            self.xb_back_test_account.available_funds -= order.margin + order.transaction_cost
            self.xb_back_test_account.frozen_capital += order.margin + order.transaction_cost
            if order.effect == CLOSE:
                if order.side == SIDE_BUY:
                    self.short_position_dict[order.order_book_id].frozen_position += order.quantity
                    self.short_position_dict[order.order_book_id].frozen_td_position += order.close_td_pos
                else:
                    self.long_position_dict[order.order_book_id].frozen_position += order.quantity
                    self.long_position_dict[order.order_book_id].frozen_td_position += order.close_td_pos
        elif order.status == PartTradedNotQueueing or order.status == FILLED:
            self.xb_back_test_account.available_funds += order.margin / order.quantity * order.quantity \
                                                         + order.transaction_cost
            self.xb_back_test_account.frozen_capital -= order.margin / order.quantity * order.quantity + \
                                                        order.transaction_cost
            if order.effect == CLOSE:
                if order.side == SIDE_BUY:
                    self.short_position_dict[order.order_book_id].frozen_position -= order.quantity
                    self.short_position_dict[order.order_book_id].frozen_td_position -= order.close_td_pos
                else:
                    self.long_position_dict[order.order_book_id].frozen_position -= order.quantity
                    self.long_position_dict[order.order_book_id].frozen_td_position -= order.close_td_pos

        elif order.status == CANCELLED:
            self.xb_back_test_account.available_funds += order.margin * (order.unfilled_quantity / order.quantity) \
                                                         - order.transaction_cost * (
                                                                 order.unfilled_quantity / order.quantity)
            self.xb_back_test_account.frozen_capital -= order.margin * (order.unfilled_quantity / order.quantity) \
                                                        + order.transaction_cost * (
                                                                order.unfilled_quantity / order.quantity)
            if order.effect == CLOSE:
                if order.side == SIDE_BUY:
                    self.short_position_dict[order.order_book_id].frozen_position -= order.unfilled_quantity
                    self.short_position_dict[order.order_book_id].frozen_td_position -= order.unfilled_close_td_pos
                else:
                    self.long_position_dict[order.order_book_id].frozen_position -= order.unfilled_quantity
                    self.long_position_dict[order.order_book_id].frozen_td_position -= order.unfilled_close_td_pos

    def on_future_rtn_trade(self, trade):
        if trade.account_id != self.account:
            return

        self.xb_back_test_trade_dict[trade.trade_id] = trade
        bar_dict = QuotationData.get_instance().bar_dict
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if trade.direction == OPEN:  # 开仓

            if trade.business == SIDE_BUY:
                # 开仓做多
                today_symbol_list = self.today_long_symbol_list
                self.xb_back_test_profit.day_purchase += trade.price * trade.volume

                if trade.contract_code in self.long_position_dict.keys():
                    xb_back_test_position = self.long_position_dict[trade.contract_code]

                    xb_back_test_position.hold_price = (
                                                               xb_back_test_position.hold_price * xb_back_test_position.position + trade.price * trade.volume) / (
                                                               xb_back_test_position.position + trade.volume)

                    xb_back_test_position.price = (
                                                          xb_back_test_position.price * xb_back_test_position.position + trade.price * trade.volume) / (
                                                          xb_back_test_position.position + trade.volume)

                    xb_back_test_position.position += trade.volume
                    xb_back_test_position.td_position += trade.volume
                    xb_back_test_position.cost += trade.cost
                    xb_back_test_position.margin += trade.margin

                else:
                    xb_back_test_position = PandaBacktestPosition()
                    xb_back_test_position.back_id = run_info.run_id
                    xb_back_test_position.account_id = self.account
                    xb_back_test_position.contract_code = trade.contract_code
                    xb_back_test_position.contract_name = trade.contract_name
                    xb_back_test_position.type = 1
                    xb_back_test_position.direction = trade.business
                    xb_back_test_position.price = trade.price
                    xb_back_test_position.settlement = bar_dict[trade.contract_code].settlement
                    xb_back_test_position.hold_price = trade.price
                    xb_back_test_position.position = trade.volume
                    xb_back_test_position.td_position = trade.volume
                    xb_back_test_position.cost = trade.cost
                    xb_back_test_position.margin += trade.margin
                    xb_back_test_position.last_price = bar_dict[trade.contract_code].last
                    xb_back_test_position.gmt_create = strategy_context.trade_date
                    xb_back_test_position.round_lot = trade.round_lot
                    self.long_position_dict[trade.contract_code] = xb_back_test_position
                    event_bus = self.context.event_bus
                    event = Event(
                        ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_SUB,
                        symbol_list=[trade.contract_code],
                        sub_type=0)
                    event_bus.publish_event(event)

            else:  # 开仓做空

                today_symbol_list = self.today_short_symbol_list
                self.xb_back_test_profit.day_put += trade.price * trade.volume

                if trade.contract_code in self.short_position_dict.keys():
                    xb_back_test_position = self.short_position_dict[trade.contract_code]
                    xb_back_test_position.price = (
                                                          xb_back_test_position.price * xb_back_test_position.position + trade.price * trade.volume) / (
                                                          xb_back_test_position.position + trade.volume)

                    xb_back_test_position.hold_price = (
                                                               xb_back_test_position.hold_price * xb_back_test_position.position + trade.price * trade.volume) / (
                                                               xb_back_test_position.position + trade.volume)
                    xb_back_test_position.position += trade.volume
                    xb_back_test_position.td_position += trade.volume
                    xb_back_test_position.cost += trade.cost
                    xb_back_test_position.margin += trade.margin

                else:
                    xb_back_test_position = PandaBacktestPosition()
                    xb_back_test_position.back_id = run_info.run_id
                    xb_back_test_position.account_id = self.account
                    xb_back_test_position.contract_code = trade.contract_code
                    xb_back_test_position.contract_name = trade.contract_name
                    xb_back_test_position.type = 1
                    xb_back_test_position.direction = trade.business
                    xb_back_test_position.price = trade.price
                    xb_back_test_position.settlement = bar_dict[trade.contract_code].settlement
                    xb_back_test_position.hold_price = trade.price
                    xb_back_test_position.position = trade.volume
                    xb_back_test_position.td_position = trade.volume
                    xb_back_test_position.cost = trade.cost
                    xb_back_test_position.margin += trade.margin
                    xb_back_test_position.last_price = bar_dict[trade.contract_code].last
                    xb_back_test_position.gmt_create = strategy_context.trade_date
                    xb_back_test_position.round_lot = trade.round_lot
                    self.short_position_dict[trade.contract_code] = xb_back_test_position
                    event_bus = self.context.event_bus
                    event = Event(
                        ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_SUB,
                        symbol_list=[trade.contract_code],
                        sub_type=0)
                    event_bus.publish_event(event)

            today_symbol_list[trade.contract_code].insert(0, (trade.volume, trade.price))

            self.xb_back_test_account.margin += trade.margin

        else:  # 平仓
            symbol = trade.contract_code
            trade_date = int(strategy_context.trade_date)
            margin_info = self.future_margin_map.get_future_margin_info(symbol, trade_date)

            if margin_info['name'] == '未知':
                # print('平仓未获取到每日保证金，日期：【%s】，代码：【%s】' % (str(trade_date), str(symbol)))
                instrument_info = self.future_info_map[symbol]
                margin_rate = instrument_info['ftfirsttransmargin'] / 100 * run_info.margin_multiplier
            else:
                margin_rate = margin_info['long_margin'] / 100
                if margin_rate == 0:
                    margin_rate = margin_info['margin'] / 100
                # print('平仓每日保证金比例，日期：【%s】，代码：【%s】，保证金比例：【%s】,old_margin_rate:【%s】,差：【%s】' % (
                #     str(trade_date), str(symbol), str(margin_rate),old_margin_rate,margin_rate-old_margin_rate))
            # self.redis_client.setRedis('margin_rate_' + str(trade_date) + "_" + symbol, margin_rate, 120)
            self._cache_set(trade_date, symbol, margin_rate)
            if trade.business == SIDE_BUY:
                # 平空仓
                self.xb_back_test_profit.day_purchase += trade.price
                xb_back_test_position = self.short_position_dict[trade.contract_code]
                today_symbol_list = self.today_short_symbol_list

            else:
                # 平多仓
                self.xb_back_test_profit.day_put += trade.price
                xb_back_test_position = self.long_position_dict[trade.contract_code]
                today_symbol_list = self.today_long_symbol_list

            left_pos = trade.volume
            if trade.is_td_close == 0:
                # 普通平仓
                if xb_back_test_position.position - xb_back_test_position.td_position > 0:
                    left_pos = trade.volume - (xb_back_test_position.position - xb_back_test_position.td_position)
                    if xb_back_test_position.position - trade.volume == 0:
                        xb_back_test_position.hold_price = 0
                    else:
                        xb_back_test_position.hold_price = (xb_back_test_position.hold_price *
                                                            xb_back_test_position.position - trade.volume *
                                                            xb_back_test_position.pre_settlement) / \
                                                           (xb_back_test_position.position - trade.volume)

            while True:
                if left_pos <= 0:
                    break
                else:
                    symbol_list = today_symbol_list[trade.contract_code]
                    old_quantity, old_price = symbol_list.pop()
                    if old_quantity > left_pos:
                        consumed_quantity = left_pos
                        symbol_list.append((old_quantity - left_pos, old_price))
                        left_pos = 0
                    else:
                        consumed_quantity = old_quantity
                        left_pos -= consumed_quantity
                    if consumed_quantity == xb_back_test_position.td_position:
                        xb_back_test_position.hold_price = 0
                    else:
                        if trade.is_td_close == 0:
                            xb_back_test_position.hold_price = (
                                                                       xb_back_test_position.hold_price * xb_back_test_position.td_position -
                                                                       consumed_quantity * old_price) / (
                                                                       xb_back_test_position.td_position - consumed_quantity)
                        else:
                            xb_back_test_position.hold_price = (
                                                                       xb_back_test_position.hold_price * xb_back_test_position.position -
                                                                       consumed_quantity * old_price) / (
                                                                       xb_back_test_position.position - consumed_quantity)
                    xb_back_test_position.td_position -= consumed_quantity

            xb_back_test_position.position -= trade.volume
            xb_back_test_position.cost += trade.cost
            old_margin = xb_back_test_position.margin
            xb_back_test_position.margin = xb_back_test_position.position * xb_back_test_position.round_lot * \
                                           xb_back_test_position.hold_price * margin_rate
            self.xb_back_test_account.margin += xb_back_test_position.margin - old_margin
            self.xb_back_test_account.available_funds += old_margin - xb_back_test_position.margin

        old_market_value = xb_back_test_position.market_value
        old_holding_pnl = xb_back_test_position.holding_pnl
        xb_back_test_position.market_value = xb_back_test_position.position * \
                                             xb_back_test_position.last_price * xb_back_test_position.round_lot

        if xb_back_test_position.direction == SIDE_BUY:
            xb_back_test_position.accumulate_profit = xb_back_test_position.position * \
                                                      xb_back_test_position.round_lot * (
                                                              xb_back_test_position.last_price - xb_back_test_position.price)
            xb_back_test_position.holding_pnl = xb_back_test_position.position * xb_back_test_position.round_lot * \
                                                (
                                                        xb_back_test_position.last_price - xb_back_test_position.hold_price)
            self.xb_back_test_account.market_value += xb_back_test_position.market_value - old_market_value
        else:
            xb_back_test_position.accumulate_profit = xb_back_test_position.position * \
                                                      xb_back_test_position.round_lot * (
                                                              xb_back_test_position.price - xb_back_test_position.last_price)
            xb_back_test_position.holding_pnl = xb_back_test_position.position * xb_back_test_position.round_lot * \
                                                (
                                                        xb_back_test_position.hold_price - xb_back_test_position.last_price)
            self.xb_back_test_account.market_value += (xb_back_test_position.market_value - old_market_value) * (-1)

        if trade.direction == CLOSE:
            # 计算真实成交价与最新价差价，累加到平仓盈亏上
            cj = (trade.price - xb_back_test_position.last_price) * trade.volume * xb_back_test_position.round_lot
            if trade.business == SIDE_BUY:
                cj = cj * (-1)
            xb_back_test_position.realized_pnl += old_holding_pnl - xb_back_test_position.holding_pnl + cj
            self.xb_back_test_account.realized_pnl += xb_back_test_position.realized_pnl
            self.xb_back_test_account.available_funds += old_holding_pnl - xb_back_test_position.holding_pnl + cj

        # 个人账号信息
        self.xb_back_test_account.cost += trade.cost
        self.xb_back_test_account.available_funds -= trade.margin + trade.cost
        self.xb_back_test_account.holding_pnl += xb_back_test_position.holding_pnl - old_holding_pnl
        self.xb_back_test_account.total_profit = self.xb_back_test_account.available_funds + \
                                                 self.xb_back_test_account.holding_pnl + \
                                                 self.xb_back_test_account.frozen_capital + self.xb_back_test_account.margin
        self.xb_back_test_account.add_profit = self.xb_back_test_account.total_profit - \
                                               self.xb_back_test_account.start_capital + \
                                               self.xb_back_test_account.withdraw - \
                                               self.xb_back_test_account.deposit
        self.xb_back_test_account.daily_pnl = self.xb_back_test_account.total_profit - self.xb_back_test_account.yes_total_capital + \
                                              self.xb_back_test_account.today_withdraw - \
                                              self.xb_back_test_account.today_deposit

        if xb_back_test_position.position == 0:
            if xb_back_test_position.direction == SIDE_BUY:
                del self.long_position_dict[xb_back_test_position.contract_code]
            else:
                del self.short_position_dict[xb_back_test_position.contract_code]

            if xb_back_test_position.contract_code not in self.long_position_dict.keys() and \
                    xb_back_test_position.contract_code not in self.short_position_dict.keys():
                # 取消合约的订阅
                event_bus = self.context.event_bus
                event = Event(
                    ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_UN_SUB,
                    symbol_list=[xb_back_test_position.contract_code],
                    sub_type=0)
                event_bus.publish_event(event)

    def move_cash(self, cash, move_type):
        sr_logger = RemoteLogFactory.get_sr_logger()
        if move_type == 1:
            self.xb_back_test_account.available_funds += cash
            self.xb_back_test_account.total_profit += cash
            self.xb_back_test_account.deposit += cash
            self.xb_back_test_account.today_deposit += cash
            sr_logger.info('期货账号入金成功，账号：【%s】，入金金额：【%s】' % (str(self.account), str(cash)))
            return 1
        else:

            if cash > self.xb_back_test_account.available_funds:
                sr_logger.error('期货账号出金失败，账号:【%s】，出金金额：【%s】，可用资金：【%s】' %
                                (self.account, str(cash), str(self.xb_back_test_account.available_funds)))
                return 0

            self.xb_back_test_account.available_funds -= cash
            self.xb_back_test_account.total_profit -= cash
            self.xb_back_test_account.withdraw += cash
            self.xb_back_test_account.today_withdraw += cash
            sr_logger.info('期货账号出金成功，账号：【%s】，出金金额：【%s】' % (str(self.account), str(cash)))
            return 1
