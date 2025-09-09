#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:02
# @Author : wlb
# @File   : future_positions_item.py
# @desc   : 上下文对象期货账号持仓item
from panda_backtest.backtest_common.data.future.base_future_positions_item import BaseFuturePositionsItmes
import logging

class FuturePositionsItems(BaseFuturePositionsItmes):
    def __init__(self, symbol, long_position_dict, short_position_dict,
                 strategy_long_position_dict=None, strategy_short_position_dict=None):
        self.symbol = symbol
        self.long_position_dict = long_position_dict
        self.short_position_dict = short_position_dict
        self.strategy_long_position_dict = strategy_long_position_dict
        self.strategy_short_position_dict = strategy_short_position_dict

    @property
    def order_book_id(self):
        return self.symbol

    @property
    def pnl(self):
        pnl = 0
        if self.symbol in self.long_position_dict.keys():
            pnl += self.long_position_dict[self.symbol].accumulate_profit

        if self.symbol in self.short_position_dict.keys():
            pnl += self.short_position_dict[self.symbol].accumulate_profit

        return pnl

    @property
    def daily_pnl(self):
        daily_pnl = 0
        if self.symbol in self.long_position_dict.keys():
            daily_pnl += self.long_position_dict[self.symbol].holding_pnl
            daily_pnl += self.long_position_dict[self.symbol].realized_pnl

        if self.symbol in self.short_position_dict.keys():
            daily_pnl += self.short_position_dict[self.symbol].holding_pnl
            daily_pnl += self.short_position_dict[self.symbol].realized_pnl

        return daily_pnl

    @property
    def holding_pnl(self):
        holding_pnl = 0
        if self.symbol in self.long_position_dict.keys():
            holding_pnl += self.long_position_dict[self.symbol].holding_pnl

        if self.symbol in self.short_position_dict.keys():
            holding_pnl += self.short_position_dict[self.symbol].holding_pnl

        return holding_pnl

    @property
    def realized_pnl(self):
        realized_pnl = 0
        if self.symbol in self.long_position_dict.keys():
            realized_pnl += self.long_position_dict[self.symbol].realized_pnl

        if self.symbol in self.short_position_dict.keys():
            realized_pnl += self.short_position_dict[self.symbol].realized_pnl

        return realized_pnl

    @property
    def transaction_cost(self):
        cost = 0
        if self.symbol in self.long_position_dict.keys():
            cost += self.long_position_dict[self.symbol].cost

        if self.symbol in self.short_position_dict.keys():
            cost += self.short_position_dict[self.symbol].cost

        return cost

    @property
    def margin(self):
        margin = 0
        if self.symbol in self.long_position_dict.keys():
            margin += self.long_position_dict[self.symbol].margin

        if self.symbol in self.short_position_dict.keys():
            margin += self.short_position_dict[self.symbol].margin

        return margin

    @property
    def market_value(self):
        market_value = 0
        if self.symbol in self.long_position_dict.keys():
            market_value += self.long_position_dict[self.symbol].market_value

        if self.symbol in self.short_position_dict.keys():
            market_value += self.short_position_dict[self.symbol].market_value * (-1)

        return market_value

    @property
    def buy_daily_pnl(self):
        buy_daily_pnl = 0
        if self.symbol in self.long_position_dict.keys():
            buy_daily_pnl += self.long_position_dict[self.symbol].holding_pnl
            buy_daily_pnl += self.long_position_dict[self.symbol].realized_pnl

        return buy_daily_pnl

    @property
    def buy_pnl(self):
        accumulate_profit = 0
        if self.symbol in self.long_position_dict.keys():
            accumulate_profit += self.long_position_dict[self.symbol].accumulate_profit

        return accumulate_profit

    @property
    def buy_transaction_cost(self):
        buy_transaction_cost = 0
        if self.symbol in self.long_position_dict.keys():
            buy_transaction_cost += self.long_position_dict[self.symbol].cost

        return buy_transaction_cost

    @property
    def closable_buy_quantity(self):
        closable_buy_quantity = 0
        if self.symbol in self.long_position_dict.keys():
            closable_buy_quantity += self.long_position_dict[self.symbol].position - \
                                     self.long_position_dict[self.symbol].frozen_position

        return closable_buy_quantity

    @property
    def closable_today_buy_quantity(self):
        closable_buy_quantity = 0
        if self.symbol in self.long_position_dict.keys():
            closable_buy_quantity += self.long_position_dict[self.symbol].td_position - \
                                     self.long_position_dict[self.symbol].frozen_td_position

        return closable_buy_quantity

    @property
    def buy_margin(self):
        buy_margin = 0
        if self.symbol in self.long_position_dict.keys():
            buy_margin += self.long_position_dict[self.symbol].margin

        return buy_margin

    @property
    def buy_today_quantity(self):
        buy_today_quantity = 0
        if self.symbol in self.long_position_dict.keys():
            buy_today_quantity += self.long_position_dict[self.symbol].td_position

        return buy_today_quantity

    @property
    def buy_quantity(self):
        buy_quantity = 0
        if self.symbol in self.long_position_dict.keys():
            buy_quantity += self.long_position_dict[self.symbol].position

        return int(buy_quantity)

    @property
    def buy_avg_open_price(self):
        buy_avg_open_price = 0
        if self.symbol in self.long_position_dict.keys():
            buy_avg_open_price += self.long_position_dict[self.symbol].price

        return buy_avg_open_price

    @property
    def buy_avg_holding_price(self):
        buy_avg_holding_price = 0
        if self.symbol in self.long_position_dict.keys():
            buy_avg_holding_price += self.long_position_dict[self.symbol].hold_price

        return buy_avg_holding_price

    @property
    def sell_daily_pnl(self):
        sell_daily_pnl = 0
        if self.symbol in self.short_position_dict.keys():
            sell_daily_pnl += self.short_position_dict[self.symbol].holding_pnl
            sell_daily_pnl += self.short_position_dict[self.symbol].realized_pnl

        return sell_daily_pnl

    @property
    def sell_pnl(self):
        accumulate_profit = 0
        if self.symbol in self.short_position_dict.keys():
            accumulate_profit += self.short_position_dict[self.symbol].accumulate_profit

        return accumulate_profit

    @property
    def sell_transaction_cost(self):
        sell_transaction_cost = 0
        if self.symbol in self.short_position_dict.keys():
            sell_transaction_cost += self.short_position_dict[self.symbol].cost

        return sell_transaction_cost

    @property
    def closable_sell_quantity(self):
        closable_sell_quantity = 0
        if self.symbol in self.short_position_dict.keys():
            closable_sell_quantity += self.short_position_dict[self.symbol].position - \
                                      self.short_position_dict[self.symbol].frozen_position

        return closable_sell_quantity

    @property
    def closable_today_sell_quantity(self):
        closable_sell_quantity = 0
        if self.symbol in self.short_position_dict.keys():
            closable_sell_quantity += self.short_position_dict[self.symbol].td_position - \
                                      self.short_position_dict[self.symbol].frozen_td_position

        return closable_sell_quantity

    @property
    def sell_margin(self):
        sell_margin = 0
        if self.symbol in self.short_position_dict.keys():
            sell_margin += self.short_position_dict[self.symbol].margin

        return sell_margin

    @property
    def sell_today_quantity(self):
        sell_today_quantity = 0
        if self.symbol in self.short_position_dict.keys():
            sell_today_quantity += self.short_position_dict[self.symbol].td_position

        return sell_today_quantity

    @property
    def sell_quantity(self):
        sell_quantity = 0
        if self.symbol in self.short_position_dict.keys():
            sell_quantity += self.short_position_dict[self.symbol].position

        return int(sell_quantity)

    @property
    def sell_avg_open_price(self):
        sell_avg_open_price = 0
        if self.symbol in self.short_position_dict.keys():
            sell_avg_open_price += self.short_position_dict[self.symbol].price

        return sell_avg_open_price

    @property
    def sell_avg_holding_price(self):
        sell_avg_holding_price = 0
        if self.symbol in self.short_position_dict.keys():
            sell_avg_holding_price += self.short_position_dict[self.symbol].hold_price

        return sell_avg_holding_price

    @property
    def strategy_closable_buy_quantity(self):
        closable_buy_quantity = 0
        if self.strategy_long_position_dict is not None and self.symbol in self.strategy_long_position_dict.keys():
            closable_buy_quantity += self.strategy_long_position_dict[self.symbol].position - \
                                     self.strategy_long_position_dict[self.symbol].frozen_position

        return closable_buy_quantity

    @property
    def strategy_closable_today_buy_quantity(self):
        closable_buy_quantity = 0
        if self.strategy_long_position_dict is not None and self.symbol in self.strategy_long_position_dict.keys():
            closable_buy_quantity += self.strategy_long_position_dict[self.symbol].td_position - \
                                     self.strategy_long_position_dict[self.symbol].frozen_td_position

        return closable_buy_quantity

    @property
    def strategy_buy_today_quantity(self):
        buy_today_quantity = 0
        if self.strategy_long_position_dict is not None and  self.symbol in self.strategy_long_position_dict.keys():
            buy_today_quantity += self.strategy_long_position_dict[self.symbol].td_position

        return buy_today_quantity

    @property
    def strategy_buy_quantity(self):
        buy_quantity = 0
        if self.strategy_long_position_dict is not None and self.symbol in self.strategy_long_position_dict.keys():
            buy_quantity += self.strategy_long_position_dict[self.symbol].position

        return buy_quantity

    @property
    def strategy_closable_sell_quantity(self):
        closable_sell_quantity = 0
        if self.strategy_short_position_dict is not None and self.symbol in self.strategy_short_position_dict.keys():
            closable_sell_quantity += self.strategy_short_position_dict[self.symbol].position - \
                                      self.strategy_short_position_dict[self.symbol].frozen_position

        return closable_sell_quantity

    @property
    def strategy_closable_today_sell_quantity(self):
        closable_sell_quantity = 0
        if self.strategy_short_position_dict is not None and self.symbol in self.strategy_short_position_dict.keys():
            closable_sell_quantity += self.strategy_short_position_dict[self.symbol].td_position - \
                                      self.strategy_short_position_dict[self.symbol].frozen_td_position

        return closable_sell_quantity

    @property
    def strategy_sell_today_quantity(self):
        sell_today_quantity = 0
        if self.strategy_short_position_dict is not None and self.symbol in self.strategy_short_position_dict.keys():
            sell_today_quantity += self.strategy_short_position_dict[self.symbol].td_position

        return sell_today_quantity

    @property
    def strategy_sell_quantity(self):
        sell_quantity = 0
        if self.strategy_short_position_dict is not None and self.symbol in self.strategy_short_position_dict.keys():
            sell_quantity += self.strategy_short_position_dict[self.symbol].position

        return sell_quantity