#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-9-17 下午4:27
# @Author : wlb
# @File   : fund_positions_item.py
# @desc   :
from panda_backtest.backtest_common.data.stock.base_stock_positions_item import BaseStockPositionsItmes
import logging

class FundPositionsItems(BaseStockPositionsItmes):
    def __init__(self, symbol, position_dict, strategy_context, account):
        self.symbol = symbol
        self.position_dict = position_dict
        self.strategy_context = strategy_context
        self.account = account

    @property
    def order_book_id(self):
        return self.symbol

    @property
    def pnl(self):
        pnl = 0
        if self.symbol in self.position_dict.keys():
            pnl = self.position_dict[self.symbol].accumulate_profit

        return pnl

    @property
    def quantity(self):
        quantity = 0
        if self.symbol in self.position_dict.keys():
            quantity = self.position_dict[self.symbol].position

        return quantity

    @property
    def sellable(self):
        sellable = 0
        if self.symbol in self.position_dict.keys():
            sellable = self.position_dict[self.symbol].sellable

        return sellable

    @property
    def market_value(self):
        market_value = 0
        if self.symbol in self.position_dict.keys():
            market_value = self.position_dict[self.symbol].market_value

        return market_value

    @property
    def value_percent(self):
        value_percent = 0
        if self.symbol in self.position_dict.keys():
            value_percent = self.position_dict[self.symbol].market_value / \
                            self.strategy_context.stock_account_dict[self.account].total_value

        return value_percent

    @property
    def avg_price(self):
        avg_price = 0
        if self.symbol in self.position_dict.keys():
            avg_price = self.position_dict[self.symbol].price

        return avg_price

    @property
    def position_detail_list(self):
        position_detail_list = list()
        if self.symbol in self.position_dict.keys():
            position_detail_list = self.position_dict[self.symbol].position_detail_list
        return position_detail_list
