#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-16 下午3:17
# @Author : wlb
# @File   : positions.py
# @desc   :
from panda_backtest.backtest_common.data.future.back_test.future_positions_item import FuturePositionsItems
import logging

class FuturePositions(object):

    def __init__(self, strategy_context, account):
        self.strategy_context = strategy_context
        self.account = account

    def __getitem__(self, key):

        long_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).long_position_dict
        short_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).short_position_dict

        return FuturePositionsItems(key, long_pos_dict, short_pos_dict)

    def __contains__(self, key):
        """检查期货合约是否在持仓中"""
        return key in self.keys()

    def get(self, key, default=None):
        """获取指定键的值，如果键不存在则返回默认值"""
        if key in self:
            return self[key]
        return default

    def __iter__(self):
        """返回迭代器，迭代所有键"""
        return iter(self.keys())

    def __len__(self):
        """返回持仓字典的长度"""
        return len(self.keys())

    def items(self):

        long_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).long_position_dict
        short_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).short_position_dict

        all_pos_dict = dict()

        for key in long_pos_dict.keys():
            all_pos_dict[key] = FuturePositionsItems(key, long_pos_dict, short_pos_dict)

        for key in short_pos_dict.keys():
            all_pos_dict[key] = FuturePositionsItems(key, long_pos_dict, short_pos_dict)

        return all_pos_dict.items()

    def keys(self):

        long_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).long_position_dict
        short_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).short_position_dict
        keys = list()
        for key, item in long_pos_dict.items():
            if item.position > 0:
                keys.append(key)
        for key, item in short_pos_dict.items():
            if item.position > 0:
                keys.append(key)

        all_pos_keys = list(set(keys))
        return all_pos_keys

    def values(self):

        long_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).long_position_dict
        short_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).short_position_dict

        all_pos_dict = dict()

        for key in long_pos_dict.keys():
            all_pos_dict[key] = FuturePositionsItems(key, long_pos_dict, short_pos_dict)

        for key in short_pos_dict.keys():
            all_pos_dict[key] = FuturePositionsItems(key, long_pos_dict, short_pos_dict)

        return all_pos_dict.values()
    
    def all_buy_margin(self):
        all_buy_margin = 0
        long_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).long_position_dict

        for key, item in long_pos_dict.items():
            all_buy_margin += item.margin
            
        return all_buy_margin
            
    def all_sell_margin(self):
        all_sell_margin = 0

        short_pos_dict = self.strategy_context.all_trade_reverse_result.get_future_reverse_result(
            self.account).short_position_dict

        for key, item in short_pos_dict.items():
            all_sell_margin += item.margin

        return all_sell_margin
        
    def __str__(self):
        list = []
        for key in self.keys():
            list.append(key)
        return str(list)
