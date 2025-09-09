#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/3/16 14:16
# @Author : wlb
# @File   : future_group_order.py
# @desc   :
import time
import logging

from panda_backtest.api.api import *
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class StockGroupOrder(object):

    def __init__(self, account):
        self.account = account
        self.context = CoreContext.get_instance()
        self.total_order_num = 0
        self.finish_order_num = 0
        self.order_queue_list = list()

    def init_data(self):
        pass

    def start_order(self, symbol_dict):
        sr_logger = RemoteLogFactory.get_sr_logger()
        # sr_logger.info('开始进行批量下单')
        self.handle_start_group_order(symbol_dict)

    def handle_start_group_order(self, symbol_dict):
        strategy_context = self.context.strategy_context
        stock_account = strategy_context.stock_account_dict[self.account]

        # 根据仓位计算下单
        positions = stock_account.positions

        # 本次无持仓的合约先平仓
        now_holding_symbol_set = set(positions.keys())
        for symbol in now_holding_symbol_set:
            order_position = positions[symbol]
            if symbol not in set(list(symbol_dict.keys())) or symbol_dict[symbol] == 0:
                if order_position.quantity > 0:
                    order = (symbol, CLOSE, SIDE_SELL, order_position.quantity, 0, 0)
                    self.order_queue_list.append(order)

        # 多头持仓改变
        for symbol, position in symbol_dict.items():
            if position == 0:
                continue

            target_position = position
            if symbol in positions.keys():
                order_position = positions[symbol]
                target_position = target_position - order_position.quantity

            if target_position == 0:
                continue

            if target_position > 0:
                order = (symbol, OPEN, SIDE_BUY, target_position, 0, 0)
                self.order_queue_list.append(order)
            else:
                order = (symbol, CLOSE, SIDE_SELL, -target_position, 0, 0)
                self.order_queue_list.append(order)

        if len(self.order_queue_list) > 0:
            self.handle_group_order()
        else:
            sr_logger = RemoteLogFactory.get_sr_logger()
            sr_logger.info("当前无仓位需要进行调整")

    def handle_group_order(self):
        for order in self.order_queue_list:
            symbol = order[0]
            effect = order[1]
            side = order[2]
            position = int(order[3])
            retry_num = order[4]
            order_time = order[5]
            # bar_dict = QuotationData.get_instance().bar_dict
            # bar = bar_dict[symbol]
            if time.time() >= order_time:
                if side == SIDE_BUY:
                    order_list = order_shares(self.account, symbol, position, retry_num=retry_num,
                                              remark='group_order')
                else:
                    order_list = order_shares(self.account, symbol, position * (-1), retry_num=retry_num,
                                              remark='group_order')
        self.order_queue_list.clear()
