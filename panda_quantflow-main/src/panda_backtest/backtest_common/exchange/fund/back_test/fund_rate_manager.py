#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/8/31 16:10
# @Author : wlb
# @File   : fund_rate_manager.py
# @desc   :
import time
import logging

from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY
from panda_backtest.backtest_common.exchange.fund.fund_rate import FundRate
from panda_backtest.backtest_common.system.context.core_context import CoreContext

class FundRateManager(object):
    def __init__(self, quotation_mongo_db):
        self.context = CoreContext.get_instance()
        self.fund_rate = FundRate(quotation_mongo_db)

    def init_data(self):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        self.fund_rate.init_data(run_info.rate_dict_data_str)

    def clear_cache_data(self):
        self.fund_rate.clear_cache_data()

    def get_order_rate(self, order):
        strategy_context = self.context.strategy_context
        if order.side == SIDE_BUY:
            res = self.fund_rate.get_purchase_rate_by_symbol(order.order_book_id,
                                                             strategy_context.trade_date,
                                                             order.purchase_amount, order.fund_type)
            res = res * strategy_context.run_info.commission_multiplier
            return res
        else:
            fund_account = strategy_context.fund_account_dict[order.account]
            fund_account_positions = fund_account.positions
            position_detail_list = fund_account_positions[order.order_book_id].position_detail_list

            trade_rate = 0
            order_quantity = order.quantity
            deep_index = 0
            while order_quantity > 0:
                fund_position_date_data = position_detail_list[deep_index]
                if fund_position_date_data[1] > order_quantity:
                    fill_quantity = order_quantity
                    order_quantity = 0
                else:
                    fill_quantity = fund_position_date_data[1]
                    order_quantity -= fund_position_date_data[1]

                trade_rate += self.fund_rate.get_redeem_rate_by_symbol(order.order_book_id, strategy_context.trade_date,
                                                                       fill_quantity * order.price,
                                                                       strategy_context.get_date_distance(
                                                                           fund_position_date_data[0],
                                                                           strategy_context.trade_date),
                                                                       order.fund_type)
                deep_index = deep_index + 1

            trade_rate = trade_rate * strategy_context.run_info.commission_multiplier

            return trade_rate
