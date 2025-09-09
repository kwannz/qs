#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:15
# @Author : wlb
# @File   : reverse_trade_api.py
# @desc   :
from panda_backtest.extensions.trade_reverse_future.trade.future_group_order import FutureGroupOrder
import logging

from panda_backtest.backtest_common.order.common.order_risk_control_verify import OrderRiskControlVerify

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, ORDER_CANCEL_FAILED_MESSAGE
from panda_backtest.backtest_common.order.future.common.future_order_account_verify import FutureOrderAccountVerify
from panda_backtest.backtest_common.order.future.back_test.future_order_limit_price_verify import FutureOrderLimitPriceVerify
from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.util.annotation.singleton_annotation import singleton

from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config
from panda_backtest.backtest_common.constant.strategy_constant import *
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class FutureTradeApi(object):

    def __init__(self, future_exchange):
        self.future_exchange = future_exchange
        self.context = CoreContext.get_instance()
        self.quotation_mongo_db = DatabaseHandler(config)
        self.group_order_dict = dict()

    def init_data(self):
        self.future_exchange.init_event()
        future_order_limit_price_verify = FutureOrderLimitPriceVerify(self.quotation_mongo_db)
        future_order_account_verify = FutureOrderAccountVerify()
        self.future_exchange.add_order_verify(future_order_limit_price_verify)
        self.future_exchange.add_order_verify(future_order_account_verify)
        order_risk_control_verify = OrderRiskControlVerify()
        if self.context.strategy_context.enable_risk_control:
            self.future_exchange.add_order_verify(order_risk_control_verify)

    def insert_order(self, account, order_dict):
        sr_logger = RemoteLogFactory.get_sr_logger()
        strategy_context = self.context.strategy_context
        if account not in strategy_context.future_account_dict.keys():
            if order_dict['side'] == SIDE_BUY:
                order_side = '买入'
            else:
                order_side = '卖出'
            if order_dict['effect'] == CLOSE:
                order_effect = '平仓'
            else:
                order_effect = '开仓'

            err_mes = FUTURE_ORDER_FAILED_MESSAGE % (order_dict['symbol'], str(order_dict['quantity']),
                                                     account, order_effect, order_side, str(-1),
                                                     '不存在当前期货账号')

            if order_dict.get('now_system_order', 1) == 2:
                risk_control_manager = self.context.risk_control_manager
                sr_logger.risk(risk_control_manager.get_risk_control_name(order_dict['risk_control_id']), err_mes)
            else:
                sr_logger.error(err_mes)
            return []
        return self.future_exchange.insert_order(account, order_dict)

    def cancel_order(self, account, order_id):
        sr_logger = RemoteLogFactory.get_sr_logger()
        strategy_context = self.context.strategy_context
        if account not in strategy_context.stock_account_dict.keys():
            sr_logger.error(ORDER_CANCEL_FAILED_MESSAGE % (account, order_id, '不存在当前期货账号'))
            return None
        self.future_exchange.cancel_order(account, order_id)

    def insert_group_order(self, account, long_symbol_dict, short_symbol_dict):
        if account in self.group_order_dict.keys():
            future_group_order = self.group_order_dict[account]
        else:
            future_group_order = FutureGroupOrder(account)
            self.group_order_dict[account] = future_group_order

        future_group_order.start_order(long_symbol_dict, short_symbol_dict)
