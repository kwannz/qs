#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:15
# @Author : wlb
# @File   : reverse_trade_api.py
# @desc   :
from datetime import datetime
import logging

from panda_backtest.extensions.trade_reverse_future.trade.stock_group_order import StockGroupOrder
from panda_backtest.backtest_common.order.common.order_risk_control_verify import OrderRiskControlVerify
from panda_backtest.backtest_common.order.stock.back_test.stock_order_volume_verify import StockOrderVolumeVerify
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config
from panda_backtest.backtest_common.constant.string_constant import STOCK_ORDER_FAILED_MESSAGE, ORDER_CANCEL_FAILED_MESSAGE
from panda_backtest.backtest_common.order.stock.common.stock_order_account_verify import StockOrderAccountVerify
from panda_backtest.backtest_common.order.stock.back_test.stock_order_limit_price_verify import StockOrderLimitPriceVerify
# from panda_backtest.backtest_common.order.stock.back_test.stock_order_susp_verify import StockOrderSuSpVerify
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.util.annotation.singleton_annotation import singleton
from panda_backtest.backtest_common.constant.strategy_constant import *

class ReverseTradeApi(object):

    def __init__(self, stock_exchange):
        self.stock_exchange = stock_exchange
        self.context = CoreContext.get_instance()
        self.quotation_mongo_db = DatabaseHandler(config)
        self.group_order_dict = dict()

    def init_data(self):
        self.stock_exchange.init_event()
        stock_order_account_verify = StockOrderAccountVerify()
        # stock_order_susp_verify = StockOrderSuSpVerify(self.quotation_mongo_db)
        stock_order_volume_verify = StockOrderVolumeVerify()
        stock_order_limit_price_verify = StockOrderLimitPriceVerify()
        order_risk_control_verify = OrderRiskControlVerify()
        self.stock_exchange.add_order_verify(stock_order_volume_verify)
        self.stock_exchange.add_order_verify(stock_order_account_verify)
        self.stock_exchange.add_order_verify(stock_order_limit_price_verify)
        if self.context.strategy_context.enable_risk_control:
            self.stock_exchange.add_order_verify(order_risk_control_verify)

    def insert_order(self, account, order_dict):
        return self.stock_exchange.insert_order(account, order_dict)

    def cancel_order(self, account, order_id):
        strategy_context = self.context.strategy_context
        if account not in strategy_context.stock_account_dict.keys():
            sr_logger = RemoteLogFactory.get_sr_logger()
            sr_logger.error(ORDER_CANCEL_FAILED_MESSAGE % (account, order_id, '不存在当前股票账号'))
            return None
        self.stock_exchange.cancel_order(account, order_id)

    def insert_group_order(self, account, symbol_dict):
        if account in self.group_order_dict.keys():
            stock_group_order = self.group_order_dict[account]
        else:
            stock_group_order = StockGroupOrder(account)
            self.group_order_dict[account] = stock_group_order

        stock_group_order.start_order(symbol_dict)
