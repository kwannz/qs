#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:15
# @Author : wlb
# @File   : reverse_trade_api.py
# @desc   :
from common.connector.mongodb_handler import DatabaseHandler
import logging

from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE
from panda_backtest.backtest_common.constant.string_constant import FUND_ORDER_FAILED_MESSAGE, ORDER_CANCEL_FAILED_MESSAGE
from panda_backtest.backtest_common.order.common.order_risk_control_verify import OrderRiskControlVerify
from panda_backtest.backtest_common.order.fund.common.fund_order_account_verify import FundOrderAccountVerify
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.util.annotation.singleton_annotation import singleton
from common.config.config import config
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class FundTradeApi(object):

    def __init__(self, fund_exchange):
        self.fund_exchange = fund_exchange
        self.context = CoreContext.get_instance()
        self.quotation_mongo_db = DatabaseHandler(config)

    def init_data(self):
        self.fund_exchange.init_data()
        self.fund_exchange.init_event()
        fund_order_account_verify = FundOrderAccountVerify()
        self.fund_exchange.add_order_verify(fund_order_account_verify)
        order_risk_control_verify = OrderRiskControlVerify()
        if self.context.strategy_context.enable_risk_control:
            self.fund_exchange.add_order_verify(order_risk_control_verify)

    def insert_order(self, account, order_dict):
        strategy_context = self.context.strategy_context
        if account not in strategy_context.fund_account_dict.keys():
            if order_dict['side'] == SIDE_BUY:
                order_side = '申购'
            else:
                order_side = '赎回'
            if order_dict['effect'] == CLOSE:
                order_effect = '平仓'
            else:
                order_effect = '开仓'

            sr_logger = RemoteLogFactory.get_sr_logger()
            err_mes = FUND_ORDER_FAILED_MESSAGE % (order_dict['symbol'], '--',
                                                   account, order_effect, order_side, str(-1),
                                                   '不存在当前基金账号')
            if order_dict.get('now_system_order', 1) == 2:
                risk_control_manager = self.context.risk_control_manager
                sr_logger.risk(risk_control_manager.get_risk_control_name(order_dict['risk_control_id']), err_mes)
            else:
                sr_logger.error(err_mes)
            return []
        return self.fund_exchange.insert_order(account, order_dict)

    def cancel_order(self, account, order_id):
        strategy_context = self.context.strategy_context
        if account not in strategy_context.stock_account_dict.keys():
            sr_logger = RemoteLogFactory.get_sr_logger()
            sr_logger.error(ORDER_CANCEL_FAILED_MESSAGE % (account, order_id, '不存在当前股票账号'))
            return None
        self.fund_exchange.cancel_order(account, order_id)
