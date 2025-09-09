#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:15
# @Author : wlb
# @File   : reverse_trade_api.py
# @desc   :
import json
import time

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, CANCEL_ORDER_FAILED_MESSAGE
from panda_backtest.backtest_common.data.order.real_time.future_work_order_list import FutureWorkOrderList
from panda_backtest.backtest_common.model.result.panda_backtest_trade import PandaBacktestTrade as XbBacktestTrade
from panda_backtest.backtest_common.order.common.order_risk_control_verify import OrderRiskControlVerify

from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.model.result.order import Order, FILLED, REJECTED, ACTIVE
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from common.connector.mongodb_handler import DatabaseHandler as MongoClient
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
from panda_trading.trading.exchange.future_order_account_verify import FutureOrderAccountVerify
from panda_trading.trading.exchange.future_order_limit_price_verify import FutureOrderLimitPriceVerify
from panda_trading.trading.exchange.future_order_split_manager import FutureOrderSplitManager
from panda_backtest.backtest_common.exchange.future.real_time.future_order_quotation_verify import FutureOrderQuotationVerify
from panda_trading.trading.exchange.future_order_trade_time_verify import FutureOrderTradeTimeVerify
from panda_trading.trading.extensions.real_trade.trade.future_group_order import FutureGroupOrder
from panda_trading.trading.extensions.real_trade.trade.future_group_order_with_cancel import FutureGroupOrderWithCancel
from utils.annotation.singleton_annotation import singleton
from panda_trading.real_trade_api.ctp.ctp_trade_api import CTPTradeApi, SIDE_BUY, CLOSE
from panda_trading.real_trade_api.ctp.ctp_quotation_api import CtpQuotationApi

from panda_trading.trading.extensions.real_trade.trade.future_trade_adapter import FutureTradeAdapter
from utils.log.log_factory import LogFactory
from common.config.config import config

@singleton
class FutureTradeApi(object):

    def __init__(self):
        self.logger = LogFactory.get_logger()
        self.context = CoreContext.get_instance()
        self.quotation_mongo_db = MongoClient(config).get_mongo_db()
        self.mongo_db = MongoClient(config).get_mongo_db()
        self.future_order_split_manager = FutureOrderSplitManager()
        self.order_quotation_verify = FutureOrderQuotationVerify(self.quotation_mongo_db)
        self.work_order = FutureWorkOrderList(self.mongo_db.redefine_real_order, self.context)
        self.trade_adapter = FutureTradeAdapter(self.work_order)
        self.verify_list = list()
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        self._ctp_trade_api = CTPTradeApi(run_info.run_id, self.trade_adapter)
        self._ctp_quotation_api = CtpQuotationApi(run_info.run_id, self.trade_adapter)
        self.sub_quotation_type = None
        self._ctp_quotation_account = None
        self.account_list_dict = dict()
        self.group_order_dict = dict()

    def init_data(self):
        future_order_limit_price_verify = FutureOrderLimitPriceVerify(self.quotation_mongo_db)
        future_order_trade_time_verify = FutureOrderTradeTimeVerify()
        future_order_account_verify = FutureOrderAccountVerify()
        order_risk_control_verify = OrderRiskControlVerify()
        self.verify_list.append(future_order_trade_time_verify)
        self.verify_list.append(order_risk_control_verify)
        self.verify_list.append(future_order_limit_price_verify)
        self.verify_list.append(future_order_account_verify)

    def init_future_account(self, account_list):
        for account in account_list:
            account_id = account['account_id']
            trade_ip = account['trade_server_url']
            password = account['password']
            broker_id = account['broker_id']
            app_id = account['product_info']

            if account['auth_code'] is not None:
                ctp_auth_code = account['auth_code']
            else:
                ctp_auth_code = None

            # 初始化期货行情
            trade_adapter = account['trade_adapter']
            if trade_adapter == 0:
                self.init_ctp_account(trade_ip, account_id, password, broker_id, ctp_auth_code, app_id)
                if self._ctp_quotation_account is None:
                    self._ctp_quotation_account = account_id
                    self.sub_quotation_type = 0
                    quotation_ip = account['market_server_url']
                    self.init_ctp_quotation_account(quotation_ip, account_id, password, broker_id)
            else:
                sr_logger = RemoteLogFactory.get_sr_logger()
                sr_logger.error("期货账号配置失败，不支持配置交易适配器：%s" % str(trade_adapter))
                continue
            self.account_list_dict[account_id] = account

    def init_ctp_account(self, trade_ip, account_id, password, broker_id, ctp_auth_code, app_id):
        self._ctp_trade_api.init_account_ctp_qry_thread(account_id)
        self._ctp_trade_api.init_ctp_trade_api(
            trade_ip, None, account_id, password, broker_id, ctp_auth_code, app_id)
        self._ctp_trade_api.start_account_ctp_qry_thread(account_id)

    def init_ctp_quotation_account(self, quottion_ip, account_id, password, broker_id):
        self._ctp_quotation_api.init_ctp_quotation_api(
            quottion_ip, None, account_id, password, broker_id)
        self._ctp_quotation_api.init_account_ctp_quotation_qry_thread(account_id)

    def sub_symbol(self, symbol_list):
        result = ','.join(symbol_list)

        if self._ctp_quotation_api is not None:
            self._ctp_quotation_api.sub_symbol(symbol_list)

    def init_account_info(self):
        """
        初始化查询期货资金和持仓
        :param account:
        :return:
        """
        for account in self.account_list_dict.values():
            account_id = account['account_id']
            trade_adapter = account['trade_adapter']
            if trade_adapter == 0:
                self._ctp_trade_api.init_ctp_account_info(account_id)

    def account_logout(self):
        for account in self.account_list_dict.values():
            account_id = account['account_id']
            trade_adapter = account['trade_adapter']
            if trade_adapter == 0:
                self.ctp_account_logout(account_id)

        if self.sub_quotation_type == 0:
            self.ctp_quotation_logout(self._ctp_quotation_account)

        self.work_order.clear_order()

    def ctp_account_logout(self, future_account):
        self._ctp_trade_api.account_logout(future_account)

    def ctp_quotation_logout(self, future_account):
        self._ctp_quotation_api.account_logout(future_account)

    def reset_account_login(self):
        for account in self.account_list_dict.values():
            account_id = account['account_id']
            trade_adapter = account['trade_adapter']
            if trade_adapter == 0:
                self.ctp_reset_account_login(account_id)

        if self.sub_quotation_type == 0:
            self.ctp_quotation_reset_account_login(self._ctp_quotation_account)

    def ctp_reset_account_login(self, future_account):
        self._ctp_trade_api.resume_login(future_account)

    def ctp_quotation_reset_account_login(self, future_account):
        self._ctp_quotation_api.resume_login(future_account)

    def ctp_query_account(self, future_account):
        self._ctp_trade_api.query_account(future_account)

    def ctp_query_position(self, future_account):
        self._ctp_trade_api.query_positions(future_account)

    def insert_order(self, account_id, order_dict):
        strategy_context = self.context.strategy_context
        event_bus = self.context.event_bus
        run_info = strategy_context.run_info
        order_result = Order()
        order_result.status = ACTIVE
        order_result.order_id = '0'
        order_result.order_book_id = order_dict['symbol']
        order_result.client_id = run_info.run_id
        order_result.datetime = str(strategy_context.trade_time.strftime('%Y-%m-%d %H:%M:%S'))
        order_result.date = strategy_context.now
        order_result.account = account_id
        order_result.quantity = int(order_dict['quantity'])
        order_result.unfilled_quantity = int(order_dict['quantity'])
        order_result.retry_num = order_dict.get('retry_num', 0)
        order_result.side = order_dict['side']
        order_result.effect = order_dict['effect']
        order_result.is_td_close = order_dict['is_td_close']
        order_result.market = order_dict['market']
        order_result.now_system_order = 1
        order_result.order_type = 1
        order_result.price_type = order_dict['price_type']
        order_result.price = order_dict['price']
        order_result.is_close_local = order_dict.get('is_close_local', 0)
        order_result.risk_control_id = order_dict.get('risk_control_id', 0)
        order_result.remark = order_dict.get('remark', '')
        self.logger.info('insert_order' + json.dumps(order_result.__dict__))

        order_result = self.order_quotation_verify.get_order_market_price(order_result)

        if order_result.status == REJECTED:
            self.log_order_error(order_result)
            event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, order=order_result)
            event_bus.publish_event(event)
            return [order_result]

        if order_result.effect == CLOSE:
            order_result_list = self.future_order_split_manager.split_close_today_order(order_result)
        else:
            order_result_list = [order_result]
        for order_result in order_result_list:
            for order_verify in self.verify_list:
                if not order_verify.can_submit_order(account_id, order_result):
                    order_result.status = REJECTED
                    self.log_order_error(order_result)
                    event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, order=order_result)
                    event_bus.publish_event(event)
                    break
            if order_result.status != REJECTED:
                account = self.account_list_dict[account_id]
                if account['trade_adapter'] == 0:
                    self._ctp_trade_api.insert_order(account_id, order_result)
        return order_result_list

    def cancel_future_order(self, order_id, account_id, risk_control_client=None):
        if account_id not in self.account_list_dict.keys():
            err_mes = CANCEL_ORDER_FAILED_MESSAGE % (
                account_id,
                '不存在当前期货账号')
            sr_logger = RemoteLogFactory.get_sr_logger()
            if risk_control_client is not None:
                risk_control_manager = self.context.risk_control_manager
                sr_logger.risk(risk_control_manager.get_risk_control_name(risk_control_client), err_mes)
            else:
                sr_logger.error(err_mes)
            return False

        res = False
        account = self.account_list_dict[account_id]
        if account['trade_adapter'] == 0:
            res = self._ctp_trade_api.cancel_order(account_id, order_id)

        return res

    def log_order_error(self, order_result):
        sr_logger = RemoteLogFactory.get_sr_logger()
        if order_result.now_system_order == 2:
            risk_control_manager = self.context.risk_control_manager
            sr_logger.risk(risk_control_manager.get_risk_control_name(order_result.risk_control_id),
                           order_result.message)
        else:
            sr_logger.error(order_result.message)

    def insert_group_order(self, account, long_symbol_dict, short_symbol_dict):
        if account in self.group_order_dict.keys():
            future_group_order = self.group_order_dict[account]
        else:
            future_group_order = FutureGroupOrder(account, self.work_order)
            # future_group_order = FutureGroupOrderWithCancel(account, self.work_order)
            self.group_order_dict[account] = future_group_order

        future_group_order.start_order(long_symbol_dict, short_symbol_dict)

    def on_future_order_cancel(self, order):
        if order.account in self.group_order_dict.keys():
            future_group_order = self.group_order_dict[order.account]
            future_group_order.on_future_order_cancel(order)

    def on_future_rtn_trade(self, trade):
        if trade.account_id in self.group_order_dict.keys():
            future_group_order = self.group_order_dict[trade.account_id]
            future_group_order.on_future_rtn_trade(trade)
