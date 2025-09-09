import os
import logging

from panda_backtest import project_dir

from common.connector.mongodb_handler import DatabaseHandler
from panda_backtest.extensions.trade_reverse_future.trade.fund_trade_api import FundTradeApi
from panda_backtest.backtest_common.exception.error_exception import ErrorException
from panda_backtest.backtest_common.exception.strategy_exception_builder import StrategyExceptionBuilder
from panda_backtest.backtest_common.exchange.fund.back_test.fund_exchange import FundExchange
from panda_backtest.backtest_common.exchange.future.back_test.future_exchange import FutureExchange
from panda_backtest.backtest_common.exchange.common.back_test.quotation_subscribe import QuotationSubscribe
from panda_backtest.backtest_common.exchange.stock.back_test.stock_exchange import StockExchange

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.backtest_common.system.event.event import *
from panda_backtest.extensions.trade_reverse_future.result.all_result import AllTradeReverseResult
from panda_backtest.extensions.trade_reverse_future.trade.reverse_trade_api import ReverseTradeApi
from panda_backtest.extensions.trade_reverse_future.trade.future_trade_api import FutureTradeApi
from panda_backtest.server.tools import export_dataframe_to_file
import time
from common.config.config import config
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class ReverseOperationProxy(object):

    def __init__(self, context):
        self.all_trade_reverse_result = None
        self._context = context
        self.quotation_mongo_db = DatabaseHandler(config)
        self.quotation_subscribe = QuotationSubscribe()
        self.stock_exchange = StockExchange(self.quotation_mongo_db)
        self.trade_api = ReverseTradeApi(self.stock_exchange)

        cost_rate_file = os.path.join(project_dir,
                                      'cost_rate.json')
        self.future_exchange = FutureExchange(self.quotation_mongo_db, cost_rate_file)
        self.fund_exchange = FundExchange(self.quotation_mongo_db)
        self.future_trade_api = FutureTradeApi(self.future_exchange)
        self.fund_trade_api = FundTradeApi(self.fund_exchange)

    def init_data(self):
        strategy_context = self._context.strategy_context
        self.all_trade_reverse_result = AllTradeReverseResult()
        strategy_context.init_all_result(self.all_trade_reverse_result)
        self.init_stock_account()
        self.init_future_account()
        self.init_fund_account()
        self.all_trade_reverse_result.init_data()
        self.trade_api.init_data()
        self.future_trade_api.init_data()
        self.fund_trade_api.init_data()

    def init_event(self):
        event_bus = self._context.event_bus
        event_bus.register_handle(ConstantEvent.SYSTEM_NEW_DATE, self.new_date)
        event_bus.register_handle(ConstantEvent.SYSTEM_END_DATE, self.end_date)
        event_bus.register_handle(ConstantEvent.SYSTEM_DAY_START, self.day_start)
        event_bus.register_handle(ConstantEvent.SYSTEM_HANDLE_BAR, self.sys_handle_bar)

        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_RTN_ORDER, self.sys_stock_rtn_order)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_RTN_TRADE, self.sys_stock_rtn_trade)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_QUOTATION_CHANGE, self.sys_stock_quotation_change)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_ORDER_CANCEL, self.sys_stock_order_cancel)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_DIVIDEND, self.sys_stock_dividend)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_QUOTATION_START_SUB, self.sys_sub_stock)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_QUOTATION_START_UN_SUB, self.sys_un_sub_stock)
        event_bus.register_handle(ConstantEvent.SYSTEM_ETF_SPLIT, self.sys_etf_split)

        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, self.sys_future_rtn_order)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_RTN_TRADE, self.sys_future_rtn_trade)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_QUOTATION_CHANGE, self.sys_future_quotation_change)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, self.sys_future_order_cancel)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_BURNED, self.sys_future_burned)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_SETTLE, self.sys_future_settle)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_DELIVERY, self.sys_future_delivery)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_SUB, self.sys_sub_future)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_UN_SUB, self.sys_un_sub_future)

        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_RTN_ORDER, self.sys_fund_rtn_order)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_RTN_TRADE, self.sys_fund_rtn_trade)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_QUOTATION_CHANGE, self.sys_fund_quotation_change)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_ORDER_CANCEL, self.sys_fund_order_cancel)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_DIVIDEND, self.sys_fund_dividend)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_SPLIT, self.sys_fund_split)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_QUOTATION_START_SUB, self.sys_sub_fund)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_QUOTATION_START_UN_SUB, self.sys_un_sub_fund)

        event_bus.register_handle(ConstantEvent.SYSTEM_CALCULATE_RESULT, self.show_result)

    def init_stock_account(self):
        strategy_context = self._context.strategy_context
        run_info = strategy_context.run_info
        account_type = run_info.account_type
        if account_type != 1 and account_type != 3 and account_type != 5:
            if run_info.stock_account is None:
                stock_account_list = ['15032863']
            else:
                stock_account_list = [run_info.stock_account]

            for account in stock_account_list:
                self.all_trade_reverse_result.add_stock_account(account)
                strategy_context.add_stock_account(account)

    def init_future_account(self):
        strategy_context = self._context.strategy_context
        run_info = strategy_context.run_info
        account_type = run_info.account_type
        if account_type != 0 and account_type != 3 and account_type != 4:
            if run_info.future_account is None:
                future_account_list = ['5588']
            else:
                future_account_list = [run_info.future_account]
            for future_account in future_account_list:
                self.all_trade_reverse_result.add_future_account(future_account)
                strategy_context.add_future_account(future_account)

    def init_fund_account(self):
        strategy_context = self._context.strategy_context
        run_info = strategy_context.run_info
        account_type = run_info.account_type
        if account_type != 0 and account_type != 1 and account_type != 2:
            if run_info.fund_account is None:
                fund_account_list = ['2233']
            else:
                fund_account_list = [run_info.fund_account]
            for fund_account in fund_account_list:
                self.all_trade_reverse_result.add_fund_account(fund_account)
                strategy_context.add_fund_account(fund_account)

    def sys_handle_bar(self):
        strategy_context = self._context.strategy_context
        event_bus = self._context.event_bus
        bar_data = QuotationData.get_instance().bar_dict
        run_info = strategy_context.run_info

        if run_info.matching_type == 1:
            bar_data.change_last_field(0)
        self.quotation_subscribe.start_quotation_play(time_type=0)

        if strategy_context.enable_risk_control:
            event = Event(
                ConstantEvent.RISK_CONTROL_HANDLE_BAR,
                context=strategy_context,
                data=bar_data)
            event_bus.publish_event(event)

        try:
            day_start_time = time.time()
            event = Event(
                ConstantEvent.STRATEGY_HANDLE_BAR,
                context=strategy_context,
                data=bar_data)
            event_bus.publish_event(event)
            # print('STRATEGY_HANDLE_BAR耗时：===》' + str(time.time() - day_start_time))

        except Exception as e:
            raise ErrorException(StrategyExceptionBuilder.build_strategy_run_exception_msg(), '00001', None)

        if run_info.matching_type == 1:
            bar_data.change_last_field(1)
            self.quotation_subscribe.start_quotation_play(time_type=1)

    def new_date(self):
        strategy_context = self._context.strategy_context
        self.quotation_subscribe.init_cache_data()

        self.fund_exchange.new_date()

        self.quotation_subscribe.init_daily_data()

        self.all_trade_reverse_result.new_date()

        event_bus = self._context.event_bus

        if strategy_context.enable_risk_control:
            event = Event(
                ConstantEvent.RISK_CONTROL_TRADING_BEFORE,
                context=strategy_context)
            event_bus.publish_event(event)

        try:
            event = Event(
                ConstantEvent.STRATEGY_TRADING_BEFORE,
                context=strategy_context)
            event_bus.publish_event(event)
        except Exception as e:
            raise ErrorException(StrategyExceptionBuilder.build_strategy_run_exception_msg(), '00001', None)

    def end_date(self):
        self.stock_exchange.end_date()
        self.future_exchange.end_date()
        self.fund_exchange.end_date()
        self.all_trade_reverse_result.end_date()
        self.all_trade_reverse_result.save_strategy_account_info()
        event_bus = self._context.event_bus
        strategy_context = self._context.strategy_context
        day_start_time = time.time()
        if strategy_context.enable_risk_control:
            event = Event(
                ConstantEvent.RISK_CONTROL_TRADING_AFTER,
                context=strategy_context)
            event_bus.publish_event(event)
        # print('end_dateRISK_CONTROL_TRADING_AFTER：===》' + str(time.time() - day_start_time))

        try:
            event = Event(
                ConstantEvent.STRATEGY_TRADING_AFTER,
                context=strategy_context)
            event_bus.publish_event(event)
            # print('end_dateSTRATEGY_TRADING_AFTER：===》' + str(time.time() - day_start_time))

        except Exception as e:
            raise ErrorException(StrategyExceptionBuilder.build_strategy_run_exception_msg(), '00001', None)
        self.stock_exchange.etf_split()
        self.future_exchange.future_settle()
        self.fund_exchange.fund_split()
        self.fund_exchange.fund_dividend()

    def day_start(self):
        self.all_trade_reverse_result.day_start()
        strategy_context = self._context.strategy_context
        self.stock_exchange.day_start()
        self.fund_exchange.day_start()
        if strategy_context.is_trade_date():
            self.quotation_subscribe.start_fund_quotation_play()

        event_bus = self._context.event_bus

        if strategy_context.enable_risk_control:
            event = Event(
                ConstantEvent.RISK_CONTROL_DAY_BEFORE,
                context=strategy_context)
            event_bus.publish_event(event)

        try:
            event = Event(
                ConstantEvent.STRATEGY_DAY_BEFORE,
                context=strategy_context)
            event_bus.publish_event(event)

        except Exception as e:
            raise ErrorException(StrategyExceptionBuilder.build_strategy_run_exception_msg(), '00001', None)

    def sys_stock_rtn_order(self, order):
        self.all_trade_reverse_result.on_stock_rtn_order(order)

    def sys_stock_rtn_trade(self, trade):
        self.all_trade_reverse_result.on_stock_rtn_trade(trade)

    def sys_stock_quotation_change(self, bar_data):
        self.all_trade_reverse_result.refresh_stock_position(bar_data)
        strategy_context = self._context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 0:
            self.all_trade_reverse_result.refresh_standard_symbol(bar_data)

    def sys_stock_order_cancel(self, order):
        sr_logger = RemoteLogFactory.get_sr_logger()
        if order.now_system_order == 2:
            risk_control_manager = self._context.risk_control_manager
            sr_logger.risk(risk_control_manager.get_risk_control_name(order.risk_control_id),
                           order.message)
        else:
            sr_logger.error(order.message)
        bar_data = QuotationData.get_instance().bar_dict
        event_bus = self._context.event_bus
        strategy_context = self._context.strategy_context
        event = Event(
            ConstantEvent.STOCK_ORDER_CANCEL,
            context=strategy_context,
            order=order,
            bar_dict=bar_data)
        event_bus.publish_event(event)

    def sys_stock_dividend(self, dividend):
        self.all_trade_reverse_result.on_rtn_dividend(dividend)

    def sys_etf_split(self, etf_split):
        self.all_trade_reverse_result.on_etf_rtn_split(etf_split)

    def sys_fund_dividend(self, dividend):
        self.all_trade_reverse_result.on_fund_rtn_dividend(dividend)

    def sys_fund_split(self, fund_split):
        self.all_trade_reverse_result.on_fund_rtn_split(fund_split)

    def sys_future_rtn_order(self, order):
        self.all_trade_reverse_result.on_future_rtn_order(order)

    def sys_future_rtn_trade(self, trade):
        self.all_trade_reverse_result.on_future_rtn_trade(trade)

    def sys_future_quotation_change(self, bar_data):
        self.all_trade_reverse_result.refresh_future_position(bar_data)
        strategy_context = self._context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 1:
            self.all_trade_reverse_result.refresh_standard_symbol(bar_data)

    def sys_future_order_cancel(self, order):
        sr_logger = RemoteLogFactory.get_sr_logger()
        if order.now_system_order == 2:
            risk_control_manager = self._context.risk_control_manager
            sr_logger.risk(risk_control_manager.get_risk_control_name(order.risk_control_id),
                          order.message)
        else:
            sr_logger.error(order.message)
        bar_data = QuotationData.get_instance().bar_dict
        event_bus = self._context.event_bus
        strategy_context = self._context.strategy_context
        event = Event(
            ConstantEvent.FUTURE_ORDER_CANCEL,
            order=order,
            context=strategy_context,
            bar_dict=bar_data)
        event_bus.publish_event(event)

    def sys_future_burned(self, account):
        self.all_trade_reverse_result.on_future_burned(account)

    def sys_future_settle(self):
        self.all_trade_reverse_result.on_future_settle()

    def sys_future_delivery(self, future_symbol):
        self.all_trade_reverse_result.on_future_delivery(future_symbol)

    def sys_fund_rtn_order(self, order):
        self.all_trade_reverse_result.on_fund_rtn_order(order)

    def sys_fund_rtn_trade(self, trade):
        self.all_trade_reverse_result.on_fund_rtn_trade(trade)

    def sys_fund_quotation_change(self, bar_data):
        self.all_trade_reverse_result.refresh_fund_position(bar_data)
        strategy_context = self._context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 2:
            self.all_trade_reverse_result.refresh_standard_symbol(bar_data)

    def sys_fund_order_cancel(self, order):
        bar_data = QuotationData.get_instance().bar_dict
        event_bus = self._context.event_bus
        strategy_context = self._context.strategy_context
        event = Event(
            ConstantEvent.FUND_ORDER_CANCEL,
            context=strategy_context,
            order=order,
            bar_dict=bar_data)
        event_bus.publish_event(event)

    def sys_sub_stock(self, symbol_list, sub_type):
        strategy_context = self._context.strategy_context
        strategy_context.sub_stock_symbol(symbol_list, sub_type)

    def sys_un_sub_stock(self, symbol_list, sub_type):
        """
        :param symbol_list:
        :param sub_type: 1：取消订单订阅的合约
        :return:
        """
        strategy_context = self._context.strategy_context
        strategy_context.un_sub_stock_symbol(symbol_list, sub_type)

    def sys_sub_future(self, symbol_list, sub_type):
        strategy_context = self._context.strategy_context
        strategy_context.sub_future_symbol(symbol_list, sub_type)

    def sys_un_sub_future(self, symbol_list, sub_type):
        strategy_context = self._context.strategy_context
        strategy_context.un_sub_future_symbol(symbol_list, sub_type)

    def sys_sub_fund(self, symbol_list, sub_type):
        strategy_context = self._context.strategy_context
        strategy_context.sub_fund_symbol(symbol_list, sub_type)

    def sys_un_sub_fund(self, symbol_list, sub_type):
        strategy_context = self._context.strategy_context
        strategy_context.un_sub_fund_symbol(symbol_list, sub_type)

    def subscribe(self, symbol_list):
        pass

    def place_order(self, account_id, order_dict):
        """
        下单
        :param account_id:
        :param order_dict:
        :return:
        """
        return self.trade_api.insert_order(account_id, order_dict)

    def place_future_order(self, account_id, order_dict):
        """
        下单
        :param account_id:
        :param order:
        :return:
        """
        return self.future_trade_api.insert_order(account_id, order_dict)

    def place_fund_order(self, account_id, order):
        """
        基金下单
        :param account_id:
        :param order:
        :return:
        """
        return self.fund_trade_api.insert_order(account_id, order)

    def insert_future_group_order(self, account, long_symbol_dict, short_symbol_dict):
        self.future_trade_api.insert_group_order(account, long_symbol_dict, short_symbol_dict)

    def insert_stock_group_order(self, account, symbol_dict, price_type=0):
        self.trade_api.insert_group_order(account, symbol_dict)

    def cancel_order(self, account_id, order_id):
        """
        撤单
        :param account_id:
        :param order_id:
        :return:
        """
        self.trade_api.cancel_order(account_id, order_id)

    def cancel_future_order(self, account_id, order_id):
        pass

    def get_open_fund_orders(self):
        """
        获取所有未成交订单
        :return:
        """
        pass

    def pub_data(self, pub_key, json_data):
        sr_logger = RemoteLogFactory.get_sr_logger()
        sr_logger.error("回测不支持pub_data功能")

    def sub_data(self, sub_keys, call_back):
        sr_logger = RemoteLogFactory.get_sr_logger()
        sr_logger.error("回测不支持sub_data功能")

    def show_result(self):
        self.all_trade_reverse_result.save_strategy_result()

    def export_data_to_file(self, data_frame):
        strategy_context = self._context.strategy_context
        run_info = strategy_context.run_info
        date = time.strftime("%Y%m%d")
        save_time = time.strftime("%H%M%S")
        export_dataframe_to_file(
            data_frame,
            run_info.run_id,
            date,
            save_time,
            run_info.run_type)

    def cash_moving(self, from_account, to_account, cash, move_type):
        return self.all_trade_reverse_result.cash_moving(from_account, to_account, cash, move_type)

    def draw(self, data):
        self.all_trade_reverse_result.draw(data)
