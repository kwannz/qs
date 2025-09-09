from panda_backtest.backtest_common.exception.error_exception import ErrorException
from panda_backtest.backtest_common.system.event.event import *
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from common.connector.mongodb_handler import DatabaseHandler as MongoClient
from common.connector.redis_client import RedisClient
from panda_backtest.system.panda_log import SRLogger
from panda_trading.trading.extensions.real_trade.trade.future_trade_api import FutureTradeApi
import time
from panda_trading.trading.extensions.real_trade.result.all_result import AllTradeReverseResult
from panda_trading.trading.restore.restore_strategy import RestoreStrategy
from panda_trading.trading.sub_pub.strategy_sub_pub import StrategySubPub
import common.config as config

class ReverseOperationProxy(object):

    def __init__(self, context):
        self.trade_api = None
        self.future_trade_api = None
        self.context = context
        self.event_bus = self.context.event_bus
        self.all_trade_reverse_result = AllTradeReverseResult()
        # self.mysql_client = MysqlClient.get_mysql_client()
        self.mongo_client = MongoClient(config).get_mongo_db()
        self.stock_quotation_account = None
        self.redis_client = RedisClient()
        self.strategy_sub_pub = StrategySubPub()
        self.strategy_sub_flag = False

    def init_data(self):
        strategy_context = self.context.strategy_context
        self.future_trade_api = FutureTradeApi()
        self.all_trade_reverse_result = AllTradeReverseResult()
        strategy_context.init_all_result(self.all_trade_reverse_result)
        self.init_stock_account()
        self.init_future_account()
        self.all_trade_reverse_result.init_data()
        self.future_trade_api.init_data()
        strategy_context.init_future_trade_time_dict()

    def init_event(self):
        event_bus = self.context.event_bus
        event_bus.register_handle(ConstantEvent.SYSTEM_NEW_DATE, self.new_date)
        event_bus.register_handle(ConstantEvent.SYSTEM_END_DATE, self.end_date)
        event_bus.register_handle(ConstantEvent.SYSTEM_DAY_START, self.day_start)
        event_bus.register_handle(ConstantEvent.SYSTEM_NIGHT_END, self.night_end)
        event_bus.register_handle(ConstantEvent.SYSTEM_HANDLE_BAR, self.sys_handle_bar)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_RTN_ORDER, self.sys_stock_rtn_order)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_RTN_TRADE, self.sys_stock_rtn_trade)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_RTN_TRANSFER, self.sys_stock_rtn_transfer)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_QUOTATION_CHANGE, self.sys_stock_quotation_change)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_ORDER_CANCEL, self.sys_stock_order_cancel)

        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_ALL_POSITION_REFRESH, self.sys_stock_all_pos_refresh)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_TRADE_POSITION_REFRESH, self.sys_stock_trade_pos_refresh)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_ASSET_REFRESH, self.sys_stock_asset_refresh)
        event_bus.register_handle(ConstantEvent.SYSTEM_STOCK_QUOTATION_START_SUB, self.sys_sub_stock_symbol)

        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, self.sys_future_rtn_order)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_RTN_TRADE, self.sys_future_rtn_trade)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_QUOTATION_CHANGE, self.sys_future_quotation_change)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, self.sys_future_order_cancel)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_RTN_TRANSFER, self.sys_future_rtn_transfer)

        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_ALL_POSITION_REFRESH, self.sys_future_all_pos_refresh)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_TRADE_POSITION_REFRESH, self.sys_future_trade_pos_refresh)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_ASSET_REFRESH, self.sys_future_asset_refresh)
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_SUB, self.sys_sub_future_symbol)

        event_bus.register_handle(ConstantEvent.SYSTEM_DAILY_DATA_SAVE, self.sys_daily_data_save)

        event_bus.register_handle(ConstantEvent.SYSTEM_RESTORE_STRATEGY, self.sys_restore_strategy)

    def init_stock_account(self):
        """
        初始化股票账号
        :param account_list:
        :param mock_id:
        :return:
        """
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        account_type = run_info.account_type
        if account_type != 1 and account_type != 3 and account_type != 5:
            if run_info.stock_account is None:
                stock_account_list = ['15032863']
            else:
                stock_account_list = [run_info.stock_account]

            account_list_cur = self.mongo_client.real_stock_account_info.find(
                {'account_id': {'$in': stock_account_list}})
            account_list = list(account_list_cur)
            if len(account_list) == 0:
                account_list = list()
            for account in account_list:
                account_id = account['account_id']
                self.all_trade_reverse_result.add_account(account_id)
                self.context.strategy_context.add_account(account_id)
            self.trade_api.init_stock_account(account_list)

    def init_future_account(self):
        """
        初始化账号
        :param account_list:
        :param mock_id:
        :return:
        """
        sub_market_flag = True
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        account_type = run_info.account_type
        if account_type != 0 and account_type != 3 and account_type != 5:
            if run_info.future_account is None:
                future_account_list = ['5588']
            else:
                future_account_list = [run_info.future_account]

            account_list_cur = self.mongo_client.real_future_account_info.find(
                {'account_id': {'$in': future_account_list}})
            future_account_list = list(account_list_cur)
            if future_account_list is False or future_account_list is None:
                future_account_list = []
            for account in future_account_list:
                account_id = account['account_id']

                self.all_trade_reverse_result.add_future_account(account_id)
                self.context.strategy_context.add_future_account(account_id)

                self.future_trade_api.init_future_account(future_account_list)

    def init_account_info(self):
        self.trade_api.init_account_info()

    def init_future_account_info(self):
        self.future_trade_api.init_account_info()

    def sys_handle_bar(self):
        quotation_data = QuotationData.get_instance()
        data = quotation_data.bar_dict
        try:
            event = Event(
                ConstantEvent.STRATEGY_HANDLE_BAR,
                context=self.context.strategy_context,
                data=data)
            event_bus = self.context.event_bus
            event_bus.publish_event(event)

        except ErrorException as e:
            SRLogger.error(e.message)

    def new_date(self):
        # 期货账号重新登录
        self.all_trade_reverse_result.new_date()
        strategy_context = self.context.strategy_context
        strategy_context.sub_stock_symbol_list = strategy_context.sub_strategy_stock_symbol_list
        strategy_context.sub_future_symbol_list = strategy_context.sub_strategy_future_symbol_list

        self.context.strategy_context.init_future_trade_time_dict()
        self.future_trade_api.reset_account_login()

        time.sleep(3)
        self.all_trade_reverse_result.save_account_info()

        try:
            event = Event(
                ConstantEvent.STRATEGY_TRADING_BEFORE,
                context=self.context.strategy_context)
            self.event_bus.publish_event(event)
        except ErrorException as e:
            SRLogger.error(e.message)

    def end_date(self):
        self.init_future_account_info()
        self.init_account_info()
        time.sleep(3)
        self.all_trade_reverse_result.save_account_info()
        self.all_trade_reverse_result.save_daily_result()

        # 期货账号登出
        self.future_trade_api.account_logout()

        # 股票账号登出
        self.trade_api.account_logout()

        self.all_trade_reverse_result.end_date()
        try:
            event = Event(
                ConstantEvent.STRATEGY_TRADING_AFTER,
                context=self.context.strategy_context)
            event_bus = self.context.event_bus
            event_bus.publish_event(event)

        except ErrorException as e:
            SRLogger.error(e.message)

    def day_start(self):
        # 期货账号重新登录
        self.future_trade_api.reset_account_login()

        # 股票账号重新登录
        self.trade_api.reset_account_login()

        time.sleep(3)
        self.all_trade_reverse_result.save_account_info()

        self.all_trade_reverse_result.day_start()
        try:
            event = Event(
                ConstantEvent.STRATEGY_DAY_BEFORE,
                context=self.context.strategy_context)
            self.event_bus.publish_event(event)
        except ErrorException as e:
            SRLogger.error(e.message)

    def subscribe(self, symbol_list):
        pass

    def place_order(self, account_id, order):
        """
        下单
        :param account_id:
        :param order:
        :return:
        """

        return self.trade_api.insert_order(account_id, order)

    def place_future_order(self, account_id, order):
        """
        下单
        :param account_id:
        :param order:
        :return:
        """
        print('place_future_order')

        return self.future_trade_api.insert_order(account_id, order)

    def insert_future_group_order(self, account, long_symbol_dict, short_symbol_dict):
        self.future_trade_api.insert_group_order(account, long_symbol_dict, short_symbol_dict)

    def insert_stock_group_order(self, account, symbol_dict, price_type=0):
        self.trade_api.insert_group_order(account, symbol_dict, price_type)

    def close_strategy_position(self, account, symbol, side, close_strategy_today_amount, close_strategy_yes_amount):
        return self.future_trade_api.close_strategy_position(account, symbol, side, close_strategy_today_amount,
                                                             close_strategy_yes_amount)

    def cancel_order(self, account_id, order_id, risk_control_client=None):
        """
        撤单
        :param account_id:
        :param order_id:
        :return:
        """
        self.trade_api.cancel_stock_order(order_id, account_id, risk_control_client)

    def cancel_future_order(self, account_id, order_id):
        self.future_trade_api.cancel_future_order(order_id, account_id)

    def pub_data(self, pub_key, json_data):
        try:
            self.strategy_sub_pub.pub_data(pub_key, json_data)
        except Exception as e:
            SRLogger.error("推送异常，异常内容：" + str(e))

    def sub_data(self, sub_keys, call_back):
        try:
            strategy_context = self.context.strategy_context

            if self.strategy_sub_flag is False:
                self.strategy_sub_pub.init_sub_strategy_signal(strategy_context, sub_keys, call_back)
            else:
                SRLogger.error("不可进行重复自定义订阅")
        except Exception as e:
            SRLogger.error("订阅异常，异常内容：" + str(e))

    def night_end(self):
        # 期货账号登出
        self.init_future_account_info()
        time.sleep(3)
        self.all_trade_reverse_result.save_account_info()
        self.future_trade_api.account_logout()

    def sys_stock_rtn_order(self, order):
        self.all_trade_reverse_result.on_stock_rtn_order(order)

    def sys_stock_rtn_trade(self, trade):
        self.all_trade_reverse_result.on_stock_rtn_trade(trade)
        self.trade_api.on_stock_rtn_trade(trade)

    def sys_stock_rtn_transfer(self, xb_real_withdraw_deposit):
        self.all_trade_reverse_result.on_stock_rtn_transfer(xb_real_withdraw_deposit)

    def sys_stock_quotation_change(self, bar_data):
        self.all_trade_reverse_result.refresh_stock_position(bar_data)
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 0:
            self.all_trade_reverse_result.refresh_standard_symbol(bar_data)

    def sys_stock_order_cancel(self, order):
        self.trade_api.on_stock_order_cancel(order)

        bar_data = QuotationData.get_instance().bar_dict
        event_bus = self.context.event_bus
        strategy_context = self.context.strategy_context
        event = Event(
            ConstantEvent.STOCK_ORDER_CANCEL,
            context=strategy_context,
            order=order,
            bar_dict=bar_data)
        event_bus.publish_event(event)

    def sys_future_rtn_order(self, order):
        self.all_trade_reverse_result.on_future_rtn_order(order)

    def sys_future_rtn_trade(self, trade):
        self.all_trade_reverse_result.on_future_rtn_trade(trade)
        self.future_trade_api.on_future_rtn_trade(trade)

    def sys_future_rtn_transfer(self, xb_real_withdraw_deposit):
        self.all_trade_reverse_result.on_future_rtn_transfer(xb_real_withdraw_deposit)

    def sys_future_quotation_change(self, bar_data):
        self.all_trade_reverse_result.refresh_future_position(bar_data)
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if run_info.standard_type == 1:
            self.all_trade_reverse_result.refresh_standard_symbol(bar_data)

    def sys_future_order_cancel(self, order):

        self.future_trade_api.on_future_order_cancel(order)

        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if order.now_system_order == 1 and order.client_id == run_info.run_id:
            bar_data = QuotationData.get_instance().bar_dict
            event_bus = self.context.event_bus
            event = Event(
                ConstantEvent.FUTURE_ORDER_CANCEL,
                context=strategy_context,
                order=order,
                bar_dict=bar_data)
            event_bus.publish_event(event)

    def sys_future_asset_refresh(self, xb_back_test_account):
        self.all_trade_reverse_result.refresh_future_asset(xb_back_test_account)

    def sys_stock_asset_refresh(self, xb_back_test_account):
        self.all_trade_reverse_result.refresh_stock_asset(xb_back_test_account)

    def calculate_daily_result(self):
        """计算按日统计的交易结果"""
        self.all_trade_reverse_result.calculate_all_result()

    def show_daily_result(self, benchmark_name):
        self.all_trade_reverse_result.show_all_result(benchmark_name)

    def cash_moving(self, from_account, to_account, cash, move_type):
        pass

    def get_today_order(self, account_id, order_id):
        if order_id in self.all_trade_reverse_result.get_trade_reverse_reslult(account_id).work_order.keys():
            return self.all_trade_reverse_result.get_trade_reverse_reslult(account_id).work_order[order_id]

    def get_today_future_order(self, account_id, order_id):
        if order_id in self.all_trade_reverse_result.get_future_reverse_result(account_id).work_order.keys():
            return self.all_trade_reverse_result.get_future_reverse_result(account_id).work_order[order_id]

    def get_today_work_order(self, account_id):
        work_order_list = list()
        for account_order in self.all_trade_reverse_result.get_trade_reverse_reslult(account_id).work_order.values():
            if account_order.status == 0 or account_order.status == 1 or account_order.status == 6:
                work_order_list.append(account_order)
        return work_order_list

    def get_today_work_future_order(self, account_id):
        work_future_order_list = list()
        for account_order in self.all_trade_reverse_result.get_future_reverse_reslult(account_id).work_order.values():
            if account_order.status == 0 or account_order.status == 1 or account_order.status == 6:
                work_future_order_list.append(account_order)
        return work_future_order_list

    def sys_daily_data_save(self):
        self.all_trade_reverse_result.save_account_info()

    def sys_sub_future_symbol(self, symbol_list, sub_type=0):
        """
        订阅期货
        :param symbol_list:
        :param sub_type: 0：系统订阅（持仓，订单），1：策略自主订阅
        :return:
        """
        strategy_context = self.context.strategy_context
        if symbol_list is None:
            self.future_trade_api.sub_symbol(strategy_context.sub_future_symbol_list)
        else:
            if sub_type == 1:
                strategy_context.sub_strategy_future_symbol(symbol_list)
            else:
                strategy_context.sub_future_symbol(symbol_list)
            self.future_trade_api.sub_symbol(symbol_list)

    def sys_sub_stock_symbol(self, symbol_list, sub_type=0):
        strategy_context = self.context.strategy_context
        if symbol_list is None:
            self.trade_api.sub_symbol(strategy_context.sub_stock_symbol_list)
        else:
            if sub_type == 1:
                strategy_context.sub_strategy_stock_symbol(symbol_list)
            else:
                strategy_context.sub_stock_symbol(symbol_list)
            self.trade_api.sub_symbol(symbol_list)

    def sub_future_symbol(self, symbol_list):
        self.sys_sub_future_symbol(symbol_list, sub_type=1)

    def sub_stock(self, symbol_list):
        self.sys_sub_stock_symbol(symbol_list, sub_type=1)

    def sys_future_all_pos_refresh(self, account, position_dict):
        self.all_trade_reverse_result.on_future_all_pos_refresh(account, position_dict)

    def sys_future_trade_pos_refresh(self, account, position_dict):
        self.all_trade_reverse_result.on_future_trade_pos_refresh(account, position_dict)

    def sys_stock_all_pos_refresh(self, account, position_dict):
        self.all_trade_reverse_result.on_stock_all_pos_refresh(account, position_dict)

    def sys_stock_trade_pos_refresh(self, account, position_dict):
        self.all_trade_reverse_result.on_stock_trade_pos_refresh(account, position_dict)

    def sys_restore_strategy(self):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        restore_strategy = RestoreStrategy()
        restore_strategy.start_restore_read(run_info.run_id, strategy_context)
        restore_strategy.init_restore_save(run_info.run_id, strategy_context)
