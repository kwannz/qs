import copy

import time
from collections import defaultdict

import math

from panda_backtest.backtest_common.constant.strategy_constant import CLOSE
from panda_backtest.backtest_common.data.order.common.work_order_list import WorkOrderList
from panda_backtest.backtest_common.exchange.fund.back_test.fund_rate_manager import FundRateManager
from panda_backtest.backtest_common.exchange.fund.fund_bonus_manager import FundBonusManager
from panda_backtest.backtest_common.exchange.fund.fund_info_map import FundInfoMap
from panda_backtest.backtest_common.exchange.fund.fund_split_manager import FundSplitManager
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
from panda_backtest.backtest_common.model.result.panda_backtest_trade import PandaBacktestTrade, OPEN
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.order.fund.common.fund_order_builder import FundOrderBuilder
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.model.result.order import Order, ACTIVE, REJECTED, SIDE_BUY, FILLED, CANCELLED
from panda_backtest.backtest_common.system.event.event import ConstantEvent, Event

class FundExchange(object):
    def __init__(self, quotation_mongo_db):
        self.context = CoreContext.get_instance()
        self.work_order = WorkOrderList()
        self.all_order = defaultdict(dict)
        self.order_verify_chain = list()
        self.order_count = 0
        self.trade_count = 0
        self.quotation_mongo_db = quotation_mongo_db
        self.fund_info_map = FundInfoMap(self.quotation_mongo_db)
        self.fund_order_builder = FundOrderBuilder(self.fund_info_map)
        self.fund_rate_manager = FundRateManager(self.quotation_mongo_db)
        self.fund_dividend_manager = FundBonusManager(self.quotation_mongo_db)
        self.fund_split_manager = FundSplitManager(self.quotation_mongo_db)

    def add_order_verify(self, order_verify):
        self.order_verify_chain.append(order_verify)

    def init_data(self):
        self.fund_rate_manager.init_data()

    def init_event(self):
        event_bus = self.context.event_bus
        event_bus.register_handle(ConstantEvent.SYSTEM_FUND_ORDER_CROSS, self.sys_cross_order)

    def day_start(self):
        """
        每日开始，判断是否到账，进行分红
        """
        pass

    def fund_split(self):
        strategy_context = self.context.strategy_context
        if strategy_context.is_trade_date():
            self.fund_split_manager.get_fund_split()

    def fund_dividend(self):
        strategy_context = self.context.strategy_context
        if strategy_context.is_trade_date():
            event_bus = self.context.event_bus
            order_list = self.work_order.get_fund_arrive_order_list(strategy_context.trade_date)
            if len(order_list) > 0:
                for order in order_list:
                    event = Event(ConstantEvent.SYSTEM_FUND_RTN_ORDER, order=order)
                    event_bus.publish_event(event)
                self.work_order.remove_fund_order_arrive_date(strategy_context.trade_date)
            self.fund_dividend_manager.get_fund_bonus()

    def new_date(self):
        strategy_context = self.context.strategy_context
        symbol_list = self.work_order.get_fund_cross_date_symbol_list(strategy_context.trade_date)
        event_bus = self.context.event_bus
        event = Event(
            ConstantEvent.SYSTEM_FUND_QUOTATION_START_UN_SUB,
            symbol_list=None,
            sub_type=1)
        event_bus.publish_event(event)
        if len(symbol_list) > 0:
            event = Event(
                ConstantEvent.SYSTEM_FUND_QUOTATION_START_SUB,
                symbol_list=symbol_list,
                sub_type=1)
            event_bus.publish_event(event)

    def end_date(self):
        """
        每日结束，对未成交订单进行撤单
        :return:
        """
        self.fund_rate_manager.clear_cache_data()

    def log_order_error(self, order_result):
        sr_logger = RemoteLogFactory.get_sr_logger()
        if order_result.now_system_order == 2:
            risk_control_manager = self.context.risk_control_manager
            sr_logger.risk(risk_control_manager.get_risk_control_name(order_result.risk_control_id), order_result.message)
        else:
            sr_logger.error(order_result.message)

    def insert_order(self, account, order_dict):
        event_bus = self.context.event_bus
        order_result = self.fund_order_builder.init_fund_order(account, order_dict)

        if order_result.status == REJECTED:
            self.log_order_error(order_result)
            event = Event(ConstantEvent.SYSTEM_FUND_RTN_ORDER, order=order_result)
            event_bus.publish_event(event)
            event = Event(ConstantEvent.SYSTEM_FUND_ORDER_CANCEL, order=order_result)
            event_bus.publish_event(event)
            return order_result

        for chain_item in self.order_verify_chain:
            if not chain_item.can_submit_order(account, order_result):
                order_result.status = REJECTED
                self.log_order_error(order_result)
                self.log_order_error(order_result)
                self.all_order[account][order_result.order_id] = order_result
                event = Event(ConstantEvent.SYSTEM_FUND_RTN_ORDER, order=order_result)
                event_bus.publish_event(event)
                event = Event(ConstantEvent.SYSTEM_FUND_ORDER_CANCEL, order=order_result)
                event_bus.publish_event(event)
                self.all_order[account][order_result.order_id] = order_result
                return [order_result]

        if order_result.fund_cover_old == 1:
            order_list = self.work_order.get_order_list(symbol=order_result.order_book_id)
            for order in order_list:
                if order.side == order_result.side:
                    self.cancel_order(account, order.order_id)

        event = Event(ConstantEvent.SYSTEM_FUND_RTN_ORDER, order=order_result)
        event_bus.publish_event(event)
        self.all_order[account][order_result.order_id] = order_result
        self.work_order.add_order(order_result, is_fund_order=True)
        return [order_result]

    def cancel_order(self, account, order_id):
        event_bus = self.context.event_bus

        order_list = self.work_order.get_order_list(order_id)
        if len(order_list) == 0:
            return False

        order = order_list[0]
        if order.status == CANCELLED or order.status == FILLED or \
                order.status == REJECTED:
            return False
        else:
            order.status = CANCELLED

            event = Event(ConstantEvent.SYSTEM_FUND_RTN_ORDER, order=order)
            event_bus.publish_event(event)

            event = Event(ConstantEvent.SYSTEM_FUND_ORDER_CANCEL, order=order)
            event_bus.publish_event(event)
            self.work_order.remove_order(order_id, is_fund_order=True)
            return True

    def sys_cross_order(self, bar_data=None):
        strategy_context = self.context.strategy_context
        if bar_data is None or bar_data.symbol == '' or bar_data.unit_nav == 0:
            return
        order_list = self.work_order.get_order_list(symbol=bar_data.symbol, cross_date=strategy_context.trade_date)
        for order in order_list:
            self.handle_cross_order(order)

    def handle_cross_order(self, order):
        bar_dict = QuotationData.get_instance().bar_dict
        bar_data = bar_dict[order.order_book_id]

        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        event_bus = self.context.event_bus

        hq_price = float(bar_data.unit_nav)

        cross = True

        if cross:
            # 推送成交数据
            self.trade_count += 1  # 成交编号自增1
            trade_id = str(self.trade_count)
            trade = PandaBacktestTrade()
            trade.back_id = run_info.run_id
            trade.now_system_order = order.now_system_order
            trade.client_id = order.client_id
            trade.account_id = order.account
            trade.contract_code = order.order_book_id
            trade.contract_name = order.order_book_name
            trade.trade_id = trade_id
            trade.order_id = order.order_id
            trade.direction = order.effect
            trade.price = hq_price
            trade.business = order.side
            trade.type = 2
            trade.gmt_create = strategy_context.trade_date
            trade.gmt_create_time = strategy_context.hms

            order.price = hq_price

            if order.side == SIDE_BUY:
                rate = self.fund_rate_manager.get_order_rate(order)
                trade.cost = rate
                order.transaction_cost = rate
                trade_quantity = (order.purchase_amount - rate) / hq_price
                order.filled_quantity = math.floor(trade_quantity * 10000) / 10000
                order.unfilled_quantity = 0
                trade.volume = order.filled_quantity
                order.status = FILLED

                # 推送委托数据
                self.work_order.remove_order(order.order_id, is_fund_order=True)
                event = Event(ConstantEvent.SYSTEM_FUND_RTN_ORDER, order=order)
                event_bus.publish_event(event)

            else:
                rate = self.fund_rate_manager.get_order_rate(order)
                trade.cost = rate
                order.transaction_cost = rate
                order.filled_quantity = order.quantity

                latency_date = strategy_context.get_next_count_date(strategy_context.now, int(order.latency_date))
                trade.trade_amount = order.filled_quantity * hq_price
                trade.volume = order.filled_quantity
                trade.fund_arrive_date = latency_date
                order.fund_arrive_date = latency_date
                order.redeem_frozen_amount = trade.trade_amount
                order.status = FILLED
                self.work_order.remove_order(order.order_id, is_fund_order=True)
                self.work_order.add_fund_order_arrive_date(order)

            event = Event(ConstantEvent.SYSTEM_FUND_RTN_TRADE, trade=trade)
            event_bus.publish_event(event)

    def restore_read(self, redis_client, key, hkey):
        self.work_order.restore_read(redis_client, key, hkey)

    def restore_save(self, redis_client, key, hkey):
        self.work_order.restore_save(redis_client, key, hkey)

