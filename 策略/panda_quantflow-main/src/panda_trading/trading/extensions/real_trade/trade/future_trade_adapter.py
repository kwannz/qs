#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/10 下午1:38
# @Author : wlb
# @File   : future_trade_adapter.py
# @desc   :
import time
from datetime import datetime

from panda_backtest.backtest_common.constant.strategy_constant import REJECTED, SIDE_BUY, CLOSE
from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE
from panda_backtest.backtest_common.data.order.real_time.future_work_order_list import FutureWorkOrderList
from panda_backtest.backtest_common.model.result.order import Order
from panda_backtest.backtest_common.system.event.event import ConstantEvent, Event

from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
from utils.log.log_factory import LogFactory


class FutureTradeAdapter(object):
    def __init__(self, work_order):
        self.context = CoreContext.get_instance()
        self.logger = LogFactory.get_logger()
        self.work_order = work_order

    def on_insert_order(self, account, order):
        self.logger.info("发生订单插入，订单id:{}".format(order.order_id))
        date = str(datetime.now().strftime('%Y%m%d'))
        self.work_order.save_work_order(account, order, date)

    def on_system_cancel_order(self, account, order):
        self.logger.info("发生订单撤单回报，订单id:{}".format(order.order_id))
        self.log_order_error(order)
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, order=order)
        event_bus.publish_event(event)

    def on_rsp_qry_trading_account(self, account, xb_back_test_account):
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_FUTURE_ASSET_REFRESH, xb_back_test_account=xb_back_test_account)
        event_bus.publish_event(event)

    def on_rsp_qry_investor_position(self, account, account_position_dict, req_id):
        event_bus = self.context.event_bus

        if req_id == 0:
            # 成交回报刷下持仓的金额等情况
            event = Event(ConstantEvent.SYSTEM_FUTURE_TRADE_POSITION_REFRESH, account=account,
                          position_dict=account_position_dict)

        else:
            event = Event(ConstantEvent.SYSTEM_FUTURE_ALL_POSITION_REFRESH, account=account,
                          position_dict=account_position_dict)
        event_bus.publish_event(event)

    def on_rtn_order(self, account, order):
        self.logger.info("发生订单回报，订单id:{0}".format(order.order_id))
        date = str(datetime.now().strftime('%Y%m%d'))
        self.work_order.save_work_order(account, order, date)
        # 回报
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, order=order)
        event_bus.publish_event(event)

    def get_old_work_order(self, account, order_id, relative_order_sys_iD):
        date = str(datetime.now().strftime('%Y%m%d'))
        if relative_order_sys_iD is not None and relative_order_sys_iD != '':
            # 条件单触发的订单
            self.logger.info('这是条件单触发的订单')
            date = str(datetime.now().strftime('%Y%m%d'))
            # 条件单通过relative_order_sys_iD关联id获取对应单子
            old_order = self.work_order.get_work_order_by_sys_id(account, relative_order_sys_iD, date)
            if old_order:
                order_id = old_order.order_id
            else:
                # 兼容融航情况，普通单也有relative_order_sys_iD这个字段
                old_order = self.work_order.get_work_order_by_order_id(account, order_id, date)
        else:
            old_order = self.work_order.get_work_order_by_order_id(account, order_id, date)

        if old_order is None:
            # 三方订单
            self.logger.info('当前是三方订单')
            old_order = Order()
            old_order.client_id = '0000001'
            old_order.now_system_order = 0
            old_order.date = str(datetime.now().strftime('%Y%m%d'))
            old_order.datetime = str(datetime.now())

        return old_order

    def get_work_order_by_order_id(self, account, order_id):
        date = str(datetime.now().strftime('%Y%m%d'))
        return self.work_order.get_work_order_by_order_id(account, order_id, date)

    def get_work_order_by_order_sys_id(self, account, order_sys_id):
        date = str(datetime.now().strftime('%Y%m%d'))
        return self.work_order.get_work_order_by_sys_id(account, order_sys_id, date)

    def on_rtn_trade(self, account, trade):
        # trade.trade_date = self.context.strategy_context.trade_date
        self.logger.info("发生订单成交回报，成交id:【%s】" % str(trade.trade_id))
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_TRADE, trade=trade)
        event_bus.publish_event(event)

    def on_rtn_transfer(self, account, xb_real_withdraw_deposit):
        self.logger.info("发生期货转账回报，转账记录:{}".format(xb_real_withdraw_deposit.__dict__))
        event_bus = self.context.event_bus
        print('发生了SYSTEM_FUTURE_RTN_TRANSFER')
        event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_TRANSFER, xb_real_withdraw_deposit=xb_real_withdraw_deposit)
        event_bus.publish_event(event)

    def on_tick_back(self, bar_data):
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_FUTURE_QUOTATION_CHANGE, bar_data=bar_data)
        event_bus.publish_event(event)

    def on_account_login(self, account):
        pass

    def on_quotation_account_login(self, account):
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_SUB, sub_symbol_list=None)
        event_bus.publish_event(event)

    def log_order_error(self, order_result):
        order_result.is_reject_by_risk = 1
        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        err_mes = FUTURE_ORDER_FAILED_MESSAGE % (
            order_result.order_book_id, str(order_result.quantity),
            order_result.account, order_effect, order_side, order_result.order_id,
            order_result.message)

        sr_logger = RemoteLogFactory.get_sr_logger()
        if order_result.now_system_order == 2:
            risk_control_manager = self.context.risk_control_manager
            sr_logger.risk(risk_control_manager.get_risk_control_name(order_result.risk_control_id),
                           err_mes)
        else:
            sr_logger.error(err_mes)
