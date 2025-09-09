import json
import logging

from collections import defaultdict

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, SYMBOL_CAN_NOT_CROSS
from panda_backtest.backtest_common.data.future.future_margin_map import FutureMarginMap
from panda_backtest.backtest_common.exchange.future.back_test.future_order_split_manager import FutureOrderSplitManager
from panda_backtest.backtest_common.exchange.future.back_test.future_settle_manager import FutureSettleManager
from panda_backtest.backtest_common.data.order.common.work_order_list import WorkOrderList
from panda_backtest.backtest_common.constant.strategy_constant import FILLED, CANCELLED, PartTradedNotQueueing, CLOSE

from panda_backtest.backtest_common.exchange.future.back_test.future_rate_manager import FutureRateManager
from panda_backtest.backtest_common.model.result.panda_backtest_trade import PandaBacktestTrade

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.order.future.common.future_order_builder import FutureOrderBuilder

from panda_backtest.backtest_common.system.event.event import ConstantEvent, Event

from panda_backtest.backtest_common.model.result.order import Order, ACTIVE, SIDE_BUY, OPEN, LIMIT, MARKET, REJECTED

from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.order.common.order_quotation_verify import OrderQuotationVerify

class FutureExchange(object):
    def __init__(self, quotation_mongo_db, rate_file):
        self.context = CoreContext.get_instance()
        self.work_order = WorkOrderList()
        self.all_order = defaultdict(dict)
        self.order_verify_chain = list()
        self.order_count = 0
        self.trade_count = 0
        self.quotation_mongo_db = quotation_mongo_db
        self.future_info_map = FutureInfoMap(self.quotation_mongo_db)
        self.order_quotation_verify = OrderQuotationVerify()
        self.future_rate_manager = FutureRateManager(rate_file)
        self.future_order_split_manager = FutureOrderSplitManager()
        self.future_settle_manager = FutureSettleManager()
        self.future_order_builder = FutureOrderBuilder(self.future_info_map, self.future_rate_manager)
        self.future_margin_map = FutureMarginMap()

    def init_event(self):
        event_bus = self.context.event_bus
        event_bus.register_handle(ConstantEvent.SYSTEM_FUTURE_ORDER_CROSS, self.sys_cross_order)

    def add_order_verify(self, order_verify):
        self.order_verify_chain.append(order_verify)

    def insert_order(self, account, order_dict):
        event_bus = self.context.event_bus
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        order_result = self.future_order_builder.init_future_order(account, order_dict)

        if order_result.status == REJECTED:
            # self.log_order_error(order_result)
            event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, order=order_result)
            event_bus.publish_event(event)
            event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, order=order_result)
            event_bus.publish_event(event)
            return [order_result]

        order_result.effect = order_dict['effect']
        order_result = self.future_order_split_manager.split_close_today_order(order_result)

        for chain_item in self.order_verify_chain:
            if not chain_item.can_submit_order(account, order_result):
                order_result.status = REJECTED
                # self.log_order_error(order_result)
                self.all_order[account][order_result.order_id] = order_result
                event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, order=order_result)
                event_bus.publish_event(event)
                event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, order=order_result)
                event_bus.publish_event(event)
                self.all_order[account][order_result.order_id] = order_result
                return [order_result]
        event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, order=order_result)
        event_bus.publish_event(event)
        self.all_order[account][order_result.order_id] = order_result

        self.work_order.add_order(order_result)

        # 订阅
        event_bus = self.context.event_bus
        event = Event(
            ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_SUB,
            symbol_list=[order_result.order_book_id],
            sub_type=1)
        event_bus.publish_event(event)

        # strategy_context.sub_future_symbol([order_result.order_book_id])
        # TODO
        # if run_info.matching_type == 0:
        #     self.handle_cross_order(order_result)
        self.handle_cross_order(order_result)
        return [order_result]

    def sys_cross_order(self, bar_data=None):
        if bar_data is None or bar_data.symbol == '':
            return
        order_list = self.work_order.get_order_list(symbol=bar_data.symbol)
        for order in order_list:
            self.handle_cross_order(order)

    def handle_cross_order(self, order):
        if order.status == FILLED and order.status == CANCELLED:
            return
        bar_dict = QuotationData.get_instance().bar_dict
        bar_data = bar_dict[order.order_book_id]

        strategy_context = self.context.strategy_context

        run_info = strategy_context.run_info

        instrument_info = self.future_info_map[order.order_book_id]

        if order.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        # TODO
        slippage = run_info.future_slippage
        # slippage = run_info.slippage

        if run_info.run_strategy_type == 0:
            if run_info.matching_type == 1:
                hq_price = bar_data.open
            else:
                hq_price = bar_data.close
        else:
            hq_price = bar_data.last

        ftminpricechg = instrument_info['ftminpricechg']
        if order.side == SIDE_BUY:
            hq_price = hq_price + int(slippage) * float(str(ftminpricechg))
        else:
            hq_price = hq_price - int(slippage) * float(str(ftminpricechg))

        buy_cross_price = hq_price  # 若买入方向限价单价格高于该价格，则会成交
        sell_cross_price = hq_price  # 若卖出方向限价单价格低于该价格，则会成交

        if order.price_type == LIMIT:  # 限价单
            if order.side == SIDE_BUY:
                cross = order.price >= buy_cross_price > 0
                trade_price = buy_cross_price
            else:
                cross = order.price <= sell_cross_price and sell_cross_price > 0
                trade_price = sell_cross_price
        else:
            cross = True
            if order.side == SIDE_BUY:
                trade_price = hq_price
            else:
                trade_price = hq_price

        if cross and bar_data.volume > 0:

            if order.quantity > bar_data.volume:
                print('期货====》')

                order.filled_quantity = int(bar_data.volume)
                order.filled_close_td_pos = int(min(order.close_td_pos,
                                                    max(0, bar_data.volume - (order.quantity - order.close_td_pos))))
                order.unfilled_quantity = int(order.quantity - order.filled_quantity)
                order.unfilled_close_td_pos = order.close_td_pos - order.filled_close_td_pos
                order.cur_filled_quantity = order.filled_quantity
                order.cur_close_td_pos = order.filled_close_td_pos
                order.status = PartTradedNotQueueing
                order.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order.order_book_id, str(order.unfilled_quantity),
                    order.account, order_effect, order_side, order.order_id,
                    SYMBOL_CAN_NOT_CROSS)
            else:
                order.filled_quantity = order.quantity
                order.filled_close_td_pos = order.close_td_pos
                order.unfilled_quantity = 0
                order.unfilled_close_td_pos = 0
                order.cur_filled_quantity = order.quantity
                order.cur_close_td_pos = order.close_td_pos
                order.status = FILLED

            # 推送成交数据
            self.trade_count += 1  # 成交编号自增1
            trade_id = str(self.trade_count)
            trade = PandaBacktestTrade()
            trade.type = 1
            trade.back_id = run_info.run_id
            trade.now_system_order = order.now_system_order
            trade.client_id = order.client_id
            trade.account_id = order.account
            trade.contract_code = order.order_book_id
            trade.contract_name = order.order_book_name
            trade.trade_id = trade_id
            trade.order_id = order.order_id
            trade.direction = order.effect
            trade.price = trade_price
            trade.business = order.side
            trade.volume = int(order.filled_quantity)
            trade.gmt_create = strategy_context.now
            trade.trade_date = strategy_context.trade_date
            trade.gmt_create_time = strategy_context.hms
            trade.round_lot = order.round_lot
            trade.close_td_pos = order.cur_close_td_pos
            trade.order_remark = order.remark

            if order.effect == OPEN:
                rate = self.future_rate_manager.get_future_cost_rate(trade.contract_code,
                                                                     trade.volume,
                                                                     trade.price * trade.volume *
                                                                     trade.round_lot,
                                                                     0, 0,
                                                                     instrument_info['emcodetype'])
                trade_date = int(strategy_context.trade_date)
                symbol=order.order_book_id
                margin_info = self.future_margin_map.get_future_margin_info(symbol, trade_date)
                if margin_info['name'] == '未知':
                    instrument_info = self.future_info_map[symbol]
                    margin_rate = instrument_info['ftfirsttransmargin'] / 100 * run_info.margin_multiplier
                else:
                    margin_rate = margin_info['long_margin'] / 100
                    if margin_rate == 0:
                        margin_rate = margin_info['margin'] / 100
                trade.margin = \
                    trade.price * trade.volume * trade.round_lot * \
                    margin_rate
            else:
                rate = self.future_rate_manager.get_future_cost_rate(trade.contract_code,
                                                                     trade.volume - trade.close_td_pos,
                                                                     (trade.volume - trade.close_td_pos) *
                                                                     trade.round_lot * trade.price,
                                                                     trade.close_td_pos,
                                                                     trade.close_td_pos * trade.round_lot * trade.price,
                                                                     instrument_info['emcodetype'])
                trade.margin = 0

            trade.cost = rate * run_info.commission_multiplier

            # 推送委托数据
            self.work_order.remove_order(order.order_id)
            event_bus = self.context.event_bus
            event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, order=order)
            event_bus.publish_event(event)

            event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_TRADE, trade=trade)
            event_bus.publish_event(event)
            self.future_settle_manager.add_settle_future(trade.contract_code, instrument_info['lasttradedate'])

            if order.status == PartTradedNotQueueing:
                event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, order=order)
                event_bus.publish_event(event)

            # 取消订阅
            if order.order_book_id not in self.work_order.symbol_index.keys():
                event_bus = self.context.event_bus
                event = Event(
                    ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_UN_SUB,
                    symbol_list=[order.order_book_id],
                    sub_type=1)
                event_bus.publish_event(event)
        else:
            order.filled_quantity = 0
            order.filled_close_td_pos = 0
            order.unfilled_quantity = int(order.quantity)
            order.unfilled_close_td_pos = int(order.close_td_pos)
            order.cur_filled_quantity = 0
            order.cur_close_td_pos = 0
            order.status = CANCELLED
            order.message = FUTURE_ORDER_FAILED_MESSAGE % (
                order.order_book_id, str(order.unfilled_quantity),
                order.account, order_effect, order_side, order.order_id,
                SYMBOL_CAN_NOT_CROSS)

            # 推送委托数据
            self.work_order.remove_order(order.order_id)
            event_bus = self.context.event_bus
            event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, order=order)
            event_bus.publish_event(event)

            event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, order=order)
            event_bus.publish_event(event)

            # 取消订阅
            if order.order_book_id not in self.work_order.symbol_index.keys():
                event_bus = self.context.event_bus
                event = Event(
                    ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_UN_SUB,
                    symbol_list=[order.order_book_id],
                    sub_type=1)
                event_bus.publish_event(event)

    def new_date(self):
        pass

    def end_date(self):
        """
        每日结束，对未成交订单进行撤单
        :return:
        """
        for order in self.work_order.get_order_list():
            self.cancel_order(order.account, order.order_id)

        self.work_order.clear()
        self.all_order.clear()

    def future_settle(self):
        strategy_context = self.context.strategy_context
        if not strategy_context.is_last_trade_date():
            self.future_settle_manager.handle_settle()

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
            event = Event(ConstantEvent.SYSTEM_FUTURE_RTN_ORDER, order=order)
            event_bus.publish_event(event)

            event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CANCEL, order=order)
            event_bus.publish_event(event)
            self.work_order.remove_order(order_id)

            # 取消订阅
            if order.order_book_id not in self.work_order.symbol_index.keys():
                event_bus = self.context.event_bus
                event = Event(
                    ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_UN_SUB,
                    symbol_list=[order.order_book_id],
                    sub_type=1)
                event_bus.publish_event(event)
            return True

    def restore_read(self, redis_client, key, hkey):
        self.work_order.restore_read(redis_client, key, hkey)

    def restore_save(self, redis_client, key, hkey):
        self.work_order.restore_save(redis_client, key, hkey)
