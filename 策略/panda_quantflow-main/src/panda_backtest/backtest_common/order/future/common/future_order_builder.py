#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/9/16 20:19
# @Author : wlb
# @File   : stock_order_builder.py
# @desc   :
from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, STOCK_GEM_QUANTITY_NOT_RIGHT, \
STOCK_QUANTITY_NOT_RIGHT
from panda_backtest.backtest_common.data.future.future_margin_map import FutureMarginMap
from panda_backtest.backtest_common.data.stock.stock_info_map import StockInfoMap
from panda_backtest.backtest_common.model.result.order import Order, ACTIVE, OPEN, SIDE_BUY, LIMIT, MARKET, REJECTED, CLOSE
from panda_backtest.backtest_common.order.common.order_quotation_verify import OrderQuotationVerify
from panda_backtest.backtest_common.system.context.core_context import CoreContext

class FutureOrderBuilder(object):

    def __init__(self, future_info_map, future_rate_manager):
        self.context = CoreContext.get_instance()
        self.order_count = 0
        self.future_info_map = future_info_map
        self.order_quotation_verify = OrderQuotationVerify()
        self.future_margin_map = FutureMarginMap()
        self.future_rate_manager = future_rate_manager

    def init_future_order(self, account, order_dict):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        trade_date = int(strategy_context.trade_date)
        self.order_count += 1
        order_id = str(self.order_count)
        order_result = Order()
        order_result.status = ACTIVE
        order_result.order_id = order_id
        order_result.client_id = run_info.run_id
        order_result.order_book_id = order_dict['symbol']
        order_result.datetime = strategy_context.trade_time
        order_result.account = account
        order_result.quantity = int(order_dict['quantity'])
        order_result.unfilled_quantity = int(order_dict['quantity'])
        order_result.client_id = order_dict.get('client_id', run_info.run_id)
        order_result.risk_control_id = order_dict.get('risk_control_id', None)
        order_result.remark = order_dict.get('remark', None)
        order_result.now_system_order = order_dict.get('now_system_order', 1)
        order_result.retry_num = order_dict.get('retry_num', 0)

        key = order_result.order_book_id
        instrument_info = self.future_info_map[key]
        order_result.order_book_name = instrument_info['name']
        order_result.round_lot = instrument_info['contractmul']
        order_result.side = order_dict['side']

        if order_dict['price_type'] == LIMIT:
            order_result.price = order_dict['price']
            order_result.price_type = LIMIT
        else:
            order_result.price = self.order_quotation_verify.get_order_market_price(order_result)
            order_result.price_type = MARKET

        if order_result.price == 0:
            order_result.status = REJECTED
            if order_result.side == SIDE_BUY:
                order_side = '买入'
            else:
                order_side = '卖出'
            if order_result.effect == CLOSE:
                order_effect = '平仓'
            else:
                order_effect = '开仓'
            print(order_dict['price_type'])
            err_mes = FUTURE_ORDER_FAILED_MESSAGE % (order_dict['symbol'], str(order_dict['quantity']),
                                                     account, order_effect, order_side, str(-1),
                                                     '报单价格为0')
            order_result.message = err_mes
            return order_result

        if order_dict['effect'] == OPEN:
            symbol = order_dict['symbol']
            margin_info = self.future_margin_map.get_future_margin_info(symbol, trade_date)
            if margin_info['name'] == '未知':
                instrument_info = self.future_info_map[symbol]
                margin_rate = instrument_info['ftfirsttransmargin'] / 100 * run_info.margin_multiplier
            else:
                margin_rate = margin_info['long_margin'] / 100
                if margin_rate == 0:
                    margin_rate = margin_info['margin'] / 100
            margin = order_result.price * order_result.quantity * \
                     order_result.round_lot * margin_rate
            order_result.margin = margin
            order_result.transaction_cost = self.future_rate_manager. \
                get_future_cost_rate(order_result.order_book_id,
                                     order_result.quantity,
                                     order_result.price * order_result.quantity *
                                     order_result.round_lot,
                                     0, 0,
                                     instrument_info['emcodetype'])
        else:
            order_result.is_td_close = order_dict['is_td_close']
            order_result.margin = 0
            order_result.transaction_cost = self.future_rate_manager. \
                get_future_cost_rate(order_result.order_book_id,
                                     order_result.quantity - order_result.close_td_pos,
                                     (
                                             order_result.quantity - order_result.close_td_pos) *
                                     order_result.round_lot * order_result.price,
                                     order_result.close_td_pos,
                                     order_result.close_td_pos * order_result.round_lot * order_result.price,
                                     instrument_info['emcodetype'])

        return order_result
