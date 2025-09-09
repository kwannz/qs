from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, SIDE_SELL
import logging

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, FUTURE_NOT_LIMIT_DATA, \
    SYMBOL_LIMIT_HIGH, \
    SYMBOL_LIMIT_LOW, ORDER_PRICE_TOO_HIGH, ORDER_PRICE_TOO_LOW
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.backtest_common.system.context.core_context import CoreContext

class FutureOrderLimitPriceVerify(object):
    def __init__(self, quotation_mongo_db):
        self.context = CoreContext.get_instance()
        self.future_info_map = FutureInfoMap(quotation_mongo_db)

    def can_submit_order(self, account, order_result):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        trade_date = strategy_context.trade_date
        bar_dict = QuotationData.get_instance().bar_dict
        bar_data_source = bar_dict.bar_data_source
        bar = bar_dict[order_result.order_book_id]
        instrument_info = self.future_info_map[order_result.order_book_id]
        # print("回测下单时标的代码"+order_result.order_book_id)
        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        limit_price_obj = bar_data_source.get_future_daily_bar(order_result.order_book_id, trade_date)
        if limit_price_obj.price_limit_range == 0:
            if instrument_info is None:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    FUTURE_NOT_LIMIT_DATA)
                return False
            else:
                limit_rate = instrument_info['ftpricelimit']
        else:
            limit_rate = limit_price_obj.price_limit_range

        trad_price = order_result.price

        if run_info.matching_type == 0:
            jz_price = bar.close
        else:
            jz_price = bar.open

        if jz_price == 0:
            return False

        if bar.high >= jz_price >= bar.low:
            pass
        else:
            if jz_price >= limit_price_obj.prev_settlement * (1 + limit_rate) and order_result.side == SIDE_BUY:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    SYMBOL_LIMIT_HIGH)
                return False
            elif jz_price <= bar.prev_settlement * (1 - limit_rate) and order_result.side == SIDE_SELL:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    SYMBOL_LIMIT_LOW)
                return False
        if jz_price >= limit_price_obj.prev_settlement * (1 + limit_rate):
            limit_high_price = jz_price
        else:
            limit_high_price = limit_price_obj.prev_settlement * (1 + limit_rate)

        if jz_price <= limit_price_obj.prev_settlement * (1 - limit_rate):
            limit_low_price = jz_price
        else:
            limit_low_price = limit_price_obj.prev_settlement * (1 - limit_rate)

        if bar.high >= trad_price >= bar.low:
            return True
        else:
            if trad_price > limit_high_price:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_PRICE_TOO_HIGH %
                    (str(order_result.price), str(limit_high_price)))
                return False
            elif trad_price < limit_low_price:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_PRICE_TOO_LOW %
                    (str(order_result.price), str(limit_high_price)))
                return False

        return True
